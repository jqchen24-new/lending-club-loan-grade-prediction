import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import joblib
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction import DictVectorizer
import torch.nn.functional as F

class LoanDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class LoanGradeMLP(nn.Module):
    def __init__(self, input_dim, num_classes=7):
        super(LoanGradeMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256),       nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128),       nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        return self.network(x)

class LoanGradePredictor:
    def __init__(self, device=None):
        self.dv        = DictVectorizer(sparse=False)
        self.scaler    = StandardScaler()
        self.le        = LabelEncoder()
        self.input_dim = None
        self.model     = None
        self.device    = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _make_loader(self, X_tensor, y_tensor, shuffle=False):
        return DataLoader(
            LoanDataset(X_tensor, y_tensor),
            batch_size=512, shuffle=shuffle,
            pin_memory=True if torch.cuda.is_available() else False,
            num_workers=0
        )

    def _prepare_X(self, records, fit=False):
        if isinstance(records, pd.DataFrame):
            records = records.to_dict(orient='records')
        X = self.dv.fit_transform(records) if fit else self.dv.transform(records)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        X = self.scaler.fit_transform(X) if fit else self.scaler.transform(X)
        return X

    def transform(self, records):
        """
        Transform raw records/DataFrame into the model's numeric input space.
        Returns (X, feature_names) where X is a dense numpy array.
        """
        X = self._prepare_X(records, fit=False)
        feature_names = getattr(self.dv, "feature_names_", None)
        if feature_names is None:
            feature_names = []
        return X, list(feature_names)

    def predict_proba(self, df):
        X = self._prepare_X(df)
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            logits = self.model(X_tensor)
            probs = F.softmax(logits, dim=1).cpu().numpy()
        return probs

    def predict(self, df):
        X = self._prepare_X(df)
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            preds    = self.model(X_tensor).argmax(1).cpu().numpy()
        return self.le.inverse_transform(preds)

    def save(self, path='predictor.pkl'):
        self.model  = self.model.cpu()
        self.device = torch.device('cpu')
        joblib.dump(self, path)
        print(f"Saved: {path}")

    @classmethod
    def load(cls, path='predictor.pkl'):
        predictor        = joblib.load(path)
        predictor.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        predictor.model  = predictor.model.to(predictor.device)
        predictor.model.eval()
        return predictor
    
    def fit(self, df_train, df_val, num_epochs=50):
        X_train = self._prepare_X(df_train.drop('grade', axis=1), fit=True)
        X_val   = self._prepare_X(df_val.drop('grade', axis=1),   fit=False)

        y_train_enc = self.le.fit_transform(df_train['grade'])
        y_val_enc   = self.le.transform(df_val['grade'])

        self.input_dim = X_train.shape[1]
        self.model     = LoanGradeMLP(self.input_dim, num_classes=7).to(self.device)

        train_loader = self._make_loader(
            torch.FloatTensor(X_train), torch.LongTensor(y_train_enc), shuffle=True
        )
        val_loader = self._make_loader(
            torch.FloatTensor(X_val), torch.LongTensor(y_val_enc)
        )

        class_counts  = np.bincount(y_train_enc)
        class_weights = torch.FloatTensor(1.0 / class_counts).to(self.device)
        class_weights = class_weights / class_weights.sum()

        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=3, factor=0.5
        )

        best_val_loss    = float('inf')
        best_weights     = None
        patience_counter = 0
        early_stop       = 10

        for epoch in range(num_epochs):
            # Train
            self.model.train()
            total_loss, correct = 0, 0
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device, non_blocking=True)
                y_batch = y_batch.to(self.device, non_blocking=True)
                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss    = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                correct    += (outputs.argmax(1) == y_batch).sum().item()
            train_loss = total_loss / len(train_loader)
            train_acc  = correct / len(train_loader.dataset)

            # Eval
            self.model.eval()
            total_loss, correct = 0, 0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(self.device, non_blocking=True)
                    y_batch = y_batch.to(self.device, non_blocking=True)
                    outputs = self.model(X_batch)
                    loss    = criterion(outputs, y_batch)
                    total_loss += loss.item()
                    correct    += (outputs.argmax(1) == y_batch).sum().item()
            val_loss = total_loss / len(val_loader)
            val_acc  = correct / len(val_loader.dataset)

            scheduler.step(val_loss)
            print(f"Epoch {epoch+1:02d} | Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.4f}")

            if val_loss < best_val_loss:
                best_val_loss    = val_loss
                best_weights     = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
                print("  -> Best model saved")
            else:
                patience_counter += 1
                if patience_counter >= early_stop:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        # Restore best weights
        self.model.load_state_dict(best_weights)
        self.model.eval()
        return self

    def evaluate(self, df, label='Evaluation'):
        from sklearn.metrics import classification_report, f1_score
        y_true = df['grade'].values
        y_pred = self.predict(df.drop('grade', axis=1))
        print(f"\n── {label} ──")
        print(classification_report(y_true, y_pred, target_names=list('ABCDEFG')))
        print(f"Weighted F1: {f1_score(y_true, y_pred, average='weighted'):.4f}")
        print(f"Macro F1:    {f1_score(y_true, y_pred, average='macro'):.4f}")