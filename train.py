import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import kagglehub
import os
import joblib
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import classification_report, f1_score

# ── Device ────────────────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ── Dataset class ─────────────────────────────────────────────────────────────
class LoanDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ── Model architecture ────────────────────────────────────────────────────────
class LoanGradeMLP(nn.Module):
    def __init__(self, input_dim, num_classes=7):
        super(LoanGradeMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.network(x)

# ── Pipeline class ────────────────────────────────────────────────────────────
class LoanGradeNNPredictor:
    def __init__(self, num_classes=7):
        self.dv          = DictVectorizer(sparse=False)
        self.scaler      = StandardScaler()
        self.le          = LabelEncoder()
        self.num_classes = num_classes
        self.input_dim   = None
        self.model       = None
        self.device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _make_loader(self, X_tensor, y_tensor, shuffle=False):
        return DataLoader(
            LoanDataset(X_tensor, y_tensor),
            batch_size=512,
            shuffle=shuffle,
            pin_memory=True if torch.cuda.is_available() else False,
            num_workers=0
        )

    def _prepare_X(self, records, fit=False):
        if isinstance(records, pd.DataFrame):
            records = records.to_dict(orient='records')
        if fit:
            X = self.dv.fit_transform(records)
        else:
            X = self.dv.transform(records)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        if fit:
            X = self.scaler.fit_transform(X)
        else:
            X = self.scaler.transform(X)
        return X

    def fit(self, df_train, df_val, num_epochs=50):
        # Prepare features
        X_train = self._prepare_X(df_train.drop('grade', axis=1), fit=True)
        X_val   = self._prepare_X(df_val.drop('grade', axis=1),   fit=False)

        # Encode target
        y_train_enc = self.le.fit_transform(df_train['grade'])
        y_val_enc   = self.le.transform(df_val['grade'])

        # Build model
        self.input_dim = X_train.shape[1]
        self.model = LoanGradeMLP(self.input_dim, self.num_classes).to(self.device)

        # Tensors and loaders
        train_loader = self._make_loader(
            torch.FloatTensor(X_train), torch.LongTensor(y_train_enc), shuffle=True
        )
        val_loader = self._make_loader(
            torch.FloatTensor(X_val), torch.LongTensor(y_val_enc)
        )

        # Loss, optimizer, scheduler
        class_counts  = np.bincount(y_train_enc)
        class_weights = torch.FloatTensor(1.0 / class_counts).to(self.device)
        class_weights = class_weights / class_weights.sum()

        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=3, factor=0.5
        )

        best_val_loss = float('inf')

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
            print(f"Epoch {epoch+1:02d} | Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}  Acc: {val_acc:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # Store best weights inside the object
                self._best_weights = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                print("  -> Best model saved")

        # Restore best weights
        self.model.load_state_dict(self._best_weights)
        self.model.eval()
        return self

    def predict(self, df):
        """Accept a dataframe or list of dicts"""
        X = self._prepare_X(df)
        loader = self._make_loader(
            torch.FloatTensor(X),
            torch.LongTensor(np.zeros(len(X), dtype=int))  # dummy labels
        )
        self.model.eval()
        all_preds = []
        with torch.no_grad():
            for X_batch, _ in loader:
                X_batch = X_batch.to(self.device, non_blocking=True)
                preds   = self.model(X_batch).argmax(1).cpu().numpy()
                all_preds.extend(preds)
        return self.le.inverse_transform(all_preds)

    def evaluate(self, df, label='Evaluation'):
        y_true = df['grade'].values
        y_pred = self.predict(df.drop('grade', axis=1))
        print(f"\n── {label} ──")
        print(classification_report(y_true, y_pred, target_names=list('ABCDEFG')))
        print(f"Weighted F1: {f1_score(y_true, y_pred, average='weighted'):.4f}")
        print(f"Macro F1:    {f1_score(y_true, y_pred, average='macro'):.4f}")

# ── Data loading ──────────────────────────────────────────────────────────────
def load_data(nrows=75000):
    path = kagglehub.dataset_download("adarshsng/lending-club-loan-data-csv")
    df = pd.read_csv(os.path.join(path, 'loan.csv'), nrows=nrows, low_memory=False)

    drop_missing_columns = list(df.columns[df.isnull().sum() > 50000])
    df.drop(columns=drop_missing_columns, inplace=True)

    selector   = VarianceThreshold(threshold=0.01)
    numeric_cols = df.select_dtypes(include='number').columns
    selector.fit(df[numeric_cols])
    kept = numeric_cols[selector.get_support()]
    df   = df[kept.tolist() + list(df.select_dtypes(include='object').columns)]

    for var in list(df.select_dtypes(include='object').columns):
        if df[var].nunique() > 10:
            df = df.drop(columns=var)


    leakage_cols = [
        # Post-origination payment data
        'total_pymnt', 'total_pymnt_inv', 'total_rec_prncp',
        'total_rec_int', 'total_rec_late_fee', 'recoveries',
        'collection_recovery_fee', 'last_pymnt_amnt',
        'out_prncp', 'out_prncp_inv', 'disbursement_method',
        
        # Loan status after origination
        'loan_status',
        
        # Payment plan info
        'pymnt_plan',
        
        # Next payment date (post-origination)
    '   pymnt_plan', 'last_pymnt_d',
        
        # Hardship/settlement (post-origination)
        'hardship_flag', 'debt_settlement_flag',
        
        # Already removing these
        'int_rate', 'sub_grade'
    ]
    df = df.drop(columns=[c for c in leakage_cols if c in df.columns])

    df = df[df['grade'].notna()]

    return df

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print('Loading data...')
    df = load_data(nrows=500000)

    df_full_train, df_test = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df['grade']
    )
    df_train, df_val = train_test_split(
        df_full_train, test_size=0.25, random_state=42, stratify=df_full_train['grade']
    )

    print(f"Train: {df_train.shape} | Val: {df_val.shape} | Test: {df_test.shape}")

    # Train
    predictor = LoanGradeNNPredictor(num_classes=7)
    predictor.fit(df_train, df_val, num_epochs=50)

    # Evaluate
    predictor.evaluate(df_train, label='Train')
    predictor.evaluate(df_val,   label='Validation')
    predictor.evaluate(df_test,  label='Test')

    # Save — one file only
    print('\nSaving model...')
    joblib.dump(predictor, 'loan_grade_nn_model.pkl')
    print('Saved: loan_grade_nn_model.pkl')
    print('Done.')
    print('Printing feature names...')
    print(predictor.dv.feature_names_)