from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import shap


def _direction(shap_value: float) -> str:
    if shap_value > 0:
        return "increases risk"
    if shap_value < 0:
        return "decreases risk"
    return "neutral"


def _coerce_raw_scalar(raw: Any) -> Any:
    """Normalize JSON-friendly scalars; keep non-numeric strings as-is."""
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, (int, float, np.integer, np.floating)):
        return float(raw)
    if isinstance(raw, str):
        try:
            return float(raw)
        except ValueError:
            return raw
    return raw


def _display_value(feature: str, raw_input: Dict[str, Any]) -> Any:
    """
    Only values the caller supplied in raw_input — never scaled / transformed
    model inputs. For one-hot rows ``col=level``, returns the user's value for
    ``col`` when it matches this ``level``, else None. Missing keys -> None.
    """
    if "=" in feature:
        base, _, level = feature.partition("=")
        raw = raw_input.get(base)
        if raw is None:
            return None
        if str(raw).strip() != level.strip():
            return None
        return _coerce_raw_scalar(raw)

    raw = raw_input.get(feature)
    if raw is None:
        return None
    return _coerce_raw_scalar(raw)


@dataclass(frozen=True)
class ShapConfig:
    background_path: str = "background.csv"
    default_top_n: int = 10
    default_max_evals: int = 256


class ShapExplainer:
    def __init__(self, predictor, config: Optional[ShapConfig] = None):
        self.predictor = predictor
        self.config = config or ShapConfig()

        bg_df = pd.read_csv(self.config.background_path)
        background_X = bg_df.to_numpy(dtype=np.float32, copy=False)

        feature_names = getattr(self.predictor.dv, "feature_names_", None)
        if feature_names is None:
            raise RuntimeError("Predictor is missing dv.feature_names_ (was it trained/fitted?)")
        self.feature_names = list(feature_names)

        if background_X.shape[1] != len(self.feature_names):
            raise RuntimeError(
                f"background.csv has {background_X.shape[1]} columns but dv has "
                f"{len(self.feature_names)} features. Regenerate background.csv using the same predictor."
            )

        def model_f(X: np.ndarray) -> np.ndarray:
            X = np.asarray(X, dtype=np.float32)
            import torch
            with torch.no_grad():
                X_tensor = torch.from_numpy(X).to(self.predictor.device)
                logits = self.predictor.model(X_tensor)
                probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
            return probs

        masker = shap.maskers.Independent(background_X)
        self.explainer = shap.Explainer(model_f, masker, algorithm="permutation")

    def _shap_analysis(
        self,
        raw_input: Dict[str, Any],
        top_n: Optional[int] = None,
        max_evals: Optional[int] = None,
    ) -> Dict[str, Any]:
        top_n = int(top_n or self.config.default_top_n)
        requested_max_evals = int(max_evals or self.config.default_max_evals)
        max_evals = requested_max_evals

        input_df = pd.DataFrame([raw_input])
        X, _ = self.predictor.transform(input_df)
        probs = self.predictor.predict_proba(input_df)[0]
        pred_class_idx = int(np.argmax(probs))
        pred_grade = self.predictor.le.inverse_transform([pred_class_idx])[0]

        try:
            exp = self.explainer(X, max_evals=max_evals)
        except ValueError as e:
            msg = str(e)
            if "too low for the Permutation explainer" not in msg:
                raise
            m = re.search(r"=\s*(\d+)\s*!", msg)
            if not m:
                raise
            max_evals = int(m.group(1))
            exp = self.explainer(X, max_evals=max_evals)

        values = np.asarray(exp.values)
        if values.ndim == 3:
            shap_row = values[0, :, pred_class_idx]
        else:
            shap_row = values[0, :]

        x_row = X[0, :]
        order = np.argsort(np.abs(shap_row))[::-1][:top_n]

        return {
            "x_row": x_row,
            "shap_row": shap_row,
            "order": order,
            "pred_class_idx": pred_class_idx,
            "pred_grade": str(pred_grade),
            "probs": probs,
            "max_evals": max_evals,
            "requested_max_evals": requested_max_evals,
            "exp": exp,
        }

    def build_explanation_rows(
        self,
        raw_input: Dict[str, Any],
        top_n: Optional[int] = None,
        max_evals: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        s = self._shap_analysis(raw_input, top_n=top_n, max_evals=max_evals)
        shap_row, order = s["shap_row"], s["order"]
        rows: List[Dict[str, Any]] = []
        for i in order:
            name = self.feature_names[int(i)]
            sv = float(shap_row[int(i)])
            val = _display_value(name, raw_input)
            rows.append(
                {
                    "feature": name,
                    "value": val,
                    "shap_value": sv,
                    "direction": _direction(sv),
                }
            )
        return rows

    def explain_one(
        self,
        raw_input: Dict[str, Any],
        top_n: Optional[int] = None,
        max_evals: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Legacy full SHAP payload (debug / notebooks). API uses build_explanation_rows."""
        s = self._shap_analysis(raw_input, top_n=top_n, max_evals=max_evals)
        x_row, shap_row, order = s["x_row"], s["shap_row"], s["order"]
        pred_class_idx = s["pred_class_idx"]
        probs = s["probs"]
        max_evals = s["max_evals"]
        requested_max_evals = s["requested_max_evals"]
        exp = s["exp"]

        top_features: List[Dict[str, Any]] = [
            {
                "name": self.feature_names[int(i)],
                "value": float(x_row[int(i)]),
                "shap_value": float(shap_row[int(i)]),
            }
            for i in order
        ]

        base_values = getattr(exp, "base_values", None)
        expected_value: Optional[float] = None
        if base_values is not None:
            base_arr = np.asarray(base_values)
            if base_arr.ndim == 2:
                expected_value = float(base_arr[0, pred_class_idx])
            elif base_arr.ndim == 1:
                expected_value = float(base_arr[0])

        classes = list(self.predictor.le.classes_)
        proba_by_class = {str(classes[i]): float(probs[i]) for i in range(len(classes))}

        out: Dict[str, Any] = {
            "predicted_grade": s["pred_grade"],
            "predicted_class_index": pred_class_idx,
            "predicted_proba": proba_by_class,
            "expected_value": expected_value,
            "top_features": top_features,
            "algorithm": "permutation",
            "max_evals": max_evals,
        }
        if max_evals != requested_max_evals:
            out["max_evals_requested"] = requested_max_evals
        return out
