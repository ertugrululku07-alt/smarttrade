"""
Meta-Label Predictor v1.2 (Unified Model Loading)

v1.2 Fixes:
  - MODEL_DIR path unified (uses ai/models)
  - Model format: dict with 'model', 'features', 'meta' keys (v1.2 trainer compat)
  - Legacy tuple format (model, features, meta) still supported
  - Mutable default argument fixed
  - get_regime_stats key format corrected
  - Fallback model search without timeframe suffix
  - NaN/Inf protection in feature extraction
"""

import os
import numpy as np
import pandas as pd
import joblib
from typing import Dict, Optional, Tuple, List

from ai.regime_detector import Regime

# Model directory: ai/models (same level as this file's parent)
MODEL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models"
)


class MetaPredictor:
    """
    Loads trained meta-models and performs prediction.
    Supports both dict format (v1.2 trainer) and legacy tuple format.
    """

    def __init__(self, timeframes: Optional[List[str]] = None):
        if timeframes is None:
            timeframes = ["15m", "1h", "4h"]
        self.timeframes = timeframes
        self.models: Dict[str, Dict] = {}
        self._load_models()

    def _load_models(self):
        """Load all regime + timeframe models with fallback."""
        loaded = 0

        if not os.path.exists(MODEL_DIR):
            print(f"  [WARN] Model directory not found: {MODEL_DIR}")
            return

        for tf in self.timeframes:
            for regime in Regime:
                # Primary path: meta_{regime}_{tf}.joblib
                model_path = os.path.join(
                    MODEL_DIR, f"meta_{regime.value}_{tf}.joblib"
                )

                # Fallback chain for different save locations
                if not os.path.exists(model_path):
                    candidates = [
                        # models/meta/meta_{regime}_{tf}.joblib (trainer CLI default)
                        os.path.join(MODEL_DIR, "meta", f"meta_{regime.value}_{tf}.joblib"),
                        # models/meta_{regime}.joblib (no tf suffix)
                        os.path.join(MODEL_DIR, f"meta_{regime.value}.joblib"),
                        # models/meta/meta_{regime}.joblib (subdir, no tf)
                        os.path.join(MODEL_DIR, "meta", f"meta_{regime.value}.joblib"),
                    ]
                    found = False
                    for candidate in candidates:
                        if os.path.exists(candidate):
                            model_path = candidate
                            found = True
                            break
                    if not found:
                        continue

                try:
                    raw = joblib.load(model_path)
                    model, features, meta = self._parse_model_data(raw)

                    if model is None:
                        print(f"  [WARN] Invalid model data: {model_path}")
                        continue

                    key = f"{regime.value}_{tf}"
                    threshold = 0.60
                    if isinstance(meta, dict):
                        # Support both 'best_threshold' and nested metrics
                        threshold = max(0.55, meta.get('best_threshold', 0.60))
                        if 'metrics' in meta and isinstance(meta['metrics'], dict):
                            # v1.2 trainer stores metrics inside model_data
                            pass

                    self.models[key] = {
                        'model': model,
                        'features': list(features) if features else [],
                        'meta': meta if isinstance(meta, dict) else {},
                        'threshold': threshold,
                    }
                    loaded += 1

                except Exception as e:
                    print(f"  [ERROR] Meta-model load failed ({regime.value}_{tf}): {e}")

        print(f"  [MODEL] MetaPredictor: {loaded} model(s) loaded ({', '.join(self.timeframes)})")

    def _parse_model_data(self, raw):
        """
        Parse model data from joblib file.
        Supports:
          - Dict format (v1.2): {'model': ..., 'features': [...], 'meta': {...}}
          - Tuple format (legacy): (model, features, meta)
          - Raw model object (oldest format)
        """
        if isinstance(raw, dict):
            model = raw.get('model')
            features = raw.get('features', [])
            meta = raw.get('metadata', raw.get('meta', {}))
            # If metrics are stored separately, merge them
            if 'metrics' in raw and isinstance(raw['metrics'], dict):
                if isinstance(meta, dict):
                    meta.update(raw['metrics'])
            return model, features, meta

        if isinstance(raw, (tuple, list)) and len(raw) >= 2:
            model = raw[0]
            features = raw[1] if len(raw) > 1 else []
            meta = raw[2] if len(raw) > 2 else {}
            return model, features, meta

        # Raw model object
        features = getattr(raw, 'feature_names_in_', [])
        if hasattr(raw, 'get_booster'):
            try:
                features = raw.get_booster().feature_names or features
            except Exception:
                pass
        return raw, features, {}

    def predict(
        self,
        df: pd.DataFrame,
        regime: Regime,
        signal_direction: str,
        signal_confidence: float,
        timeframe: str = "1h"
    ) -> Tuple[float, bool, float, str]:
        """
        Meta-model prediction with timeframe awareness.

        Returns: (meta_confidence, should_trade, threshold, reason)
        """
        regime_val = regime.value if isinstance(regime, Regime) else str(regime)
        key = f"{regime_val}_{timeframe}"

        # TF fallback: 1h yoksa 4h, 4h yoksa 1h dene
        if key not in self.models:
            tf_fallback_order = ['4h', '1h', '15m']
            found_key = None
            for alt_tf in tf_fallback_order:
                alt_key = f"{regime_val}_{alt_tf}"
                if alt_key in self.models:
                    found_key = alt_key
                    break
            if found_key:
                key = found_key
            else:
                fallback_conf = signal_confidence * 0.80
                return fallback_conf, fallback_conf > 0.55, 0.55, f"No model for {regime_val}"

        model_info = self.models[key]
        model = model_info['model']
        feature_cols = model_info['features']
        threshold = model_info['threshold']

        try:
            # Last bar data
            last_row = df.iloc[-1]

            # Meta-features (signal info)
            meta_values = {
                'signal_is_long': float(signal_direction == 'LONG'),
                'signal_confidence': float(signal_confidence),
                'regime_trending': float(regime_val == 'trending'),
                'regime_mean_rev': float(regime_val == 'mean_reverting'),
                'regime_high_vol': float(regime_val == 'high_volatile'),
                'regime_low_vol': float(regime_val == 'low_volatile'),
            }

            # Build feature vector in model's expected order
            row_data = {}
            missing_features = []

            for col in feature_cols:
                if col in meta_values:
                    row_data[col] = meta_values[col]
                elif col in last_row.index:
                    val = last_row[col]
                    if pd.isna(val) or np.isinf(val):
                        row_data[col] = 0.0
                    else:
                        row_data[col] = float(val)
                else:
                    row_data[col] = 0.0
                    missing_features.append(col)

            if missing_features and len(missing_features) > len(feature_cols) * 0.5:
                # Too many features missing - model prediction unreliable
                fallback_conf = signal_confidence * 0.75
                return (
                    fallback_conf, fallback_conf > 0.55, 0.55,
                    f"Too many features missing ({len(missing_features)}/{len(feature_cols)})"
                )

            # Build DataFrame with correct column order
            X = pd.DataFrame([row_data], columns=feature_cols)
            X = X.astype(np.float32)

            # Replace any remaining NaN/Inf
            X = X.replace([np.inf, -np.inf], 0.0).fillna(0.0)

            # Prediction
            proba = model.predict_proba(X)[0]
            meta_conf = float(proba[1]) if len(proba) >= 2 else float(proba[0])

            should_trade = meta_conf >= threshold

            return meta_conf, should_trade, threshold, "OK"

        except Exception as e:
            print(f"  [ERROR] MetaPredictor prediction ({regime_val}): {e}")
            fallback_conf = signal_confidence * 0.70
            return fallback_conf, fallback_conf > 0.55, 0.55, f"Error: {str(e)[:80]}"

    def get_regime_stats(self) -> Dict:
        """Statistics for loaded models."""
        stats = {}

        # Loaded models
        for key, info in self.models.items():
            meta = info.get('meta', {})
            stats[key] = {
                'loaded': True,
                'precision': meta.get('precision', '?'),
                'recall': meta.get('recall', '?'),
                'f1': meta.get('f1', '?'),
                'threshold': info.get('threshold', 0.60),
                'n_features': len(info.get('features', [])),
            }

        # Missing models
        for tf in self.timeframes:
            for regime in Regime:
                key = f"{regime.value}_{tf}"
                if key not in stats:
                    stats[key] = {'loaded': False}

        return stats

    @property
    def is_ready(self) -> bool:
        return len(self.models) > 0

    @property
    def loaded_regimes(self) -> list:
        return list(self.models.keys())
