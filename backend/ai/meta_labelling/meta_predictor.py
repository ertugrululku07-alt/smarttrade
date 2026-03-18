"""
Meta-Label Predictor v2.0 (LightGBM + Legacy XGBoost)

v2.0 Changes:
  - LightGBM model support (v3.0 trainer)
  - XGBoost backward compatibility preserved
  - Automatic model type detection (lgb vs xgb)

v1.2 preserved:
  - MODEL_DIR path unified (uses ai/models)
  - Legacy tuple format (model, features, meta) still supported
  - Fallback model search without timeframe suffix
  - NaN/Inf protection in feature extraction
"""

import os
import numpy as np
import pandas as pd
import joblib
from typing import Dict, Optional, Tuple, List

from ai.regime_detector import Regime

# ICT feature extraction (lazy import)
_ict_core = None
_ict_features = None

def _ensure_ict_imports():
    """Lazy import for ict_core and ict_features."""
    global _ict_core, _ict_features
    if _ict_core is None:
        try:
            from ai import ict_core as ic
            from ai import ict_features as iff
            _ict_core = ic
            _ict_features = iff
        except ImportError:
            pass
    return _ict_core is not None

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
        self.last_debug: Optional[Dict] = None
        self._load_models()

    def _load_models(self):
        """Load all regime + timeframe models with fallback."""
        loaded = 0

        if not os.path.exists(MODEL_DIR):
            print(f"  [WARN] Model directory not found: {MODEL_DIR}")
            return

        # v3.0: merged regimes (directional, volatile) + standard regimes
        all_regime_values = [r.value for r in Regime] + ['directional', 'volatile']

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
                        # best_threshold from training maximizes WR but can be too strict
                        # for live trading (e.g. AUC=0.66 models can't output 0.78+).
                        # Cap at MAX_LIVE_THRESHOLD to maintain reasonable pass rate.
                        MAX_LIVE_THRESHOLD = 0.65
                        raw_thr = meta.get('best_threshold', 0.60)
                        threshold = max(0.55, min(raw_thr, MAX_LIVE_THRESHOLD))
                        if 'metrics' in meta and isinstance(meta['metrics'], dict):
                            pass

                    # Detect model type
                    model_type = type(model).__name__
                    self.models[key] = {
                        'model': model,
                        'features': list(features) if features else [],
                        'meta': meta if isinstance(meta, dict) else {},
                        'threshold': threshold,
                        'model_type': model_type,
                    }
                    loaded += 1

                except Exception as e:
                    print(f"  [ERROR] Meta-model load failed ({regime.value}_{tf}): {e}")

        # v3.0: Load merged regime models (directional, volatile)
        for tf in self.timeframes:
            for merged_name in ['directional', 'volatile']:
                key = f"{merged_name}_{tf}"
                if key in self.models:
                    continue
                candidates = [
                    os.path.join(MODEL_DIR, f"meta_{merged_name}_{tf}.joblib"),
                    os.path.join(MODEL_DIR, "meta", f"meta_{merged_name}_{tf}.joblib"),
                ]
                for cpath in candidates:
                    if os.path.exists(cpath):
                        try:
                            raw = joblib.load(cpath)
                            model, features, meta = self._parse_model_data(raw)
                            if model is None:
                                continue
                            threshold = 0.60
                            if isinstance(meta, dict):
                                MAX_LIVE_THRESHOLD = 0.65
                                raw_thr = meta.get('best_threshold', 0.60)
                                threshold = max(0.55, min(raw_thr, MAX_LIVE_THRESHOLD))
                            self.models[key] = {
                                'model': model,
                                'features': list(features) if features else [],
                                'meta': meta if isinstance(meta, dict) else {},
                                'threshold': threshold,
                                'model_type': type(model).__name__,
                            }
                            loaded += 1
                        except Exception as e:
                            print(f"  [ERROR] Merged model load failed ({key}): {e}")
                        break

        types = set(v.get('model_type', '?') for v in self.models.values())
        print(f"  [MODEL] MetaPredictor: {loaded} model(s) loaded ({', '.join(self.timeframes)}) [{', '.join(types)}]")

    def _parse_model_data(self, raw):
        """
        Parse model data from joblib file.
        Supports:
          - Dict format (v1.2): {'model': ..., 'features': [...], 'meta': {...}}
          - Tuple format (legacy): (model, features, meta)
          - Raw model object (oldest format)
          - Both LightGBM and XGBoost models (sklearn API compatible)
        """
        if isinstance(raw, dict):
            model = raw.get('model')
            features = raw.get('features', [])
            meta = raw.get('metadata', raw.get('meta', {}))
            if 'metrics' in raw and isinstance(raw['metrics'], dict):
                if isinstance(meta, dict):
                    meta.update(raw['metrics'])
            return model, features, meta

        if isinstance(raw, (tuple, list)) and len(raw) >= 2:
            model = raw[0]
            features = raw[1] if len(raw) > 1 else []
            meta = raw[2] if len(raw) > 2 else {}
            return model, features, meta

        # Raw model object (LightGBM or XGBoost)
        features = getattr(raw, 'feature_names_in_', [])
        # XGBoost booster fallback
        if hasattr(raw, 'get_booster'):
            try:
                features = raw.get_booster().feature_names or features
            except Exception:
                pass
        # LightGBM booster fallback
        if hasattr(raw, 'feature_name_'):
            try:
                features = raw.feature_name_ or features
            except Exception:
                pass
        return raw, features, {}

    def predict(
        self,
        df: pd.DataFrame,
        regime: Regime,
        signal_direction: str,
        signal_confidence: float,
        timeframe: str = "1h",
        debug: bool = False,
    ) -> Tuple[float, bool, float, str]:
        """
        Meta-model prediction with timeframe awareness.

        Returns: (meta_confidence, should_trade, threshold, reason)
        When debug=True, self.last_debug is populated with feature diagnostics.
        """
        regime_val = regime.value if isinstance(regime, Regime) else str(regime)
        key = f"{regime_val}_{timeframe}"
        self.last_debug = None  # reset

        # v3.0: Merged regime fallback map
        MERGED_FALLBACK = {
            'trending': 'directional',
            'mean_reverting': 'directional',
            'high_volatile': 'volatile',
            'low_volatile': 'volatile',
        }

        # TF fallback: 1h yoksa 4h, 4h yoksa 1h dene
        # Sonra merged regime fallback dene
        if key not in self.models:
            tf_fallback_order = ['4h', '1h', '15m']
            found_key = None
            # 1) Aynı regime, farklı TF
            for alt_tf in tf_fallback_order:
                alt_key = f"{regime_val}_{alt_tf}"
                if alt_key in self.models:
                    found_key = alt_key
                    break
            # 2) Merged regime fallback
            if not found_key and regime_val in MERGED_FALLBACK:
                merged_val = MERGED_FALLBACK[regime_val]
                merged_key = f"{merged_val}_{timeframe}"
                if merged_key in self.models:
                    found_key = merged_key
                else:
                    for alt_tf in tf_fallback_order:
                        alt_key = f"{merged_val}_{alt_tf}"
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

            # ICT features: model ict_* istiyorsa canlı hesapla
            ict_values = {}
            needs_ict = any(c.startswith('ict_') for c in feature_cols)
            if needs_ict and _ensure_ict_imports():
                try:
                    cp = float(df['close'].iloc[-1])
                    atr_val = float(df['atr'].iloc[-1]) if 'atr' in df.columns else cp * 0.01
                    if np.isnan(atr_val) or atr_val <= 0:
                        atr_val = cp * 0.01
                    analysis = _ict_core.analyze(df.tail(80), signal_direction, 3, 2)
                    ict_values = _ict_features.extract_ict_features(
                        analysis, cp, atr_val, signal_direction
                    )
                except Exception as e_ict:
                    print(f"    [WARN] ICT feature extraction failed: {e_ict}")

            # Build feature vector in model's expected order
            row_data = {}
            missing_features = []
            zero_features = []
            ok_features = []
            feature_details = {}

            for col in feature_cols:
                if col in meta_values:
                    row_data[col] = meta_values[col]
                    feature_details[col] = {"value": meta_values[col], "source": "signal"}
                elif col in ict_values:
                    row_data[col] = ict_values[col]
                    if ict_values[col] != 0.0:
                        ok_features.append(col)
                    else:
                        zero_features.append(col)
                    feature_details[col] = {"value": round(ict_values[col], 4), "source": "ict"}
                elif col in last_row.index:
                    val = last_row[col]
                    if pd.isna(val) or np.isinf(val):
                        row_data[col] = 0.0
                        zero_features.append(col)
                        feature_details[col] = {"value": 0.0, "source": "nan_replaced"}
                    else:
                        row_data[col] = float(val)
                        if float(val) == 0.0:
                            zero_features.append(col)
                        else:
                            ok_features.append(col)
                        feature_details[col] = {"value": round(float(val), 4), "source": "df"}
                else:
                    row_data[col] = 0.0
                    missing_features.append(col)
                    feature_details[col] = {"value": 0.0, "source": "MISSING"}

            # Debug output
            n_total = len(feature_cols)
            n_missing = len(missing_features)
            n_zero = len(zero_features)
            n_ok = len(ok_features)
            n_signal = len([c for c in feature_cols if c in meta_values])

            debug_info = {
                "model_key": key,
                "total_features": n_total,
                "ok_nonzero": n_ok,
                "signal_features": n_signal,
                "zero_or_nan": n_zero,
                "missing_from_df": n_missing,
                "missing_list": missing_features[:20],
                "zero_list": zero_features[:20],
                "feature_values": {k: v for k, v in feature_details.items()},
            }
            self.last_debug = debug_info

            # Console log (always)
            print(f"\n  [META DEBUG] {key} | OK:{n_ok} Signal:{n_signal} Zero:{n_zero} Missing:{n_missing}/{n_total}")
            if missing_features:
                print(f"    MISSING: {missing_features[:10]}")
            if zero_features:
                print(f"    ZERO: {zero_features[:10]}")

            if n_missing > n_total * 0.5:
                fallback_conf = signal_confidence * 0.75
                return (
                    fallback_conf, fallback_conf > 0.55, 0.55,
                    f"Too many features missing ({n_missing}/{n_total})"
                )

            # Build DataFrame with correct column order
            X = pd.DataFrame([row_data], columns=feature_cols)
            X = X.astype(np.float32)
            X = X.replace([np.inf, -np.inf], 0.0).fillna(0.0)

            # Prediction
            proba = model.predict_proba(X)[0]
            meta_conf = float(proba[1]) if len(proba) >= 2 else float(proba[0])
            should_trade = meta_conf >= threshold

            print(f"    → Prediction: {meta_conf:.4f} (thr={threshold:.2f}) {'✅ PASS' if should_trade else '❌ REJECT'}")

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
