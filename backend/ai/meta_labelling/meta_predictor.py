"""
Meta-Label Predictor v1.1

Düzeltme: Feature mismatch hatası çözüldü
  - Eksik feature'lar 0.0 ile doldurulur
  - XGBoost'un beklediği feature sırası korunur
"""

import os
import numpy as np
import pandas as pd
import joblib
from typing import Dict, Optional, Tuple

from ai.regime_detector import Regime

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")


class MetaPredictor:
    """
    Eğitilmiş meta-modelleri yükler ve prediction yapar.
    """

    def __init__(self, timeframes: list = ["15m", "1h"]):
        self.timeframes = timeframes
        self.models: Dict[str, Dict] = {} # Key format: "regime_timeframe"
        self._load_models()

    def _load_models(self):
        """Tüm rejim ve timeframe modellerini yükle"""
        loaded = 0
        for tf in self.timeframes:
            for regime in Regime:
                model_path = os.path.join(
                    MODEL_DIR, f"meta_{regime.value}_{tf}.joblib"
                )
                if os.path.exists(model_path):
                    try:
                        model_data = joblib.load(model_path)
                        if isinstance(model_data, tuple) and len(model_data) == 3:
                            model, features, meta = model_data
                        else:
                            model = model_data
                            features = getattr(model, "feature_names_in_", [])
                            meta = {}

                        key = f"{regime.value}_{tf}"
                        self.models[key] = {
                            'model': model,
                            'features': features,
                            'meta': meta,
                            'threshold': max(0.60, meta.get('best_threshold', 0.60)),
                        }
                        loaded += 1
                    except Exception as e:
                        print(f"  ERROR: Meta-model yuklenemedi ({regime.value} - {tf}): {e}")

        print(f"  [MODEL] MetaPredictor: {loaded} model yuklendi ({', '.join(self.timeframes)})")

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
        """
        regime_val = regime.value if isinstance(regime, Regime) else regime
        key = f"{regime_val}_{timeframe}"

        if key not in self.models:
            fallback_conf = signal_confidence * 0.80
            return fallback_conf, fallback_conf > 0.55, 0.55, f"No model for {key}"

        model_info = self.models[key]
        model = model_info['model']
        feature_cols = model_info['features']
        threshold = model_info['threshold']

        try:
            # ── Son bar'ın verilerini al ─────────────────────
            last_row = df.iloc[-1]

            # ── Meta-feature'ları hazırla ────────────────────
            meta_values = {
                'signal_is_long': float(signal_direction == 'LONG'),
                'signal_confidence': float(signal_confidence),
                'regime_trending': float(regime_val == 'trending'),
                'regime_mean_rev': float(regime_val == 'mean_reverting'),
                'regime_high_vol': float(regime_val == 'high_volatile'),
                'regime_low_vol': float(regime_val == 'low_volatile'),
            }

            # ── Modelin beklediği TÜM feature'ları sırayla oluştur ──
            row_data = {}
            for col in feature_cols:
                if col in meta_values:
                    # Meta-feature (sinyal bilgisi)
                    row_data[col] = meta_values[col]
                elif col in last_row.index:
                    # DataFrame'den al
                    val = last_row[col]
                    if pd.isna(val) or np.isinf(val):
                        row_data[col] = 0.0
                    else:
                        row_data[col] = float(val)
                else:
                    # Eksik feature → 0.0 ile doldur
                    row_data[col] = 0.0

            # ── DataFrame oluştur (model feature sırası) ─────
            X = pd.DataFrame([row_data], columns=feature_cols)
            X = X.astype(np.float32)

            # ── Prediction ───────────────────────────────────
            proba = model.predict_proba(X)[0]
            meta_conf = float(proba[1]) if len(proba) >= 2 else float(proba[0])

            should_trade = meta_conf >= threshold

            return meta_conf, should_trade, threshold, "Success"

        except Exception as e:
            print(f"  ERROR: MetaPredictor hata ({regime_val}): {e}")
            fallback_conf = signal_confidence * 0.70
            return fallback_conf, fallback_conf > 0.55, 0.55, f"Error: {str(e)}"

    def get_regime_stats(self) -> Dict:
        """Yüklü modellerin istatistikleri"""
        stats = {}
        for regime_val, info in self.models.items():
            meta = info.get('meta', {})
            stats[regime_val] = {
                'loaded': True,
                'accuracy': meta.get('accuracy', '?'),
                'filtered_wr': meta.get('filtered_wr', '?'),
                'best_threshold': meta.get('best_threshold', 0.55),
                'best_threshold_wr': meta.get('best_threshold_wr', '?'),
                'auc': meta.get('auc', '?'),
                'n_features': len(info.get('features', [])),
            }

        for regime in Regime:
            if regime.value not in stats:
                stats[regime.value] = {'loaded': False}

        return stats

    @property
    def is_ready(self) -> bool:
        return len(self.models) > 0

    @property
    def loaded_regimes(self) -> list:
        return list(self.models.keys())
