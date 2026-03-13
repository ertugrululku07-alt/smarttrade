"""
Meta-Label Trainer v2.0 (Context-Aware)

KRİTİK DEĞİŞİKLİK:
  v1.3: RSI, MACD, BB, Stoch → sinyal üret → AYNI feature'ları meta-model'e ver
  v2.0: RSI, MACD, BB, Stoch → sinyal üret → CONTEXT feature'ları meta-model'e ver

  Döngüsel bilgi sorunu çözüldü.

v2.0 Improvements:
  - Context-aware feature architecture (circular dependency çözümü)
  - v1.3 fixes preserved (no double balancing, degenerate penalty)
  - 4h HTF data integration in CLI
  - Quality verdict system (GOOD/OK/WEAK)
  - Feature category breakdown (CTX/SIG/HTF/TCH)
  - WR improvement tracking
"""

import os
import gc
import json
import time
import warnings
import traceback

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from typing import Dict, List, Optional, Tuple
from datetime import datetime
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, accuracy_score, brier_score_loss,
)


# ══════════════════════════════════════════════════════════════
# Optuna (opsiyonel import)
# ══════════════════════════════════════════════════════════════

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False


# ══════════════════════════════════════════════════════════════
# Sabitler
# ══════════════════════════════════════════════════════════════

VERSION = "2.0"

REGIME_VALUES = [
    'trending',
    'mean_reverting',
    'high_volatile',
    'low_volatile',
]

# ─────────────────────────────────────────────────────────────
# FEATURE MİMARİSİ v2.0
#
# ESKİ (v1.3):
#   Signal üretici feature'lar → Meta-model = DÖNGÜSEL
#
# YENİ (v2.0):
#   Sinyal meta → Context → HTF → Minimal teknik = YENİ BİLGİ
# ─────────────────────────────────────────────────────────────

# Grup 1: Sinyal meta-data
SIGNAL_FEATURES = [
    'signal_is_long',
    'signal_confidence',
]

# Grup 2: CONTEXT — sinyal üretiminde KULLANILMAYAN feature'lar
CONTEXT_FEATURES = [
    # Zaman
    'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
    # Rejim kontekst
    'trend_duration', 'vol_regime_ratio', 'regime_age',
    # Diverjans
    'price_vol_divergence', 'rsi_price_divergence', 'roc_divergence',
    # Hareket kalitesi
    'move_cleanliness', 'trend_consistency', 'momentum_quality',
    # Fiyat pozisyonu
    'price_position_50', 'range_position',
    # Volume profili
    'vol_relative_50', 'vol_trend', 'vol_climax',
    # Candle yapısı
    'body_ratio', 'is_doji', 'upper_wick_ratio', 'lower_wick_ratio',
    # Ardışıklık
    'consec_direction', 'consec_magnitude',
    # EMA mesafeleri (ATR normalize)
    'dist_ema200_atr', 'dist_ema50_atr', 'dist_vwap_atr',
    # Mikro yapı
    'bar_range_vs_atr', 'spread_estimate',
    # İnteraksiyon
    'vol_x_volatility', 'trend_x_clean',
]

# Grup 3: HTF (multi-timeframe)
HTF_FEATURES = [
    'htf_bias_numeric', 'htf_trend_strength', 'htf_rsi',
    'htf_price_vs_ema200', 'htf_ema_alignment', 'htf_structure_numeric',
]

# Grup 4: Minimal teknik (sinyal üretiminde doğrudan kullanılmayan)
MINIMAL_TECHNICAL = [
    'adx',          # Trend gücü ölçer, sinyal üretmez
    'atr_pct',      # Volatilite seviyesi
    'hurst',        # Fraktal yapı
    'vol_ratio_20', # Volume relatif
]

# Feature'lardan düşürülecekler
DROP_FEATURES = [
    'funding_rate', 'funding_rate_ma8', 'funding_rate_trend',
    'open_interest_norm', 'long_short_ratio',
]

# v1.3'teki DÖNGÜSEL feature'lar — KASITLI olarak HARİÇ
# Bunlar sinyal üretiminde kullanılıyor, meta-model'e VERİLMEYECEK
EXCLUDED_CIRCULAR = [
    'rsi', 'macd_hist', 'bb_width', 'bb_pos', 'stoch_k',
    'ema_cross', 'di_plus', 'di_minus', 'di_diff',
    'efficiency_ratio', 'zscore_20', 'adx_accel',
    'volatility_10', 'momentum_10', 'atr_rank_50',
    'macd', 'macd_signal',
]

# Purge & Split parametreleri
DEFAULT_PURGE_BARS = 16
TRAIN_PCT = 0.65
VAL_PCT = 0.15

# Feature kalite eşiği
MAX_NAN_RATIO = 0.50       # %50'den fazla NaN → feature düşür
MIN_SAMPLES_PER_REGIME = 100
MIN_SAMPLES_AFTER_SPLIT = 30


# ══════════════════════════════════════════════════════════════
# Ana Trainer Sınıfı
# ══════════════════════════════════════════════════════════════

class MetaTrainer:
    """
    Her regime için ayrı XGBoost meta-model eğitir.

    Eğitim Pipeline:
      1. Feature listesi oluştur (mevcut kolonlardan)
      2. Feature kalite kontrolü (NaN oranı)
      3. Purged train/val/test split
      4. Optuna hyperparameter optimization (val set ile)
      5. Final model eğitimi (train, val ile early stop)
      6. Test set üzerinde değerlendirme
      7. Threshold sweep
      8. Model + metadata kaydet
    """

    def __init__(
        self,
        model_dir: str = "models/meta",
        use_mtf: bool = True,
        optimize: bool = True,
        n_trials: int = 25,
        purge_bars: int = DEFAULT_PURGE_BARS,
    ):
        self.model_dir = model_dir
        self.use_mtf = use_mtf
        self.optimize = optimize and OPTUNA_AVAILABLE
        self.n_trials = n_trials
        self.purge_bars = purge_bars

        os.makedirs(model_dir, exist_ok=True)

    # ══════════════════════════════════════════════════
    # ANA EĞİTİM
    # ══════════════════════════════════════════════════

    def train_all_regimes(
        self,
        meta_df: pd.DataFrame,
        version: str = VERSION,
        timeframe: str = "1h",
    ) -> Dict:
        """Tüm regime'ler için model eğit."""

        if meta_df.empty:
            print("  [ERROR] Meta-DataFrame boş. Eğitim yapılamaz.")
            return {}

        start_time = time.time()

        print(f"\n  {'='*60}")
        print(f"  META-TRAINER v{VERSION}")
        print(f"  Samples: {len(meta_df):,} | TF: {timeframe}")
        print(f"  MTF: {'ON' if self.use_mtf else 'OFF'} | "
              f"Optuna: {'ON' if self.optimize else 'OFF'}")
        print(f"  {'='*60}")

        # Regime kolonu tespit et  [FIX #5]
        regime_col = self._find_regime_column(meta_df)
        if regime_col is None:
            print("  [ERROR] Regime kolonu bulunamadı!")
            return {}

        print(f"  Regime column: '{regime_col}'")

        # Hangi regime'ler mevcut?
        available_regimes = self._get_available_regimes(
            meta_df, regime_col
        )

        results = {}

        for regime_val in available_regimes:
            regime_df = meta_df[
                meta_df[regime_col] == regime_val
            ].copy()

            if len(regime_df) < MIN_SAMPLES_PER_REGIME:
                print(
                    f"\n  [SKIP] {regime_val}: Yetersiz veri "
                    f"({len(regime_df)} < {MIN_SAMPLES_PER_REGIME})"
                )
                results[regime_val] = {
                    "status": "skipped",
                    "samples": len(regime_df),
                }
                continue

            result = self._train_single_regime(
                regime_df=regime_df,
                regime_val=regime_val,
                timeframe=timeframe,
                version=version,
            )

            results[regime_val] = result

            gc.collect()

        # Özet
        elapsed = time.time() - start_time
        self._print_summary(results, elapsed)

        return results

    # ══════════════════════════════════════════════════
    # TEK REGİME EĞİTİMİ
    # ══════════════════════════════════════════════════

    def _train_single_regime(
        self,
        regime_df: pd.DataFrame,
        regime_val: str,
        timeframe: str,
        version: str,
    ) -> Dict:
        """Tek bir regime için tam eğitim pipeline'ı."""

        print(f"\n  {'-'*50}")
        print(f"  [TRAIN] {regime_val.upper()}")
        print(f"  Samples: {len(regime_df):,}")

        try:
            # ── 1. Feature Listesi (v2.0 — context-based) ──
            features = self._build_feature_list(regime_df)

            if len(features) < 5:
                print(f"  [SKIP] Çok az feature: {len(features)}")
                return {"status": "insufficient_features"}

            # ── 2. Feature Kalite Kontrolü ──
            features = self._filter_quality_features(
                regime_df, features
            )

            # Feature category breakdown
            n_sig = sum(1 for f in features if f in SIGNAL_FEATURES)
            n_ctx = sum(1 for f in features if f in CONTEXT_FEATURES)
            n_htf = sum(1 for f in features if f.startswith('htf_'))
            n_tech = sum(1 for f in features if f in MINIMAL_TECHNICAL)

            print(f"  Features: {len(features)} "
                  f"(sig={n_sig} ctx={n_ctx} htf={n_htf} tech={n_tech})")

            # Döngüsel feature uyarısı
            circular_found = [f for f in features if f in EXCLUDED_CIRCULAR]
            if circular_found:
                print(f"  [WARN] Döngüsel feature bulundu (çıkarılıyor): {circular_found}")
                features = [f for f in features if f not in EXCLUDED_CIRCULAR]
                print(f"  Features (temiz): {len(features)}")

            # ── 3. X, y Hazırlama ──
            X, y = self._prepare_xy(regime_df, features)

            if X is None or y is None:
                return {"status": "data_preparation_error"}

            # Label dağılımı
            n_pos = int((y == 1).sum())
            n_neg = int((y == 0).sum())
            raw_wr = n_pos / len(y) * 100 if len(y) > 0 else 0

            print(
                f"  Labels: Correct={n_pos} Wrong={n_neg} | "
                f"Raw WR={raw_wr:.1f}%"
            )

            if n_pos < 10 or n_neg < 10:
                print("  [SKIP] Çok dengesiz veri")
                return {"status": "imbalanced_data"}

            # ── 4. Purged Split ──  [FIX #1, #3]
            splits = self._purged_split(X, y)

            if splits is None:
                return {"status": "split_error"}

            X_train, y_train, X_val, y_val, X_test, y_test = splits

            print(
                f"  Split: Train={len(X_train)} "
                f"Val={len(X_val)} Test={len(X_test)}"
            )

            # ── 5. Optuna ──  [FIX #4]
            scale_pw = max(1.0, n_neg / max(1.0, n_pos))

            if self.optimize:
                print(
                    f"  [OPT] Optuna {self.n_trials} trials..."
                )
                best_params = self._run_optuna(
                    X_train, y_train, X_val, y_val,
                    scale_pw,
                )
            else:
                best_params = self._default_params(scale_pw)

            # ── 6. Final Model ──  [FIX #1, #6, #7]
            model = self._train_final_model(
                X_train, y_train, X_val, y_val,
                best_params,
            )

            # ── 7. Test Değerlendirme ──  [FIX #8, #9]
            metrics = self._evaluate_model(
                model, X_test, y_test, raw_wr
            )

            # ── 8. Feature Importance ──
            importance = self._get_feature_importance(
                model, features
            )
            self._print_feature_importance(importance)

            # ── 9. Kaydet ──
            save_path = self._save_model(
                model=model,
                features=features,
                regime_val=regime_val,
                timeframe=timeframe,
                version=version,
                metrics=metrics,
                params=best_params,
                importance=importance,
                raw_wr=raw_wr,
                split_sizes={
                    'train': len(X_train),
                    'val': len(X_val),
                    'test': len(X_test),
                },
            )

            return {
                "status": "trained",
                "samples": len(regime_df),
                "features": len(features),
                "raw_wr": round(raw_wr, 1),
                "model_path": save_path,
                **metrics,
            }

        except Exception as e:
            print(f"  [ERROR] {regime_val}: {e}")
            traceback.print_exc()
            return {"status": "error", "error": str(e)}

    # ══════════════════════════════════════════════════
    # FEATURE MANAGEMENT  [FIX #2, #10, #11]
    # ══════════════════════════════════════════════════

    def _build_feature_list(
        self,
        df: pd.DataFrame,
    ) -> List[str]:
        """
        v2.0 Feature seçimi:
          1. Signal meta     (2 feature)
          2. Context         (~33 feature) ← ANA GRUP
          3. HTF             (6 feature)
          4. Minimal tech    (4 feature)

        RSI, MACD, BB, Stoch vb. DAHIL EDİLMEZ (döngüsel).
        """
        available = set(df.columns)
        features = []
        seen = set()

        # 1. Signal meta
        for f in SIGNAL_FEATURES:
            if f in available and f not in seen:
                features.append(f)
                seen.add(f)

        # 2. CONTEXT (ana grup)
        for f in CONTEXT_FEATURES:
            if f in available and f not in seen:
                features.append(f)
                seen.add(f)

        # 3. HTF
        if self.use_mtf:
            for f in HTF_FEATURES:
                if f in available and f not in seen:
                    features.append(f)
                    seen.add(f)

        # 4. Minimal technical
        for f in MINIMAL_TECHNICAL:
            if (f in available
                    and f not in seen
                    and f not in DROP_FEATURES):
                features.append(f)
                seen.add(f)

        return features

    def _filter_quality_features(
        self,
        df: pd.DataFrame,
        features: List[str],
    ) -> List[str]:
        """
        NaN oranı çok yüksek feature'ları filtrele.

        [FIX #10]: >%50 NaN olan feature'lar düşürülür.
        """
        quality_features = []
        dropped = []

        for f in features:
            if f not in df.columns:
                continue

            nan_ratio = df[f].isna().mean()

            if nan_ratio > MAX_NAN_RATIO:
                dropped.append(f"{f}({nan_ratio:.0%})")
            else:
                quality_features.append(f)

        if dropped:
            print(
                f"  [WARN] NaN nedeniyle düşürülen: "
                f"{', '.join(dropped)}"
            )

        return quality_features

    def _prepare_xy(
        self,
        df: pd.DataFrame,
        features: List[str],
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """X matrix ve y vector hazırla."""
        try:
            # meta_label kolonu
            if 'meta_label' not in df.columns:
                print("  [ERROR] 'meta_label' kolonu yok")
                return None, None

            # NaN/inf temizle
            clean = df[features + ['meta_label']].copy()
            clean = clean.replace([np.inf, -np.inf], np.nan)
            clean = clean.dropna(subset=['meta_label'])

            if len(clean) < MIN_SAMPLES_PER_REGIME:
                print(
                    f"  [ERROR] Temizlik sonrası yetersiz: "
                    f"{len(clean)}"
                )
                return None, None

            # Fill remaining NaN with column median
            for f in features:
                if clean[f].isna().any():
                    median_val = clean[f].median()
                    if np.isnan(median_val):
                        median_val = 0.0
                    clean[f] = clean[f].fillna(median_val)

            X = clean[features].astype(np.float32).values
            y = clean['meta_label'].values.astype(np.int32).ravel()

            # y validation
            unique = set(np.unique(y[~np.isnan(y)]))
            if not unique.issubset({0, 1}):
                warnings.warn(f"meta_label beklenmeyen değerler: {unique}")
                y = (y > 0).astype(np.int32)

            return X, y

        except Exception as e:
            print(f"  [ERROR] X/y hazırlama: {e}")
            return None, None

    # ══════════════════════════════════════════════════
    # PURGED SPLIT  [FIX #1, #3]
    # ══════════════════════════════════════════════════

    def _purged_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Optional[Tuple]:
        """
        Train / Val / Test split with purge gaps.

        [FIX #1]: Ayrı train/val/test — data leakage yok.
        [FIX #3]: Purge gap ile look-ahead bias koruması.

        Layout:
          [--- Train ---|purge|--- Val ---|purge|--- Test ---]
        """
        n = len(X)
        purge = self.purge_bars
        total_purge = 2 * purge
        usable = n - total_purge

        if usable < MIN_SAMPLES_AFTER_SPLIT * 3:
            print(
                f"  [ERROR] Purge sonrası yetersiz: "
                f"{usable} < {MIN_SAMPLES_AFTER_SPLIT * 3}"
            )
            return None

        n_train = int(usable * TRAIN_PCT)
        n_val = int(usable * VAL_PCT)

        train_end = n_train
        val_start = train_end + purge
        val_end = val_start + n_val
        test_start = val_end + purge

        # Minimum kontrol
        n_test = n - test_start
        if (n_train < MIN_SAMPLES_AFTER_SPLIT
                or n_val < MIN_SAMPLES_AFTER_SPLIT // 2
                or n_test < MIN_SAMPLES_AFTER_SPLIT // 2):
            print("  [ERROR] Split sonrası parçalar çok küçük")
            return None

        return (
            X[:train_end], y[:train_end],
            X[val_start:val_end], y[val_start:val_end],
            X[test_start:], y[test_start:],
        )

    # ══════════════════════════════════════════════════
    # OPTUNA  [FIX #4, #12]
    # ══════════════════════════════════════════════════

    def _run_optuna(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        scale_pos_weight: float,
    ) -> Dict:
        """
        Optuna hyperparameter optimization.

        [FIX #4]: Val set ile değerlendirme (CV yerine).
        Purge gap zaten split'te uygulandı.

        Skor: (Prec×0.30 + AUC×0.50 + F1×0.20) × balance - penalty
        """
        if not OPTUNA_AVAILABLE:
            return self._default_params(scale_pos_weight)

        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int(
                    'n_estimators', 100, 500
                ),
                'max_depth': trial.suggest_int('max_depth', 3, 7),
                'learning_rate': trial.suggest_float(
                    'learning_rate', 0.01, 0.1, log=True
                ),
                'subsample': trial.suggest_float(
                    'subsample', 0.6, 0.95
                ),
                'colsample_bytree': trial.suggest_float(
                    'colsample_bytree', 0.5, 0.95
                ),
                'min_child_weight': trial.suggest_int(
                    'min_child_weight', 1, 15
                ),
                'gamma': trial.suggest_float('gamma', 0.0, 1.5),
                'reg_alpha': trial.suggest_float(
                    'reg_alpha', 1e-5, 5.0, log=True
                ),
                'reg_lambda': trial.suggest_float(
                    'reg_lambda', 1e-5, 5.0, log=True
                ),
                'scale_pos_weight': scale_pos_weight,
                'eval_metric': 'logloss',
                'random_state': 42,
                'tree_method': 'hist',
                'n_jobs': -1,
            }

            model = xgb.XGBClassifier(
                **params,
                early_stopping_rounds=20,
            )
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )

            preds = model.predict(X_val)
            proba = model.predict_proba(X_val)[:, 1]

            prec = precision_score(y_val, preds, zero_division=0)
            rec = recall_score(y_val, preds, zero_division=0)
            f1 = f1_score(y_val, preds, zero_division=0)

            try:
                auc = roc_auc_score(y_val, proba)
            except Exception:
                auc = 0.5

            # Degenerate solution penalty: if recall>95% model is
            # likely predicting everything as positive (no filtering)
            if rec > 0.95:
                penalty = 0.15
            elif rec > 0.90:
                penalty = 0.05
            else:
                penalty = 0.0

            # Balance penalty: precision and recall should be
            # reasonably close for a useful filter
            balance = 1.0 - abs(prec - rec) * 0.1

            score = (prec * 0.30 + auc * 0.50 + f1 * 0.20) * balance - penalty
            return score

        study = optuna.create_study(direction='maximize')
        study.optimize(
            objective,
            n_trials=self.n_trials,
            show_progress_bar=False,
        )

        print(f"  [OPT] Best: {study.best_value:.4f}")

        best = study.best_params
        best.update({
            'scale_pos_weight': scale_pos_weight,
            'eval_metric': 'logloss',
            'random_state': 42,
            'tree_method': 'hist',
            'n_jobs': -1,
        })

        return best

    def _default_params(self, scale_pos_weight: float) -> Dict:
        """Optuna olmadığında default parametreler."""
        return {
            'n_estimators': 300,
            'max_depth': 5,
            'learning_rate': 0.03,
            'subsample': 0.8,
            'colsample_bytree': 0.7,
            'min_child_weight': 5,
            'gamma': 0.3,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'scale_pos_weight': scale_pos_weight,
            'eval_metric': 'logloss',
            'random_state': 42,
            'tree_method': 'hist',
            'n_jobs': -1,
        }

    # ══════════════════════════════════════════════════
    # FINAL MODEL  [FIX #1, #6, #7]
    # ══════════════════════════════════════════════════

    def _train_final_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        params: Dict,
    ) -> xgb.XGBClassifier:
        """
        Final model eğitimi.

        [FIX #1]: Sadece train verisi ile eğitim (test görmez).
        [FIX #6]: Early stopping ile overfit koruması.
        [FIX #7]: scale_pos_weight ile balance (params icinde).
        """
        model = xgb.XGBClassifier(
            **params,
            early_stopping_rounds=30,
        )

        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        return model

    # ══════════════════════════════════════════════════
    # DEĞERLENDİRME  [FIX #8, #9]
    # ══════════════════════════════════════════════════

    def _evaluate_model(
        self,
        model: xgb.XGBClassifier,
        X_test: np.ndarray,
        y_test: np.ndarray,
        raw_wr: float,
    ) -> Dict:
        """
        Test seti üzerinde model değerlendirme.

        [FIX #8]: AUC + Brier eklendi.
        [FIX #9]: Threshold sweep eklendi.
        """
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        try:
            auc = roc_auc_score(y_test, y_proba)
        except Exception:
            auc = 0.5

        brier = brier_score_loss(y_test, y_proba)

        # Filtered Win Rate
        trade_mask = y_pred == 1
        if trade_mask.sum() > 0:
            filtered_wr = float(y_test[trade_mask].mean() * 100)
            n_trades = int(trade_mask.sum())
        else:
            filtered_wr = 0.0
            n_trades = 0

        # ── Threshold Sweep ──
        best_thr = 0.50
        best_thr_wr = filtered_wr
        best_thr_n = n_trades

        for thr in np.arange(0.40, 0.80, 0.02):
            thr_mask = y_proba >= thr
            n = int(thr_mask.sum())
            if n > max(10, len(y_test) * 0.05):
                thr_wr = float(y_test[thr_mask].mean() * 100)
                if thr_wr > best_thr_wr:
                    best_thr_wr = thr_wr
                    best_thr = float(thr)
                    best_thr_n = n

        # WR improvement
        wr_improv = filtered_wr - raw_wr

        # Quality verdict
        if auc >= 0.62 and wr_improv >= 3.0:
            verdict = "✅ GOOD"
        elif auc >= 0.58 and wr_improv >= 1.0:
            verdict = "⚠️  OK"
        else:
            verdict = "❌ WEAK"

        # Rapor
        print(f"\n  Metrikler:")
        print(f"  Accuracy    : {acc*100:.1f}%")
        print(f"  Precision   : {prec*100:.1f}%")
        print(f"  Recall      : {rec*100:.1f}%")
        print(f"  F1          : {f1*100:.1f}%")
        print(f"  AUC         : {auc:.4f}")
        print(f"  Brier       : {brier:.4f}")
        print(f"  Raw WR      : {raw_wr:.1f}%")
        print(f"  Filtered WR : {filtered_wr:.1f}% ({n_trades} trades)")
        print(f"  WR Improv.  : {wr_improv:+.1f}%")
        print(f"  Best Thr    : {best_thr:.2f} → {best_thr_wr:.1f}% ({best_thr_n} trades)")
        print(f"  Verdict     : {verdict}")

        return {
            'accuracy': round(acc * 100, 2),
            'precision': round(prec * 100, 2),
            'recall': round(rec * 100, 2),
            'f1': round(f1 * 100, 2),
            'auc': round(auc, 4),
            'brier': round(brier, 4),
            'filtered_wr': round(filtered_wr, 1),
            'n_trades_test': n_trades,
            'wr_improvement': round(wr_improv, 1),
            'best_threshold': round(best_thr, 2),
            'best_threshold_wr': round(best_thr_wr, 1),
            'best_threshold_n': best_thr_n,
            'verdict': verdict,
        }

    # ══════════════════════════════════════════════════
    # FEATURE IMPORTANCE
    # ══════════════════════════════════════════════════

    def _get_feature_importance(
        self,
        model: xgb.XGBClassifier,
        features: List[str],
    ) -> List[Tuple[str, float]]:
        """Feature importance listesi (sıralı)."""
        try:
            importances = model.feature_importances_
            if len(importances) != len(features):
                return []

            pairs = list(zip(features, [float(v) for v in importances]))
            return sorted(pairs, key=lambda x: x[1], reverse=True)

        except Exception:
            return []

    def _print_feature_importance(
        self,
        importance: List[Tuple[str, float]],
        top_n: int = 12,
    ):
        """Feature importance ASCII tablosu with category tags."""
        if not importance:
            return

        print(f"\n  Top {top_n} Features:")
        print(f"  {'-'*55}")

        for fname, imp in importance[:top_n]:
            bar = '█' * int(imp * 50)
            if fname.startswith('htf_'):
                tag = 'HTF'
            elif fname in CONTEXT_FEATURES:
                tag = 'CTX'
            elif fname in SIGNAL_FEATURES:
                tag = 'SIG'
            elif fname in MINIMAL_TECHNICAL:
                tag = 'TCH'
            else:
                tag = '???'
            print(f"  {fname:25s} {imp:.4f} {bar} [{tag}]")

        # Category totals
        ctx = sum(v for f, v in importance if f in CONTEXT_FEATURES)
        htf = sum(v for f, v in importance if f.startswith('htf_'))
        tch = sum(v for f, v in importance if f in MINIMAL_TECHNICAL)
        sig = sum(v for f, v in importance if f in SIGNAL_FEATURES)

        print(f"  {'-'*55}")
        print(f"  Totals: CTX={ctx:.1%} SIG={sig:.1%} HTF={htf:.1%} TCH={tch:.1%}")

        # Döngüsel feature kontrolü
        circular = [f for f, _ in importance if f in EXCLUDED_CIRCULAR]
        if circular:
            print(f"  ⚠️  DÖNGÜSEL FEATURE TESPİT: {circular}")

    # ══════════════════════════════════════════════════
    # MODEL KAYDETME
    # ══════════════════════════════════════════════════

    def _save_model(
        self,
        model: xgb.XGBClassifier,
        features: List[str],
        regime_val: str,
        timeframe: str,
        version: str,
        metrics: Dict,
        params: Dict,
        importance: List[Tuple],
        raw_wr: float,
        split_sizes: Dict,
    ) -> str:
        """Model + metadata kaydet."""
        save_path = os.path.join(
            self.model_dir,
            f"meta_{regime_val}_{timeframe}.joblib",
        )

        meta_info = {
            'regime': regime_val,
            'timeframe': timeframe,
            'version': version,
            'trained_at': datetime.now().isoformat(),
            'mtf_enabled': self.use_mtf,
            'n_features': len(features),
            'n_context': sum(1 for f in features if f in CONTEXT_FEATURES),
            'n_htf': sum(1 for f in features if f.startswith('htf_')),
            'raw_wr': round(raw_wr, 1),
            **metrics,
            'split': split_sizes,
            'params': params,
        }

        model_data = {
            'model': model,
            'features': features,
            'meta_info': meta_info,
            'feature_importance': importance,
        }

        # Geriye uyumlu format: (model, features, meta_info)
        joblib.dump(
            (model, features, meta_info),
            save_path,
        )

        print(f"  Saved: {save_path}")
        return save_path

    # ══════════════════════════════════════════════════
    # YARDIMCI
    # ══════════════════════════════════════════════════

    def _find_regime_column(
        self,
        df: pd.DataFrame,
    ) -> Optional[str]:
        """
        Regime kolonunu otomatik tespit et.

        [FIX #5]: _regime, ml_regime, regime desteği.
        """
        for col_name in ['_regime', 'ml_regime', 'regime']:
            if col_name in df.columns:
                return col_name

        return None

    def _get_available_regimes(
        self,
        df: pd.DataFrame,
        regime_col: str,
    ) -> List[str]:
        """Mevcut regime'leri sıralı döndür."""
        actual = df[regime_col].unique().tolist()

        # Bilinen sıralama
        ordered = [
            r for r in REGIME_VALUES
            if r in actual
        ]

        # Bilinmeyen regime'ler
        unknown = [
            r for r in actual
            if r not in REGIME_VALUES and r != 'unknown'
        ]

        return ordered + unknown

    def _print_summary(self, results: Dict, elapsed: float):
        """Özet rapor."""
        print(f"\n  {'='*65}")
        print(f"  META-TRAINER v{VERSION} SUMMARY | {elapsed:.1f}s")
        print(f"  {'='*65}")

        for rv, res in results.items():
            if res.get('status') == 'trained':
                v = res.get('verdict', '?')
                print(
                    f"  {v:6s} {rv:18s} "
                    f"AUC={res['auc']:.3f} "
                    f"FiltWR={res['filtered_wr']:.0f}% "
                    f"Improv={res.get('wr_improvement', 0):+.1f}% "
                    f"Thr={res['best_threshold']:.2f} "
                    f"n={res.get('n_trades_test', 0)}"
                )
            else:
                print(
                    f"  [--]   {rv:18s} "
                    f"{res.get('status', '?')}"
                )

        print(f"  {'='*65}")


# ═══════════════════════════════════════════════════════════════
# Bulk Entry Point
# ═══════════════════════════════════════════════════════════════

def train_meta_models_bulk(
    meta_df: pd.DataFrame,
    model_dir: str = "models/meta",
    use_mtf: bool = True,
    timeframe: str = "1h",
    **kwargs,
) -> Dict:
    """Tüm regime'ler için model eğit (wrapper)."""
    trainer = MetaTrainer(
        model_dir=model_dir,
        use_mtf=use_mtf,
        n_trials=kwargs.get('n_trials', 25),
        purge_bars=kwargs.get('purge_bars', DEFAULT_PURGE_BARS),
    )
    return trainer.train_all_regimes(
        meta_df, timeframe=timeframe
    )


# ═══════════════════════════════════════════════════════════════
# CLI (v2.0 — 4h veri + context features)
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=f"Meta-Trainer v{VERSION}"
    )
    parser.add_argument('--timeframe', type=str, default='1h')
    parser.add_argument('--limit', type=int, default=10000)
    parser.add_argument('--trials', type=int, default=25)

    top_50 = (
        "BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,XRPUSDT,ADAUSDT,"
        "DOGEUSDT,AVAXUSDT,SHIBUSDT,DOTUSDT,"
        "LINKUSDT,TRXUSDT,MATICUSDT,LTCUSDT,BCHUSDT,"
        "UNIUSDT,ATOMUSDT,XMRUSDT,ETCUSDT,ICPUSDT,"
        "XLMUSDT,FILUSDT,HBARUSDT,APTUSDT,LDOUSDT,"
        "ARBUSDT,MKRUSDT,VETUSDT,OPUSDT,INJUSDT,"
        "GRTUSDT,RNDRUSDT,THETAUSDT,FTMUSDT,SNXUSDT,"
        "AAVEUSDT,SANDUSDT,AXSUSDT,EGLDUSDT,NEOUSDT,"
        "KAVAUSDT,CHZUSDT,GALAUSDT,ENJUSDT,ZILUSDT,"
        "CRVUSDT,LUNA2USDT,MANAUSDT,ROSEUSDT,DYDXUSDT"
    )
    parser.add_argument('--symbols', type=str, default=top_50)
    parser.add_argument('--model-dir', type=str, default='ai/models/meta')
    args = parser.parse_args()

    symbols = [s.strip() for s in args.symbols.split(',') if s.strip()]

    print(f"\n[CLI] Meta-Trainer v{VERSION} (Context-Aware)")
    print(f"  Symbols: {len(symbols)} | TF: {args.timeframe}")

    import sys
    from pathlib import Path

    backend_dir = str(Path(__file__).resolve().parent.parent.parent)
    if backend_dir not in sys.path:
        sys.path.insert(0, backend_dir)

    from backtest.data_fetcher import DataFetcher
    from backtest.signals import add_all_indicators, add_meta_context_features
    from ai.xgboost_trainer import generate_features
    from ai.meta_labelling.meta_label_generator import MetaLabelGenerator

    fetcher = DataFetcher('binance')
    all_meta_dfs = []

    for sym in symbols:
        print(f"\n[DATA] {sym}...")
        try:
            # 1h (veya seçilen TF) veri
            raw_df = fetcher.fetch_ohlcv(
                sym, args.timeframe, limit=args.limit
            )
            if raw_df is None or len(raw_df) < 500:
                print(f"  [SKIP] Yetersiz veri")
                continue

            if not isinstance(raw_df.index, pd.DatetimeIndex):
                if 'timestamp' in raw_df.columns:
                    raw_df.set_index('timestamp', inplace=True)
                else:
                    raw_df.index = pd.to_datetime(raw_df.index)
            raw_df = raw_df.sort_index()

            # Indicators + features + CONTEXT (v2.0)
            df = add_all_indicators(raw_df)
            df = generate_features(df)
            df = add_meta_context_features(df)

            # ── HTF veri (v2.0) — TF'ye göre dinamik ──
            htf_map = {'5m': '1h', '15m': '1h', '1h': '4h', '4h': '1d'}
            htf_tf = htf_map.get(args.timeframe, '4h')
            htf_ratio = {'5m': 12, '15m': 4, '1h': 4, '4h': 6}.get(args.timeframe, 4)

            df_4h = None
            try:
                htf_limit = max(args.limit // htf_ratio, 500)
                raw_htf = fetcher.fetch_ohlcv(sym, htf_tf, limit=htf_limit)

                if raw_htf is not None and len(raw_htf) >= 200:
                    if not isinstance(raw_htf.index, pd.DatetimeIndex):
                        if 'timestamp' in raw_htf.columns:
                            raw_htf.set_index('timestamp', inplace=True)
                        else:
                            raw_htf.index = pd.to_datetime(raw_htf.index)
                    raw_htf = raw_htf.sort_index()
                    df_4h = add_all_indicators(raw_htf)
                    print(f"  [OK] HTF({htf_tf}): {len(df_4h)} bars")
                else:
                    print(f"  [WARN] HTF({htf_tf}) yetersiz")
            except Exception as e_htf:
                print(f"  [WARN] HTF({htf_tf}) hata: {e_htf}")

            # Meta-labels (v2.0: context features + 4h)
            generator = MetaLabelGenerator(
                timeframe=args.timeframe,
                progress_interval=5000,
            )
            meta_df, stats = generator.generate(
                df, df_4h=df_4h, symbol=sym, verbose=True,
            )

            if not meta_df.empty:
                all_meta_dfs.append(meta_df)

        except Exception as e:
            print(f"  [ERROR] {sym}: {e}")
            traceback.print_exc()

    if not all_meta_dfs:
        print("\n[FAIL] Meta-data üretilemedi!")
        exit(1)

    total_meta_df = pd.concat(all_meta_dfs, axis=0)
    print(f"\n[CLI] Combined: {total_meta_df.shape}")

    # v2.0 context feature varlık kontrolü
    ctx_present = [c for c in CONTEXT_FEATURES if c in total_meta_df.columns]
    ctx_missing = [c for c in CONTEXT_FEATURES if c not in total_meta_df.columns]
    print(f"  Context features: {len(ctx_present)}/{len(CONTEXT_FEATURES)} mevcut")
    if ctx_missing:
        print(f"  Missing: {ctx_missing[:10]}...")

    results = train_meta_models_bulk(
        meta_df=total_meta_df,
        model_dir=args.model_dir,
        timeframe=args.timeframe,
        n_trials=args.trials,
        use_mtf=True,
    )

    print(f"\n✅ Meta-Label eğitimi tamamlandı (v{VERSION})!")
    print(json.dumps(results, indent=2, default=str))