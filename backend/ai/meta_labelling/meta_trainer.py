"""
Meta-Label Trainer v1.0

Her rejim için ayrı binary XGBoost meta-model eğitir.

Meta-model hedefi:
  "Primary sinyal (rule-based) DOĞRU çıkacak mı?"
  1 = Evet (TP vurdu)
  0 = Hayır (SL vurdu)

Bu BINARY classification olduğu için:
  - SHORT bias sorunu YOK
  - balanced weight MÜKEMMEL çalışır
  - %55-65 accuracy hedeflenebilir
"""

import os
import sys
import gc
import json
import warnings
import numpy as np
import pandas as pd
import joblib
import time as _time
from datetime import datetime
from typing import List, Optional, Dict, Tuple
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, f1_score,
    precision_score, recall_score, roc_auc_score,
    brier_score_loss,
)
from sklearn.utils.class_weight import compute_sample_weight

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backtest.data_fetcher import DataFetcher
from backtest.signals import add_all_indicators
from ai.data_sources.futures_data import enrich_ohlcv_with_futures
from ai.xgboost_trainer import generate_features, FEATURE_COLS, _update_progress
from ai.regime_detector import Regime
from ai.meta_labelling.meta_label_generator import (
    MetaLabelGenerator, generate_meta_labels_bulk,
)

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")

DEFAULT_SYMBOLS = [
    'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT', 'ADAUSDT',
    'DOGEUSDT', 'SHIBUSDT', 'AVAXUSDT', 'DOTUSDT', 'LINKUSDT', 'MATICUSDT',
    'UNIUSDT', 'LTCUSDT', 'ATOMUSDT', 'ETCUSDT', 'BCHUSDT', 'ALGOUSDT',
    'VETUSDT', 'FILUSDT', 'AAVEUSDT', 'GALAUSDT', 'SANDUSDT', 'MANAUSDT',
    'FTMUSDT', 'NEARUSDT', 'RUNEUSDT', 'EGLDUSDT', 'CRVUSDT', 'CHZUSDT',
    'AXSUSDT', 'THETAUSDT', 'ENJUSDT', 'SNXUSDT', 'GRTUSDT', 'MKRUSDT',
    'COMPUSDT', 'YFIUSDT', 'ZILUSDT', 'BATUSDT', 'WAVESUSDT', 'ONTUSDT',
    'QTUMUSDT', 'OMGUSDT', 'NEOUSDT', 'EOSUSDT', 'XTZUSDT', 'DASHUSDT',
    'ZECUSDT', 'XMRUSDT', 'XLMUSDT', 'TRXUSDT', 'IOTAUSDT', 'KSMUSDT',
    'SUSHIUSDT', '1INCHUSDT', 'OCEANUSDT', 'ROSEUSDT', 'CELOUSDT',
    'KAVAUSDT', 'SRMUSDT', 'RAYUSDT', 'LRCUSDT', 'RENUSDT', 'BALUSDT',
    'BANDUSDT', 'KNCUSDT', 'RLCUSDT', 'CTKUSDT', 'STXUSDT', 'TRBUSDT',
    'MDTUSDT', 'INJUSDT', 'TIAUSDT', 'WLDUSDT', 'JTOUSDT', 'ORDIUSDT',
]

MAKER_FEE = 0.0004


# ═══════════════════════════════════════════════════════════════════
# TF Konfigürasyonu (meta-label için)
# ═══════════════════════════════════════════════════════════════════

def _get_meta_tf_config(timeframe: str) -> Dict:
    configs = {
        "1m":  {"lookahead": 16, "trail_act": 0.6, "min_data_bars": 200},
        "3m":  {"lookahead": 32, "trail_act": 0.6, "min_data_bars": 200},
        "5m":  {"lookahead": 24, "trail_act": 0.6, "min_data_bars": 250},
        "15m": {"lookahead": 16, "trail_act": 0.6, "min_data_bars": 300},
        "30m": {"lookahead": 12, "trail_act": 0.6, "min_data_bars": 300},
        "1h":  {"lookahead": 12, "trail_act": 0.6, "min_data_bars": 300},
        "4h":  {"lookahead": 12, "trail_act": 0.6, "min_data_bars": 300},
        "1d":  {"lookahead": 10, "trail_act": 0.6, "min_data_bars": 250},
    }
    return configs.get(timeframe, {"lookahead": 16, "trail_act": 0.6, "min_data_bars": 300})


# ═══════════════════════════════════════════════════════════════════
# Meta Feature Seti
# ═══════════════════════════════════════════════════════════════════

# Orijinal feature'lar + meta-specific feature'lar
META_EXTRA_FEATURES = [
    'signal_is_long',
    'signal_confidence',
    'regime_trending',
    'regime_mean_rev',
    'regime_high_vol',
    'regime_low_vol',
]

# Feature'lardan çıkarılacaklar (meta-model için gereksiz)
META_DROP_FEATURES = [
    'funding_rate', 'funding_rate_ma8', 'funding_rate_trend',
    'open_interest_norm', 'long_short_ratio',
]


def _get_meta_features(base_features: List[str]) -> List[str]:
    """Meta-model için feature listesi oluştur"""
    features = [f for f in base_features if f not in META_DROP_FEATURES]
    features.extend(META_EXTRA_FEATURES)
    return features


# ═══════════════════════════════════════════════════════════════════
# Purged Split (meta-label için)
# ═══════════════════════════════════════════════════════════════════

def _purged_split_meta(
    X: pd.DataFrame,
    y: pd.Series,
    purge_bars: int,
    train_pct: float = 0.65,
    val_pct: float = 0.15,
) -> Tuple:
    """Train / Val / Test split with purge gaps"""
    n = len(X)
    gaps = 2 * purge_bars
    usable = n - gaps

    n_train = int(usable * train_pct)
    n_val = int(usable * val_pct)

    train_end = n_train
    val_start = train_end + purge_bars
    val_end = val_start + n_val
    test_start = val_end + purge_bars

    return (
        X.iloc[:train_end], y.iloc[:train_end],
        X.iloc[val_start:val_end], y.iloc[val_start:val_end],
        X.iloc[test_start:], y.iloc[test_start:],
    )


# ═══════════════════════════════════════════════════════════════════
# Optuna — Binary Meta-Model
# ═══════════════════════════════════════════════════════════════════

def _optuna_optimize_binary(
    X_train, y_train, X_val, y_val,
    n_trials: int = 25,
) -> Dict:
    """Binary classification için Optuna optimizasyonu"""
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        print("   ⚠️ Optuna yüklü değil, default params kullanılıyor")
        return {}

    sw = compute_sample_weight('balanced', y_train)

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 200, 800),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 15),
            'gamma': trial.suggest_float('gamma', 0.0, 1.5),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-5, 5.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-5, 5.0, log=True),
        }

        m = XGBClassifier(
            **params, objective='binary:logistic',
            random_state=42, tree_method='hist',
            early_stopping_rounds=25, eval_metric='logloss', n_jobs=-1,
        )
        m.fit(X_train, y_train, sample_weight=sw,
              eval_set=[(X_val, y_val)], verbose=False)

        preds = m.predict(X_val)
        proba = m.predict_proba(X_val)[:, 1]

        f1 = f1_score(y_val, preds, zero_division=0)
        precision = precision_score(y_val, preds, zero_division=0)

        try:
            auc = roc_auc_score(y_val, proba)
        except Exception:
            auc = 0.5

        # Precision (Win Rate) ağırlıklı skor
        # Yanlış sinyal (False Positive) bizim en büyük düşmanımız.
        # Bu yüzden Precision ağırlığını artırıyoruz.
        score = precision * 0.50 + auc * 0.30 + f1 * 0.20

        return score

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    print(f"   ✅ Optuna best score: {study.best_value:.4f}")
    return study.best_params


# ═══════════════════════════════════════════════════════════════════
# Ana Meta-Trainer
# ═══════════════════════════════════════════════════════════════════

class MetaTrainer:
    """
    Her rejim için ayrı binary meta-model eğitir.

    Kullanım:
        trainer = MetaTrainer(timeframe='15m')
        results = trainer.train_all(symbols=DEFAULT_SYMBOLS, limit=3000)
    """

    def __init__(self, timeframe: str = "1h"):
        self.timeframe = timeframe
        self.tf_config = _get_meta_tf_config(timeframe)

    def train_all(
        self,
        symbols: Optional[List[str]] = None,
        limit: int = 3000,
        n_trials: int = 25,
        use_cache: bool = False,
    ) -> Dict:
        """
        Tüm süreç:
          1. Veri çek
          2. Feature'ları hesapla
          3. Meta-label üret
          4. Her rejim için model eğit
          5. Kaydet
        """
        if symbols is None:
            symbols = DEFAULT_SYMBOLS

        lookahead = self.tf_config['lookahead']
        trail_act = self.tf_config['trail_act']
        min_bars = self.tf_config['min_data_bars']

        print(f"\n{'═' * 65}")
        print(f"  🧠 META-LABEL TRAINER v1.0")
        print(f"  TF: {self.timeframe} | Lookahead: {lookahead} | "
              f"Coins: {len(symbols)}")
        print(f"{'═' * 65}\n")

        # ── 1. Veri Çekme ────────────────────────────────────
        print("📥 Veri çekiliyor...")
        fetcher = DataFetcher('binance')
        all_dfs = {}
        skipped = []

        for idx, sym in enumerate(symbols):
            try:
                if use_cache:
                    try:
                        from ai.data_cache import get_cached_ohlcv
                        df = get_cached_ohlcv(sym, self.timeframe, limit=limit, fetcher=fetcher)
                    except ImportError:
                        df = fetcher.fetch_ohlcv(sym, self.timeframe, limit=limit)
                else:
                    df = fetcher.fetch_ohlcv(sym, self.timeframe, limit=limit)

                if df is None or df.empty or len(df) < min_bars:
                    skipped.append(sym)
                    continue

                df = add_all_indicators(df)
                df = enrich_ohlcv_with_futures(df, sym, silent=True)
                df = generate_features(df)

                all_dfs[sym] = df

                if not use_cache:
                    _time.sleep(0.25)

                if (idx + 1) % 10 == 0:
                    print(f"   [{idx + 1}/{len(symbols)}] {len(all_dfs)} coin yüklendi")

            except Exception as e:
                print(f"   ⚠️ {sym}: {e}")
                skipped.append(sym)

        print(f"   ✅ {len(all_dfs)} coin yüklendi, {len(skipped)} atlandı\n")

        if not all_dfs:
            return {"error": "No data"}

        # ── 2. Meta-Label Üretimi ────────────────────────────
        print("🏷️  Meta-label üretiliyor...")
        meta_df, meta_stats = generate_meta_labels_bulk(
            all_dfs,
            lookahead=lookahead,
            trail_activation=trail_act,
            verbose=True,
        )

        if meta_df.empty:
            return {"error": "No meta-labels generated"}

        del all_dfs
        gc.collect()

        # ── 3. Her Rejim İçin Model Eğit ─────────────────────
        print("\n🎓 Rejim bazlı meta-model eğitimi...")

        base_features = FEATURE_COLS.copy()
        meta_features = _get_meta_features(base_features)

        # Kullanılabilir feature'ları filtrele
        available_features = [f for f in meta_features if f in meta_df.columns]
        print(f"   📡 Kullanılabilir feature: {len(available_features)}")

        results = {}
        regime_models = {}

        for regime in Regime:
            regime_val = regime.value
            regime_mask = meta_df['_regime'] == regime_val
            regime_df = meta_df[regime_mask].copy()

            if len(regime_df) < 100:
                print(f"\n   ⏭️  {regime_val}: Yetersiz veri ({len(regime_df)} sample), atlandı")
                results[regime_val] = {"status": "skipped", "samples": len(regime_df)}
                continue

            print(f"\n{'─' * 55}")
            print(f"   🎯 {regime_val.upper()} rejimi eğitiliyor...")
            print(f"   Samples: {len(regime_df):,}")

            # NaN temizle
            regime_df = regime_df.dropna(subset=available_features + ['meta_label'])
            regime_df = regime_df.replace([np.inf, -np.inf], np.nan)
            regime_df = regime_df.dropna(subset=available_features)

            if len(regime_df) < 80:
                print(f"   ⏭️  NaN temizleme sonrası yetersiz ({len(regime_df)})")
                results[regime_val] = {"status": "skipped_after_clean", "samples": len(regime_df)}
                continue

            X = regime_df[available_features].astype(np.float32)
            y = regime_df['meta_label'].astype(int)

            # Label dağılımı
            n_correct = int((y == 1).sum())
            n_wrong = int((y == 0).sum())
            raw_wr = n_correct / len(y) * 100
            print(f"   Label: Correct={n_correct} Wrong={n_wrong} | Raw WR=%{raw_wr:.1f}")

            # Purged split
            try:
                X_train, y_train, X_val, y_val, X_test, y_test = \
                    _purged_split_meta(X, y, purge_bars=lookahead)
            except Exception as e:
                print(f"   ⚠️ Split hatası: {e}")
                results[regime_val] = {"status": "split_error", "error": str(e)}
                continue

            print(f"   Split: Train={len(X_train)} Val={len(X_val)} Test={len(X_test)}")

            if len(X_train) < 50 or len(X_val) < 20 or len(X_test) < 20:
                print(f"   ⏭️  Split sonrası yetersiz veri")
                results[regime_val] = {"status": "insufficient_split"}
                continue

            # ── Optuna ───────────────────────────────────────
            print(f"   🔍 Optuna {n_trials} trial...")
            best_params = _optuna_optimize_binary(
                X_train, y_train, X_val, y_val, n_trials=n_trials,
            )

            # ── Final Model ──────────────────────────────────
            final_params = {
                'n_estimators': best_params.get('n_estimators', 500),
                'learning_rate': best_params.get('learning_rate', 0.03),
                'max_depth': best_params.get('max_depth', 6),
                'subsample': best_params.get('subsample', 0.8),
                'colsample_bytree': best_params.get('colsample_bytree', 0.7),
                'min_child_weight': best_params.get('min_child_weight', 5),
                'gamma': best_params.get('gamma', 0.3),
                'reg_alpha': best_params.get('reg_alpha', 0.1),
                'reg_lambda': best_params.get('reg_lambda', 1.0),
            }

            sw = compute_sample_weight('balanced', y_train)

            model = XGBClassifier(
                **final_params, objective='binary:logistic',
                random_state=42, tree_method='hist',
                early_stopping_rounds=30, eval_metric='logloss', n_jobs=-1,
            )
            model.fit(
                X_train, y_train, sample_weight=sw,
                eval_set=[(X_val, y_val)], verbose=False,
            )

            # ── Evaluation ───────────────────────────────────
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]

            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)

            try:
                auc = roc_auc_score(y_test, y_proba)
            except Exception:
                auc = 0.5

            brier = brier_score_loss(y_test, y_proba)

            # Win rate simülasyonu
            # Meta-model "trade et" dediğinde gerçekten kazanıyor mu?
            trade_mask = y_pred == 1
            if trade_mask.sum() > 0:
                filtered_wr = y_test[trade_mask].mean() * 100
                n_trades = trade_mask.sum()
                n_filtered_out = (~trade_mask).sum()
            else:
                filtered_wr = 0.0
                n_trades = 0
                n_filtered_out = len(y_test)

            # Confidence threshold sweep
            best_threshold = 0.50
            best_threshold_wr = filtered_wr
            for thr in np.arange(0.45, 0.75, 0.02):
                thr_mask = y_proba >= thr
                if thr_mask.sum() > 10:
                    thr_wr = y_test[thr_mask].mean() * 100
                    if thr_wr > best_threshold_wr:
                        best_threshold_wr = thr_wr
                        best_threshold = thr

            print(f"\n   📊 {regime_val.upper()} Sonuçları:")
            print(f"   Accuracy   : %{acc * 100:.1f}")
            print(f"   F1 Score   : %{f1 * 100:.1f}")
            print(f"   Precision  : %{prec * 100:.1f}")
            print(f"   Recall     : %{rec * 100:.1f}")
            print(f"   AUC        : {auc:.4f}")
            print(f"   Brier      : {brier:.4f}")
            print(f"   Raw WR     : %{raw_wr:.1f}")
            print(f"   Filtered WR: %{filtered_wr:.1f} ({n_trades} trades, "
                  f"{n_filtered_out} filtered)")
            print(f"   Best Thr   : {best_threshold:.2f} → WR=%{best_threshold_wr:.1f}")

            # ── Kaydet ───────────────────────────────────────
            os.makedirs(MODEL_DIR, exist_ok=True)
            save_path = os.path.join(
                MODEL_DIR, f"meta_{regime_val}_{self.timeframe}.joblib"
            )

            meta_info = {
                "regime": regime_val,
                "timeframe": self.timeframe,
                "trained_at": datetime.now().isoformat(),
                "lookahead": lookahead,
                "accuracy": round(acc * 100, 2),
                "f1": round(f1 * 100, 2),
                "precision": round(prec * 100, 2),
                "recall": round(rec * 100, 2),
                "auc": round(auc, 4),
                "brier": round(brier, 4),
                "raw_wr": round(raw_wr, 1),
                "filtered_wr": round(filtered_wr, 1),
                "best_threshold": round(best_threshold, 2),
                "best_threshold_wr": round(best_threshold_wr, 1),
                "n_train": len(X_train),
                "n_val": len(X_val),
                "n_test": len(X_test),
                "n_trades_test": int(n_trades),
                "best_params": final_params,
                "version": "1.0",
            }

            joblib.dump((model, available_features, meta_info), save_path)
            print(f"   💾 Saved: {save_path}")

            results[regime_val] = {
                "status": "trained",
                "accuracy": round(acc * 100, 2),
                "f1": round(f1 * 100, 2),
                "precision": round(prec * 100, 2),
                "auc": round(auc, 4),
                "raw_wr": round(raw_wr, 1),
                "filtered_wr": round(filtered_wr, 1),
                "best_threshold": round(best_threshold, 2),
                "best_threshold_wr": round(best_threshold_wr, 1),
                "samples": len(regime_df),
                "model_path": save_path,
            }

            regime_models[regime_val] = model

            del X, y, X_train, y_train, X_val, y_val, X_test, y_test
            gc.collect()

        # ── Final Rapor ──────────────────────────────────────
        print(f"\n{'═' * 65}")
        print(f"  🏆 META-LABEL EĞİTİM ÖZETİ ({self.timeframe})")
        print(f"{'═' * 65}")

        for regime_val, res in results.items():
            if res.get('status') == 'trained':
                print(f"  ✅ {regime_val:16}: "
                      f"Acc=%{res['accuracy']:5.1f} "
                      f"F1=%{res['f1']:5.1f} "
                      f"Prec=%{res['precision']:5.1f} "
                      f"RawWR=%{res['raw_wr']:.0f} "
                      f"FilteredWR=%{res['filtered_wr']:.0f} "
                      f"BestThr={res['best_threshold']:.2f}→%{res['best_threshold_wr']:.0f}")
            else:
                print(f"  ⏭️  {regime_val:16}: {res.get('status', 'unknown')}")

        print(f"{'═' * 65}\n")

        return {
            "regime_results": results,
            "meta_stats": meta_stats,
            "timeframe": self.timeframe,
            "total_coins": len(symbols) - len(skipped),
            "skipped_coins": skipped,
        }


# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Meta-Label Trainer v1.0")
    parser.add_argument('--timeframe', type=str, default='1h')
    parser.add_argument('--limit', type=int, default=3000)
    parser.add_argument('--trials', type=int, default=25)
    parser.add_argument('--symbols', type=str, default='')
    parser.add_argument('--cache', action='store_true')
    args = parser.parse_args()

    syms = [s.strip() for s in args.symbols.split(',') if s.strip()] or None

    trainer = MetaTrainer(timeframe=args.timeframe)
    results = trainer.train_all(
        symbols=syms,
        limit=args.limit,
        n_trials=args.trials,
        use_cache=args.cache,
    )

    print("\n✅ Meta-Label eğitimi tamamlandı!")
    print(json.dumps({
        k: v for k, v in results.items()
        if k not in ('meta_stats',)
    }, indent=2, default=str))
