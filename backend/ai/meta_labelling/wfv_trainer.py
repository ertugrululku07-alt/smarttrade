"""
Walk-Forward Validation (WFV) Trainer v1.1

v1.1 Fixes:
  - ASCII-only log messages (Unicode emoji removed for Windows cp1252)
  - Safe FEATURE_COLS usage with missing column protection
  - use_label_encoder removed (deprecated in XGBoost 1.6+)
  - Better error handling for missing columns
"""

import os
import pandas as pd
import numpy as np
import datetime
import warnings
import time
from typing import List, Dict
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

# Project imports
from ai.meta_labelling.meta_label_generator import MetaLabelGenerator
from ai.regime_detector import Regime
from backtest.data_fetcher import DataFetcher
from backtest.signals import add_all_indicators
from ai.xgboost_trainer import generate_features, FEATURE_COLS


class WFVTrainer:
    def __init__(self, timeframe: str = "1h", n_splits: int = 5, threshold: float = 0.60):
        self.timeframe = timeframe
        self.n_splits = n_splits
        self.threshold = threshold
        self.generator = MetaLabelGenerator(timeframe=timeframe)
        self.purge_bars = 10  # Meta-label forward window size

    def validate_regime(self, regime: Regime, symbols: List[str]) -> Dict:
        """
        Walk-Forward Validation for each symbol, leak-free and methodological.
        """
        print(f"\n{'='*60}")
        print(f" [WFV] VALIDATION: {regime.value} ({self.timeframe})")
        print(f"{'='*60}")

        fetcher = DataFetcher('binance')
        all_fold_results = []
        
        for sym in symbols:
            try:
                # 1. Raw data fetch and index prep
                raw_df = fetcher.fetch_ohlcv(sym, self.timeframe, limit=3000)
                if raw_df is None or len(raw_df) < 500:
                    continue

                # DatetimeIndex guarantee
                if not isinstance(raw_df.index, pd.DatetimeIndex):
                    if 'timestamp' in raw_df.columns:
                        raw_df.set_index('timestamp', inplace=True)
                    else:
                        raw_df.index = pd.to_datetime(raw_df.index)
                
                raw_df = raw_df.sort_index()
                print(f" [DATA] Processing {sym}...")

                # 2. TimeSeriesSplit (symbol-level isolation)
                tscv = TimeSeriesSplit(n_splits=self.n_splits)
                
                for fold, (train_idx, test_idx) in enumerate(tscv.split(raw_df)):
                    # 3. Prevent indicator leakage (fold-level feature gen)
                    buffer = 200
                    start_idx = max(0, train_idx[0] - buffer)
                    
                    # Only take this fold's window (with lookback buffer)
                    fold_raw = raw_df.iloc[start_idx : test_idx[-1] + 1].copy()
                    original_index = fold_raw.index.copy()
                    
                    # Feature gen and meta-labeling (isolated within window)
                    fold_data = add_all_indicators(fold_raw)
                    fold_data = generate_features(fold_data)
                    
                    # INDEX GUARD: preserve index integrity
                    if not fold_data.index.equals(original_index):
                        fold_data = fold_data.reindex(original_index).ffill()
                    
                    meta_df, _ = self.generator.generate(fold_data, symbol=sym, verbose=False)
                    if meta_df.empty:
                        continue

                    # Regime filter
                    regime_df = meta_df[meta_df['_regime'] == regime.value]
                    if len(regime_df) < 30:
                        continue

                    # 4. Split by TSS index boundaries (expanding window)
                    train_boundary_ts = raw_df.index[train_idx[-1]]
                    
                    fold_train = regime_df[regime_df.index <= train_boundary_ts]
                    fold_test = regime_df[regime_df.index > train_boundary_ts]

                    # 5. PURGING: prevent train tail labels from peeking into test
                    if len(fold_train) > self.purge_bars:
                        fold_train = fold_train.iloc[:-self.purge_bars]

                    if len(fold_train) < 30 or len(fold_test) < 10:
                        continue

                    # Safe feature column selection
                    available_cols = [c for c in FEATURE_COLS if c in fold_train.columns]
                    if len(available_cols) < 5:
                        print(f"   [WARN] {sym} fold {fold}: Too few features ({len(available_cols)})")
                        continue

                    X_train = fold_train[available_cols].fillna(0).astype('float32')
                    y_train = fold_train['meta_label']
                    X_test = fold_test[available_cols].fillna(0).astype('float32')
                    y_test = fold_test['meta_label']

                    # 6. Model training (optimized params)
                    model = XGBClassifier(
                        n_estimators=100,
                        max_depth=4,
                        learning_rate=0.01,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        eval_metric='logloss',
                        verbosity=0,
                        random_state=42
                    )
                    model.fit(X_train, y_train)

                    # 7. Prediction and scoring (crash-protected)
                    probs = model.predict_proba(X_test)[:, 1]
                    
                    if y_test.nunique() < 2:
                        auc = 0.5
                    else:
                        auc = roc_auc_score(y_test, probs)
                    
                    # Threshold application (high-confidence signals only)
                    preds = (probs > self.threshold).astype(int)
                    fold_wr = float(y_test[preds == 1].mean()) if preds.sum() > 0 else 0.0

                    all_fold_results.append({
                        'symbol': sym,
                        'fold': fold,
                        'auc': auc,
                        'wr': fold_wr,
                        'samples': len(y_test),
                        'passed': int(preds.sum())
                    })

            except Exception as e:
                print(f" [WARN] Error loading {sym}: {e}")

        # 8. Statistical reporting
        return self._summarize(all_fold_results, regime.value)

    def _summarize(self, results: List[Dict], regime_name: str) -> Dict:
        if not results:
            print(f" [FAIL] Insufficient results (Regime: {regime_name})")
            return {'is_reliable': False, 'avg_wr': 0}

        avg_auc = np.mean([r['auc'] for r in results])
        all_wrs = [r['wr'] for r in results]  # Include 0s (survivor bias prevention)
        avg_wr = np.mean(all_wrs)
        std_wr = np.std(all_wrs) if len(all_wrs) > 1 else 0
        active_folds = sum(1 for r in results if r['passed'] > 0)

        print(f"{'-'*60}")
        print(f" [REPORT] REGIME: {regime_name.upper()}")
        print(f" [REPORT] AVG AUC: {avg_auc:.3f}")
        print(f" [REPORT] AVG WR : {avg_wr:.1%} (+/-{std_wr:.1%})")
        print(f" [REPORT] ACTIVE FOLDS: {active_folds}/{len(results)}")
        
        is_reliable = avg_wr > 0.55 and std_wr < 0.12 and active_folds > (len(results) * 0.5)
        
        if is_reliable:
            print(" [OK] Model is reliable and stable.")
        else:
            print(" [FAIL] Insufficient performance or high variance.")

        return {
            'regime': regime_name,
            'avg_auc': avg_auc,
            'avg_wr': avg_wr,
            'std_wr': std_wr,
            'is_reliable': is_reliable,
            'active_folds': active_folds,
            'total_samples': sum(r['samples'] for r in results)
        }

if __name__ == "__main__":
    # Test run
    trainer = WFVTrainer(n_splits=5, threshold=0.65)
    trainer.validate_regime(Regime.MEAN_REVERTING, ["BTCUSDT", "ETHUSDT", "SOLUSDT"])
