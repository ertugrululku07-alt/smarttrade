"""
Walk-Forward Validation (WFV) Trainer v1.0

Bu araç, meta-labelling modellerini sadece tek bir train/test split ile değil,
zaman içinde kayan pencereler (Time Series Split) ile valide eder.
Bu sayede modelin farklı piyasa koşullarında (Boğa/Ayı) ne kadar tutarlı olduğu ölçülür.
"""

import os
import pandas as pd
import numpy as np
import joblib
from typing import List, Dict
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, precision_score
from xgboost import XGBClassifier

from ai.meta_labelling.meta_label_generator import MetaLabelGenerator
from ai.regime_detector import Regime

class WFVTrainer:
    def __init__(self, timeframe: str = "1h", n_splits: int = 5):
        self.timeframe = timeframe
        self.n_splits = n_splits
        self.generator = MetaLabelGenerator(timeframe=timeframe)

    def validate_regime(self, regime: Regime, symbols: List[str] = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]):
        print(f"\n{'='*60}")
        print(f" 🔍 WFV VALIDATION: {regime.value} ({self.timeframe})")
        print(f"{'='*60}")

        from backtest.data_fetcher import DataFetcher
        from backtest.signals import add_all_indicators
        from ai.xgboost_trainer import generate_features
        
        fetcher = DataFetcher('binance')
        df_all = []
        
        for sym in symbols:
            try:
                print(f"  Fetching {sym} ({self.timeframe})...")
                df = fetcher.fetch_ohlcv(sym, self.timeframe, limit=3000)
                if df is None or len(df) < 500:
                    continue
                
                df = add_all_indicators(df)
                df = generate_features(df)
                
                # Meta-label üret
                meta_df, stats = self.generator.generate(df, symbol=sym, verbose=False)
                
                if not meta_df.empty:
                    # Sadece ilgili rejimi seç
                    regime_df = meta_df[meta_df['_regime'] == regime.value]
                    if not regime_df.empty:
                        df_all.append(regime_df)
            except Exception as e:
                print(f"  Error loading {sym}: {e}")

        if not df_all:
            print("  [WFV] Yeterli veri bulunamadi (Rejim: {}).".format(regime.value))
            return

        full_df = pd.concat(df_all).sort_index()
        
        from ai.xgboost_trainer import FEATURE_COLS
        X = full_df[FEATURE_COLS]
        y = full_df['meta_label']

        # 2. TimeSeriesSplit
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        
        fold_results = []
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            if len(y_test) < 10 or y_train.nunique() < 2:
                continue

            # Model eğit (Hızlı parametrelerle)
            model = XGBClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss'
            )
            model.fit(X_train, y_train)

            # Tahmin ve Skor
            probs = model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, probs)
            
            # Threshold uygula (0.60)
            preds = (probs > 0.60).astype(int)
            filtered_wr = 0
            if preds.sum() > 0:
                filtered_wr = y_test[preds == 1].mean()

            fold_results.append({
                'fold': fold,
                'auc': auc,
                'wr': filtered_wr,
                'samples': len(y_test),
                'passed': preds.sum()
            })
            
            print(f"  Fold {fold}: Samples={len(y_test):4} | AUC={auc:.3f} | Filtered WR={filtered_wr:.1%}")

        # 3. İstatistikler
        if not fold_results:
            print("  [WFV] Fold sonuclari uretilemedi.")
            return

        avg_auc = np.mean([r['auc'] for r in fold_results])
        avg_wr = np.mean([r['wr'] for r in fold_results if r['wr'] > 0])
        std_wr = np.std([r['wr'] for r in fold_results if r['wr'] > 0])

        print(f"{'-'*60}")
        print(f"  📊 ORTALAMA AUC: {avg_auc:.3f}")
        print(f"  📊 ORTALAMA WR : {avg_wr:.1%} (±{std_wr:.1%})")
        
        if std_wr > 0.12:
            print("  ⚠️ UYARI: Model tutarsiz (Yüksek varyans!)")
        elif avg_wr > 0.55:
            print("  ✅ SONUÇ: Model güvenilir ve tutarlı.")
        else:
            print("  ❌ SONUÇ: Model performansı yetersiz.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--timeframe', type=str, default='1h')
    parser.add_argument('--regime', type=str, default='mean_reverting')
    args = parser.parse_args()

    trainer = WFVTrainer(timeframe=args.timeframe)
    trainer.validate_regime(Regime(args.regime))
