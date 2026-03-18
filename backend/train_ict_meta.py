"""
ICT/SMC Meta-Model Eğitim Scripti
──────────────────────────────────
1. Top coin'lerden 1h + 4h veri çek
2. ICT label generator ile meta-label üret
3. Meta-trainer ile LightGBM modelleri eğit
"""

import sys
import time
import traceback
from pathlib import Path

backend_dir = str(Path(__file__).resolve().parent)
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

import pandas as pd
from backtest.data_fetcher import DataFetcher
from backtest.signals import add_all_indicators, add_meta_context_features
from ai.meta_labelling.ict_label_generator import ICTLabelGenerator
from ai.meta_labelling.meta_trainer import MetaTrainer

# ── Config ──
SYMBOLS = [
    "BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT",
    "ADA/USDT", "DOGE/USDT", "AVAX/USDT", "DOT/USDT", "LINK/USDT",
    "LTC/USDT", "UNI/USDT", "ATOM/USDT", "FIL/USDT", "ARB/USDT",
    "OP/USDT", "INJ/USDT", "APT/USDT", "NEAR/USDT", "FTM/USDT",
]
LIMIT_1H = 5000
LIMIT_4H = 2000
MODEL_DIR = "ai/models/meta"
N_TRIALS = 25

def main():
    print("=" * 60)
    print("  ICT/SMC META-MODEL EĞİTİMİ")
    print("=" * 60)

    fetcher = DataFetcher('binance')
    all_meta_dfs = []
    total_start = time.time()

    for idx, sym in enumerate(SYMBOLS, 1):
        print(f"\n[{idx}/{len(SYMBOLS)}] {sym}")

        try:
            # 1h veri
            raw_1h = fetcher.fetch_ohlcv(sym, '1h', limit=LIMIT_1H)
            if raw_1h is None or len(raw_1h) < 500:
                print(f"  [SKIP] Yetersiz 1h veri")
                continue

            if not isinstance(raw_1h.index, pd.DatetimeIndex):
                if 'timestamp' in raw_1h.columns:
                    raw_1h.set_index('timestamp', inplace=True)
                else:
                    raw_1h.index = pd.to_datetime(raw_1h.index)
            raw_1h = raw_1h.sort_index()

            df_1h = add_all_indicators(raw_1h)
            try:
                df_1h = add_meta_context_features(df_1h)
            except Exception:
                pass

            print(f"  1h: {len(df_1h)} bars")

            # 4h veri
            df_4h = None
            try:
                raw_4h = fetcher.fetch_ohlcv(sym, '4h', limit=LIMIT_4H)
                if raw_4h is not None and len(raw_4h) >= 200:
                    if not isinstance(raw_4h.index, pd.DatetimeIndex):
                        if 'timestamp' in raw_4h.columns:
                            raw_4h.set_index('timestamp', inplace=True)
                        else:
                            raw_4h.index = pd.to_datetime(raw_4h.index)
                    raw_4h = raw_4h.sort_index()
                    df_4h = add_all_indicators(raw_4h)
                    print(f"  4h: {len(df_4h)} bars")
            except Exception as e:
                print(f"  [WARN] 4h hata: {e}")

            # ICT Label Generation
            gen = ICTLabelGenerator(
                lookback=60,
                lookahead=16,
                min_structs=2,
                progress_interval=2000,
            )

            meta_df, stats = gen.generate(
                df_1h, df_4h=df_4h, symbol=sym, verbose=True
            )

            if not meta_df.empty:
                meta_df['_symbol'] = sym
                all_meta_dfs.append(meta_df)
                print(f"  ✅ {len(meta_df)} label üretildi")
            else:
                print(f"  [SKIP] Label üretilemedi")

        except Exception as e:
            print(f"  [ERROR] {sym}: {e}")
            traceback.print_exc()

    if not all_meta_dfs:
        print("\n[FAIL] Hiç meta-data üretilemedi!")
        return

    # Birleştir
    total_meta_df = pd.concat(all_meta_dfs, axis=0, ignore_index=True)
    print(f"\n{'=' * 60}")
    print(f"  TOPLAM: {len(total_meta_df)} sample, "
          f"{len(total_meta_df.columns)} kolon")

    # Label dağılımı
    if 'meta_label' in total_meta_df.columns:
        n1 = (total_meta_df['meta_label'] == 1).sum()
        n0 = (total_meta_df['meta_label'] == 0).sum()
        print(f"  Labels: Correct={n1} Wrong={n0} "
              f"WR={n1/(n1+n0)*100:.1f}%")

    # ICT feature varlık kontrolü
    ict_cols = [c for c in total_meta_df.columns if c.startswith('ict_')]
    htf_cols = [c for c in total_meta_df.columns if c.startswith('htf_')]
    print(f"  ICT features: {len(ict_cols)}")
    print(f"  HTF features: {len(htf_cols)}")

    # Regime dağılımı
    regime_col = '_regime'
    if regime_col in total_meta_df.columns:
        print(f"  Regimes: {total_meta_df[regime_col].value_counts().to_dict()}")

    print(f"{'=' * 60}")

    # ── Eğitim ──
    trainer = MetaTrainer(
        model_dir=MODEL_DIR,
        use_mtf=True,
        optimize=True,
        n_trials=N_TRIALS,
    )

    # Regime kolonu düzelt
    if '_regime' in total_meta_df.columns and 'regime' not in total_meta_df.columns:
        total_meta_df['regime'] = total_meta_df['_regime']

    results = trainer.train_all_regimes(
        total_meta_df,
        timeframe="1h",
    )

    elapsed = time.time() - total_start
    print(f"\n[DONE] Toplam süre: {elapsed:.0f}s")

    return results


if __name__ == "__main__":
    main()
