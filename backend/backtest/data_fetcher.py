import ccxt
import pandas as pd
from typing import Optional

class DataFetcher:
    def __init__(self, exchange_id: str = 'binance'):
        # CCXT exchange instance'ını asenkron modda başlatıyoruz
        exchange_class = getattr(ccxt, exchange_id)
        self.exchange = exchange_class({
            'enableRateLimit': True,
            'timeout': 15000,
        })
        
    def fetch_ohlcv(self, symbol: str, timeframe: str, since: Optional[str] = None, limit: int = 1000) -> pd.DataFrame:
        """
        Belirtilen parite için OHLCV verisi çeker ve Pandas DataFrame döner.
        Limit 1000'den büyükse, sayfalama (pagination) yaparak geçmişe doğru çeker.
        """
        try:
            timeframe_specs = self.exchange.timeframes
            # Timeframe'i milisaniyeye çevirme
            tf_ms = self.exchange.parse_timeframe(timeframe) * 1000
            
            since_timestamp = None
            if since:
                since_timestamp = self.exchange.parse8601(since)
                
            all_ohlcv = []
            max_limit = 1000 # Binance ve çoğu borsa için
            
            # Eğer 'since' verilmemişse, şu andan itibaren geriye dönük hesapla
            if not since_timestamp:
                now = self.exchange.milliseconds()
                target_since = now - (limit * tf_ms)
                current_since = target_since
            else:
                current_since = since_timestamp
                
            remaining = limit
            
            while remaining > 0:
                fetch_limit = min(remaining, max_limit)
                
                ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, since=current_since, limit=fetch_limit)
                
                if not ohlcv:
                    break
                    
                all_ohlcv.extend(ohlcv)
                remaining -= len(ohlcv)
                
                # Bir sonraki fetch için since parametresini güncelle
                # Son alınan mumun zaman damgasına grafiğin süresini ekle
                current_since = ohlcv[-1][0] + tf_ms 
                
                # API limitlerine takılmamak için biraz bekle
                self.exchange.sleep(self.exchange.rateLimit / 1000)
                
            df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Çok fazla gelmişse veya duplicates varsa temizle
            df = df[~df.index.duplicated(keep='first')]
            
            return df.tail(limit)
            
        except Exception as e:
            print(f"Error fetching data from {self.exchange.id} for {symbol}: {e}")
            return pd.DataFrame()
            
    def fetch_multi_tf(self, symbol: str, timeframes: list = None, limit: int = 500) -> dict:
        """
        Çoklu timeframe veri çekme (Phase 1: 4h Trend + Primary).
        """
        if timeframes is None:
            timeframes = ['4h', '1h', '15m']
            
        result = {}
        for tf in timeframes:
            try:
                df = self.fetch_ohlcv(symbol, tf, limit=limit)
                if not df.empty:
                    result[tf] = df
            except Exception as e:
                print(f"Error fetching {tf} for {symbol}: {e}")
        return result

    def close(self):
        """Exchange session'ı kapatır"""
        self.exchange.close()

# Kullanım örneği için (Test amacıyla)
def test_fetch():
    fetcher = DataFetcher('binance')
    print("Fetching BTC/USDT 1h data...")
    df = fetcher.fetch_ohlcv('BTC/USDT', '1h', limit=5)
    print(df)

if __name__ == "__main__":
    test_fetch()
