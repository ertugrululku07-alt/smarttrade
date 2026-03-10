import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from collections import deque
from loguru import logger

class MarketMicrostructureAnalyzer:
    """Piyasa mikroyapısını analiz eder"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.order_book_history = deque(maxlen=window_size)
        self.trade_history = deque(maxlen=window_size)
        
    async def analyze_order_book(self, bids: List[List[float]], asks: List[List[float]]) -> Dict:
        """Order book analizi"""
        
        bids = np.array(bids, dtype=float)
        asks = np.array(asks, dtype=float)
        
        # Bid-ask spread
        best_bid = bids[0][0] if len(bids) > 0 else 0
        best_ask = asks[0][0] if len(asks) > 0 else 0
        spread = best_ask - best_bid
        spread_pct = spread / best_bid if best_bid > 0 else 0
        
        # Order book imbalance
        bid_volume = np.sum(bids[:, 1])
        ask_volume = np.sum(asks[:, 1])
        
        if bid_volume + ask_volume > 0:
            imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume)
        else:
            imbalance = 0
            
        # Depth (market derinliği)
        depth_bid_10pct = self.calculate_depth(bids, best_bid * 0.9)
        depth_ask_10pct = self.calculate_depth(asks, best_ask * 1.1)
        
        return {
            'spread': float(spread),
            'spread_pct': float(spread_pct),
            'imbalance': float(imbalance),
            'depth_bid': float(depth_bid_10pct),
            'depth_ask': float(depth_ask_10pct),
            'bid_volume': float(bid_volume),
            'ask_volume': float(ask_volume)
        }
        
    def calculate_depth(self, orders: np.ndarray, price_level: float) -> float:
        """Belirli fiyat seviyesine kadar olan derinliği hesapla"""
        
        if len(orders) == 0:
            return 0
            
        # price_level altındaki/kalan tüm emirlerin hacmi
        if orders[0][0] < price_level:  # Bid
            mask = orders[:, 0] >= price_level
        else:  # Ask
            mask = orders[:, 0] <= price_level
            
        return float(np.sum(orders[mask, 1]))
        
    async def detect_spoofing(self, order_book_history: List) -> float:
        """Spoofing (sahte emir) tespiti"""
        
        if len(order_book