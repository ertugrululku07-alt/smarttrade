import networkx as nx
import numpy as np
from typing import Dict, List, Tuple, Optional
from web3 import Web3
from collections import defaultdict
from loguru import logger
import asyncio

class OnChainAnalyzer:
    """Zincir içi verileri analiz eder"""
    
    def __init__(self, w3: Web3, moralis_api_key: str):
        self.w3 = w3
        self.moralis_api_key = moralis_api_key
        self.holder_graph = nx.DiGraph()
        self.risk_thresholds = {
            'concentration': 0.3,      # %30'dan fazlası tek cüzdanda
            'dev_dump': 0.05,           # %5'ten fazla dev satışı
            'liquidity_lock': 0.5,       # %50'den azı kilitli
            'mint_authority': 0.7,       # Mint yetkisi açıksa
            'honeypot': 0.8               # Satış engeli varsa
        }
        
    async def analyze_token(self, token_address: str, chain: str = 'ethereum') -> Dict:
        """Bir token'ı detaylı analiz et"""
        
        logger.info(f"Analyzing token: {token_address} on {chain}")
        
        results = {
            'token_address': token_address,
            'chain': chain,
            'risk_score': 0,
            'signals': [],
            'metrics': {}
        }
        
        try:
            # 1. Holder konsantrasyonu analizi
            concentration_score = await self.analyze_holder_concentration(token_address)
            results['metrics']['concentration'] = concentration_score
            
            if concentration_score > self.risk_thresholds['concentration']:
                results['signals'].append({
                    'type': 'HIGH_CONCENTRATION',
                    'severity': 'high',
                    'message': f"Holder concentration: {concentration_score:.2%}"
                })
                results['risk_score'] += 30
                
            # 2. Dev cüzdan aktivitesi
            dev_activity_score = await self.analyze_dev_wallets(token_address)
            results['metrics']['dev_activity'] = dev_activity_score
            
            if dev_activity_score > self.risk_thresholds['dev_dump']:
                results['signals'].append({
                    'type': 'DEV_DUMPING',
                    'severity': 'critical',
                    'message': f"Dev wallets dumped {dev_activity_score:.2%}"
                })
                results['risk_score'] += 50
                
            # 3. Likidite analizi
            liquidity_score = await self.analyze_liquidity(token_address)
            results['metrics']['liquidity'] = liquidity_score
            
            if liquidity_score < self.risk_thresholds['liquidity_lock']:
                results['signals'].append({
                    'type': 'LIQUIDITY_UNLOCKED',
                    'severity': 'critical',
                    'message': f"Liquidity lock: {liquidity_score:.2%}"
                })
                results['risk_score'] += 40
                
            # 4. Akıllı kontrat analizi
            contract_score = await self.analyze_contract(token_address)
            results['metrics']['contract'] = contract_score
            
            if contract_score > self.risk_thresholds['honeypot']:
                results['signals'].append({
                    'type': 'HONEYPOT',
                    'severity': 'critical',
                    'message': "Contract has honeypot characteristics"
                })
                results['risk_score'] += 80
                
            # Normalize risk score (0-100)
            results['risk_score'] = min(results['risk_score'], 100)
            
        except Exception as e:
            logger.error(f"Token analysis failed: {e}")
            results['risk_score'] = -1  # Error
            
        return results
        
    async def analyze_holder_concentration(self, token_address: str) -> float:
        """Top holder konsantrasyonunu analiz et"""
        
        try:
            # Moralis ile top holder'ları çek
            # Bu kısım gerçek API çağrısı gerektirir
            # Simülasyon için dummy data:
            
            # Top 10 holder yüzdesi
            top_10_percentage = np.random.uniform(0.2, 0.8)
            
            return top_10_percentage
            
        except Exception as e:
            logger.error(f"Holder concentration analysis failed: {e}")
            return 0.5  # Orta risk varsay
            
    async def analyze_dev_wallets(self, token_address: str) -> float:
        """Geliştirici cüzdanlarını analiz et"""
        
        try:
            # Son 24 saatte dev satışlarını bul
            # Simülasyon:
            
            dev_sales = np.random.uniform(0, 0.2)
            return dev_sales
            
        except Exception as e:
            logger.error(f"Dev wallet analysis failed: {e}")
            return 0.1
            
    async def analyze_liquidity(self, token_address: str) -> float:
        """Likidite durumunu analiz et"""
        
        try:
            # Likidite kilidi kontrolü
            # Simülasyon:
            
            locked_percentage = np.random.uniform(0, 1)
            return locked_percentage
            
        except Exception as e:
            logger.error(f"Liquidity analysis failed: {e}")
            return 0.3
            
    async def analyze_contract(self, token_address: str) -> float:
        """Akıllı kontrat kodunu analiz et"""
        
        try:
            # Kontrat bytecode'unu getir
            bytecode = self.w3.eth.get_code(Web3.to_checksum_address(token_address))
            
            # Tehlikeli fonksiyon imzalarını ara
            dangerous_signatures = [
                '0x40c10f19',  # mint
                '0x2e1a7d4d',  # withdraw
                '0x8da5cb5b',  # owner
                '0xf2fde38b',  # transferOwnership
                '0x95d89b41',  # symbol
            ]
            
            honeypot_score = 0
            for sig in dangerous_signatures:
                if sig in bytecode.hex():
                    honeypot_score += 0.2
                    
            return min(honeypot_score, 1.0)
            
        except Exception as e:
            logger.error(f"Contract analysis failed: {e}")
            return 0.5
            
    async def build_holder_graph(self, token_address: str, depth: int = 2) -> nx.DiGraph:
        """Holder'lar arasındaki ilişki grafiğini oluştur"""
        
        G = nx.DiGraph()
        
        try:
            # Token holder'larını getir
            # Simülasyon:
            
            for i in range(100):
                G.add_node(f"holder_{i}", balance=np.random.uniform(0, 1000))
                
            # Transferleri ekle
            for i in range(200):
                from_node = f"holder_{np.random.randint(0, 100)}"
                to_node = f"holder_{np.random.randint(0, 100)}"
                amount = np.random.uniform(1, 100)
                
                G.add_edge(from_node, to_node, amount=amount)
                
        except Exception as e:
            logger.error(f"Holder graph building failed: {e}")
            
        return G