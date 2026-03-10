import asyncio
import aiohttp
import websockets
from typing import Dict, List, Optional, Any
from web3 import Web3
from moralis import evm_api
from solana.rpc.async_api import AsyncClient
from loguru import logger
import json
import redis
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

class MempoolWatcher:
    """Mempool'daki bekleyen işlemleri izler"""
    
    def __init__(self, w3: Web3):
        self.w3 = w3
        self.pending_txs = []
        self.interesting_patterns = [
            '0x095ea7b3',  # approve
            '0x40c10f19',  # mint
            '0x89afcb44',  # transferOwnership
            '0x2e1a7d4d',  # withdraw
        ]
        
    async def watch_mempool(self):
        """Mempool'u dinle ve şüpheli işlemleri yakala"""
        pending_filter = self.w3.eth.filter('pending')
        
        while True:
            try:
                pending_hashes = pending_filter.get_new_entries()
                
                for tx_hash in pending_hashes:
                    try:
                        tx = self.w3.eth.get_transaction(tx_hash)
                        
                        # Şüpheli işlem kontrolü
                        if self.is_suspicious_transaction(tx):
                            self.pending_txs.append({
                                'hash': tx_hash.hex(),
                                'from': tx['from'],
                                'to': tx['to'],
                                'value': tx['value'],
                                'input': tx['input'][:10],  # İlk 10 byte
                                'timestamp': datetime.now()
                            })
                            
                            # Alert gönder
                            await self.alert_suspicious_tx(tx)
                            
                    except Exception as e:
                        continue
                        
            except Exception as e:
                logger.error(f"Mempool watch error: {e}")
                await asyncio.sleep(1)

class DEXPairListener:
    """Yeni DEX pair'lerini dinler"""
    
    def __init__(self, w3: Web3, factory_addresses: Dict[str, str]):
        self.w3 = w3
        self.factories = factory_addresses
        self.new_pairs = []
        
        # Uniswap V2 PairCreated event signature
        self.pair_created_event = '0x0d3648bd0f6ba80134a33ba9275ac585d9d315f0ad8355cddefde31afa28d0e9'
        
    async def listen_new_pairs(self):
        """Yeni oluşturulan token pair'lerini dinle"""
        
        for chain, factory in self.factories.items():
            contract = self.w3.eth.contract(
                address=factory,
                abi=self.get_factory_abi()
            )
            
            # Event filter oluştur
            event_filter = contract.events.PairCreated.create_filter(
                fromBlock='latest'
            )
            
            while True:
                try:
                    for event in event_filter.get_new_entries():
                        pair_info = {
                            'chain': chain,
                            'token0': event.args.token0,
                            'token1': event.args.token1,
                            'pair': event.args.pair,
                            'timestamp': datetime.now()
                        }
                        
                        self.new_pairs.append(pair_info)
                        
                        # Yeni token'ı hemen analiz et
                        await self.analyze_new_token(pair_info)
                        
                except Exception as e:
                    logger.error(f"New pair listener error: {e}")
                    
                await asyncio.sleep(1)

class DataHarvester:
    """Ana veri toplama katmanı"""
    
    def __init__(self, config):
        self.config = config
        self.redis_client = redis.Redis.from_url(config.db.redis_url)
        
        # Web3 bağlantıları
        self.w3_eth = Web3(Web3.HTTPProvider(config.web3.eth_rpc))
        self.w3_bsc = Web3(Web3.HTTPProvider(config.web3.bsc_rpc))
        self.solana_client = AsyncClient(config.web3.solana_rpc)
        
        # İzleyiciler
        self.mempool_watcher = MempoolWatcher(self.w3_eth)
        self.dex_listener = DEXPairListener(
            self.w3_eth,
            {
                'ethereum': '0x5C69bEe701ef814a2B6a3EDD4B1652CB9cc5aA6f',  # Uniswap V2
                'bsc': '0xcA143Ce32Fe78f1f7019d7d551a6402fC5350c73',  # PancakeSwap
            }
        )
        
        # Data buffers
        self.buffer_size = 1000
        self.onchain_buffer = []
        self.social_buffer = []
        self.market_buffer = []
        
    async def harvest_onchain(self):
        """Zincir üstü verileri topla"""
        
        # Son blokları getir
        latest_block = self.w3_eth.eth.block_number
        
        # Son 10 bloğu analiz et
        for block_num in range(latest_block - 10, latest_block):
            try:
                block = self.w3_eth.eth.get_block(block_num, full_transactions=True)
                
                # Büyük işlemleri bul (> 100k USD)
                large_txs = []
                for tx in block.transactions:
                    if tx['value'] > 10**18:  # 1 ETH
                        tx_info = {
                            'hash': tx['hash'].hex(),
                            'from': tx['from'],
                            'to': tx['to'],
                            'value': tx['value'],
                            'block': block_num,
                            'timestamp': datetime.now()
                        }
                        large_txs.append(tx_info)
                        
                        # Redis'e kaydet
                        self.redis_client.lpush(
                            f"large_txs:{block_num}",
                            json.dumps(tx_info)
                        )
                
                self.onchain_buffer.append({
                    'block': block_num,
                    'tx_count': len(block.transactions),
                    'large_txs': large_txs,
                    'gas_used': block.gasUsed,
                    'timestamp': datetime.now()
                })
                
            except Exception as e:
                logger.error(f"Block harvest error: {e}")
                
        # Buffer boyutunu kontrol et
        if len(self.onchain_buffer) > self.buffer_size:
            self.onchain_buffer = self.onchain_buffer[-self.buffer_size:]
            
    async def harvest_social(self):
        """Sosyal medya verilerini topla"""
        
        # Twitter'dan trend topicleri çek
        async with aiohttp.ClientSession() as session:
            headers = {
                'Authorization': f'Bearer {self.config.social.twitter_bearer_token}'
            }
            
            # Kripto ile ilgili tweet'leri ara
            queries = [
                'crypto OR bitcoin OR ethereum',
                '$BTC OR $ETH OR $SOL',
                'new token OR presale OR airdrop',
                'rug pull OR scam OR honeypot'
            ]
            
            for query in queries:
                params = {
                    'query': query,
                    'max_results': 100,
                    'tweet.fields': 'created_at,public_metrics,author_id'
                }
                
                async with session.get(
                    'https://api.twitter.com/2/tweets/search/recent',
                    headers=headers,
                    params=params
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        for tweet in data.get('data', []):
                            self.social_buffer.append({
                                'platform': 'twitter',
                                'content': tweet['text'],
                                'author': tweet['author_id'],
                                'metrics': tweet['public_metrics'],
                                'timestamp': datetime.now()
                            })
                            
            await asyncio.sleep(60)  # Rate limit
            
    async def harvest_market(self):
        """Piyasa verilerini topla (order book, trades)"""
        
        # Binance WebSocket'ine bağlan
        async with websockets.connect('wss://stream.binance.com:9443/ws') as ws:
            # 20 popüler pariteyi dinle
            symbols = ['btcusdt', 'ethusdt', 'bnbusdt', 'solusdt', 'xrpusdt']
            
            for symbol in symbols:
                # Depth stream (order book)
                await ws.send(json.dumps({
                    'method': 'SUBSCRIBE',
                    'params': [f'{symbol}@depth20@100ms'],
                    'id': 1
                }))
                
                # Trade stream
                await ws.send(json.dumps({
                    'method': 'SUBSCRIBE',
                    'params': [f'{symbol}@trade'],
                    'id': 2
                }))
                
            while True:
                try:
                    message = await ws.recv()
                    data = json.loads(message)
                    
                    if 'bids' in data:  # Depth update
                        self.market_buffer.append({
                            'type': 'depth',
                            'symbol': data['s'].lower(),
                            'bids': data['bids'][:10],  # İlk 10 bid
                            'asks': data['asks'][:10],  # İlk 10 ask
                            'timestamp': datetime.now()
                        })
                        
                    elif 'p' in data:  # Trade update
                        self.market_buffer.append({
                            'type': 'trade',
                            'symbol': data['s'].lower(),
                            'price': float(data['p']),
                            'quantity': float(data['q']),
                            'side': 'buy' if data['m'] else 'sell',
                            'timestamp': datetime.now()
                        })
                        
                except Exception as e:
                    logger.error(f"Market harvest error: {e}")
                    break
                    
    async def get_features(self) -> pd.DataFrame:
        """Tüm toplanan verilerden feature matrix oluştur"""
        
        features = []
        
        # On-chain features
        if self.onchain_buffer:
            df_onchain = pd.DataFrame(self.onchain_buffer)
            features.append({
                'large_tx_count': len(df_onchain['large_txs'].sum()),
                'avg_gas': df_onchain['gas_used'].mean(),
                'tx_volume': df_onchain['tx_count'].sum()
            })
            
        # Social features
        if self.social_buffer:
            df_social = pd.DataFrame(self.social_buffer)
            features.append({
                'tweet_volume': len(df_social),
                'avg_likes': df_social['metrics'].apply(
                    lambda x: x.get('like_count', 0)
                ).mean()
            })
            
        # Market features
        if self.market_buffer:
            df_market = pd.DataFrame(self.market_buffer)
            trades = df_market[df_market['type'] == 'trade']
            
            if not trades.empty:
                features.append({
                    'price_volatility': trades['price'].std() / trades['price'].mean(),
                    'buy_sell_ratio': len(trades[trades['side']=='buy']) / len(trades),
                    'trade_frequency': len(trades) / 60  # trades per minute
                })
                
        return pd.DataFrame([features]) if features else pd.DataFrame()
        
    async def run(self):
        """Ana döngü - tüm harvest işlemlerini başlat"""
        
        logger.info("Starting Data Harvester...")
        
        tasks = [
            self.harvest_onchain(),
            self.harvest_social(),
            self.harvest_market(),
            self.mempool_watcher.watch_mempool(),
            self.dex_listener.listen_new_pairs()
        ]
        
        await asyncio.gather(*tasks)