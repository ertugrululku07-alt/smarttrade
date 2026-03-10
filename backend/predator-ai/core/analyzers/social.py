import asyncio
import aiohttp
from transformers import pipeline
import numpy as np
from typing import Dict, List, Tuple
from collections import Counter
import re
from loguru import logger

class BotActivityDetector:
    """Sosyal medyadaki bot aktivitelerini tespit eder"""
    
    def __init__(self):
        self.bot_patterns = [
            r'\b(buy|sell|pump|dump|moon|lambo|gem|100x)\b',  # Spam kelimeler
            r'https?://\S+',  # Linkler
            r'@\w+',  # Mention'lar
        ]
        
    def detect(self, texts: List[str]) -> float:
        """Verilen metinlerdeki bot oranını hesapla"""
        
        bot_count = 0
        total_count = len(texts)
        
        for text in texts:
            bot_score = 0
            
            # Pattern eşleştirme
            for pattern in self.bot_patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    bot_score += 0.3
                    
            # Çok kısa mesajlar
            if len(text.split()) < 5:
                bot_score += 0.2
                
            # Çok fazla emoji
            emoji_count = len(re.findall(r'[^\w\s]', text))
            if emoji_count > 5:
                bot_score += 0.2
                
            if bot_score > 0.5:
                bot_count += 1
                
        return bot_count / total_count if total_count > 0 else 0

class SocialSentimentAnalyzer:
    """Sosyal medya duygu analizi"""
    
    def __init__(self):
        # Sentiment analysis pipeline
        self.sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest"
        )
        
        # Bot dedektörü
        self.bot_detector = BotActivityDetector()
        
        # Cache
        self.sentiment_cache = {}
        self.cache_ttl = 300  # 5 dakika
        
    async def analyze_project(self, project_name: str, keywords: List[str]) -> Dict:
        """Bir proje hakkındaki sosyal medya duyarlılığını analiz et"""
        
        logger.info(f"Analyzing social sentiment for: {project_name}")
        
        # Farklı platformlardan veri topla
        tasks = [
            self.analyze_twitter(project_name, keywords),
            self.analyze_telegram(project_name, keywords),
            self.analyze_discord(project_name, keywords)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Sonuçları birleştir
        all_texts = []
        platform_scores = []
        
        for result in results:
            if isinstance(result, dict):
                all_texts.extend(result.get('texts', []))
                platform_scores.append(result)
                
        # Genel duygu analizi
        if all_texts:
            sentiments = self.sentiment_pipeline(all_texts[:100])  # İlk 100 ile sınırla
            
            sentiment_score = np.mean([
                1 if s['label'] == 'POSITIVE' else -1 if s['label'] == 'NEGATIVE' else 0
                for s in sentiments
            ])
            
            # Güven skoru
            confidence = np.mean([s['score'] for s in sentiments])
            
        else:
            sentiment_score = 0
            confidence = 0
            
        # Bot aktivitesi
        bot_ratio = self.bot_detector.detect(all_texts)
        
        # Hype metrikleri
        mention_count = len(all_texts)
        unique_users = len(set([t.get('user_id') for t in results if 'user_id' in t]))
        
        return {
            'project': project_name,
            'sentiment_score': float(sentiment_score),
            'confidence': float(confidence),
            'bot_activity': float(bot_ratio),
            'mention_count': mention_count,
            'unique_users': unique_users,
            'platforms': platform_scores
        }
        
    async def analyze_twitter(self, project_name: str, keywords: List[str]) -> Dict:
        """Twitter analizi"""
        
        # Burada Twitter API çağrıları yapılacak
        # Simülasyon:
        
        await asyncio.sleep(1)
        
        texts = [
            f"This {project_name} project looks amazing! #crypto #gem",
            f"Just invested in {project_name}, moon soon! 🚀",
            f"Warning: {project_name} might be a scam, do your own research",
            f"{project_name} team is very responsive in Telegram, great project!"
        ]
        
        return {
            'platform': 'twitter',
            'texts': texts,
            'user_ids': [f"user_{i}" for i in range(len(texts))]
        }
        
    async def analyze_telegram(self, project_name: str, keywords: List[str]) -> Dict:
        """Telegram analizi"""
        
        # Simülasyon
        await asyncio.sleep(1)
        
        texts = [
            f"{project_name} to the moon! 🚀",
            f"When {project_name} listing on Binance?",
            f"Dev just sold some tokens, be careful!"
        ]
        
        return {
            'platform': 'telegram',
            'texts': texts,
            'user_ids': [f"tg_user_{i}" for i in range(len(texts))]
        }
        
    async def analyze_discord(self, project_name: str, keywords: List[str]) -> Dict:
        """Discord analizi"""
        
        # Simülasyon
        await asyncio.sleep(1)
        
        texts = [
            f"Check out the new {project_name} whitepaper",
            f"AMA with {project_name} team tomorrow",
            f"Price update: {project_name} up 20% today"
        ]
        
        return {
            'platform': 'discord',
            'texts': texts,
            'user_ids': [f"dc_user_{i}" for i in range(len(texts))]
        }