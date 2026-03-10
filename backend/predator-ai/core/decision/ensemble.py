import numpy as np
import pandas as pd
from xgboost import XGBClassifier
import torch
import torch.nn as nn
from typing import Dict, List, Tuple
import joblib
from loguru import logger
import asyncio

class LSTMModel(nn.Module):
    """LSTM tabanlı zaman serisi modeli"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True,
            dropout=0.2
        )
        self.fc = nn.Linear(hidden_dim, 3)  # 3 class (LONG, SHORT, HOLD)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        output = self.fc(last_output)
        return self.softmax(output)

class GraphNeuralNetwork(nn.Module):
    """Graf sinir ağı - holder ilişkileri için"""
    
    def __init__(self, node_features: int = 64, edge_features: int = 32):
        super().__init__()
        self.node_encoder = nn.Linear(node_features, 128)
        self.edge_encoder = nn.Linear(edge_features, 64)
        self.gnn_layer1 = nn.Linear(192, 256)
        self.gnn_layer2 = nn.Linear(256, 128)
        self.fc = nn.Linear(128, 3)
        
    def forward(self, node_features, edge_features, adjacency):
        # Basitleştirilmiş GNN forward
        x = torch.relu(self.node_encoder(node_features))
        e = torch.relu(self.edge_encoder(edge_features))
        
        # Message passing simülasyonu
        x = torch.cat([x, e.mean(dim=1)], dim=1)
        x = torch.relu(self.gnn_layer1(x))
        x = torch.relu(self.gnn_layer2(x))
        
        return torch.softmax(self.fc(x), dim=1)

class TransformerModel(nn.Module):
    """Transformer - sosyal medya analizi için"""
    
    def __init__(self, vocab_size: int = 10000, d_model: int = 256, nhead: int = 8):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, batch_first=True),
            num_layers=4
        )
        self.fc = nn.Linear(d_model, 3)
        
    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        x = x.mean(dim=1)  # Global average pooling
        return torch.softmax(self.fc(x), dim=1)

class EnsembleDecisionMaker:
    """Topluluk öğrenmesi tabanlı karar mekanizması"""
    
    def __init__(self, config):
        self.config = config
        self.weights = config.ai.ensemble_weights
        
        # Modelleri yükle
        self.models = {}
        self.load_models()
        
        # Feature scaler
        self.scaler = joblib.load('models/scaler.pkl') if os.path.exists('models/scaler.pkl') else None
        
    def load_models(self):
        """Önceden eğitilmiş modelleri yükle"""
        
        try:
            # XGBoost
            if os.path.exists('models/xgboost.pkl'):
                self.models['xgboost'] = joblib.load('models/xgboost.pkl')
            else:
                self.models['xgboost'] = XGBClassifier(n_estimators=100)
                
            # LSTM
            if os.path.exists('models/lstm.pth'):
                self.models['lstm'] = LSTMModel(input_dim=50)
                self.models['lstm'].load_state_dict(torch.load('models/lstm.pth'))
                self.models['lstm'].eval()
                
            # GNN
            if os.path.exists('models/gnn.pth'):
                self.models['graph_nn'] = GraphNeuralNetwork()
                self.models['graph_nn'].load_state_dict(torch.load('models/gnn.pth'))
                self.models['graph_nn'].eval()
                
            # Transformer
            if os.path.exists('models/transformer.pth'):
                self.models['transformer'] = TransformerModel()
                self.models['transformer'].load_state_dict(torch.load('models/transformer.pth'))
                self.models['transformer'].eval()
                
            logger.info(f"Loaded {len(self.models)} models")
            
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            
    async def predict(self, features: Dict) -> Dict:
        """Tüm modellerle tahmin yap"""
        
        predictions = {}
        confidences = {}
        
        # XGBoost tahmini
        if 'xgboost' in self.models:
            try:
                # Feature vektörü hazırla
                X = self.prepare_features(features)
                
                # Tahmin
                pred = self.models['xgboost'].predict_proba([X])[0]
                predictions['xgboost'] = pred
                confidences['xgboost'] = float(np.max(pred))
                
            except Exception as e:
                logger.error(f"XGBoost prediction failed: {e}")
                
        # LSTM tahmini
        if 'lstm' in self.models:
            try:
                # Zaman serisi feature'ları
                X_lstm = self.prepare_timeseries(features)
                
                with torch.no_grad():
                    X_tensor = torch.FloatTensor(X_lstm).unsqueeze(0)  # Add batch dim
                    pred = self.models['lstm'](X_tensor).numpy()[0]
                    
                predictions['lstm'] = pred
                confidences['lstm'] = float(np.max(pred))
                
            except Exception as e:
                logger.error(f"LSTM prediction failed: {e}")
                
        # GNN tahmini
        if 'graph_nn' in self.models:
            try:
                # Graf feature'ları
                node_features, edge_features, adjacency = self.prepare_graph_features(features)
                
                with torch.no_grad():
                    pred = self.models['graph_nn'](
                        torch.FloatTensor(node_features),
                        torch.FloatTensor(edge_features),
                        torch.FloatTensor(adjacency)
                    ).numpy()[0]
                    
                predictions['graph_nn'] = pred
                confidences['graph_nn'] = float(np.max(pred))
                
            except Exception as e:
                logger.error(f"GNN prediction failed: {e}")
                
        # Transformer tahmini
        if 'transformer' in self.models:
            try:
                # Sosyal medya token'ları
                X_text = self.prepare_text_features(features)
                
                with torch.no_grad():
                    pred = self.models['transformer'](torch.LongTensor(X_text).unsqueeze(0)).numpy()[0]
                    
                predictions['transformer'] = pred
                confidences['transformer'] = float(np.max(pred))
                
            except Exception as e:
                logger.error(f"Transformer prediction failed: {e}")
                
        # Weighted voting ile final tahmin
        final_pred = self.weighted_vote(predictions)
        
        # Karar ve güven skoru
        action = np.argmax(final_pred)
        confidence = float(final_pred[action])
        
        # Action mapping: 0=SHORT, 1=HOLD, 2=LONG
        action_map = {0: 'SHORT', 1: 'HOLD', 2: 'LONG'}
        
        return {
            'action': action_map[action],
            'action_id': int(action),
            'confidence': confidence,
            'probabilities': {
                'SHORT': float(final_pred[0]),
                'HOLD': float(final_pred[1]),
                'LONG': float(final_pred[2])
            },
            'model_predictions': predictions,
            'model_confidences': confidences
        }
        
    def weighted_vote(self, predictions: Dict) -> np.ndarray:
        """Ağırlıklı oylama ile final tahmini hesapla"""
        
        if not predictions:
            return np.array([0.33, 0.34, 0.33])  # Default
            
        weighted_sum = np.zeros(3)
        total_weight = 0
        
        for model_name, pred in predictions.items():
            weight = self.weights.get(model_name, 0.25)
            weighted_sum += pred * weight
            total_weight += weight
            
        if total_weight > 0:
            return weighted_sum / total_weight
        else:
            return np.array([0.33, 0.34, 0.33])
            
    def prepare_features(self, features: Dict) -> np.ndarray:
        """Feature vektörü hazırla"""
        
        # Feature'ları düzleştir
        flat_features = []
        
        for key, value in features.items():
            if isinstance(value, (int, float)):
                flat_features.append(float(value))
            elif isinstance(value, dict):
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, (int, float)):
                        flat_features.append(float(subvalue))
                        
        # Pad veya truncate
        if len(flat_features) > 100:
            flat_features = flat_features[:100]
        elif len(flat_features) < 100:
            flat_features.extend([0] * (100 - len(flat_features)))
            
        return np.array(flat_features)
        
    def prepare_timeseries(self, features: Dict) -> np.ndarray:
        """Zaman serisi feature'ları hazırla"""
        
        # Örnek: son 20 periyot
        timeseries = np.random.randn(20, 50)  # Simülasyon
        return timeseries
        
    def prepare_graph_features(self, features: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Graf feature'ları hazırla"""
        
        # Simülasyon
        node_features = np.random.randn(10, 64)
        edge_features = np.random.randn(10, 10, 32)
        adjacency = np.random.randn(10, 10)
        
        return node_features, edge_features, adjacency
        
    def prepare_text_features(self, features: Dict) -> np.ndarray:
        """Text feature'ları hazırla"""
        
        # Simülasyon
        tokens = np.random.randint(0, 10000, (20,))
        return tokens