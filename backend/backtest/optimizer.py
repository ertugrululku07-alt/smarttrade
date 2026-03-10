import optuna
import pandas as pd
from typing import List, Dict, Any
from schemas import OptimizationRequest
from backtest.engine import BacktestEngine

class StrategyOptimizer:
    """
    Kullanıcının Visual Bot Designer üzerinden tasarladığı stratejinin
    içerisindeki parametreleri 'optuna' kullanarak optimize eder.
    """
    def __init__(self, request: OptimizationRequest, df: pd.DataFrame):
        self.request = request
        self.df = df
        self.base_strategy = [b.model_dump() for b in request.strategy]
        self.best_params = {}
        
    def _create_trial_strategy(self, trial: optuna.Trial) -> List[Dict[str, Any]]:
        """
        Optuna'dan gelen yeni hiperparametre denemeleriyle (trial) 
        kullanıcının orijinal stratejisini geçici olarak harmanlar.
        Şu anlık MVP olarak RSI ve Mantık kapılarının değerleri optimize edilir.
        """
        modified_strategy = []
        for block in self.base_strategy:
            new_block = block.copy()
            if new_block['type'] == 'indicator' and 'rsi' in new_block['id'].lower():
                # RSI Periyodu için 7 ile 30 arası AI Optimizasyonu
                new_block['opt_period'] = trial.suggest_int(f"rsi_period_{new_block['id']}", 7, 30)
            elif new_block['type'] == 'logic' and 'if_condition' in new_block['id'].lower():
                # Eşik değeri (Örn RSI 30 seviyesi) için 20 ile 80 arası AI Optimizasyonu
                new_block['opt_threshold'] = trial.suggest_int(f"threshold_{new_block['id']}", 20, 80)
            
            modified_strategy.append(new_block)
            
        return modified_strategy

    def _objective(self, trial: optuna.Trial) -> float:
        """
        Her denemede (Trial) Backtest motorunu o anki random parametrelerle
        çalıştırıp kâr/zarar (Total PnL) skorunu geri döndürür.
        """
        # 1. Trial'a özel strateji üret
        trial_strategy = self._create_trial_strategy(trial)
        
        # 2. Backtest Motoruna ver ve çalıştır
        engine = BacktestEngine(self.df, initial_balance=self.request.initial_balance)
        
        # Gelecekte Engine içinde 'opt_period' ve 'opt_threshold' gibi 
        # Optuna fieldlarını okuması için Engine'de ufak güncellemeler yapılmalı.
        engine.load_strategy(trial_strategy)
        results = engine.run()
        
        # 3. Objective fonksiyonunun hedefi: PnL (Kâr oranını) maksimize etmektir.
        return results['total_pnl']

    def run_optimization(self) -> Dict[str, Any]:
        """
        AI optimizasyon sürecini başlatır ve en kârlı sonucu döner.
        """
        # Hedefimiz (objective) PnL'yi 'maximize' etmektir.
        study = optuna.create_study(direction="maximize")
        
        # Tanımlı n_trials (örn: 50 kombinasyon) kadar simülasyon yap
        study.optimize(self._objective, n_trials=self.request.n_trials)
        
        self.best_params = study.best_params
        best_pnl = study.best_value
        
        return {
            "success": True,
            "best_pnl": best_pnl,
            "best_parameters": self.best_params,
            "total_trials": len(study.trials)
        }
