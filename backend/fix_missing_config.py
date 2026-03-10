import os

filepath = 'd:/SmartTrade/backend/ai/director_ai.py'

with open(filepath, 'r', encoding='utf-8') as f:
    text = f.read()

cfg_str = '''@dataclass
class DirectorConfig:
    """Direktör AI ayarları."""
    min_confluence: float = 0.40
    min_expert_score: float = 0.40
    min_trades_for_weight: int = 10
    max_trade_history: int = 50
    weight_decay: float = 0.95
    weight_bounds: tuple = (0.5, 2.0)

'''

if 'class DirectorConfig:' not in text:
    idx = text.find('class DirectorDecision:')
    if idx != -1:
        # Find the @dataclass before it
        idx = text.rfind('@dataclass', 0, idx)
        text = text[:idx] + cfg_str + text[idx:]
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(text)
        print('Inserted DirectorConfig')
    else:
        print('Could not find DirectorDecision')
else:
    print('DirectorConfig already present')
