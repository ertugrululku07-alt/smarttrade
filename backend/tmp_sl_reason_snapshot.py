import json, os
from datetime import datetime

dir_path = r"d:\SmartTrade\backend\logs\live_runs\ict24h_20260319_223624"
progress_path = os.path.join(dir_path, 'progress.json')
out_path = os.path.join(dir_path, 'sl_reason_snapshot.json')

data = {}
if os.path.exists(progress_path):
    with open(progress_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

status = data.get('status', {})
closed = status.get('closed_trades', []) or []
losses = [t for t in closed if float(t.get('pnl', 0) or 0) <= 0]
reasons = {}
for t in losses:
    r = str(t.get('exit_reason', 'UNKNOWN'))
    reasons[r] = reasons.get(r, 0) + 1

snapshot = {
    'generated_at': datetime.utcnow().isoformat() + 'Z',
    'run_id': os.path.basename(dir_path),
    'elapsed_minutes': data.get('elapsed_minutes', 0),
    'kpi': {
        'balance': status.get('balance', 0),
        'open_trades_count': status.get('open_trades_count', 0),
        'closed_trades_count': status.get('closed_trades_count', 0),
        'trades_count': data.get('trades_count', 0),
    },
    'loss_reason_breakdown': reasons,
    'loss_samples': losses[-10:],
}

with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(snapshot, f, indent=2)

print(out_path)
