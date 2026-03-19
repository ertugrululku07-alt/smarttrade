import json
import os
import time
import traceback
from datetime import datetime
from live_trader import LivePaperTrader

state_dir = os.environ.get('LIVE_TRADER_STATE_DIR', '')
progress_path = os.path.join(state_dir, 'progress.json')
final_path = os.path.join(state_dir, 'final_report.json')

tr = LivePaperTrader(initial_balance=1000.0, leverage=10)
tr.apply_profile_core_v2()
tr.start()

start_ts = time.time()
end_ts = start_ts + 24*3600

try:
    while time.time() < end_ts:
        st = tr.get_status()
        analytics = tr.get_analytics()
        payload = {
            'updated_at': datetime.utcnow().isoformat() + 'Z',
            'elapsed_minutes': round((time.time() - start_ts)/60.0, 2),
            'status': st,
            'analytics': analytics.get('analytics', {}),
            'trades_count': len(analytics.get('trades', [])),
        }
        with open(progress_path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2)
        time.sleep(60)
except Exception:
    with open(os.path.join(state_dir, 'error.txt'), 'w', encoding='utf-8') as f:
        f.write(traceback.format_exc())
finally:
    try:
        tr.stop()
    except Exception:
        pass

    analytics = tr.get_analytics()
    trades = analytics.get('trades', [])
    losses = [t for t in trades if float(t.get('pnl', 0) or 0) <= 0]
    by_reason = {}
    for t in losses:
        r = str(t.get('exit_reason', 'UNKNOWN'))
        by_reason[r] = by_reason.get(r, 0) + 1

    report = {
        'completed_at': datetime.utcnow().isoformat() + 'Z',
        'state_dir': state_dir,
        'duration_hours_target': 24,
        'final_status': tr.get_status(),
        'analytics': analytics.get('analytics', {}),
        'total_trades': len(trades),
        'loss_trades': len(losses),
        'loss_exit_reasons': by_reason,
        'sample_loss_trades': losses[-20:],
    }
    with open(final_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
