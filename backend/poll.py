import time
import json
import sys

while True:
    try:
        with open('d:/SmartTrade/backend/ai/models/training_progress.json', 'r') as f:
            data = json.load(f)
            print(f"[{data.get('progress')}%] {data.get('detail')} - Status: {data.get('status')}")
            if data.get('status') in ['done', 'error']:
                break
    except Exception as e:
        print(e)
    time.sleep(5)
