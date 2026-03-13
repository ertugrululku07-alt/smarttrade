import json
import os

# Path to the state file
STATE_FILE = "backend/live_trader_state.json"

def reset_risk():
    if not os.path.exists(STATE_FILE):
        print(f"File not found: {STATE_FILE}")
        return

    try:
        with open(STATE_FILE, 'r', encoding='utf-8') as f:
            state = json.load(f)

        print(f"Current Balance: ${state.get('balance', 0):.2f}")
        print(f"Current Consecutive Losses: {state.get('consecutive_losses', 0)}")

        # Reset risk values
        state['consecutive_losses'] = 0
        state['symbol_consecutive_losses'] = {}
        
        with open(STATE_FILE, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=4)

        print("Risk limits (consecutive losses) reset successfully.")
        print("When you restart the robot, the 'Daily Limit' and 'Consecutive Loss' halt will be cleared.")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    reset_risk()
