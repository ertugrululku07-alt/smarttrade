import subprocess
import os

def run_test():
    print("Running terminal_test.py and capturing output to final_test.log...")
    try:
        with open('final_test.log', 'w', encoding='utf-8') as f:
            subprocess.run(['python', 'terminal_test.py'], stdout=f, stderr=subprocess.STDOUT)
        print("Done. Read final_test.log for results.")
    except Exception as e:
        print(f"Runner failed: {e}")

if __name__ == "__main__":
    run_test()
