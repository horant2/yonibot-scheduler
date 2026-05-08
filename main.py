import os
import time
import requests
import anthropic

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")

def send_telegram(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": message})

def run_yonibot():
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    message = client.messages.create(
        model="claude-opus-4-5-20251101",
        max_tokens=1024,
        messages=[{"role": "user", "content": "You are YoniBot. Scan arXiv q-fin papers and live market signals. Output a trade brief now."}]
    )
    return message.content[0].text

while True:
    try:
        result = run_yonibot()
        send_telegram(result)
        print("Brief sent.")
    except Exception as e:
        send_telegram(f"YoniBot error: {e}")
        print(f"Error: {e}")
    time.sleep(900)
