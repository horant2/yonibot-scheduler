import os
import time
import requests
import anthropic

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

def send_telegram(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": message[:4096]})

def ask_misfit(name, persona, signal):
    msg = client.messages.create(
        model="claude-opus-4-5-20251101",
        max_tokens=512,
        messages=[{"role": "user", "content": f"{persona}\n\nYoniBot signal:\n{signal}\n\nGive your verdict in 3-5 sentences. Be direct."}]
    )
    return msg.content[0].text

MISFITS = [
    ("Soros", "You ARE George Soros. Find the hidden peg. Where is the lie everyone believes and when does it break?"),
    ("Druckenmiller", "You ARE Stanley Druckenmiller. State your stop level, your target, and your conviction size."),
    ("PTJ", "You ARE Paul Tudor Jones. Check the chart pattern. Does a 5 to 1 risk/reward setup exist? Where is the stop?"),
    ("Jane Street", "You ARE Jane Street quant engine. Give edge percentage, Kelly size, max drawdown, and verdict: TRADE or NO TRADE."),
    ("Tepper", "You ARE David Tepper. Read the policy backdrop. Is the Fed with us or against us on this trade?"),
    ("Andurand", "You ARE Pierre Andurand. What does this mean for physical commodity flows and geopolitical supply disruption?"),
]

def run_cycle():
    yoni = client.messages.create(
        model="claude-opus-4-5-20251101",
        max_tokens=1024,
        messages=[{"role": "user", "content": "You are YoniBot. Scan arXiv q-fin papers and live market signals. Output a concise trade signal with asset, direction, entry zone, stop, and target."}]
    )
    signal = yoni.content[0].text

    brief = f"YONIBOT SIGNAL\n{signal}\n\nTHE MISFITS DEBATE\n"
    for name, persona in MISFITS:
        verdict = ask_misfit(name, persona, signal)
        brief += f"\n{name.upper()}:\n{verdict}\n"

    send_telegram(brief)
    print("Misfits brief sent.")

while True:
    try:
        run_cycle()
    except Exception as e:
        send_telegram(f"Misfits error: {e}")
        print(f"Error: {e}")
    time.sleep(900)
