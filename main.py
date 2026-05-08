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
        messages=[{"role": "user", "content": f"{persona}\n\nYoniBot signal:\n{signal}\n\nGive your verdict in 3-5 sentences. End your response with either VOTE: TRADE or VOTE: PASS on its own line."}]
    )
    return msg.content[0].text

def ask_jane_street(signal, verdicts):
    debate = "\n\n".join([f"{n}:\n{v}" for n, v in verdicts])
    msg = client.messages.create(
        model="claude-opus-4-5-20251101",
        max_tokens=512,
        messages=[{"role": "user", "content": f"You ARE Jane Street quant engine. The Misfits have debated this signal.\n\nSIGNAL:\n{signal}\n\nDEBATE:\n{debate}\n\nCalculate edge, Kelly size, and max drawdown. You have veto power. End with either VETO: BLOCKED or VETO: APPROVED on its own line."}]
    )
    return msg.content[0].text

MISFITS = [
    ("Soros", "You ARE George Soros. Find the hidden peg. Where is the lie everyone believes and when does it break?"),
    ("Druckenmiller", "You ARE Stanley Druckenmiller. State your stop level, your target, and your conviction size."),
    ("PTJ", "You ARE Paul Tudor Jones. Check the chart pattern. Does a 5 to 1 risk/reward setup exist? Where is the stop?"),
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

    verdicts = []
    vote_count = 0
    brief = f"YONIBOT SIGNAL\n{signal}\n\nTHE MISFITS DEBATE\n"

    for name, persona in MISFITS:
        verdict = ask_misfit(name, persona, signal)
        verdicts.append((name, verdict))
        brief += f"\n{name.upper()}:\n{verdict}\n"
        if "VOTE: TRADE" in verdict:
            vote_count += 1

    jane = ask_jane_street(signal, verdicts)
    brief += f"\nJANE STREET:\n{jane}\n"

    approved = "VETO: APPROVED" in jane
    majority = vote_count >= 3

    if majority and approved:
        verdict_line = f"VERDICT: EXECUTE -- {vote_count}/5 voted TRADE. Jane Street approved."
    elif not approved:
        verdict_line = f"VERDICT: BLOCKED -- Jane Street vetoed. ({vote_count}/5 voted TRADE)"
    else:
        verdict_line = f"VERDICT: PASS -- Only {vote_count}/5 voted TRADE. No majority."

    brief += f"\n{verdict_line}"
    send_telegram(brief)
    print(f"Brief sent. {verdict_line}")

while True:
    try:
        run_cycle()
    except Exception as e:
        send_telegram(f"Misfits error: {e}")
        print(f"Error: {e}")
    time.sleep(900)
