import os
import time
import requests
import anthropic

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
FINNHUB_API_KEY = os.environ.get("FINNHUB_API_KEY")
ALPHA_VANTAGE_API_KEY = os.environ.get("ALPHA_VANTAGE_API_KEY")

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

def send_telegram(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": message[:4096]})

def get_technical_signals():
    signals = []
    symbols = ["SPY", "QQQ", "GLD", "USO"]
    for symbol in symbols:
        try:
            rsi = requests.get(f"https://www.alphavantage.co/query?function=RSI&symbol={symbol}&interval=daily&time_period=14&series_type=close&apikey={ALPHA_VANTAGE_API_KEY}").json()
            rsi_val = list(rsi["Technical Analysis: RSI"].values())[0]["RSI"]
            macd = requests.get(f"https://www.alphavantage.co/query?function=MACD&symbol={symbol}&interval=daily&series_type=close&apikey={ALPHA_VANTAGE_API_KEY}").json()
            macd_val = list(macd["Technical Analysis: MACD"].values())[0]
            bbands = requests.get(f"https://www.alphavantage.co/query?function=BBANDS&symbol={symbol}&interval=daily&time_period=20&series_type=close&apikey={ALPHA_VANTAGE_API_KEY}").json()
            bb = list(bbands["Technical Analysis: BBANDS"].values())[0]
            bb_width = float(bb["Real Upper Band"]) - float(bb["Real Lower Band"])
            signals.append(f"{symbol}: RSI={rsi_val}, MACD={macd_val['MACD']}, Signal={macd_val['MACD_Signal']}, BB_Width={bb_width:.2f}")
        except:
            pass
    return "\n".join(signals)

def get_market_data():
    data = []
    try:
        crypto = requests.get("https://api.coingecko.com/api/v3/simple/price?ids=bitcoin,ethereum,solana&vs_currencies=usd&include_24hr_change=true").json()
        data.append(f"BTC: ${crypto['bitcoin']['usd']} ({crypto['bitcoin']['usd_24h_change']:.1f}%)")
        data.append(f"ETH: ${crypto['ethereum']['usd']} ({crypto['ethereum']['usd_24h_change']:.1f}%)")
        data.append(f"SOL: ${crypto['solana']['usd']} ({crypto['solana']['usd_24h_change']:.1f}%)")
    except:
        pass
    try:
        fear = requests.get("https://api.alternative.me/fng/").json()
        data.append(f"Fear & Greed: {fear['data'][0]['value']} ({fear['data'][0]['value_classification']})")
    except:
        pass
    return "\n".join(data)

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
    technical = get_technical_signals()
    market = get_market_data()

    yoni = client.messages.create(
        model="claude-opus-4-5-20251101",
        max_tokens=1024,
        messages=[{"role": "user", "content": f"""You are YoniBot. You have real statistical data. Analyze it and identify genuine anomalies only.

TECHNICAL INDICATORS:
{technical}

MARKET DATA:
{market}

Rules:
- RSI below 30 is oversold. RSI above 70 is overbought.
- MACD crossing above signal line is bullish. Below is bearish.
- Narrow Bollinger Band width means compression. Breakout coming.
- Only generate a signal if at least two indicators confirm the same direction.
- If no genuine anomaly exists, say NO SIGNAL and explain why.

Output: Asset, Direction, Entry Zone, Stop, Target, and which indicators confirm."""}]
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
