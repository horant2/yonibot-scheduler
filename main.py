import os
import time
import requests
import anthropic
import yfinance as yf
import pandas as pd
import numpy as np

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
FINNHUB_API_KEY = os.environ.get("FINNHUB_API_KEY")
ALPHA_VANTAGE_API_KEY = os.environ.get("ALPHA_VANTAGE_API_KEY")

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

def send_telegram(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": message[:4096]})

def calc_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = -delta.clip(upper=0).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def get_technical_signals():
    signals = []
    symbols = ["SPY", "QQQ", "GLD", "USO", "TLT", "DXY", "BTC-USD", "ETH-USD", "SOL-USD"]
    for symbol in symbols:
        try:
            df = yf.download(symbol, period="60d", interval="1d", progress=False)
            close = df["Close"].squeeze()
            rsi = calc_rsi(close).iloc[-1]
            ema12 = close.ewm(span=12).mean()
            ema26 = close.ewm(span=26).mean()
            macd = ema12 - ema26
            signal_line = macd.ewm(span=9).mean()
            macd_val = macd.iloc[-1]
            signal_val = signal_line.iloc[-1]
            sma20 = close.rolling(20).mean()
            std20 = close.rolling(20).std()
            bb_upper = sma20 + 2 * std20
            bb_lower = sma20 - 2 * std20
            bb_width = (bb_upper - bb_lower).iloc[-1]
            price = close.iloc[-1]
            signals.append(f"{symbol}: Price={price:.2f}, RSI={rsi:.1f}, MACD={macd_val:.3f}, Signal={signal_val:.3f}, BB_Width={bb_width:.2f}")
        except:
            pass
    return "\n".join(signals)

def get_yield_curve():
    try:
        t2 = yf.download("^IRX", period="5d", progress=False)["Close"].iloc[-1].item()
        t10 = yf.download("^TNX", period="5d", progress=False)["Close"].iloc[-1].item()
        t30 = yf.download("^TYX", period="5d", progress=False)["Close"].iloc[-1].item()
        spread = t10 - t2
        return f"2Y={t2:.2f}% 10Y={t10:.2f}% 30Y={t30:.2f}% Spread(10-2)={spread:.2f}%"
    except:
        return "Yield curve unavailable"

def get_congressional_trades():
    try:
        r = requests.get("https://www.quiverquant.com/sources/congresstrading", timeout=5)
        return "Congressional trading data: check quiverquant.com for latest insider moves"
    except:
        return "Congressional data unavailable"

def get_crypto_funding():
    try:
        r = requests.get("https://api.coinglass.com/pub/v2/futures/funding_rate?symbol=BTC", timeout=5)
        data = r.json()
        if data.get("data"):
            rate = data["data"][0].get("fundingRate", "N/A")
            return f"BTC Funding Rate: {rate}"
    except:
        pass
    return "Crypto funding data unavailable"

def get_fear_greed():
    try:
        fear = requests.get("https://api.alternative.me/fng/").json()
        return f"Fear & Greed: {fear['data'][0]['value']} ({fear['data'][0]['value_classification']})"
    except:
        return "Fear & Greed unavailable"

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
    yields = get_yield_curve()
    funding = get_crypto_funding()
    fear = get_fear_greed()
    congress = get_congressional_trades()

    context = f"""TECHNICAL INDICATORS (9 assets):
{technical}

YIELD CURVE:
{yields}

CRYPTO FUNDING RATES:
{funding}

SENTIMENT:
{fear}

CONGRESSIONAL TRADING:
{congress}"""

    yoni = client.messages.create(
        model="claude-opus-4-5-20251101",
        max_tokens=1024,
        messages=[{"role": "user", "content": f"""You are YoniBot. You have real statistical data across equities, bonds, commodities, and crypto. Find genuine anomalies only.

{context}

Rules:
- RSI below 30 is oversold. RSI above 70 is overbought.
- MACD crossing above signal line is bullish. Below is bearish.
- Narrow Bollinger Band width means compression and imminent breakout.
- Inverted yield curve (negative spread) signals recession risk.
- Negative crypto funding rate means shorts paying longs -- mean reversion long setup.
- Only generate a signal if at least two indicators confirm the same direction.
- If no genuine anomaly exists say NO SIGNAL and explain why.

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
