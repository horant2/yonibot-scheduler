import os
import time
import requests
import anthropic
import yfinance as yf
import pandas as pd
import numpy as np
from exa_py import Exa

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
FINNHUB_API_KEY = os.environ.get("FINNHUB_API_KEY")
ALPHA_VANTAGE_API_KEY = os.environ.get("ALPHA_VANTAGE_API_KEY")
EXA_API_KEY = os.environ.get("EXA_API_KEY")
ALPACA_API_KEY = os.environ.get("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY")
ALPACA_BASE_URL = "https://paper-api.alpaca.markets"

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
exa = Exa(api_key=EXA_API_KEY)

def send_telegram(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    chunks = [message[i:i+4000] for i in range(0, len(message), 4000)]
    for chunk in chunks:
        requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": chunk})
        time.sleep(1)

def alpaca_request(method, endpoint, data=None):
    headers = {
        "APCA-API-KEY-ID": ALPACA_API_KEY,
        "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
        "Content-Type": "application/json"
    }
    url = f"{ALPACA_BASE_URL}{endpoint}"
    if method == "GET":
        return requests.get(url, headers=headers).json()
    elif method == "POST":
        return requests.post(url, headers=headers, json=data).json()
    elif method == "DELETE":
        return requests.delete(url, headers=headers).json()

def get_account():
    return alpaca_request("GET", "/v2/account")

def get_positions():
    return alpaca_request("GET", "/v2/positions")

def close_position(symbol):
    return alpaca_request("DELETE", f"/v2/positions/{symbol}")

def submit_order(symbol, qty, side):
    return alpaca_request("POST", "/v2/orders", {
        "symbol": symbol,
        "qty": qty,
        "side": side,
        "type": "market",
        "time_in_force": "day"
    })

def submit_stop_loss(symbol, qty, side, stop_price):
    return alpaca_request("POST", "/v2/orders", {
        "symbol": symbol,
        "qty": qty,
        "side": side,
        "type": "stop",
        "stop_price": str(round(stop_price, 2)),
        "time_in_force": "gtc"
    })

def calc_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = -delta.clip(upper=0).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def get_technical_signals():
    signals = []
    symbols = ["SPY", "QQQ", "GLD", "USO", "TLT", "BTC-USD", "ETH-USD", "SOL-USD"]
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
            bb_width = (4 * std20).iloc[-1]
            price = close.iloc[-1]
            signals.append(f"{symbol}: Price={price:.2f}, RSI={rsi:.1f}, MACD={macd_val:.3f}, Signal={signal_val:.3f}, BB_Width={bb_width:.2f}")
        except:
            pass
    return "\n".join(signals)

def get_yield_curve():
    try:
        t2 = yf.download("^IRX", period="5d", progress=False)["Close"].squeeze().iloc[-1]
        t10 = yf.download("^TNX", period="5d", progress=False)["Close"].squeeze().iloc[-1]
        t30 = yf.download("^TYX", period="5d", progress=False)["Close"].squeeze().iloc[-1]
        spread = t10 - t2
        return f"2Y={t2:.2f}% 10Y={t10:.2f}% 30Y={t30:.2f}% Spread(10-2)={spread:.2f}%"
    except:
        return "Yield curve unavailable"

def get_fear_greed():
    try:
        fear = requests.get("https://api.alternative.me/fng/").json()
        return f"Fear & Greed: {fear['data'][0]['value']} ({fear['data'][0]['value_classification']})"
    except:
        return "Fear & Greed unavailable"

def get_arxiv_signals():
    try:
        results = exa.search_and_contents(
            "quantitative finance trading anomaly alpha signal 2025 2026",
            num_results=3,
            include_domains=["arxiv.org"],
            text={"max_characters": 500}
        )
        papers = []
        for r in results.results:
            papers.append(f"PAPER: {r.title}\n{r.text[:300]}")
        return "\n\n".join(papers)
    except Exception as e:
        return f"arXiv unavailable: {e}"

def get_market_news():
    try:
        results = exa.search_and_contents(
            "market moving news macro trading today",
            num_results=5,
            include_domains=["reuters.com", "bloomberg.com", "ft.com", "wsj.com", "cnbc.com"],
            text={"max_characters": 300}
        )
        news = []
        for r in results.results:
            news.append(f"{r.title}: {r.text[:200]}")
        return "\n\n".join(news)
    except Exception as e:
        return f"News unavailable: {e}"

def get_geopolitical_news():
    try:
        results = exa.search_and_contents(
            "Iran Strait of Hormuz oil supply disruption geopolitical risk today",
            num_results=3,
            text={"max_characters": 300}
        )
        news = []
        for r in results.results:
            news.append(f"{r.title}: {r.text[:200]}")
        return "\n\n".join(news)
    except Exception as e:
        return f"Geopolitical news unavailable: {e}"

def get_congressional_trading():
    try:
        results = exa.search_and_contents(
            "congress senator representative stock trade purchase sale disclosure 2026",
            num_results=5,
            include_domains=["quiverquant.com", "capitoltrades.com", "housestockwatcher.com", "senatestockwatcher.com"],
            text={"max_characters": 300}
        )
        trades = []
        for r in results.results:
            trades.append(f"{r.title}: {r.text[:200]}")
        return "\n\n".join(trades) if trades else "No recent congressional trades found"
    except Exception as e:
        return f"Congressional trading unavailable: {e}"

def get_position_size(vote_count, buying_power):
    if vote_count == 5:
        pct = 0.10
        label = "MAXIMUM CONVICTION -- 10% of book"
    elif vote_count == 4:
        pct = 0.08
        label = "HIGH CONVICTION -- 8% of book"
    else:
        pct = 0.05
        label = "BASE CONVICTION -- 5% of book"
    size = min(buying_power * pct, 10000)
    return size, label

def check_stop_losses():
    try:
        positions = get_positions()
        if not isinstance(positions, list):
            return
        closed = []
        for pos in positions:
            symbol = pos["symbol"]
            current_price = float(pos["current_price"])
            side = pos["side"]
            unrealized_pnl_pct = float(pos["unrealized_plpc"]) * 100
            stop_triggered = False
            if side == "long" and unrealized_pnl_pct <= -5:
                stop_triggered = True
            elif side == "short" and unrealized_pnl_pct <= -5:
                stop_triggered = True
            if stop_triggered:
                close_position(symbol)
                closed.append(f"STOP LOSS: Closed {symbol} at ${current_price:.2f}. Loss: {unrealized_pnl_pct:.1f}%")
        if closed:
            send_telegram("STOP LOSS TRIGGERED\n" + "\n".join(closed))
    except Exception as e:
        print(f"Stop loss check error: {e}")

def extract_ticker(signal):
    for ticker in ["QQQ", "SPY", "GLD", "USO", "TLT"]:
        if ticker in signal:
            return ticker
    return None

def extract_direction(signal):
    signal_upper = signal.upper()
    if "SHORT" in signal_upper:
        return "sell"
    if "LONG" in signal_upper or "BUY" in signal_upper:
        return "buy"
    return None

def execute_trade(signal, vote_count):
    try:
        ticker = extract_ticker(signal)
        direction = extract_direction(signal)
        if not ticker or not direction:
            return "No executable equity trade identified."
        account = get_account()
        buying_power = float(account["buying_power"])
        position_size, size_label = get_position_size(vote_count, buying_power)
        price_data = yf.download(ticker, period="1d", interval="1m", progress=False)
        price = float(price_data["Close"].squeeze().iloc[-1])
        qty = max(1, int(position_size / price))
        order = submit_order(ticker, qty, direction)
        order_id = order.get("id", "unknown")
        stop_side = "sell" if direction == "buy" else "buy"
        stop_price = price * 0.95 if direction == "buy" else price * 1.05
        submit_stop_loss(ticker, qty, stop_side, stop_price)
        return f"TRADE EXECUTED: {direction.upper()} {qty} shares of {ticker} at ~${price:.2f}\nSize: {size_label}\nStop: ${stop_price:.2f}\nOrder ID: {order_id}"
    except Exception as e:
        return f"Trade execution failed: {e}"

def ask_misfit(name, persona, signal):
    msg = client.messages.create(
        model="claude-opus-4-5-20251101",
        max_tokens=400,
        messages=[{"role": "user", "content": f"{persona}\n\nYoniBot signal:\n{signal}\n\nGive your verdict in 2-3 sentences maximum. Be brutal and direct. End with either VOTE: TRADE or VOTE: PASS on its own line."}]
    )
    return msg.content[0].text

def ask_jane_street(signal, verdicts):
    debate = "\n\n".join([f"{n}:\n{v}" for n, v in verdicts])
    msg = client.messages.create(
        model="claude-opus-4-5-20251101",
        max_tokens=400,
        messages=[{"role": "user", "content": f"You ARE Jane Street quant engine. The Misfits have debated this signal.\n\nSIGNAL:\n{signal}\n\nDEBATE:\n{debate}\n\nCalculate edge, Kelly size, and max drawdown in 2-3 sentences. You have veto power. End with either VETO: BLOCKED or VETO: APPROVED on its own line."}]
    )
    return msg.content[0].text

MISFITS = [
    ("Soros", "You ARE George Soros. Find the hidden peg. Where is the lie everyone believes and when does it break?"),
    ("Druckenmiller", "You ARE Stanley Druckenmiller. State your stop level, your target, and your conviction size."),
    ("PTJ", "You ARE Paul Tudor Jones. Does a 5 to 1 risk/reward setup exist? Where is the stop?"),
    ("Tepper", "You ARE David Tepper. Is the Fed with us or against us on this trade? What are congressional insiders doing?"),
    ("Andurand", "You ARE Pierre Andurand. What does this mean for physical commodity flows and geopolitical supply disruption?"),
]

def run_cycle():
    check_stop_losses()

    technical = get_technical_signals()
    yields = get_yield_curve()
    fear = get_fear_greed()
    arxiv = get_arxiv_signals()
    news = get_market_news()
    geo = get_geopolitical_news()
    congress = get_congressional_trading()

    context = f"""TECHNICAL INDICATORS:
{technical}

YIELD CURVE:
{yields}

SENTIMENT:
{fear}

LATEST ARXIV QUANT PAPERS:
{arxiv}

MARKET NEWS:
{news}

GEOPOLITICAL NEWS:
{geo}

CONGRESSIONAL TRADING:
{congress}"""

    yoni = client.messages.create(
        model="claude-opus-4-5-20251101",
        max_tokens=1024,
        messages=[{"role": "user", "content": f"""You are YoniBot. You have real statistical data, live news, academic research, and congressional trading intelligence. Find genuine anomalies only.

{context}

Rules:
- RSI below 30 is oversold. RSI above 70 is overbought.
- MACD crossing above signal line is bullish. Below is bearish.
- Narrow Bollinger Band width means compression and imminent breakout.
- Inverted yield curve signals recession risk.
- Congressional buying in a sector is a leading indicator -- politicians have information advantage.
- Only generate a signal if at least two indicators confirm the same direction AND news or congressional activity supports the thesis.
- If no genuine anomaly exists say NO SIGNAL and explain why.
- Keep output concise: Asset, Direction, Entry Zone, Stop, Target, confirming indicators only.
- Output maximum 300 words."""}]
    )
    signal = yoni.content[0].text

    verdicts = []
    vote_count = 0

    for name, persona in MISFITS:
        verdict = ask_misfit(name, persona, signal)
        verdicts.append((name, verdict))
        if "VOTE: TRADE" in verdict:
            vote_count += 1

    jane = ask_jane_street(signal, verdicts)
    approved = "VETO: APPROVED" in jane
    majority = vote_count >= 3

    if majority and approved:
        trade_result = execute_trade(signal, vote_count)
        verdict_line = f"VERDICT: EXECUTE -- {vote_count}/5 voted TRADE. Jane Street approved.\n{trade_result}"
    elif not approved:
        verdict_line = f"VERDICT: BLOCKED -- Jane Street vetoed. ({vote_count}/5 voted TRADE)"
    else:
        verdict_line = f"VERDICT: PASS -- Only {vote_count}/5 voted TRADE. No majority."

    msg1 = f"YONIBOT SIGNAL\n{signal}"
    msg2 = "THE MISFITS DEBATE\n"
    for name, verdict in verdicts:
        msg2 += f"\n{name.upper()}:\n{verdict}\n"
    msg3 = f"JANE STREET:\n{jane}\n\n{verdict_line}"

    send_telegram(msg1)
    time.sleep(2)
    send_telegram(msg2)
    time.sleep(2)
    send_telegram(msg3)
    print(f"Brief sent. {verdict_line}")

while True:
    try:
        run_cycle()
    except Exception as e:
        send_telegram(f"Misfits error: {e}")
        print(f"Error: {e}")
    time.sleep(900)
