import os
import time
import requests
import anthropic
import yfinance as yf
import pandas as pd
import numpy as np
from exa_py import Exa
from datetime import datetime

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
TELEGRAM_PERFORMANCE_TOKEN = os.environ.get("TELEGRAM_PERFORMANCE_TOKEN")
TELEGRAM_PERFORMANCE_CHAT_ID = os.environ.get("TELEGRAM_PERFORMANCE_CHAT_ID")
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

def send_performance(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_PERFORMANCE_TOKEN}/sendMessage"
    chunks = [message[i:i+4000] for i in range(0, len(message), 4000)]
    for chunk in chunks:
        requests.post(url, json={"chat_id": TELEGRAM_PERFORMANCE_CHAT_ID, "text": chunk})
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

def keepalive():
    try:
        get_account()
        print(f"Keepalive ping {datetime.utcnow().strftime('%H:%M:%S')}")
    except:
        pass

def smart_sleep(total_seconds):
    interval = 480
    elapsed = 0
    while elapsed < total_seconds:
        sleep_chunk = min(interval, total_seconds - elapsed)
        time.sleep(sleep_chunk)
        elapsed += sleep_chunk
        if elapsed < total_seconds:
            keepalive()

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

def get_misfit_knowledge(name, trader_query):
    try:
        results = exa.search_and_contents(
            trader_query,
            num_results=3,
            text={"max_characters": 400}
        )
        knowledge = []
        for r in results.results:
            knowledge.append(f"{r.title}: {r.text[:300]}")
        return "\n\n".join(knowledge)
    except:
        return ""

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
                msg = f"STOP LOSS HIT\nSymbol: {symbol}\nPrice: ${current_price:.2f}\nLoss: {unrealized_pnl_pct:.1f}%\nTime: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}"
                closed.append(msg)
                send_performance(msg)
        if closed:
            send_telegram("STOP LOSS TRIGGERED\n" + "\n".join(closed))
    except Exception as e:
        print(f"Stop loss check error: {e}")

def report_open_positions():
    try:
        positions = get_positions()
        if not isinstance(positions, list) or len(positions) == 0:
            return
        report = f"OPEN POSITIONS -- {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}\n\n"
        for pos in positions:
            symbol = pos["symbol"]
            qty = pos["qty"]
            entry = float(pos["avg_entry_price"])
            current = float(pos["current_price"])
            pnl_pct = float(pos["unrealized_plpc"]) * 100
            pnl_dollar = float(pos["unrealized_pl"])
            report += f"{symbol}: {qty} shares\nEntry: ${entry:.2f} | Now: ${current:.2f}\nP&L: ${pnl_dollar:.2f} ({pnl_pct:.1f}%)\n\n"
        send_performance(report)
    except Exception as e:
        print(f"Position report error: {e}")

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
        result = f"TRADE EXECUTED: {direction.upper()} {qty} shares of {ticker} at ~${price:.2f}\nSize: {size_label}\nStop: ${stop_price:.2f}\nOrder ID: {order_id}"
        perf_log = f"NEW TRADE LOGGED\nTime: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}\nAsset: {ticker}\nDirection: {direction.upper()}\nEntry: ${price:.2f}\nQty: {qty}\nSize: {size_label}\nStop: ${stop_price:.2f}\nVotes: {vote_count}/5\nOrder ID: {order_id}"
        send_performance(perf_log)
        return result
    except Exception as e:
        return f"Trade execution failed: {e}"

def ask_misfit(name, persona, signal, extra_knowledge=""):
    knowledge_context = f"\n\nRECENT RESEARCH ON YOUR TRADING STYLE:\n{extra_knowledge}" if extra_knowledge else ""
    msg = client.messages.create(
        model="claude-opus-4-5-20251101",
        max_tokens=400,
        messages=[{"role": "user", "content": f"{persona}{knowledge_context}\n\nYoniBot signal:\n{signal}\n\nGive your verdict in 2-3 sentences maximum. Be brutal and direct. End with either VOTE: TRADE or VOTE: PASS on its own line."}]
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
    (
        "Soros",
        "You ARE George Soros. You broke the Bank of England on Black Wednesday 1992 by identifying the lie that the British pound could hold its peg. Find the hidden peg in every market. Where is the lie everyone believes and when do the defenders run out of ammunition?",
        "George Soros Black Wednesday 1992 pound sterling ERM trade reflexivity biggest win interview"
    ),
    (
        "Druckenmiller",
        "You ARE Stanley Druckenmiller. You called the Deutsche Mark collapse alongside Soros and doubled the position. You think about what you can lose before what you can make. State your stop level first, then your target, then your conviction size.",
        "Stanley Druckenmiller biggest trade win asymmetric bet risk management interview Ira Sohn"
    ),
    (
        "PTJ",
        "You ARE Paul Tudor Jones. You called Black Monday 1987 by studying 1929 patterns. You never get out of bed for less than 5 to 1 risk reward. Does a 5 to 1 setup exist here? Where is the hard stop?",
        "Paul Tudor Jones Black Monday 1987 prediction trading rules risk management documentary interview"
    ),
    (
        "Tepper",
        "You ARE David Tepper. You made 7 billion dollars in 2009 buying bank bonds when everyone thought the system was ending because you read the government would not let banks fail. Read the policy backdrop. Is the Fed with us or against us?",
        "David Tepper 2009 bank trade Appaloosa biggest win Fed policy interview CNBC"
    ),
    (
        "Andurand",
        "You ARE Pierre Andurand. You called the 2008 oil spike, the 2014 crash, and the 2022 Russia Ukraine energy crisis by tracking physical flows before they showed in price. Read tanker movements, refinery margins, and geopolitical chokepoints. What does the physical world say?",
        "Pierre Andurand oil trade 2008 2022 biggest win physical commodity flows interview letter"
    ),
]

cycle_count = 0
misfit_knowledge_cache = {}
knowledge_refresh_cycles = 8

def run_cycle():
    global cycle_count, misfit_knowledge_cache
    cycle_count += 1

    check_stop_losses()

    if cycle_count % 4 == 0:
        report_open_positions()

    if cycle_count % knowledge_refresh_cycles == 1:
        print("Refreshing Misfit knowledge bases...")
        for name, persona, query in MISFITS:
            misfit_knowledge_cache[name] = get_misfit_knowledge(name, query)
            time.sleep(2)

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
- Congressional buying in a sector is a leading indicator.
- Only generate a signal if at least two indicators confirm the same direction AND news or congressional activity supports the thesis.
- If no genuine anomaly exists say NO SIGNAL and explain why.
- Keep output concise: Asset, Direction, Entry Zone, Stop, Target, confirming indicators only.
- Output maximum 300 words."""}]
    )
    signal = yoni.content[0].text

    verdicts = []
    vote_count = 0

    for name, persona, query in MISFITS:
        knowledge = misfit_knowledge_cache.get(name, "")
        verdict = ask_misfit(name, persona, signal, knowledge)
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

    smart_sleep(900)

while True:
    try:
        run_cycle()
    except Exception as e:
        send_telegram(f"Misfits error: {e}")
        print(f"Error: {e}")
        smart_sleep(900)
        
