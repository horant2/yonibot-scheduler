  import os
import time
import requests
import anthropic
import yfinance as yf
import pandas as pd
import numpy as np
from exa_py import Exa
from datetime import datetime, timedelta
import pytz

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

def alpaca_request(method, endpoint, data=None, params=None):
    headers = {
        "APCA-API-KEY-ID": ALPACA_API_KEY,
        "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
        "Content-Type": "application/json"
    }
    url = f"{ALPACA_BASE_URL}{endpoint}"
    if method == "GET":
        return requests.get(url, headers=headers, params=params).json()
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

def submit_order(symbol, qty, side, notional=None):
    order = {
        "symbol": symbol,
        "side": side,
        "type": "market",
        "time_in_force": "day"
    }
    if notional:
        order["notional"] = str(round(notional, 2))
    else:
        order["qty"] = qty
    return alpaca_request("POST", "/v2/orders", order)

def submit_stop_loss(symbol, qty, side, stop_price):
    return alpaca_request("POST", "/v2/orders", {
        "symbol": symbol,
        "qty": qty,
        "side": side,
        "type": "stop",
        "stop_price": str(round(stop_price, 2)),
        "time_in_force": "gtc"
    })

def is_market_hours():
    et = pytz.timezone("America/New_York")
    now = datetime.now(et)
    if now.weekday() >= 5:
        return False
    market_open = now.replace(hour=9, minute=30, second=0)
    market_close = now.replace(hour=16, minute=0, second=0)
    return market_open <= now <= market_close

def get_option_contract(underlying, direction, days_out=30):
    try:
        exp_date = (datetime.now() + timedelta(days=days_out)).strftime("%Y-%m-%d")
        exp_date_max = (datetime.now() + timedelta(days=days_out + 14)).strftime("%Y-%m-%d")
        contract_type = "call" if direction == "buy" else "put"
        params = {
            "underlying_symbols": underlying,
            "type": contract_type,
            "expiration_date_gte": exp_date,
            "expiration_date_lte": exp_date_max,
            "status": "active",
            "limit": 20
        }
        result = alpaca_request("GET", "/v2/options/contracts", params=params)
        contracts = result.get("option_contracts", [])
        if not contracts:
            return None
        price_data = yf.download(underlying, period="1d", interval="1m", progress=False)
        current_price = float(price_data["Close"].squeeze().iloc[-1])
        best_contract = None
        min_diff = float("inf")
        for c in contracts:
            strike = float(c.get("strike_price", 0))
            diff = abs(strike - current_price)
            if diff < min_diff:
                min_diff = diff
                best_contract = c
        return best_contract
    except Exception as e:
        print(f"Options contract error: {e}")
        return None

def execute_options_trade(underlying, direction, position_size):
    try:
        contract = get_option_contract(underlying, direction)
        if not contract:
            return f"No options contract found for {underlying}"
        symbol = contract["symbol"]
        strike = contract["strike_price"]
        expiry = contract["expiration_date"]
        contract_type = "CALL" if direction == "buy" else "PUT"
        order = {
            "symbol": symbol,
            "qty": 1,
            "side": "buy",
            "type": "market",
            "time_in_force": "day"
        }
        result = alpaca_request("POST", "/v2/orders", order)
        order_id = result.get("id", "unknown")
        msg = f"OPTIONS EXECUTED: BUY {contract_type} on {underlying}\nStrike: ${strike} | Expiry: {expiry}\nOrder ID: {order_id}"
        send_performance(f"NEW OPTIONS TRADE\nTime: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}\n{msg}")
        return msg
    except Exception as e:
        return f"Options execution failed: {e}"

def execute_crypto_trade(symbol, direction, position_size):
    try:
        crypto_map = {
            "BTC": "BTC/USD",
            "ETH": "ETH/USD",
            "SOL": "SOL/USD",
            "BITCOIN": "BTC/USD",
            "ETHEREUM": "ETH/USD",
            "SOLANA": "SOL/USD"
        }
        crypto_symbol = None
        for key, val in crypto_map.items():
            if key in symbol.upper():
                crypto_symbol = val
                break
        if not crypto_symbol:
            return None
        notional = min(position_size, 5000)
        order = {
            "symbol": crypto_symbol,
            "notional": str(round(notional, 2)),
            "side": direction,
            "type": "market",
            "time_in_force": "gtc"
        }
        result = alpaca_request("POST", "/v2/orders", order)
        order_id = result.get("id", "unknown")
        msg = f"CRYPTO EXECUTED: {direction.upper()} ${notional:.0f} of {crypto_symbol}\nOrder ID: {order_id}"
        send_performance(f"NEW CRYPTO TRADE\nTime: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}\n{msg}")
        return msg
    except Exception as e:
        return f"Crypto execution failed: {e}"

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

def run_omniscient_rotation():
    tickers = ["SOXL", "TECL", "TQQQ", "FAS", "ERX", "UUP", "TMF", "BIL"]
    safe = "BIL"
    scores = {}
    prices = {}
    try:
        spy_data = yf.download("SPY", period="220d", interval="1d", progress=False)
        spy_close = spy_data["Close"].squeeze()
        spy_sma200 = spy_close.rolling(200).mean().iloc[-1]
        spy_current = spy_close.iloc[-1]
        spy_trend = spy_current > spy_sma200
    except:
        spy_trend = True
    for ticker in tickers:
        if ticker == safe:
            continue
        try:
            df = yf.download(ticker, period="100d", interval="1d", progress=False)
            close = df["Close"].squeeze()
            if len(close) < 65:
                continue
            roc_fast = (close.iloc[-1] - close.iloc[-10]) / close.iloc[-10]
            roc_med = (close.iloc[-1] - close.iloc[-22]) / close.iloc[-22]
            roc_slow = (close.iloc[-1] - close.iloc[-64]) / close.iloc[-64]
            vol = close.pct_change().rolling(21).std().iloc[-1]
            rsi = calc_rsi(close).iloc[-1]
            sma50 = close.rolling(50).mean().iloc[-1]
            price = close.iloc[-1]
            if vol == 0 or np.isnan(vol):
                vol = 0.01
            weighted_mom = (roc_fast * 0.5) + (roc_med * 0.3) + (roc_slow * 0.2)
            risk_adj_mom = weighted_mom / vol
            trend_score = 1.0 if price > sma50 else 0.5
            rsi_penalty = 1.0
            if rsi > 85:
                rsi_penalty = 0.9
            elif rsi < 30:
                rsi_penalty = 0.9
            final_score = risk_adj_mom * trend_score * rsi_penalty
            scores[ticker] = final_score
            prices[ticker] = price
        except Exception as e:
            print(f"Rotation error for {ticker}: {e}")
            continue
    if not scores:
        return None, None, None, 0
    sorted_assets = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    best_ticker = sorted_assets[0][0]
    best_score = sorted_assets[0][1]
    if not spy_trend:
        uup_score = scores.get("UUP", -999)
        if uup_score > 0 and uup_score > best_score:
            best_ticker = "UUP"
            best_score = uup_score
        elif best_score < 0:
            best_ticker = safe
            best_score = 0
    if best_score <= 0:
        best_ticker = safe
    rotation_summary = f"OMNISCIENT ROTATION SIGNAL\n"
    rotation_summary += f"SPY Trend: {'BULL' if spy_trend else 'BEAR'} (vs 200-day moving average)\n"
    rotation_summary += f"Winner: {best_ticker} (score: {best_score:.3f})\n\n"
    rotation_summary += "Full Rankings:\n"
    for t, s in sorted_assets[:5]:
        rotation_summary += f"  {t}: {s:.3f}\n"
    return best_ticker, best_score, rotation_summary, prices.get(best_ticker, 0)

def get_technical_signals():
    signals = []
    symbols = ["SPY", "QQQ", "GLD", "USO", "TLT", "BTC-USD", "ETH-USD", "SOL-USD",
               "SOXL", "TECL", "TQQQ", "FAS", "ERX"]
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

def get_position_size(vote_count, buying_power, high_conviction_rotation=False):
    if high_conviction_rotation:
        pct = 0.15
        label = "ROTATION CONVICTION -- 15% of book"
    elif vote_count == 5:
        pct = 0.10
        label = "MAXIMUM CONVICTION -- 10% of book"
    elif vote_count == 4:
        pct = 0.08
        label = "HIGH CONVICTION -- 8% of book"
    else:
        pct = 0.05
        label = "BASE CONVICTION -- 5% of book"
    size = min(buying_power * pct, 15000)
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

def extract_ticker(signal, rotation_ticker=None):
    if rotation_ticker and rotation_ticker not in ["BIL", "UUP"]:
        return rotation_ticker
    for ticker in ["TQQQ", "SOXL", "TECL", "FAS", "ERX", "QQQ", "SPY", "GLD", "USO", "TLT"]:
        if ticker in signal:
            return ticker
    return None

def extract_crypto(signal):
    signal_upper = signal.upper()
    for crypto in ["BITCOIN", "BTC", "ETHEREUM", "ETH", "SOLANA", "SOL"]:
        if crypto in signal_upper:
            return crypto
    return None

def extract_direction(signal):
    signal_upper = signal.upper()
    if "SHORT" in signal_upper:
        return "sell"
    return "buy"

def execute_trade(signal, vote_count, rotation_ticker=None, high_conviction_rotation=False):
    try:
        account = get_account()
        buying_power = float(account["buying_power"])
        position_size, size_label = get_position_size(vote_count, buying_power, high_conviction_rotation)
        direction = extract_direction(signal)
        market_open = is_market_hours()

        crypto_asset = extract_crypto(signal)
        if crypto_asset and not market_open:
            result = execute_crypto_trade(crypto_asset, direction, position_size)
            if result:
                return f"{result}\nSize: {size_label}"

        if market_open:
            ticker = extract_ticker(signal, rotation_ticker)
            if not ticker:
                return "No executable trade identified."

            options_underlyings = {
                "TQQQ": "QQQ", "TECL": "QQQ", "SOXL": "SOXX",
                "FAS": "XLF", "ERX": "XLE", "QQQ": "QQQ",
                "SPY": "SPY", "GLD": "GLD", "USO": "USO", "TLT": "TLT"
            }

            if high_conviction_rotation and ticker in options_underlyings:
                underlying = options_underlyings[ticker]
                options_result = execute_options_trade(underlying, direction, position_size)
                price_data = yf.download(ticker, period="1d", interval="1m", progress=False)
                price = float(price_data["Close"].squeeze().iloc[-1])
                equity_qty = max(1, int((position_size * 0.5) / price))
                equity_order = submit_order(ticker, equity_qty, direction)
                equity_id = equity_order.get("id", "unknown")
                stop_price = price * 0.95 if direction == "buy" else price * 1.05
                submit_stop_loss(ticker, equity_qty, "sell" if direction == "buy" else "buy", stop_price)
                result = f"DUAL EXECUTION:\n{options_result}\nEQUITY: {direction.upper()} {equity_qty} {ticker} at ~${price:.2f}\nStop: ${stop_price:.2f}\nSize: {size_label}"
            else:
                price_data = yf.download(ticker, period="1d", interval="1m", progress=False)
                price = float(price_data["Close"].squeeze().iloc[-1])
                qty = max(1, int(position_size / price))
                order = submit_order(ticker, qty, direction)
                order_id = order.get("id", "unknown")
                stop_price = price * 0.95 if direction == "buy" else price * 1.05
                submit_stop_loss(ticker, qty, "sell" if direction == "buy" else "buy", stop_price)
                result = f"TRADE EXECUTED: {direction.upper()} {qty} shares of {ticker} at ~${price:.2f}\nSize: {size_label}\nStop: ${stop_price:.2f}\nOrder ID: {order_id}"

            send_performance(f"NEW TRADE\nTime: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}\n{result}")
            return result
        else:
            return f"Market closed. Crypto only mode. No equity signal executed."
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
        "You ARE George Soros. You broke the Bank of England on Black Wednesday 1992. Find the hidden peg in every market. Where is the lie everyone believes and when do the defenders run out of ammunition?",
        "George Soros Black Wednesday 1992 pound sterling ERM trade reflexivity biggest win interview"
    ),
    (
        "Druckenmiller",
        "You ARE Stanley Druckenmiller. You think about what you can lose before what you can make. State your stop level first, then your target, then your conviction size.",
        "Stanley Druckenmiller biggest trade win asymmetric bet risk management interview Ira Sohn"
    ),
    (
        "PTJ",
        "You ARE Paul Tudor Jones. You called Black Monday 1987. You never get out of bed for less than 5 to 1 risk reward. Does a 5 to 1 setup exist here? Where is the hard stop?",
        "Paul Tudor Jones Black Monday 1987 prediction trading rules risk management documentary interview"
    ),
    (
        "Tepper",
        "You ARE David Tepper. You made 7 billion dollars in 2009 buying bank bonds. Read the policy backdrop. Is the Fed with us or against us?",
        "David Tepper 2009 bank trade Appaloosa biggest win Fed policy interview CNBC"
    ),
    (
        "Andurand",
        "You ARE Pierre Andurand. You called the 2008 oil spike and the 2022 energy crisis by tracking physical flows. Read tanker movements, refinery margins, and geopolitical chokepoints.",
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

    market_open = is_market_hours()
    rotation_ticker, rotation_score, rotation_summary, rotation_price = run_omniscient_rotation()
    high_conviction_rotation = rotation_ticker not in ["BIL", "UUP", None] and rotation_score > 0.5

    technical = get_technical_signals()
    yields = get_yield_curve()
    fear = get_fear_greed()
    arxiv = get_arxiv_signals()
    news = get_market_news()
    geo = get_geopolitical_news()
    congress = get_congressional_trading()

    rotation_context = rotation_summary if rotation_summary else "Rotation signal unavailable"
    market_status = "OPEN -- equities, options, and crypto available" if market_open else "CLOSED -- crypto only, markets reopen Monday 9:30 AM ET"

    yoni_push = ""
    if high_conviction_rotation:
        yoni_push = f"""
YONIBOT OVERRIDE -- HIGH CONVICTION ROTATION:
The Omniscient Rotation Strategy (2,324% backtest 2019-2026) has selected {rotation_ticker} score {rotation_score:.3f}. SPY is in bull trend. This is statistically validated momentum rotation. YoniBot is pushing hard. Misfits need a very high bar to vote PASS."""

    context = f"""MARKET STATUS: {market_status}

OMNISCIENT ROTATION SIGNAL (2,324% backtest 2019-2026):
{rotation_context}

TECHNICAL INDICATORS:
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
{congress}
{yoni_push}"""

    weekend_crypto_note = "" if market_open else "\nMARKET IS CLOSED. Focus on crypto signals only. Bitcoin, Ethereum, Solana trade 24/7. Generate crypto signals or NO SIGNAL."

    yoni = client.messages.create(
        model="claude-opus-4-5-20251101",
        max_tokens=1024,
        messages=[{"role": "user", "content": f"""You are YoniBot. You have real statistical data, live news, academic research, congressional intelligence, and a backtested rotation strategy with 2,324% returns.

{context}
{weekend_crypto_note}

Rules:
- The Omniscient Rotation signal is your highest conviction tool.
- RSI below 30 is oversold. RSI above 70 is overbought.
- MACD crossing above signal line is bullish. Below is bearish.
- Narrow Bollinger Band width means compression and imminent breakout.
- During market hours: equity, options, and crypto signals valid.
- After hours and weekends: crypto signals only.
- When rotation signal is high conviction, push hard for the trade.
- If no genuine anomaly exists say NO SIGNAL and explain why.
- Output: Asset, Direction, Entry Zone, Stop, Target, confirming indicators.
- Maximum 300 words."""}]
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
        trade_result = execute_trade(signal, vote_count, rotation_ticker if high_conviction_rotation else None, high_conviction_rotation)
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
