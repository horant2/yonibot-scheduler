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

INCEPTION_VALUE = 100000

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
exa = Exa(api_key=EXA_API_KEY)

# FIVE NON-NEGOTIABLE RULES -- DRUCKENMILLER MODEL
MAX_ORDERS_PER_CYCLE = 2          # Knight Capital prevention
DAILY_LOSS_LIMIT = 0.03           # One circuit breaker only
FRIDAY_SHORT_CUTOFF_HOUR = 14     # No new shorts after 2 PM ET Friday
FRIDAY_SHORT_CLOSE_HOUR = 15      # Close losing shorts by 3:30 PM ET Friday
FRIDAY_SHORT_CLOSE_MINUTE = 30

# POSITION SIZING -- aggressive when conviction is high
CONVICTION_SIZING = {5: 0.05, 4: 0.08, 3: 0.05}
ROTATION_SIZE = 0.15
MAX_SINGLE_POSITION = 0.15
VIX_REDUCE_THRESHOLD = 35
VIX_STOP_THRESHOLD = 50

# ALLOCATION GUARDRAILS -- not gates, just caps
MAX_EQUITY_PCT = 0.60
MAX_CRYPTO_PCT = 0.25
MAX_LEVERAGED_PCT = 0.45
DUPLICATE_SIGNAL_BLOCKS = 2

LEVERAGED_ETFS = {"TQQQ", "SOXL", "TECL", "FAS", "ERX", "TMF", "SPXL", "UPRO"}

recent_signals = {}
daily_start_value = None
trades_halted_today = False
orders_this_cycle = 0

misfit_scorecard = {
    "Soros": {"correct": 0, "total": 0},
    "Druckenmiller": {"correct": 0, "total": 0},
    "PTJ": {"correct": 0, "total": 0},
    "Tepper": {"correct": 0, "total": 0},
    "Andurand": {"correct": 0, "total": 0},
    "Jane Street": {"correct": 0, "total": 0}
}

trade_history = []

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
        try:
            r = requests.post(url, json={"chat_id": TELEGRAM_PERFORMANCE_CHAT_ID, "text": chunk}, timeout=10)
            print(f"Performance send status: {r.status_code}")
        except Exception as e:
            print(f"Performance send error: {e}")
        time.sleep(1)

def format_trade_alert(ticker, direction, qty, price, stop_price, size_label, vote_count, voters_for, voters_against, reason):
    action = "Bought" if direction == "buy" else "Sold Short"
    total_bet = qty * price
    emoji = "🟢" if direction == "buy" else "🔴"
    against_text = f"\nWho disagreed: {', '.join(voters_against)}" if voters_against else ""
    return f"""{emoji} THE MISFITS JUST TRADED

{action} {qty} shares of {ticker} at ${price:.2f}
Bet size: ${total_bet:,.0f} ({size_label})
Stop loss set at: ${stop_price:.2f}

Why: {reason}

Who agreed ({vote_count}/5): {', '.join(voters_for)}{against_text}
Jane Street approved the math.

-- Satis House Consulting"""

def format_crypto_alert(crypto_symbol, direction, notional, reason, vote_count, voters_for, voters_against):
    action = "Bought" if direction == "buy" else "Sold"
    emoji = "🟢" if direction == "buy" else "🔴"
    against_text = f"\nWho disagreed: {', '.join(voters_against)}" if voters_against else ""
    return f"""{emoji} THE MISFITS JUST TRADED CRYPTO

{action} ${notional:,.0f} of {crypto_symbol}
Markets are closed but crypto never sleeps.

Why: {reason}

Who agreed ({vote_count}/5): {', '.join(voters_for)}{against_text}
Jane Street approved the math.

-- Satis House Consulting"""

def format_stop_loss_alert(symbol, exit_price, pnl_dollar, pnl_pct):
    result = f"+${abs(pnl_dollar):,.0f} profit" if pnl_dollar >= 0 else f"-${abs(pnl_dollar):,.0f} loss"
    return f"""🛑 STOP LOSS TRIGGERED

Closed {symbol} position
Exit price: ${exit_price:.2f}
Result: {result} ({pnl_pct:+.1f}%)

The Misfits cut the position. Capital protected.
On to the next signal.

-- Satis House Consulting"""

def format_friday_close_alert(symbol, exit_price, pnl_dollar, pnl_pct):
    result = f"+${abs(pnl_dollar):,.0f} profit" if pnl_dollar >= 0 else f"-${abs(pnl_dollar):,.0f} loss"
    return f"""📅 FRIDAY RISK MANAGEMENT

Closed short position in {symbol} before weekend.
Exit price: ${exit_price:.2f}
Result: {result} ({pnl_pct:+.1f}%)

Short positions do not survive weekends.
Capital protected.

-- Satis House Consulting"""

def format_position_report(portfolio_state, daily_pnl=None):
    portfolio_value = portfolio_state["portfolio_value"] if portfolio_state else INCEPTION_VALUE
    net_profit = portfolio_value - INCEPTION_VALUE
    net_pct = (net_profit / INCEPTION_VALUE) * 100
    profit_emoji = "📈" if net_profit >= 0 else "📉"
    net_profit_line = f"{profit_emoji} Net profit since inception: {'+' if net_profit >= 0 else ''}${net_profit:,.0f} ({'+' if net_pct >= 0 else ''}{net_pct:.2f}%)"

    if not portfolio_state or not portfolio_state["positions"]:
        return f"""📊 MISFITS PORTFOLIO UPDATE

No open positions right now.
Sitting in cash, waiting for the right signal.

Total value: ${portfolio_value:,.0f}
{net_profit_line}

-- Satis House Consulting"""

    lines = ["📊 MISFITS PORTFOLIO UPDATE\n"]
    lines.append(f"Total value: ${portfolio_value:,.2f}")

    if daily_pnl is not None:
        daily_emoji = "📈" if daily_pnl >= 0 else "📉"
        lines.append(f"Today: {daily_emoji} {'+' if daily_pnl >= 0 else ''}{daily_pnl*100:.2f}%")

    lines.append("\nOpen positions:")
    for symbol, pos in portfolio_state["positions"].items():
        pnl_emoji = "✅" if pos["unrealized_pnl"] >= 0 else "⚠️"
        direction = "Long" if pos["side"] == "long" else "Short"
        lines.append(f"{pnl_emoji} {symbol} ({direction}) -- {'+' if pos['unrealized_pnl'] >= 0 else ''}{pos['unrealized_pct']:.1f}% since entry")

    lines.append(f"\n{net_profit_line}")
    lines.append("\n-- Satis House Consulting")
    return "\n".join(lines)

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

def cancel_all_orders():
    return alpaca_request("DELETE", "/v2/orders")

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

def verify_order_filled(order_id, max_wait=30):
    for _ in range(6):
        time.sleep(5)
        try:
            order = alpaca_request("GET", f"/v2/orders/{order_id}")
            status = order.get("status", "unknown")
            if status in ["filled", "partially_filled"]:
                return True
            if status in ["canceled", "rejected", "expired"]:
                print(f"Order {order_id} failed: {status}")
                return False
        except Exception as e:
            print(f"Order verify error: {e}")
    return False

def submit_stop_loss_atomic(symbol, qty, side, stop_price):
    try:
        result = alpaca_request("POST", "/v2/orders", {
            "symbol": symbol,
            "qty": qty,
            "side": side,
            "type": "stop",
            "stop_price": str(round(stop_price, 2)),
            "time_in_force": "gtc"
        })
        if not result.get("id"):
            print(f"Stop loss failed for {symbol} -- closing position")
            close_position(symbol)
            send_performance(f"⚠️ SAFETY CLOSE\n\nCould not set stop loss on {symbol}.\nPosition closed to protect capital.\n\n-- Satis House Consulting")
            return False
        return True
    except Exception as e:
        print(f"Stop loss atomic error {symbol}: {e} -- closing")
        close_position(symbol)
        return False

def is_market_hours():
    et = pytz.timezone("America/New_York")
    now = datetime.now(et)
    if now.weekday() >= 5:
        return False
    market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
    return market_open <= now <= market_close

def is_friday_short_blocked():
    et = pytz.timezone("America/New_York")
    now = datetime.now(et)
    return now.weekday() == 4 and now.hour >= FRIDAY_SHORT_CUTOFF_HOUR

def should_close_friday_shorts():
    et = pytz.timezone("America/New_York")
    now = datetime.now(et)
    return now.weekday() == 4 and (now.hour > FRIDAY_SHORT_CLOSE_HOUR or
           (now.hour == FRIDAY_SHORT_CLOSE_HOUR and now.minute >= FRIDAY_SHORT_CLOSE_MINUTE))

def close_friday_shorts(portfolio_state):
    if not portfolio_state or not portfolio_state["positions"]:
        return
    for symbol, pos in portfolio_state["positions"].items():
        if pos["side"] == "short":
            try:
                close_position(symbol)
                msg = format_friday_close_alert(symbol, pos["current_price"], pos["unrealized_pnl"], pos["unrealized_pct"])
                send_performance(msg)
                print(f"Friday close: {symbol}")
            except Exception as e:
                print(f"Friday close error {symbol}: {e}")

def get_vix():
    try:
        vix = yf.download("^VIX", period="1d", interval="1m", progress=False)
        return float(vix["Close"].squeeze().iloc[-1])
    except:
        return 20.0

def get_portfolio_state():
    try:
        account = get_account()
        positions = get_positions()
        portfolio_value = float(account.get("portfolio_value", 100000))
        buying_power = float(account.get("buying_power", 50000))

        position_map = {}
        equity_value = 0
        crypto_value = 0
        leveraged_value = 0

        if isinstance(positions, list):
            for pos in positions:
                symbol = pos["symbol"]
                market_val = float(pos.get("market_value", 0))
                side = pos.get("side", "long")
                qty = float(pos.get("qty", 0))
                avg_entry = float(pos.get("avg_entry_price", 0))
                current_price = float(pos.get("current_price", 0))
                unrealized_pnl = float(pos.get("unrealized_pl", 0))
                unrealized_pct = float(pos.get("unrealized_plpc", 0)) * 100

                position_map[symbol] = {
                    "side": side, "qty": qty, "market_value": market_val,
                    "avg_entry": avg_entry, "current_price": current_price,
                    "unrealized_pnl": unrealized_pnl, "unrealized_pct": unrealized_pct,
                    "pct_of_portfolio": abs(market_val) / portfolio_value if portfolio_value > 0 else 0
                }

                if "USD" in symbol:
                    crypto_value += abs(market_val)
                elif symbol in LEVERAGED_ETFS:
                    leveraged_value += abs(market_val)
                else:
                    equity_value += abs(market_val)

        return {
            "portfolio_value": portfolio_value, "buying_power": buying_power,
            "positions": position_map,
            "equity_pct": equity_value / portfolio_value if portfolio_value > 0 else 0,
            "crypto_pct": crypto_value / portfolio_value if portfolio_value > 0 else 0,
            "leveraged_pct": leveraged_value / portfolio_value if portfolio_value > 0 else 0
        }
    except Exception as e:
        print(f"Portfolio state error: {e}")
        return None

def check_execution_rules(ticker, direction, position_size, portfolio_state, vix):
    global trades_halted_today, daily_start_value, orders_this_cycle

    if orders_this_cycle >= MAX_ORDERS_PER_CYCLE:
        return False, "Max orders per cycle -- Knight Capital rule"

    if trades_halted_today:
        return False, "Daily loss limit hit"

    if portfolio_state is None:
        return False, "Portfolio state unavailable"

    portfolio_value = portfolio_state["portfolio_value"]

    if daily_start_value is not None:
        daily_pnl = (portfolio_value - daily_start_value) / daily_start_value
        if daily_pnl <= -DAILY_LOSS_LIMIT:
            trades_halted_today = True
            send_performance(f"⚡ CIRCUIT BREAKER\n\nPortfolio down {abs(daily_pnl)*100:.1f}% today.\nAll trading paused until tomorrow.\nCapital protected.\n\n-- Satis House Consulting")
            return False, "Daily loss limit triggered"

    if vix >= VIX_STOP_THRESHOLD:
        return False, f"VIX {vix:.0f} -- above danger threshold"

    if direction == "sell" and is_friday_short_blocked():
        return False, "Friday short rule -- no new shorts after 2 PM ET"

    signal_key = f"{ticker}_{direction}"
    if signal_key in recent_signals and recent_signals[signal_key] < DUPLICATE_SIGNAL_BLOCKS:
        return False, "Duplicate signal -- waiting"

    positions = portfolio_state["positions"]
    if ticker in positions:
        existing = positions[ticker]
        if existing["side"] == "long" and direction == "buy":
            return False, f"Already long {ticker}"
        if existing["side"] == "short" and direction == "sell":
            return False, f"Already short {ticker}"

    is_crypto = any(c in ticker.upper() for c in ["BTC", "ETH", "SOL"])
    is_leveraged = ticker in LEVERAGED_ETFS

    if is_leveraged and not is_market_hours():
        return False, "Leveraged ETF -- market hours only"

    if not is_crypto and not is_market_hours():
        return False, "Market closed"

    if vix >= VIX_REDUCE_THRESHOLD:
        pass

    if is_leveraged and portfolio_state["leveraged_pct"] >= MAX_LEVERAGED_PCT:
        return False, "Leveraged allocation cap reached"
    elif is_crypto and portfolio_state["crypto_pct"] >= MAX_CRYPTO_PCT:
        return False, "Crypto allocation cap reached"
    elif not is_crypto and not is_leveraged and portfolio_state["equity_pct"] >= MAX_EQUITY_PCT:
        return False, "Equity allocation cap reached"

    return True, "Clear"

def get_position_size(vote_count, portfolio_value, vix, high_conviction_rotation=False):
    if high_conviction_rotation:
        base_pct = ROTATION_SIZE
        label = "15% of portfolio -- rotation conviction"
    elif vote_count == 5:
        base_pct = 0.10
        label = "10% of portfolio -- maximum conviction"
    elif vote_count == 4:
        base_pct = 0.08
        label = "8% of portfolio -- high conviction"
    else:
        base_pct = 0.05
        label = "5% of portfolio -- base conviction"

    if vix >= VIX_REDUCE_THRESHOLD:
        base_pct = base_pct * 0.5
        label = f"reduced to {base_pct*100:.0f}% -- VIX elevated at {vix:.0f}"

    size = min(portfolio_value * base_pct, portfolio_value * MAX_SINGLE_POSITION)
    return size, label

def update_training_loop(ticker, voters_for, voters_against, outcome_pnl_pct):
    trade_was_profitable = outcome_pnl_pct > 0
    for name in voters_for:
        if name in misfit_scorecard:
            misfit_scorecard[name]["total"] += 1
            if trade_was_profitable:
                misfit_scorecard[name]["correct"] += 1
    for name in voters_against:
        if name in misfit_scorecard:
            misfit_scorecard[name]["total"] += 1
            if not trade_was_profitable:
                misfit_scorecard[name]["correct"] += 1

def build_scorecard_context():
    lines = []
    for name, scores in misfit_scorecard.items():
        total = scores["total"]
        if total > 0:
            win_rate = scores["correct"] / total * 100
            lines.append(f"{name}: {win_rate:.0f}% win rate ({scores['correct']}/{total} trades)")
    return "MISFIT TRACK RECORD:\n" + "\n".join(lines) if lines else ""

def keepalive():
    try:
        get_account()
        print(f"Keepalive {datetime.now(pytz.utc).strftime('%H:%M:%S')}")
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
        spy_trend = spy_close.iloc[-1] > spy_sma200
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
            rsi_penalty = 0.9 if (rsi > 85 or rsi < 30) else 1.0
            final_score = risk_adj_mom * trend_score * rsi_penalty
            scores[ticker] = final_score
            prices[ticker] = price
        except Exception as e:
            print(f"Rotation error {ticker}: {e}")
    if not scores:
        return None, None, None, 0
    sorted_assets = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    best_ticker, best_score = sorted_assets[0]
    if not spy_trend:
        uup_score = scores.get("UUP", -999)
        if uup_score > 0 and uup_score > best_score:
            best_ticker, best_score = "UUP", uup_score
        elif best_score < 0:
            best_ticker, best_score = safe, 0
    if best_score <= 0:
        best_ticker = safe
    summary = f"OMNISCIENT ROTATION\nSPY: {'BULL' if spy_trend else 'BEAR'}\nWinner: {best_ticker} ({best_score:.3f})\n"
    for t, s in sorted_assets[:5]:
        summary += f"  {t}: {s:.3f}\n"
    return best_ticker, best_score, summary, prices.get(best_ticker, 0)

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
            macd = (ema12 - ema26).iloc[-1]
            signal_line = (ema12 - ema26).ewm(span=9).mean().iloc[-1]
            bb_width = (4 * close.rolling(20).std()).iloc[-1]
            price = close.iloc[-1]
            signals.append(f"{symbol}: ${price:.2f} RSI={rsi:.0f} MACD={macd:.3f} Signal={signal_line:.3f} BB={bb_width:.2f}")
        except:
            pass
    return "\n".join(signals)

def get_yield_curve():
    try:
        t2 = yf.download("^IRX", period="5d", progress=False)["Close"].squeeze().iloc[-1]
        t10 = yf.download("^TNX", period="5d", progress=False)["Close"].squeeze().iloc[-1]
        t30 = yf.download("^TYX", period="5d", progress=False)["Close"].squeeze().iloc[-1]
        return f"2Y={t2:.2f}% 10Y={t10:.2f}% 30Y={t30:.2f}% Spread={t10-t2:.2f}%"
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
            num_results=5, include_domains=["arxiv.org"],
            text={"max_characters": 500}
        )
        return "\n\n".join([f"PAPER: {r.title}\n{r.text[:400]}" for r in results.results])
    except Exception as e:
        return f"arXiv unavailable: {e}"

def get_market_news():
    try:
        results = exa.search_and_contents(
            "market moving news macro trading today", num_results=7,
            include_domains=["reuters.com", "bloomberg.com", "ft.com", "wsj.com", "cnbc.com"],
            text={"max_characters": 400}
        )
        return "\n\n".join([f"{r.title}: {r.text[:300]}" for r in results.results])
    except Exception as e:
        return f"News unavailable: {e}"

def get_geopolitical_news():
    try:
        results = exa.search_and_contents(
            "Iran Strait Hormuz oil supply disruption geopolitical risk today",
            num_results=5, text={"max_characters": 400}
        )
        return "\n\n".join([f"{r.title}: {r.text[:300]}" for r in results.results])
    except Exception as e:
        return f"Geo news unavailable: {e}"

def get_congressional_trading():
    try:
        results = exa.search_and_contents(
            "congress senator stock trade purchase sale disclosure 2026", num_results=5,
            include_domains=["quiverquant.com", "capitoltrades.com", "housestockwatcher.com", "senatestockwatcher.com"],
            text={"max_characters": 300}
        )
        trades = [f"{r.title}: {r.text[:200]}" for r in results.results]
        return "\n\n".join(trades) if trades else "No recent congressional trades"
    except Exception as e:
        return f"Congressional unavailable: {e}"

def get_deep_misfit_knowledge(name, queries):
    blocks = []
    for query in queries:
        try:
            results = exa.search_and_contents(query, num_results=3, text={"max_characters": 500})
            for r in results.results:
                blocks.append(f"SOURCE: {r.title}\n{r.text[:400]}")
            time.sleep(1)
        except Exception as e:
            print(f"Knowledge error {name}: {e}")
    return "\n\n".join(blocks)

def check_stop_losses(portfolio_state):
    try:
        if portfolio_state is None:
            return
        for symbol, pos in portfolio_state["positions"].items():
            unrealized_pct = pos["unrealized_pct"]
            side = pos["side"]
            stop_triggered = (side == "long" and unrealized_pct <= -5) or (side == "short" and unrealized_pct <= -5)
            if stop_triggered:
                close_position(symbol)
                msg = format_stop_loss_alert(symbol, pos["current_price"], pos["unrealized_pnl"], unrealized_pct)
                send_performance(msg)
                send_telegram(f"Stop loss hit on {symbol}.")
                voters_for, voters_against = [], []
                for trade in reversed(trade_history):
                    if trade.get("ticker") == symbol:
                        voters_for = trade.get("voters_for", [])
                        voters_against = trade.get("voters_against", [])
                        break
                update_training_loop(symbol, voters_for, voters_against, unrealized_pct / 100)
    except Exception as e:
        print(f"Stop loss error: {e}")

def report_open_positions(portfolio_state):
    try:
        if portfolio_state is None:
            return
        daily_pnl = None
        if daily_start_value and portfolio_state:
            daily_pnl = (portfolio_state["portfolio_value"] - daily_start_value) / daily_start_value
        send_performance(format_position_report(portfolio_state, daily_pnl))
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
    for crypto in ["BITCOIN", "BTC", "ETHEREUM", "ETH", "SOLANA", "SOL"]:
        if crypto in signal.upper():
            return crypto
    return None

def extract_direction(signal):
    return "sell" if "SHORT" in signal.upper() else "buy"

def extract_reason(signal):
    lines = signal.strip().split("\n")
    for line in lines:
        if any(word in line.lower() for word in ["because", "thesis", "confirmed", "why", "momentum", "breakout", "oversold", "overbought"]):
            return line.strip()
    return lines[0].strip() if lines else "Multiple indicators confirmed"

def execute_trade(signal, vote_count, verdicts, portfolio_state, vix, rotation_ticker=None, high_conviction_rotation=False):
    global recent_signals, trade_history, orders_this_cycle

    if portfolio_state is None:
        return "Portfolio unavailable"

    portfolio_value = portfolio_state["portfolio_value"]
    position_size, size_label = get_position_size(vote_count, portfolio_value, vix, high_conviction_rotation)
    direction = extract_direction(signal)
    reason = extract_reason(signal)
    market_open = is_market_hours()
    results = []

    voters_for = [name for name, verdict in verdicts if "VOTE: TRADE" in verdict]
    voters_against = [name for name, verdict in verdicts if "VOTE: PASS" in verdict]

    crypto_asset = extract_crypto(signal)
    if crypto_asset and orders_this_cycle < MAX_ORDERS_PER_CYCLE:
        crypto_map = {
            "BTC": "BTC/USD", "BITCOIN": "BTC/USD",
            "ETH": "ETH/USD", "ETHEREUM": "ETH/USD",
            "SOL": "SOL/USD", "SOLANA": "SOL/USD"
        }
        crypto_symbol = crypto_map.get(crypto_asset.upper())
        if crypto_symbol:
            approved, reason_blocked = check_execution_rules(crypto_symbol, direction, position_size, portfolio_state, vix)
            if approved:
                try:
                    notional = min(position_size, 5000)
                    result = alpaca_request("POST", "/v2/orders", {
                        "symbol": crypto_symbol, "notional": str(round(notional, 2)),
                        "side": direction, "type": "market", "time_in_force": "gtc"
                    })
                    orders_this_cycle += 1
                    send_performance(format_crypto_alert(crypto_symbol, direction, notional, reason, vote_count, voters_for, voters_against))
                    recent_signals[f"{crypto_symbol}_{direction}"] = 0
                    trade_history.append({"ticker": crypto_symbol, "direction": direction, "entry_price": None, "voters_for": voters_for, "voters_against": voters_against})
                    results.append(f"Crypto: {crypto_symbol}")
                except Exception as e:
                    results.append(f"Crypto failed: {e}")
            else:
                results.append(f"Crypto blocked: {reason_blocked}")

    if market_open and orders_this_cycle < MAX_ORDERS_PER_CYCLE:
        ticker = extract_ticker(signal, rotation_ticker)
        if ticker:
            approved, reason_blocked = check_execution_rules(ticker, direction, position_size, portfolio_state, vix)
            if approved:
                try:
                    existing = portfolio_state["positions"].get(ticker, {})
                    if existing:
                        ex_side = existing.get("side")
                        if (ex_side == "long" and direction == "sell") or (ex_side == "short" and direction == "buy"):
                            close_position(ticker)
                            time.sleep(2)

                    price_data = yf.download(ticker, period="1d", interval="1m", progress=False)
                    price = float(price_data["Close"].squeeze().iloc[-1])
                    qty = max(1, int(position_size / price))

                    order = submit_order(ticker, qty, direction)
                    order_id = order.get("id", "unknown")

                    filled = verify_order_filled(order_id)
                    if not filled:
                        results.append(f"{ticker} order did not fill -- no position taken")
                    else:
                        orders_this_cycle += 1
                        stop_price = price * 0.95 if direction == "buy" else price * 1.05
                        stop_side = "sell" if direction == "buy" else "buy"
                        stop_ok = submit_stop_loss_atomic(ticker, qty, stop_side, stop_price)
                        if stop_ok:
                            send_performance(format_trade_alert(ticker, direction, qty, price, stop_price, size_label, vote_count, voters_for, voters_against, reason))
                            recent_signals[f"{ticker}_{direction}"] = 0
                            trade_history.append({"ticker": ticker, "direction": direction, "entry_price": price, "voters_for": voters_for, "voters_against": voters_against})
                            results.append(f"Equity: {ticker} filled and protected")
                        else:
                            results.append(f"Equity: {ticker} -- position closed, stop failed")
                except Exception as e:
                    results.append(f"Equity failed: {e}")
            else:
                results.append(f"Equity blocked: {reason_blocked}")

    return "\n".join(results) if results else "No trades executed"

def ask_misfit(name, persona, signal, knowledge=""):
    scorecard = build_scorecard_context()
    knowledge_ctx = f"\n\nDEEP KNOWLEDGE:\n{knowledge}" if knowledge else ""
    score_ctx = f"\n\nTRACK RECORD:\n{scorecard}" if scorecard else ""
    msg = client.messages.create(
        model="claude-opus-4-5-20251101", max_tokens=400,
        messages=[{"role": "user", "content": f"{persona}{knowledge_ctx}{score_ctx}\n\nYoniBot signal:\n{signal}\n\n2-3 sentences. Brutal and direct. End with VOTE: TRADE or VOTE: PASS."}]
    )
    return msg.content[0].text

def ask_jane_street(signal, verdicts):
    debate = "\n\n".join([f"{n}:\n{v}" for n, v in verdicts])
    scorecard = build_scorecard_context()
    score_ctx = f"\n\nTRACK RECORD:\n{scorecard}" if scorecard else ""
    msg = client.messages.create(
        model="claude-opus-4-5-20251101", max_tokens=400,
        messages=[{"role": "user", "content": f"You ARE Jane Street quant engine.{score_ctx}\n\nSIGNAL:\n{signal}\n\nDEBATE:\n{debate}\n\nEdge, Kelly size, max drawdown. End with VETO: BLOCKED or VETO: APPROVED."}]
    )
    return msg.content[0].text

MISFITS = [
    ("Soros",
     "You ARE George Soros. Find the hidden peg. Where is the lie everyone believes and when do the defenders run out of ammunition?",
     ["George Soros Black Wednesday 1992 pound sterling ERM trade mechanics",
      "George Soros reflexivity theory philosophy interview 2024 2025",
      "George Soros biggest trades currency crisis methodology",
      "Soros Fund Management macro strategy current market views 2025 2026"]),
    ("Druckenmiller",
     "You ARE Stanley Druckenmiller. State your stop first, then target, then conviction size. Never average a loser.",
     ["Stanley Druckenmiller biggest trade Deutsche Mark Soros 1992",
      "Stanley Druckenmiller philosophy concentration asymmetric bet",
      "Stanley Druckenmiller current macro views Federal Reserve 2025 2026",
      "Druckenmiller risk management stop loss position sizing methodology"]),
    ("PTJ",
     "You ARE Paul Tudor Jones. Never get out of bed for less than 5 to 1. Does a 5 to 1 setup exist? Where is the hard stop?",
     ["Paul Tudor Jones Black Monday 1987 prediction",
      "Paul Tudor Jones trading rules risk management 5 to 1",
      "Paul Tudor Jones current market views macro 2025 2026",
      "PTJ Tudor Investment technical analysis trend following"]),
    ("Tepper",
     "You ARE David Tepper. Read the policy backdrop. Is the Federal Reserve with us or against us?",
     ["David Tepper 2009 bank trade how he made billions Appaloosa",
      "David Tepper Federal Reserve policy reading investment strategy",
      "David Tepper current market views Fed policy 2025 2026",
      "Tepper Appaloosa credit equity macro investing methodology"]),
    ("Andurand",
     "You ARE Pierre Andurand. Read tanker movements, refinery margins, geopolitical chokepoints. What does the physical world say?",
     ["Pierre Andurand oil trade 2008 2022 how he called the move",
      "Pierre Andurand physical commodity flows tanker market methodology",
      "Andurand Capital oil market views 2025 2026",
      "Pierre Andurand geopolitical risk Hormuz energy supply disruption"]),
]

cycle_count = 0
misfit_knowledge_cache = {}
knowledge_refresh_cycles = 8

def send_startup_message():
    send_performance("""🚀 MISFITS SYSTEM ONLINE

The Misfits are watching the markets.

You will receive updates here when:
📈 A trade is executed
🛑 A stop loss triggers
📅 Friday shorts closed before weekend
⚡ Circuit breaker fires
📊 Hourly portfolio updates

Five rules protecting every trade:
1. Max 2 orders per cycle
2. Every order verified before stop loss placed
3. Daily 3% loss limit circuit breaker
4. No new shorts after 2 PM ET Fridays
5. Stop loss placed atomically with every entry

-- Satis House Consulting""")

def run_cycle():
    global cycle_count, misfit_knowledge_cache, daily_start_value
    global trades_halted_today, recent_signals, orders_this_cycle

    cycle_count += 1
    orders_this_cycle = 0

    et = pytz.timezone("America/New_York")
    now_et = datetime.now(et)

    if now_et.hour == 9 and now_et.minute < 30:
        daily_start_value = None
        trades_halted_today = False
        cancel_all_orders()

    for key in list(recent_signals.keys()):
        recent_signals[key] += 1
        if recent_signals[key] > DUPLICATE_SIGNAL_BLOCKS:
            del recent_signals[key]

    portfolio_state = get_portfolio_state()
    if daily_start_value is None and portfolio_state:
        daily_start_value = portfolio_state["portfolio_value"]

    vix = get_vix()

    check_stop_losses(portfolio_state)

    if should_close_friday_shorts():
        close_friday_shorts(portfolio_state)
        portfolio_state = get_portfolio_state()

    if cycle_count % 4 == 0:
        report_open_positions(portfolio_state)

    if cycle_count % knowledge_refresh_cycles == 1:
        print("Refreshing knowledge bases...")
        for name, persona, queries in MISFITS:
            print(f"  Loading {name}...")
            misfit_knowledge_cache[name] = get_deep_misfit_knowledge(name, queries)
            time.sleep(3)

    market_open = is_market_hours()
    rotation_ticker, rotation_score, rotation_summary, _ = run_omniscient_rotation()
    high_conviction_rotation = rotation_ticker not in ["BIL", "UUP", None] and rotation_score > 0.5 and market_open

    technical = get_technical_signals()
    yields = get_yield_curve()
    fear = get_fear_greed()
    arxiv = get_arxiv_signals()
    news = get_market_news()
    geo = get_geopolitical_news()
    congress = get_congressional_trading()
    scorecard = build_scorecard_context()

    portfolio_summary = ""
    if portfolio_state:
        portfolio_summary = f"""
PORTFOLIO:
Value: ${portfolio_state['portfolio_value']:,.2f}
Equity: {portfolio_state['equity_pct']*100:.1f}% | Crypto: {portfolio_state['crypto_pct']*100:.1f}% | Leveraged: {portfolio_state['leveraged_pct']*100:.1f}%
Positions: {list(portfolio_state['positions'].keys()) or 'None'}
VIX: {vix:.1f}"""

    yoni_push = f"\nHIGH CONVICTION: Rotation selected {rotation_ticker} score {rotation_score:.3f}. Push hard." if high_conviction_rotation else ""
    friday_note = "\nFRIDAY: No new short signals." if is_friday_short_blocked() else ""
    weekend_note = "" if market_open else "\nMARKET CLOSED. Crypto signals only."

    context = f"""MARKET: {'OPEN' if market_open else 'CLOSED'}
{portfolio_summary}
{scorecard}

ROTATION (2324% backtest 2019-2026):
{rotation_summary or 'Unavailable'}

TECHNICALS:
{technical}

YIELD CURVE: {yields}
SENTIMENT: {fear}

ARXIV:
{arxiv}

NEWS:
{news}

GEOPOLITICAL:
{geo}

CONGRESSIONAL:
{congress}
{yoni_push}"""

    yoni = client.messages.create(
        model="claude-opus-4-5-20251101", max_tokens=1024,
        messages=[{"role": "user", "content": f"""You are YoniBot, autonomous trading intelligence for Satis House Consulting. Pure quant signal generation. No personal investment references.

{context}
{weekend_note}
{friday_note}

Rules:
- Omniscient Rotation is highest conviction tool
- Signal only if two indicators confirm AND news supports
- Market closed means crypto only
- No short signals on Fridays after 2 PM ET
- Say NO SIGNAL if bar not met
- Output: Asset, Direction, Entry, Stop, Target, Why. Max 300 words."""}]
    )
    signal = yoni.content[0].text

    verdicts = []
    vote_count = 0
    for name, persona, queries in MISFITS:
        knowledge = misfit_knowledge_cache.get(name, "")
        verdict = ask_misfit(name, persona, signal, knowledge)
        verdicts.append((name, verdict))
        if "VOTE: TRADE" in verdict:
            vote_count += 1

    jane = ask_jane_street(signal, verdicts)
    approved = "VETO: APPROVED" in jane
    majority = vote_count >= 3

    if majority and approved:
        trade_result = execute_trade(signal, vote_count, verdicts, portfolio_state, vix,
                                     rotation_ticker if high_conviction_rotation else None,
                                     high_conviction_rotation)
        verdict_line = f"VERDICT: EXECUTE -- {vote_count}/5 voted TRADE\n{trade_result}"
    elif not approved:
        verdict_line = f"VERDICT: BLOCKED -- Jane Street vetoed ({vote_count}/5)"
    else:
        verdict_line = f"VERDICT: PASS -- Only {vote_count}/5 voted TRADE"

    send_telegram(f"YONIBOT SIGNAL\n{signal}")
    time.sleep(2)
    debate_msg = "THE MISFITS DEBATE\n"
    for name, verdict in verdicts:
        debate_msg += f"\n{name.upper()}:\n{verdict}\n"
    send_telegram(debate_msg)
    time.sleep(2)
    send_telegram(f"JANE STREET:\n{jane}\n\n{verdict_line}")
    print(f"Brief sent. {verdict_line}")

    smart_sleep(900)

while True:
    try:
        if cycle_count == 0:
            send_startup_message()
        run_cycle()
    except Exception as e:
        send_telegram(f"Misfits error: {e}")
        print(f"Error: {e}")
        smart_sleep(900)
