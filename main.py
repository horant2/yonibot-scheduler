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

MAX_POSITION_PCT = 0.15
MAX_EQUITY_PCT = 0.40
MAX_CRYPTO_PCT = 0.20
MAX_LEVERAGED_PCT = 0.30
DAILY_LOSS_LIMIT = 0.03
VIX_REDUCE_THRESHOLD = 35
VIX_STOP_THRESHOLD = 50
DUPLICATE_SIGNAL_BLOCKS = 2

recent_signals = {}
daily_start_value = None
trades_halted_today = False
last_update_id = 0

LEVERAGED_ETFS = {"TQQQ", "SOXL", "TECL", "FAS", "ERX", "TMF", "SPXL", "UPRO"}
CRYPTO_ASSETS = {"BTC/USD", "ETH/USD", "SOL/USD"}

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
        requests.post(url, json={"chat_id": TELEGRAM_PERFORMANCE_CHAT_ID, "text": chunk})
        time.sleep(1)

def format_trade_alert(ticker, direction, qty, price, stop_price, size_label, vote_count, voters_for, voters_against, reason):
    action = "Bought" if direction == "buy" else "Sold Short"
    total_bet = qty * price
    emoji = "🟢" if direction == "buy" else "🔴"
    return f"""{emoji} THE MISFITS JUST TRADED

{action} {qty} shares of {ticker} at ${price:.2f}
Bet size: ${total_bet:,.0f} ({size_label})
Stop loss set at: ${stop_price:.2f}

Why: {reason}

Who agreed ({vote_count}/5): {', '.join(voters_for)}
Who disagreed: {', '.join(voters_against) if voters_against else 'Nobody'}
Jane Street approved the math.

-- Satis House Consulting"""

def format_crypto_alert(crypto_symbol, direction, notional, reason, vote_count, voters_for, voters_against):
    action = "Bought" if direction == "buy" else "Sold"
    emoji = "🟢" if direction == "buy" else "🔴"
    return f"""{emoji} THE MISFITS JUST TRADED CRYPTO

{action} ${notional:,.0f} of {crypto_symbol}
Markets are closed but crypto never sleeps.

Why: {reason}

Who agreed ({vote_count}/5): {', '.join(voters_for)}
Who disagreed: {', '.join(voters_against) if voters_against else 'Nobody'}
Jane Street approved the math.

-- Satis House Consulting"""

def format_stop_loss_alert(symbol, exit_price, pnl_dollar, pnl_pct):
    emoji = "🛑"
    result = f"+${pnl_dollar:,.0f} profit" if pnl_dollar >= 0 else f"-${abs(pnl_dollar):,.0f} loss"
    return f"""{emoji} STOP LOSS TRIGGERED

Closed {symbol} position
Exit price: ${exit_price:.2f}
Result: {result} ({pnl_pct:+.1f}%)

The Misfits cut the position. Capital protected.
On to the next signal.

-- Satis House Consulting"""

def format_position_report(portfolio_state, daily_pnl=None):
    if not portfolio_state or not portfolio_state["positions"]:
        return f"""📊 MISFITS PORTFOLIO UPDATE

No open positions right now.
Sitting in cash, waiting for the right signal.

Cash available: ${portfolio_state['portfolio_value']:,.0f}

-- Satis House Consulting"""

    lines = [f"📊 MISFITS PORTFOLIO UPDATE\n"]
    lines.append(f"Total value: ${portfolio_state['portfolio_value']:,.2f}")

    if daily_pnl is not None:
        daily_emoji = "📈" if daily_pnl >= 0 else "📉"
        lines.append(f"Today: {daily_emoji} {'+' if daily_pnl >= 0 else ''}{daily_pnl*100:.2f}%\n")

    lines.append("Open positions:")
    for symbol, pos in portfolio_state["positions"].items():
        pnl_emoji = "✅" if pos["unrealized_pnl"] >= 0 else "⚠️"
        direction = "Long" if pos["side"] == "long" else "Short"
        lines.append(f"{pnl_emoji} {symbol} ({direction}) -- {'+' if pos['unrealized_pnl'] >= 0 else ''}{pos['unrealized_pct']:.1f}% since entry")

    buying_power = portfolio_state.get("buying_power", 0)
    lines.append(f"\nCash available: ${buying_power:,.0f}")
    lines.append("\n-- Satis House Consulting")
    return "\n".join(lines)

def format_no_signal_message():
    et = pytz.timezone("America/New_York")
    now = datetime.now(et)
    market_status = "Markets are open" if (now.weekday() < 5 and 9 <= now.hour < 16) else "Markets are closed"
    return f"""👀 THE MISFITS REVIEWED THE MARKETS

{market_status}. No trade this cycle.
The bar was not high enough. Capital is protected.

The Misfits only trade when the signal is real.

-- Satis House Consulting"""

def format_scorecard():
    lines = ["🏆 MISFIT SCORECARD\n"]
    for name, scores in misfit_scorecard.items():
        total = scores["total"]
        if total > 0:
            win_rate = scores["correct"] / total * 100
            lines.append(f"{name}: {scores['correct']}/{total} correct ({win_rate:.0f}% win rate)")
        else:
            lines.append(f"{name}: No trades yet")
    lines.append("\n-- Satis House Consulting")
    return "\n".join(lines)

def get_telegram_updates():
    global last_update_id
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_PERFORMANCE_TOKEN}/getUpdates"
        params = {"offset": last_update_id + 1, "timeout": 1}
        response = requests.get(url, params=params, timeout=5).json()
        if response.get("ok") and response.get("result"):
            return response["result"]
        return []
    except:
        return []

def handle_telegram_commands(portfolio_state, vix):
    global last_update_id, trades_halted_today
    updates = get_telegram_updates()
    for update in updates:
        last_update_id = update["update_id"]
        message = update.get("message", {})
        text = message.get("text", "").strip().lower()
        chat_id = message.get("chat", {}).get("id")
        if not text or not chat_id:
            continue

        if text in ["/positions", "positions", "show positions"]:
            daily_pnl = None
            if daily_start_value and portfolio_state:
                daily_pnl = (portfolio_state["portfolio_value"] - daily_start_value) / daily_start_value
            send_performance(format_position_report(portfolio_state, daily_pnl))

        elif text in ["/scorecard", "scorecard", "scores", "who is winning"]:
            send_performance(format_scorecard())

        elif text in ["/vix", "vix"]:
            vix_status = "Low -- normal trading" if vix < 20 else ("Elevated -- reduced position sizes" if vix < 35 else ("High -- position sizes cut 50%" if vix < 50 else "Extreme -- equity trading paused"))
            send_performance(f"📊 VIX (Market Fear Index)\n\nCurrent VIX: {vix:.1f}\nStatus: {vix_status}\n\n-- Satis House Consulting")

        elif text in ["/pause", "pause", "pause trading", "stop trading"]:
            trades_halted_today = True
            send_performance("⏸ TRADING PAUSED\n\nThe Misfits will not execute any new trades until tomorrow morning.\nExisting positions remain open.\n\n-- Satis House Consulting")

        elif text in ["/resume", "resume", "resume trading", "start trading"]:
            trades_halted_today = False
            send_performance("▶️ TRADING RESUMED\n\nThe Misfits are back on watch.\n\n-- Satis House Consulting")

        elif text in ["/rotation", "rotation", "rotation scores"]:
            send_performance("🔄 Running rotation scan... check the main channel for the next brief.")

        elif text in ["/help", "help", "commands", "what can you do"]:
            help_text = """🤖 MISFITS COMMAND CENTER

Send any of these messages to control the system:

/positions -- Show current holdings and profit/loss
/scorecard -- See which Misfits have been right
/vix -- Check market fear level
/pause -- Stop new trades (keeps existing positions)
/resume -- Restart trading after pause
/rotation -- Show current momentum scores
/help -- Show this menu

The Misfits trade automatically every 15 minutes when they find a signal strong enough.

-- Satis House Consulting"""
            send_performance(help_text)

        else:
            send_performance(f"I received your message but did not understand the command.\n\nSend /help to see what I can do.\n\n-- Satis House Consulting")

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

def get_orders(status="open"):
    return alpaca_request("GET", "/v2/orders", params={"status": status})

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
    market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
    return market_open <= now <= market_close

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
                    "side": side,
                    "qty": qty,
                    "market_value": market_val,
                    "avg_entry": avg_entry,
                    "current_price": current_price,
                    "unrealized_pnl": unrealized_pnl,
                    "unrealized_pct": unrealized_pct,
                    "pct_of_portfolio": abs(market_val) / portfolio_value if portfolio_value > 0 else 0
                }

                if "USD" in symbol:
                    crypto_value += abs(market_val)
                elif symbol in LEVERAGED_ETFS:
                    leveraged_value += abs(market_val)
                else:
                    equity_value += abs(market_val)

        return {
            "portfolio_value": portfolio_value,
            "buying_power": buying_power,
            "positions": position_map,
            "equity_value": equity_value,
            "crypto_value": crypto_value,
            "leveraged_value": leveraged_value,
            "equity_pct": equity_value / portfolio_value if portfolio_value > 0 else 0,
            "crypto_pct": crypto_value / portfolio_value if portfolio_value > 0 else 0,
            "leveraged_pct": leveraged_value / portfolio_value if portfolio_value > 0 else 0
        }
    except Exception as e:
        print(f"Portfolio state error: {e}")
        return None

def check_execution_rules(ticker, direction, position_size, portfolio_state, vix):
    global trades_halted_today, daily_start_value

    if portfolio_state is None:
        return False, "Portfolio state unavailable"

    portfolio_value = portfolio_state["portfolio_value"]
    positions = portfolio_state["positions"]

    if trades_halted_today:
        return False, "Trading paused"

    if daily_start_value is not None:
        daily_pnl_pct = (portfolio_value - daily_start_value) / daily_start_value
        if daily_pnl_pct <= -DAILY_LOSS_LIMIT:
            trades_halted_today = True
            send_performance(f"🛑 DAILY LOSS LIMIT HIT\n\nThe portfolio is down {abs(daily_pnl_pct)*100:.1f}% today.\nAll trading paused until tomorrow to protect capital.\n\n-- Satis House Consulting")
            return False, "Daily loss limit triggered"

    if vix >= VIX_STOP_THRESHOLD:
        return False, f"VIX at {vix:.1f} -- too dangerous for new equity trades"

    signal_key = f"{ticker}_{direction}"
    if signal_key in recent_signals:
        if recent_signals[signal_key] < DUPLICATE_SIGNAL_BLOCKS:
            return False, f"Same trade attempted recently -- waiting for new signal"

    if ticker in positions:
        existing = positions[ticker]
        if existing["side"] == "long" and direction == "buy":
            return False, f"Already holding {ticker} long"
        if existing["side"] == "short" and direction == "sell":
            return False, f"Already short {ticker}"

    is_crypto = any(c in ticker.upper() for c in ["BTC", "ETH", "SOL"])
    is_leveraged = ticker in LEVERAGED_ETFS

    if is_leveraged and not is_market_hours():
        return False, f"{ticker} only trades during market hours"

    if not is_crypto and not is_market_hours():
        return False, f"Market closed -- waiting for Monday"

    if is_leveraged and portfolio_state["leveraged_pct"] >= MAX_LEVERAGED_PCT:
        return False, f"Already at max leveraged ETF allocation"
    elif is_crypto and portfolio_state["crypto_pct"] >= MAX_CRYPTO_PCT:
        return False, f"Already at max crypto allocation"
    elif not is_crypto and not is_leveraged and portfolio_state["equity_pct"] >= MAX_EQUITY_PCT:
        return False, f"Already at max equity allocation"

    return True, "All checks passed"

def update_training_loop(trade_record, outcome_pnl_pct):
    global misfit_scorecard, trade_history
    voters_for = trade_record.get("voters_for", [])
    voters_against = trade_record.get("voters_against", [])
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

    trade_history.append({
        "ticker": trade_record.get("ticker"),
        "direction": trade_record.get("direction"),
        "entry": trade_record.get("entry_price"),
        "outcome_pct": outcome_pnl_pct,
        "profitable": trade_was_profitable,
        "voters_for": voters_for,
        "voters_against": voters_against,
        "timestamp": datetime.now(pytz.utc).isoformat()
    })

def build_scorecard_context():
    lines = []
    for name, scores in misfit_scorecard.items():
        total = scores["total"]
        if total > 0:
            win_rate = scores["correct"] / total * 100
            lines.append(f"{name}: {win_rate:.0f}% win rate ({scores['correct']}/{total} trades)")
    if lines:
        return "MISFIT TRACK RECORD:\n" + "\n".join(lines)
    return ""

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
            handle_telegram_commands(None, 20.0)

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
            if rsi > 85 or rsi < 30:
                rsi_penalty = 0.9
            final_score = risk_adj_mom * trend_score * rsi_penalty
            scores[ticker] = final_score
            prices[ticker] = price
        except Exception as e:
            print(f"Rotation error for {ticker}: {e}")
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
    rotation_summary += f"SPY Trend: {'BULL' if spy_trend else 'BEAR'}\n"
    rotation_summary += f"Winner: {best_ticker} (score: {best_score:.3f})\n"
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
            num_results=5,
            include_domains=["arxiv.org"],
            text={"max_characters": 500}
        )
        papers = []
        for r in results.results:
            papers.append(f"PAPER: {r.title}\n{r.text[:400]}")
        return "\n\n".join(papers)
    except Exception as e:
        return f"arXiv unavailable: {e}"

def get_market_news():
    try:
        results = exa.search_and_contents(
            "market moving news macro trading today",
            num_results=7,
            include_domains=["reuters.com", "bloomberg.com", "ft.com", "wsj.com", "cnbc.com"],
            text={"max_characters": 400}
        )
        news = []
        for r in results.results:
            news.append(f"{r.title}: {r.text[:300]}")
        return "\n\n".join(news)
    except Exception as e:
        return f"News unavailable: {e}"

def get_geopolitical_news():
    try:
        results = exa.search_and_contents(
            "Iran Strait of Hormuz oil supply disruption geopolitical risk today",
            num_results=5,
            text={"max_characters": 400}
        )
        news = []
        for r in results.results:
            news.append(f"{r.title}: {r.text[:300]}")
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

def get_deep_misfit_knowledge(name, queries):
    knowledge_blocks = []
    for query in queries:
        try:
            results = exa.search_and_contents(
                query,
                num_results=3,
                text={"max_characters": 500}
            )
            for r in results.results:
                knowledge_blocks.append(f"SOURCE: {r.title}\n{r.text[:400]}")
            time.sleep(1)
        except Exception as e:
            print(f"Knowledge fetch error for {name}: {e}")
    return "\n\n".join(knowledge_blocks)

def get_position_size(vote_count, portfolio_value, vix, high_conviction_rotation=False):
    base_pct = 0.15 if high_conviction_rotation else (0.10 if vote_count == 5 else (0.08 if vote_count == 4 else 0.05))
    if vix >= VIX_REDUCE_THRESHOLD:
        base_pct = base_pct * 0.5
        label = f"reduced size due to high volatility ({base_pct*100:.0f}% of portfolio)"
    elif high_conviction_rotation:
        label = "15% of portfolio -- rotation conviction"
    elif vote_count == 5:
        label = "10% of portfolio -- maximum conviction"
    elif vote_count == 4:
        label = "8% of portfolio -- high conviction"
    else:
        label = "5% of portfolio -- base conviction"
    size = min(portfolio_value * base_pct, 15000)
    return size, label

def check_stop_losses(portfolio_state):
    try:
        if portfolio_state is None:
            return
        positions = portfolio_state["positions"]
        for symbol, pos in positions.items():
            unrealized_pct = pos["unrealized_pct"]
            side = pos["side"]
            current_price = pos["current_price"]
            entry_price = pos["avg_entry"]
            unrealized_pnl = pos["unrealized_pnl"]
            stop_triggered = (side == "long" and unrealized_pct <= -5) or (side == "short" and unrealized_pct <= -5)
            if stop_triggered:
                close_position(symbol)
                msg = format_stop_loss_alert(symbol, current_price, unrealized_pnl, unrealized_pct)
                send_performance(msg)
                send_telegram(f"Stop loss hit on {symbol}. See performance channel for details.")
                for trade in reversed(trade_history):
                    if trade.get("ticker") == symbol and trade.get("outcome_pct") is None:
                        update_training_loop(trade, unrealized_pct / 100)
                        break
    except Exception as e:
        print(f"Stop loss check error: {e}")

def report_open_positions(portfolio_state):
    try:
        if portfolio_state is None:
            return
        daily_pnl = None
        if daily_start_value and portfolio_state:
            daily_pnl = (portfolio_state["portfolio_value"] - daily_start_value) / daily_start_value
        msg = format_position_report(portfolio_state, daily_pnl)
        send_performance(msg)
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
    if "SHORT" in signal.upper():
        return "sell"
    return "buy"

def extract_reason(signal):
    lines = signal.strip().split("\n")
    for line in lines:
        if any(word in line.lower() for word in ["because", "thesis", "signal", "confirmed", "why"]):
            return line.strip()
    return lines[0].strip() if lines else "Multiple indicators confirmed"

def execute_trade(signal, vote_count, verdicts, portfolio_state, vix, rotation_ticker=None, high_conviction_rotation=False):
    global recent_signals, trade_history

    if portfolio_state is None:
        return "Portfolio state unavailable -- trade skipped", [], []

    portfolio_value = portfolio_state["portfolio_value"]
    position_size, size_label = get_position_size(vote_count, portfolio_value, vix, high_conviction_rotation)
    direction = extract_direction(signal)
    reason = extract_reason(signal)
    market_open = is_market_hours()
    results = []

    voters_for = [name for name, verdict in verdicts if "VOTE: TRADE" in verdict]
    voters_against = [name for name, verdict in verdicts if "VOTE: PASS" in verdict]

    crypto_asset = extract_crypto(signal)
    if crypto_asset:
        crypto_map = {"BTC": "BTC/USD", "BITCOIN": "BTC/USD", "ETH": "ETH/USD", "ETHEREUM": "ETH/USD", "SOL": "SOL/USD", "SOLANA": "SOL/USD"}
        crypto_symbol = crypto_map.get(crypto_asset.upper())
        if crypto_symbol:
            approved, block_reason = check_execution_rules(crypto_symbol, direction, position_size, portfolio_state, vix)
            if approved:
                try:
                    notional = min(position_size, 5000)
                    order = {"symbol": crypto_symbol, "notional": str(round(notional, 2)), "side": direction, "type": "market", "time_in_force": "gtc"}
                    result = alpaca_request("POST", "/v2/orders", order)
                    order_id = result.get("id", "unknown")
                    msg = format_crypto_alert(crypto_symbol, direction, notional, reason, vote_count, voters_for, voters_against)
                    send_performance(msg)
                    recent_signals[f"{crypto_symbol}_{direction}"] = 0
                    trade_history.append({"ticker": crypto_symbol, "direction": direction, "entry_price": None, "voters_for": voters_for, "voters_against": voters_against, "outcome_pct": None})
                    results.append(f"Crypto trade executed: {crypto_symbol}")
                except Exception as e:
                    results.append(f"Crypto execution failed: {e}")
            else:
                results.append(f"Crypto blocked: {block_reason}")

    if market_open:
        ticker = extract_ticker(signal, rotation_ticker)
        if ticker:
            approved, block_reason = check_execution_rules(ticker, direction, position_size, portfolio_state, vix)
            if approved:
                try:
                    existing = portfolio_state["positions"].get(ticker, {})
                    if existing:
                        existing_side = existing.get("side")
                        if (existing_side == "long" and direction == "sell") or (existing_side == "short" and direction == "buy"):
                            close_position(ticker)
                            time.sleep(1)

                    price_data = yf.download(ticker, period="1d", interval="1m", progress=False)
                    price = float(price_data["Close"].squeeze().iloc[-1])
                    qty = max(1, int(position_size / price))
                    order = submit_order(ticker, qty, direction)
                    order_id = order.get("id", "unknown")
                    stop_price = price * 0.95 if direction == "buy" else price * 1.05
                    stop_side = "sell" if direction == "buy" else "buy"
                    submit_stop_loss(ticker, qty, stop_side, stop_price)

                    msg = format_trade_alert(ticker, direction, qty, price, stop_price, size_label, vote_count, voters_for, voters_against, reason)
                    send_performance(msg)
                    recent_signals[f"{ticker}_{direction}"] = 0
                    trade_history.append({"ticker": ticker, "direction": direction, "entry_price": price, "voters_for": voters_for, "voters_against": voters_against, "outcome_pct": None})
                    results.append(f"Equity trade executed: {ticker}")
                except Exception as e:
                    results.append(f"Equity execution failed: {e}")
            else:
                results.append(f"Equity blocked: {block_reason}")

    if not results:
        return "No trades executed this cycle", voters_for, voters_against
    return "\n".join(results), voters_for, voters_against

def ask_misfit(name, persona, signal, knowledge=""):
    scorecard_context = build_scorecard_context()
    knowledge_context = f"\n\nDEEP KNOWLEDGE BASE:\n{knowledge}" if knowledge else ""
    score_context = f"\n\nCURRENT TRACK RECORD:\n{scorecard_context}" if scorecard_context else ""
    msg = client.messages.create(
        model="claude-opus-4-5-20251101",
        max_tokens=400,
        messages=[{"role": "user", "content": f"{persona}{knowledge_context}{score_context}\n\nYoniBot signal:\n{signal}\n\nGive your verdict in 2-3 sentences. Be brutal and direct. End with VOTE: TRADE or VOTE: PASS on its own line."}]
    )
    return msg.content[0].text

def ask_jane_street(signal, verdicts):
    debate = "\n\n".join([f"{n}:\n{v}" for n, v in verdicts])
    scorecard = build_scorecard_context()
    score_context = f"\n\nMISFIT TRACK RECORD:\n{scorecard}" if scorecard else ""
    msg = client.messages.create(
        model="claude-opus-4-5-20251101",
        max_tokens=400,
        messages=[{"role": "user", "content": f"You ARE Jane Street quant engine.{score_context}\n\nSIGNAL:\n{signal}\n\nDEBATE:\n{debate}\n\nCalculate edge, Kelly size, max drawdown. End with VETO: BLOCKED or VETO: APPROVED on its own line."}]
    )
    return msg.content[0].text

MISFITS = [
    (
        "Soros",
        "You ARE George Soros. Find the hidden peg. Where is the lie everyone believes and when do the defenders run out of ammunition?",
        [
            "George Soros Black Wednesday 1992 pound sterling ERM trade mechanics",
            "George Soros reflexivity theory market philosophy interview 2024 2025",
            "George Soros biggest trades currency crisis methodology",
            "Soros Fund Management macro strategy current market views 2025 2026"
        ]
    ),
    (
        "Druckenmiller",
        "You ARE Stanley Druckenmiller. State your stop level first, then your target, then your conviction size. Never average a loser.",
        [
            "Stanley Druckenmiller biggest trade Deutsche Mark Soros 1992",
            "Stanley Druckenmiller investment philosophy concentration asymmetric bet",
            "Stanley Druckenmiller current macro views Federal Reserve 2025 2026",
            "Druckenmiller risk management stop loss position sizing methodology"
        ]
    ),
    (
        "PTJ",
        "You ARE Paul Tudor Jones. You never get out of bed for less than 5 to 1 risk reward. Does a 5 to 1 setup exist here? Where is the hard stop?",
        [
            "Paul Tudor Jones Black Monday 1987 prediction how he called it",
            "Paul Tudor Jones trading rules risk management 5 to 1 philosophy",
            "Paul Tudor Jones current market views macro 2025 2026 interview",
            "PTJ Tudor Investment technical analysis trend following methodology"
        ]
    ),
    (
        "Tepper",
        "You ARE David Tepper. Read the policy backdrop. Is the Federal Reserve with us or against us on this trade?",
        [
            "David Tepper 2009 bank trade how he made billions Appaloosa",
            "David Tepper Federal Reserve policy reading investment strategy",
            "David Tepper current market views Fed policy 2025 2026",
            "Tepper Appaloosa credit equity macro investing methodology"
        ]
    ),
    (
        "Andurand",
        "You ARE Pierre Andurand. Read tanker movements, refinery margins, and geopolitical chokepoints. What does the physical world say?",
        [
            "Pierre Andurand oil trade 2008 2022 how he called the move",
            "Pierre Andurand physical commodity flows tanker market methodology",
            "Andurand Capital oil market views 2025 2026 current outlook",
            "Pierre Andurand geopolitical risk Hormuz energy supply disruption framework"
        ]
    ),
]

cycle_count = 0
misfit_knowledge_cache = {}
knowledge_refresh_cycles = 8

def run_cycle():
    global cycle_count, misfit_knowledge_cache, daily_start_value, trades_halted_today, recent_signals

    cycle_count += 1

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

    handle_telegram_commands(portfolio_state, vix)
    check_stop_losses(portfolio_state)

    if cycle_count % 4 == 0:
        report_open_positions(portfolio_state)

    if cycle_count % knowledge_refresh_cycles == 1:
        print("Refreshing deep Misfit knowledge bases...")
        for name, persona, queries in MISFITS:
            print(f"  Loading knowledge for {name}...")
            misfit_knowledge_cache[name] = get_deep_misfit_knowledge(name, queries)
            time.sleep(3)

    market_open = is_market_hours()
    rotation_ticker, rotation_score, rotation_summary, rotation_price = run_omniscient_rotation()
    high_conviction_rotation = rotation_ticker not in ["BIL", "UUP", None] and rotation_score > 0.5 and market_open

    technical = get_technical_signals()
    yields = get_yield_curve()
    fear = get_fear_greed()
    arxiv = get_arxiv_signals()
    news = get_market_news()
    geo = get_geopolitical_news()
    congress = get_congressional_trading()

    scorecard_context = build_scorecard_context()
    portfolio_summary = ""
    if portfolio_state:
        portfolio_summary = f"""
CURRENT PORTFOLIO:
Value: ${portfolio_state['portfolio_value']:,.2f}
Equity: {portfolio_state['equity_pct']*100:.1f}% | Crypto: {portfolio_state['crypto_pct']*100:.1f}% | Leveraged: {portfolio_state['leveraged_pct']*100:.1f}%
Open positions: {', '.join(portfolio_state['positions'].keys()) if portfolio_state['positions'] else 'None'}
VIX: {vix:.1f}"""

    market_status = "OPEN" if market_open else "CLOSED -- crypto only"
    yoni_push = f"\nYONIBOT HIGH CONVICTION: Rotation selected {rotation_ticker} score {rotation_score:.3f}. Push hard." if high_conviction_rotation else ""

    context = f"""MARKET STATUS: {market_status}
{portfolio_summary}

{scorecard_context}

OMNISCIENT ROTATION (2324% backtest 2019-2026):
{rotation_summary if rotation_summary else 'Unavailable'}

TECHNICAL INDICATORS:
{technical}

YIELD CURVE:
{yields}

SENTIMENT:
{fear}

ARXIV QUANT PAPERS:
{arxiv}

MARKET NEWS:
{news}

GEOPOLITICAL NEWS:
{geo}

CONGRESSIONAL TRADING:
{congress}
{yoni_push}"""

    weekend_note = "" if market_open else "\nMARKET CLOSED. Crypto signals only."

    yoni = client.messages.create(
        model="claude-opus-4-5-20251101",
        max_tokens=1024,
        messages=[{"role": "user", "content": f"""You are YoniBot, autonomous trading intelligence for Satis House Consulting. Pure quantitative signal generation only. No personal investment references.

{context}
{weekend_note}

Rules:
- Omniscient Rotation is highest conviction tool.
- Only signal if two indicators confirm AND news supports.
- Consider current portfolio allocations.
- Market closed means crypto only.
- Say NO SIGNAL if nothing qualifies.
- Output: Asset, Direction, Entry Zone, Stop, Target, Why. Max 300 words."""}]
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
        trade_result, voters_for, voters_against = execute_trade(
            signal, vote_count, verdicts, portfolio_state, vix,
            rotation_ticker if high_conviction_rotation else None,
            high_conviction_rotation
        )
        verdict_line = f"VERDICT: EXECUTE -- {vote_count}/5 voted TRADE. Jane Street approved.\n{trade_result}"
    elif not approved:
        verdict_line = f"VERDICT: BLOCKED -- Jane Street vetoed. ({vote_count}/5 voted TRADE)"
        send_performance(format_no_signal_message())
    else:
        verdict_line = f"VERDICT: PASS -- Only {vote_count}/5 voted TRADE."
        send_performance(format_no_signal_message())

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
