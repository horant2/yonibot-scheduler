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
import json
import websocket
import threading

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
TELEGRAM_PERFORMANCE_TOKEN = os.environ.get("TELEGRAM_PERFORMANCE_TOKEN")
TELEGRAM_PERFORMANCE_CHAT_ID = os.environ.get("TELEGRAM_PERFORMANCE_CHAT_ID")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
EXA_API_KEY = os.environ.get("EXA_API_KEY")
ALPACA_API_KEY = os.environ.get("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY")
FRED_API_KEY = os.environ.get("FRED_API_KEY")
EIA_API_KEY = os.environ.get("EIA_API_KEY")
AISSTREAM_API_KEY = os.environ.get("AISSTREAM_API_KEY")
ALPACA_BASE_URL = "https://paper-api.alpaca.markets"
LOG_FILE = "/tmp/misfits_log.json"
INCEPTION_VALUE = 100000

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
exa = Exa(api_key=EXA_API_KEY)

MAX_ORDERS_PER_CYCLE = 2
DAILY_LOSS_LIMIT = 0.03
FRIDAY_SHORT_CUTOFF_HOUR = 14
FRIDAY_SHORT_CLOSE_HOUR = 15
FRIDAY_SHORT_CLOSE_MINUTE = 30
MAX_SINGLE_POSITION = 0.15
VIX_REDUCE_THRESHOLD = 35
VIX_STOP_THRESHOLD = 50
MAX_EQUITY_PCT = 0.60
MAX_CRYPTO_PCT = 0.25
MAX_LEVERAGED_PCT = 0.45
DUPLICATE_SIGNAL_BLOCKS = 2
QUARTER_KELLY_FRACTION = 0.25
MIN_SIGNAL_SCORE = 1.5

LEVERAGED_ETFS = {"TQQQ", "SOXL", "TECL", "FAS", "ERX", "TMF", "SPXL", "UPRO"}

TRADEABLE_UNIVERSE = {
    "SOXL": "3x Semiconductors",
    "TECL": "3x Tech",
    "TQQQ": "3x Nasdaq",
    "FAS": "3x Financials",
    "ERX": "2x Energy",
    "QQQ": "Nasdaq ETF",
    "SPY": "S&P 500 ETF",
    "GLD": "Gold ETF",
    "SLV": "Silver ETF",
    "TLT": "20Y Treasury ETF",
    "HYG": "High Yield Bond ETF",
    "USO": "Oil ETF",
    "BNO": "Brent Oil ETF",
    "UNG": "Natural Gas ETF",
    "XLE": "Energy Sector ETF",
    "XOP": "Oil & Gas ETF",
    "FRO": "Frontline Tankers",
    "VLO": "Valero Energy",
    "EEM": "Emerging Markets ETF",
    "UUP": "US Dollar ETF",
    "FXE": "Euro ETF",
    "EWZ": "Brazil ETF",
    "TUR": "Turkey ETF",
    "IWM": "Russell 2000 ETF",
    "XLF": "Financials ETF",
    "BTC/USD": "Bitcoin",
    "ETH/USD": "Ethereum",
    "SOL/USD": "Solana"
}

live_price_cache = {}
price_cache_time = {}
PRICE_CACHE_SECONDS = 60

recent_signals = {}
daily_start_value = None
trades_halted_today = False
orders_this_cycle = 0
hormuz_vessels = []
hormuz_lock = threading.Lock()
daily_scorecard_sent = False

misfit_scorecard = {
    "Soros": {"correct": 0, "total": 0, "weight": 1.0},
    "Druckenmiller": {"correct": 0, "total": 0, "weight": 1.0},
    "PTJ": {"correct": 0, "total": 0, "weight": 1.0},
    "Tepper": {"correct": 0, "total": 0, "weight": 1.0},
    "Andurand": {"correct": 0, "total": 0, "weight": 1.0}
}

session_stats = {
    "execute": 0, "pass": 0, "total": 0,
    "misfit_signals": {
        "Soros": {"fired": 0, "skipped": 0},
        "Druckenmiller": {"fired": 0, "skipped": 0},
        "PTJ": {"fired": 0, "skipped": 0},
        "Tepper": {"fired": 0, "skipped": 0},
        "Andurand": {"fired": 0, "skipped": 0}
    }
}

trade_history = []

def get_live_price(ticker):
    global live_price_cache, price_cache_time
    now = time.time()
    if ticker in live_price_cache and (now - price_cache_time.get(ticker, 0)) < PRICE_CACHE_SECONDS:
        return live_price_cache[ticker]
    try:
        yf_ticker = ticker.replace("/", "-")
        period = "2d"
        interval = "1m"
        data = yf.download(yf_ticker, period=period, interval=interval, progress=False)
        if data.empty:
            data = yf.download(yf_ticker, period="5d", interval="5m", progress=False)
        if not data.empty:
            price = float(data["Close"].squeeze().dropna().iloc[-1])
            live_price_cache[ticker] = price
            price_cache_time[ticker] = now
            return price
    except Exception as e:
        print(f"Price fetch error for {ticker}: {e}")
    return None

def get_all_live_prices():
    prices = {}
    print("Fetching live prices for tradeable universe...")
    for ticker in TRADEABLE_UNIVERSE:
        price = get_live_price(ticker)
        if price:
            prices[ticker] = price
        time.sleep(0.1)
    print(f"Got live prices for {len(prices)} assets")
    return prices

def build_price_briefing(prices):
    lines = ["LIVE MARKET PRICES (use ONLY these prices -- do NOT hallucinate prices):"]
    for ticker, name in TRADEABLE_UNIVERSE.items():
        if ticker in prices:
            lines.append(f"  {ticker} ({name}): ${prices[ticker]:.2f}")
        else:
            lines.append(f"  {ticker} ({name}): price unavailable")
    return "\n".join(lines)

def load_log():
    global session_stats, trade_history, misfit_scorecard
    try:
        if os.path.exists(LOG_FILE):
            with open(LOG_FILE, "r") as f:
                data = json.load(f)
            saved = data.get("session_stats", {})
            for key in session_stats:
                if key in saved:
                    if isinstance(session_stats[key], dict):
                        session_stats[key].update(saved.get(key, {}))
                    else:
                        session_stats[key] = saved[key]
            trade_history = data.get("trade_history", [])[-200:]
            saved_sc = data.get("misfit_scorecard", {})
            for name in misfit_scorecard:
                if name in saved_sc:
                    misfit_scorecard[name].update(saved_sc[name])
            print(f"Log loaded: {session_stats['total']} total cycles, {session_stats['execute']} executions")
    except Exception as e:
        print(f"Log load error: {e}")

def save_log():
    try:
        with open(LOG_FILE, "w") as f:
            json.dump({
                "session_stats": session_stats,
                "trade_history": trade_history[-200:],
                "misfit_scorecard": misfit_scorecard,
                "last_updated": datetime.now(pytz.utc).isoformat()
            }, f, indent=2, default=str)
    except Exception as e:
        print(f"Log save error: {e}")

def calculate_bayesian_weight(name):
    scores = misfit_scorecard.get(name, {})
    total = scores.get("total", 0)
    correct = scores.get("correct", 0)
    if total < 10:
        return 1.0
    alpha = correct + 1
    beta_val = (total - correct) + 1
    posterior_mean = alpha / (alpha + beta_val)
    return round(max(0.5, min(2.5, posterior_mean / 0.5)), 2)

def detect_environment():
    env = {
        "energy_crisis": False,
        "credit_crisis": False,
        "currency_crisis": False,
        "market_crash": False
    }
    try:
        checks = [
            ("USO", "energy_crisis", 0.12, "abs"),
            ("HYG", "credit_crisis", -0.05, "neg"),
            ("SPY", "market_crash", -0.10, "neg"),
            ("UUP", "currency_crisis", 0.05, "abs")
        ]
        for ticker, key, threshold, direction in checks:
            try:
                price = yf.download(ticker, period="30d", progress=False)["Close"].squeeze()
                ret = float((price.iloc[-1] - price.iloc[0]) / price.iloc[0])
                if direction == "abs" and abs(ret) > threshold:
                    env[key] = True
                elif direction == "neg" and ret < threshold:
                    env[key] = True
            except:
                pass
    except:
        pass
    return env

def update_misfit_weights(environment):
    for name in misfit_scorecard:
        misfit_scorecard[name]["weight"] = calculate_bayesian_weight(name)
    boosts = {
        "energy_crisis": [("Andurand", 2.0)],
        "credit_crisis": [("Tepper", 2.0), ("Druckenmiller", 1.5)],
        "currency_crisis": [("Soros", 2.0)],
        "market_crash": [("PTJ", 2.0), ("Druckenmiller", 1.5)]
    }
    for env_key, names in boosts.items():
        if environment.get(env_key):
            for name, boost in names:
                current = misfit_scorecard[name]["weight"]
                misfit_scorecard[name]["weight"] = max(current, boost)

def get_fred_data(series_id):
    try:
        r = requests.get(
            "https://api.stlouisfed.org/fred/series/observations",
            params={"series_id": series_id, "api_key": FRED_API_KEY,
                    "file_type": "json", "limit": 5, "sort_order": "desc"},
            timeout=10
        )
        obs = r.json().get("observations", [])
        if obs:
            val = obs[0]["value"]
            return float(val) if val != "." else None
    except:
        pass
    return None

def get_soros_data():
    try:
        data = {}
        usdx = get_fred_data("DTWEXBGS")
        if usdx:
            data["dollar_index"] = usdx
        for pair, ticker in [("EUR_USD", "EURUSD=X"), ("GBP_USD", "GBPUSD=X"), ("TUR", "TUR"), ("EWZ", "EWZ"), ("EEM", "EEM")]:
            try:
                price = yf.download(ticker, period="30d", progress=False)["Close"].squeeze()
                data[f"{pair}_30d_pct"] = round(float((price.iloc[-1] - price.iloc[0]) / price.iloc[0] * 100), 2)
            except:
                pass
        try:
            cot = requests.get(
                "https://publicreporting.cftc.gov/resource/jun7-fc8e.json?$limit=3&$order=report_date_as_yyyy_mm_dd DESC&$where=contract_market_name=%27EURO FX%27",
                timeout=10
            ).json()
            if cot:
                net = int(cot[0].get("noncomm_positions_long_all", 0)) - int(cot[0].get("noncomm_positions_short_all", 0))
                data["euro_fx_hedge_fund_net"] = net
        except:
            pass
        return data
    except:
        return {}

def get_druckenmiller_data():
    try:
        data = {}
        hy = get_fred_data("BAMLH0A0HYM2")
        fed = get_fred_data("WALCL")
        cc = get_fred_data("DRCCLACBS")
        if hy: data["high_yield_spread"] = hy
        if fed: data["fed_balance_sheet_billions"] = round(fed / 1000, 1)
        if cc: data["credit_card_delinquency"] = cc
        for asset in ["SPY", "QQQ", "HYG", "TLT", "IWM"]:
            try:
                price = yf.download(asset, period="60d", progress=False)["Close"].squeeze()
                data[f"{asset}_20d_momentum"] = round(float((price.iloc[-1] - price.iloc[-20]) / price.iloc[-20] * 100), 2)
            except:
                pass
        try:
            cot = requests.get(
                "https://publicreporting.cftc.gov/resource/jun7-fc8e.json?$limit=3&$order=report_date_as_yyyy_mm_dd DESC&$where=contract_market_name=%27E-MINI S%26P 500%27",
                timeout=10
            ).json()
            if cot:
                net = int(cot[0].get("noncomm_positions_long_all", 0)) - int(cot[0].get("noncomm_positions_short_all", 0))
                data["sp500_hedge_fund_net"] = net
        except:
            pass
        return data
    except:
        return {}

def get_ptj_data():
    try:
        data = {}
        vix = yf.download("^VIX", period="60d", progress=False)["Close"].squeeze()
        data["vix_current"] = round(float(vix.iloc[-1]), 1)
        data["vix_30d_avg"] = round(float(vix.rolling(30).mean().iloc[-1]), 1)
        data["vix_regime"] = "ELEVATED" if float(vix.iloc[-1]) > float(vix.rolling(30).mean().iloc[-1]) else "NORMAL"
        for ticker in ["SPY", "QQQ", "IWM", "GLD"]:
            try:
                df = yf.download(ticker, period="200d", progress=False)
                close = df["Close"].squeeze()
                vol = df["Volume"].squeeze()
                sma50 = float(close.rolling(50).mean().iloc[-1])
                price = float(close.iloc[-1])
                vol_ratio = float(vol.iloc[-1] / vol.rolling(20).mean().iloc[-1])
                data[f"{ticker}_vs_sma50_pct"] = round((price - sma50) / sma50 * 100, 2)
                data[f"{ticker}_volume_ratio"] = round(vol_ratio, 2)
            except:
                pass
        return data
    except:
        return {}

def get_tepper_data():
    try:
        data = {}
        for series, label in [
            ("BAMLH0A0HYM2", "hy_spread"),
            ("BAMLC0A0CM", "ig_spread"),
            ("DRCCLACBS", "cc_delinquency")
        ]:
            val = get_fred_data(series)
            if val: data[label] = val
        for mat, ticker in [("2Y", "^IRX"), ("10Y", "^TNX"), ("30Y", "^TYX")]:
            try:
                y = yf.download(ticker, period="30d", progress=False)["Close"].squeeze()
                data[f"treasury_{mat}"] = round(float(y.iloc[-1]), 2)
                data[f"treasury_{mat}_30d_change"] = round(float(y.iloc[-1] - y.iloc[0]), 2)
            except:
                pass
        try:
            exa_results = exa.search_and_contents(
                "Federal Reserve policy credit market 2026", num_results=2,
                text={"max_characters": 300}
            )
            data["fed_intelligence"] = " | ".join([r.title for r in exa_results.results])
        except:
            pass
        return data
    except:
        return {}

def get_andurand_data():
    try:
        data = {}
        try:
            eia_url = f"https://api.eia.gov/v2/petroleum/stoc/wstk/data/?frequency=weekly&data[]=value&facets[series][]=W_EPC0_SAX_YCUOK_MBBL&sort[0][column]=period&sort[0][direction]=desc&length=4&api_key={EIA_API_KEY}"
            r = requests.get(eia_url, timeout=10).json()
            obs = r.get("response", {}).get("data", [])
            if obs and len(obs) >= 2:
                data["cushing_stocks_mbbl"] = float(obs[0].get("value", 0))
                data["cushing_draw_mbbl"] = float(obs[0].get("value", 0)) - float(obs[1].get("value", 0))
        except:
            pass
        with hormuz_lock:
            data["vessels_near_hormuz"] = len(hormuz_vessels)
        for ticker, name in [("USO", "wti"), ("BNO", "brent"), ("UNG", "natgas"), ("ERX", "energy_2x"), ("XLE", "energy_sector"), ("FRO", "tankers")]:
            try:
                price = yf.download(ticker, period="30d", progress=False)["Close"].squeeze()
                data[f"{name}_30d_pct"] = round(float((price.iloc[-1] - price.iloc[0]) / price.iloc[0] * 100), 2)
            except:
                pass
        try:
            cot = requests.get(
                "https://publicreporting.cftc.gov/resource/jun7-fc8e.json?$limit=3&$order=report_date_as_yyyy_mm_dd DESC&$where=contract_market_name=%27CRUDE OIL%27",
                timeout=10
            ).json()
            if cot:
                net = int(cot[0].get("noncomm_positions_long_all", 0)) - int(cot[0].get("noncomm_positions_short_all", 0))
                data["crude_hedge_fund_net"] = net
        except:
            pass
        try:
            exa_results = exa.search_and_contents(
                "Strait Hormuz tanker oil Iran OPEC energy 2026", num_results=3,
                text={"max_characters": 400}
            )
            data["hormuz_intelligence"] = " | ".join([r.title for r in exa_results.results])
        except:
            pass
        return data
    except:
        return {}

def generate_misfit_signal(name, task, briefing_data, knowledge, weight, live_prices):
    weight_note = f"\nYour environment weight is {weight:.1f}x. This is YOUR setup. Be aggressive." if weight > 1.0 else ""
    data_str = json.dumps(briefing_data, default=str)[:1500]
    knowledge_str = knowledge[:600] if knowledge else ""
    scorecard = misfit_scorecard.get(name, {})
    total = scorecard.get("total", 0)
    correct = scorecard.get("correct", 0)
    record_str = f"Your track record: {correct}/{total} correct." if total > 0 else "No closed trades yet."
    price_briefing = build_price_briefing(live_prices)

    msg = client.messages.create(
        model="claude-opus-4-5-20251101",
        max_tokens=300,
        messages=[{"role": "user", "content": f"""{task}{weight_note}

{price_briefing}

YOUR SPECIALIST DATA:
{data_str}

YOUR KNOWLEDGE:
{knowledge_str}

{record_str}

CRITICAL RULES:
1. You MUST use ONLY the live prices listed above. Never invent a price.
2. Set stop loss at 5% below entry for longs, 5% above for shorts.
3. Set target at 15% above entry for longs, 15% below for shorts.
4. You MUST produce a signal. The only valid reason to skip is if markets are closed and no crypto opportunity exists.

Output EXACTLY this format using ONLY the live prices above:
TICKER: [exact ticker from the list above]
DIRECTION: [BUY or SHORT]
ENTRY: [exact live price from above -- no other number]
STOP: [entry minus 5% for BUY, entry plus 5% for SHORT]
TARGET: [entry plus 15% for BUY, entry minus 15% for SHORT]
SCORE: [your conviction 1-10]
REASON: [one sentence using your specific framework and data]

No preamble. No explanation. Just the six lines above."""}]
    )
    return msg.content[0].text

def parse_misfit_signal(text, live_prices):
    try:
        lines = text.strip().split("\n")
        result = {}
        for line in lines:
            if ":" in line:
                key, val = line.split(":", 1)
                result[key.strip().upper()] = val.strip()

        ticker = result.get("TICKER", "").strip().upper()
        if not ticker or ticker not in TRADEABLE_UNIVERSE:
            for t in TRADEABLE_UNIVERSE:
                if t in text.upper():
                    ticker = t
                    break
            if not ticker:
                return None

        live_price = live_prices.get(ticker)
        if live_price is None:
            print(f"No live price for {ticker} -- skipping signal")
            return None

        direction = "buy" if "BUY" in result.get("DIRECTION", "").upper() else "sell"

        entry = live_price

        if direction == "buy":
            stop = entry * 0.95
            target = entry * 1.15
        else:
            stop = entry * 1.05
            target = entry * 0.85

        win_size = abs(target - entry)
        loss_size = abs(entry - stop)
        risk_reward = win_size / loss_size if loss_size > 0 else 3.0

        try:
            conviction = float(result.get("SCORE", "5"))
        except:
            conviction = 5.0

        reason = result.get("REASON", "Signal from specialist data")

        return {
            "ticker": ticker,
            "direction": direction,
            "entry": entry,
            "stop": stop,
            "target": target,
            "risk_reward": round(risk_reward, 2),
            "win_size": win_size,
            "loss_size": loss_size,
            "conviction": conviction,
            "reason": reason,
            "live_price_confirmed": True
        }
    except Exception as e:
        print(f"Signal parse error: {e}")
        return None

def score_signal(signal, misfit_name, environment):
    if signal is None:
        return 0.0
    rr = signal.get("risk_reward", 2.0)
    conviction = signal.get("conviction", 5.0) / 10.0
    weight = misfit_scorecard.get(misfit_name, {}).get("weight", 1.0)
    sc = misfit_scorecard.get(misfit_name, {})
    total = sc.get("total", 0)
    correct = sc.get("correct", 0)
    bayesian = (correct + 1) / (total + 2) if total > 0 else 0.5
    composite = rr * conviction * weight * bayesian
    return round(composite, 3)

def run_contest(signals_scored):
    sorted_signals = sorted(signals_scored.items(), key=lambda x: x[1][1], reverse=True)
    if not sorted_signals:
        return []
    winners = []
    used_tickers = set()
    used_directions = {}
    for name, (signal, score) in sorted_signals:
        if score < MIN_SIGNAL_SCORE:
            continue
        ticker = signal["ticker"]
        direction = signal["direction"]
        if ticker in used_tickers:
            continue
        ticker_dir = f"{ticker}_{direction}"
        if ticker_dir in used_directions:
            continue
        winners.append((name, signal, score))
        used_tickers.add(ticker)
        used_directions[ticker_dir] = True
        if len(winners) >= MAX_ORDERS_PER_CYCLE:
            break
    return winners

def kelly_size(signal, misfit_name, portfolio_value):
    sc = misfit_scorecard.get(misfit_name, {})
    total = sc.get("total", 0)
    correct = sc.get("correct", 0)
    win_prob = (correct + 1) / (total + 2)
    win_size = signal.get("win_size", 1.0)
    loss_size = signal.get("loss_size", 1.0)
    b = win_size / loss_size if loss_size > 0 else 3.0
    kelly = max(0, (win_prob * b - (1 - win_prob)) / b)
    quarter_kelly_size = portfolio_value * kelly * QUARTER_KELLY_FRACTION
    return min(quarter_kelly_size, portfolio_value * MAX_SINGLE_POSITION)

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
            print(f"Performance send: {r.status_code}")
        except Exception as e:
            print(f"Performance send error: {e}")
        time.sleep(1)

def format_trade_alert(name, signal, position_size, score):
    ticker = signal["ticker"]
    direction = signal["direction"]
    entry = signal["entry"]
    stop = signal["stop"]
    target = signal["target"]
    rr = signal["risk_reward"]
    reason = signal["reason"]
    action = "Bought" if direction == "buy" else "Sold Short"
    emoji = "🟢" if direction == "buy" else "🔴"
    return f"""{emoji} THE MISFITS JUST TRADED

{name} called it.
{action} {ticker} at ${entry:.2f} (live price confirmed)
Bet size: ${position_size:,.0f}
Stop loss: ${stop:.2f} | Target: ${target:.2f}
Risk reward: {rr:.1f} to 1 | Contest score: {score:.2f}

Why {name}: {reason}

-- Satis House Consulting"""

def format_position_report(portfolio_state, daily_pnl=None):
    portfolio_value = portfolio_state["portfolio_value"] if portfolio_state else INCEPTION_VALUE
    net_profit = portfolio_value - INCEPTION_VALUE
    net_pct = net_profit / INCEPTION_VALUE * 100
    profit_emoji = "📈" if net_profit >= 0 else "📉"
    net_line = f"{profit_emoji} Net profit since inception: {'+' if net_profit >= 0 else ''}${net_profit:,.0f} ({'+' if net_pct >= 0 else ''}{net_pct:.2f}%)"
    if not portfolio_state or not portfolio_state["positions"]:
        return f"""📊 MISFITS PORTFOLIO UPDATE

No open positions right now.
Total value: ${portfolio_value:,.0f}
{net_line}

-- Satis House Consulting"""
    lines = ["📊 MISFITS PORTFOLIO UPDATE\n"]
    lines.append(f"Total value: ${portfolio_value:,.2f}")
    if daily_pnl is not None:
        lines.append(f"Today: {'📈' if daily_pnl >= 0 else '📉'} {'+' if daily_pnl >= 0 else ''}{daily_pnl*100:.2f}%")
    lines.append("\nOpen positions:")
    for symbol, pos in portfolio_state["positions"].items():
        pnl_emoji = "✅" if pos["unrealized_pnl"] >= 0 else "⚠️"
        direction = "Long" if pos["side"] == "long" else "Short"
        lines.append(f"{pnl_emoji} {symbol} ({direction}) -- {'+' if pos['unrealized_pnl'] >= 0 else ''}{pos['unrealized_pct']:.1f}% since entry")
    lines.append(f"\n{net_line}")
    lines.append("\n-- Satis House Consulting")
    return "\n".join(lines)

def format_daily_scorecard(portfolio_state=None):
    total = session_stats["total"]
    if total == 0:
        return "📊 MISFITS SCORECARD\n\nNo cycles recorded yet.\n\n-- Satis House Consulting"
    execute = session_stats["execute"]
    passed = session_stats["pass"]
    net_line = ""
    if portfolio_state:
        pv = portfolio_state["portfolio_value"]
        net = pv - INCEPTION_VALUE
        net_pct = net / INCEPTION_VALUE * 100
        net_line = f"\n{'📈' if net >= 0 else '📉'} Net profit since inception: {'+' if net >= 0 else ''}${net:,.0f} ({'+' if net_pct >= 0 else ''}{net_pct:.2f}%)"
    lines = ["📊 MISFITS DAILY SCORECARD\n"]
    lines.append(f"Total cycles: {total}")
    lines.append(f"✅ Trades executed: {execute} ({execute/total*100:.0f}%)")
    lines.append(f"⏭ No qualifying signal: {passed} ({passed/total*100:.0f}%)")
    lines.append(f"\nMisfit contest performance:")
    for name, votes in session_stats["misfit_signals"].items():
        total_v = votes["fired"] + votes["skipped"]
        rate = votes["fired"] / total_v * 100 if total_v > 0 else 0
        weight = misfit_scorecard.get(name, {}).get("weight", 1.0)
        sc = misfit_scorecard.get(name, {})
        record = f"{sc.get('correct',0)}/{sc.get('total',0)}" if sc.get("total", 0) > 0 else "no trades yet"
        lines.append(f"  {name}: {votes['fired']}/{total_v} signals fired ({rate:.0f}%) | record: {record} | weight: {weight:.1f}x")
    lines.append(net_line)
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
    order = {"symbol": symbol, "side": side, "type": "market", "time_in_force": "day"}
    if notional:
        order["notional"] = str(round(notional, 2))
    else:
        order["qty"] = qty
    return alpaca_request("POST", "/v2/orders", order)

def verify_order_filled(order_id):
    for _ in range(6):
        time.sleep(5)
        try:
            order = alpaca_request("GET", f"/v2/orders/{order_id}")
            status = order.get("status", "unknown")
            if status in ["filled", "partially_filled"]:
                return True
            if status in ["canceled", "rejected", "expired"]:
                return False
        except:
            pass
    return False

def submit_stop_loss_atomic(symbol, qty, side, stop_price):
    try:
        result = alpaca_request("POST", "/v2/orders", {
            "symbol": symbol, "qty": qty, "side": side,
            "type": "stop", "stop_price": str(round(stop_price, 2)),
            "time_in_force": "gtc"
        })
        if not result.get("id"):
            close_position(symbol)
            send_performance(f"⚠️ SAFETY CLOSE\n\nStop loss failed on {symbol}. Position closed.\n\n-- Satis House Consulting")
            return False
        return True
    except:
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
                result = f"+${abs(pos['unrealized_pnl']):,.0f}" if pos["unrealized_pnl"] >= 0 else f"-${abs(pos['unrealized_pnl']):,.0f}"
                send_performance(f"""📅 FRIDAY RISK MANAGEMENT

Closed short in {symbol} before weekend.
Result: {result} ({pos['unrealized_pct']:+.1f}%)
Capital protected.

-- Satis House Consulting""")
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

def check_execution_rules(ticker, direction, portfolio_state, vix):
    global trades_halted_today, daily_start_value, orders_this_cycle
    if orders_this_cycle >= MAX_ORDERS_PER_CYCLE:
        return False, "Max orders per cycle"
    if trades_halted_today:
        return False, "Daily loss limit"
    if portfolio_state is None:
        return False, "Portfolio unavailable"
    portfolio_value = portfolio_state["portfolio_value"]
    if daily_start_value is not None:
        daily_pnl = (portfolio_value - daily_start_value) / daily_start_value
        if daily_pnl <= -DAILY_LOSS_LIMIT:
            trades_halted_today = True
            send_performance(f"⚡ CIRCUIT BREAKER\n\nPortfolio down {abs(daily_pnl)*100:.1f}% today.\nAll trading paused until tomorrow.\n\n-- Satis House Consulting")
            return False, "Daily loss limit triggered"
    if vix >= VIX_STOP_THRESHOLD:
        return False, f"VIX {vix:.0f} too high"
    if direction == "sell" and is_friday_short_blocked():
        return False, "Friday short rule"
    signal_key = f"{ticker}_{direction}"
    if signal_key in recent_signals and recent_signals[signal_key] < DUPLICATE_SIGNAL_BLOCKS:
        return False, "Duplicate signal"
    positions = portfolio_state["positions"]
    if ticker in positions:
        if positions[ticker]["side"] == "long" and direction == "buy":
            return False, f"Already long {ticker}"
        if positions[ticker]["side"] == "short" and direction == "sell":
            return False, f"Already short {ticker}"
    is_crypto = any(c in ticker.upper() for c in ["BTC", "ETH", "SOL"])
    is_leveraged = ticker in LEVERAGED_ETFS
    if is_leveraged and not is_market_hours():
        return False, "Leveraged ETF market hours only"
    if not is_crypto and not is_market_hours():
        return False, "Market closed"
    if is_leveraged and portfolio_state["leveraged_pct"] >= MAX_LEVERAGED_PCT:
        return False, "Leveraged cap"
    elif is_crypto and portfolio_state["crypto_pct"] >= MAX_CRYPTO_PCT:
        return False, "Crypto cap"
    elif not is_crypto and not is_leveraged and portfolio_state["equity_pct"] >= MAX_EQUITY_PCT:
        return False, "Equity cap"
    return True, "Clear"

def execute_winner(name, signal, position_size, portfolio_state, vix):
    global recent_signals, trade_history, orders_this_cycle
    ticker = signal["ticker"]
    direction = signal["direction"]
    live_price = signal["entry"]
    stop_price = signal["stop"]

    is_crypto = any(c in ticker.upper() for c in ["BTC", "ETH", "SOL"])
    market_open = is_market_hours()

    if is_crypto:
        approved, block_reason = check_execution_rules(ticker, direction, portfolio_state, vix)
        if approved:
            try:
                notional = min(position_size, 5000)
                result = alpaca_request("POST", "/v2/orders", {
                    "symbol": ticker, "notional": str(round(notional, 2)),
                    "side": direction, "type": "market", "time_in_force": "gtc"
                })
                orders_this_cycle += 1
                recent_signals[f"{ticker}_{direction}"] = 0
                trade_history.append({"ticker": ticker, "direction": direction, "entry_price": live_price, "misfit": name, "executed": True, "timestamp": datetime.now(pytz.utc).isoformat()})
                save_log()
                return f"Crypto executed: {ticker} ${notional:,.0f} at ~${live_price:.2f}"
            except Exception as e:
                return f"Crypto failed: {e}"
        return f"Blocked: {block_reason}"

    if not market_open:
        return "Market closed -- equity signal skipped"

    approved, block_reason = check_execution_rules(ticker, direction, portfolio_state, vix)
    if not approved:
        return f"Blocked: {block_reason}"

    try:
        existing = portfolio_state["positions"].get(ticker, {})
        if existing:
            ex_side = existing.get("side")
            if (ex_side == "long" and direction == "sell") or (ex_side == "short" and direction == "buy"):
                close_position(ticker)
                time.sleep(2)

        qty = max(1, int(position_size / live_price))
        order = submit_order(ticker, qty, direction)
        order_id = order.get("id", "unknown")
        filled = verify_order_filled(order_id)

        if not filled:
            return f"{ticker} order did not fill"

        orders_this_cycle += 1
        stop_side = "sell" if direction == "buy" else "buy"
        stop_ok = submit_stop_loss_atomic(ticker, qty, stop_side, stop_price)

        if stop_ok:
            recent_signals[f"{ticker}_{direction}"] = 0
            trade_history.append({"ticker": ticker, "direction": direction, "entry_price": live_price, "misfit": name, "executed": True, "timestamp": datetime.now(pytz.utc).isoformat()})
            save_log()
            return f"Executed: {direction.upper()} {qty} {ticker} at ${live_price:.2f} | Stop: ${stop_price:.2f}"
        else:
            return f"{ticker} position closed -- stop loss failed"
    except Exception as e:
        return f"Execution failed: {e}"

def update_training_loop(ticker, misfit_name, outcome_pnl_pct):
    trade_was_profitable = outcome_pnl_pct > 0
    if misfit_name in misfit_scorecard:
        misfit_scorecard[misfit_name]["total"] += 1
        if trade_was_profitable:
            misfit_scorecard[misfit_name]["correct"] += 1
    save_log()

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
                result = f"+${abs(pos['unrealized_pnl']):,.0f}" if pos["unrealized_pnl"] >= 0 else f"-${abs(pos['unrealized_pnl']):,.0f}"
                send_performance(f"""🛑 STOP LOSS TRIGGERED

Closed {symbol}
Result: {result} ({unrealized_pct:+.1f}%)
Capital protected.

-- Satis House Consulting""")
                misfit_name = ""
                for trade in reversed(trade_history):
                    if trade.get("ticker") == symbol:
                        misfit_name = trade.get("misfit", "")
                        break
                if misfit_name:
                    update_training_loop(symbol, misfit_name, unrealized_pct / 100)
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

def start_aisstream():
    def on_message(ws, message):
        try:
            data = json.loads(message)
            if data.get("MessageType") == "PositionReport":
                meta = data.get("MetaData", {})
                pos_data = data.get("Message", {}).get("PositionReport", {})
                with hormuz_lock:
                    hormuz_vessels.append({
                        "name": meta.get("ShipName", "Unknown"),
                        "lat": pos_data.get("Latitude", 0),
                        "lon": pos_data.get("Longitude", 0),
                        "speed": pos_data.get("Sog", 0),
                        "timestamp": datetime.now(pytz.utc).isoformat()
                    })
                    if len(hormuz_vessels) > 100:
                        hormuz_vessels.pop(0)
        except:
            pass

    def on_open(ws):
        ws.send(json.dumps({
            "APIKey": AISSTREAM_API_KEY,
            "MessageType": "Subscribe",
            "BoundingBoxes": [[[21.0, 55.0], [27.0, 62.0]]],
            "FilterMessageTypes": ["PositionReport"]
        }))
        print("AISStream subscribed to Hormuz")

    def on_error(ws, error):
        print(f"AISStream error: {error}")

    def on_close(ws, *args):
        time.sleep(60)
        start_aisstream()

    def run():
        ws = websocket.WebSocketApp(
            "wss://stream.aisstream.io/v0/stream",
            on_open=on_open, on_message=on_message,
            on_error=on_error, on_close=on_close
        )
        ws.run_forever()

    threading.Thread(target=run, daemon=True).start()
    print("AISStream monitoring Hormuz")

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

MISFIT_CONFIGS = [
    (
        "Soros",
        """You ARE George Soros. You see markets through the lens of reflexivity -- self-reinforcing feedback loops that eventually collapse.

Your specialty data focuses on: currency stress, sovereign debt, emerging market capital flows, central bank reserve depletion.

Find the single best trade available RIGHT NOW from the live prices above. Lead with your specialty data but trade anything where you see a reflexivity feedback loop or unsustainable narrative. Every asset class is available to you.""",
        ["George Soros reflexivity currency crisis Black Wednesday methodology",
         "Soros Fund Management macro views sovereign debt 2025 2026"]
    ),
    (
        "Druckenmiller",
        """You ARE Stanley Druckenmiller. You never think about what you can make -- only what you can lose. Your stop is sacred.

Your specialty data focuses on: credit spreads, Federal Reserve balance sheet, macro momentum, earnings revisions.

Find the single best asymmetric trade available RIGHT NOW from the live prices above. Lead with your credit and macro data but trade anything where risk reward is extraordinary. Every asset class is available to you.""",
        ["Stanley Druckenmiller concentration asymmetric macro stop loss methodology",
         "Druckenmiller macro views credit Federal Reserve 2025 2026"]
    ),
    (
        "PTJ",
        """You ARE Paul Tudor Jones. You never get out of bed for less than 5 to 1. The chart is the truth.

Your specialty data focuses on: VIX regime, volume breakouts, momentum, technical structure.

Find the single best technical setup available RIGHT NOW from the live prices above. Lead with your volatility and momentum data but trade anything showing a clean technical setup with defined risk. Every asset class is available to you.""",
        ["Paul Tudor Jones 5 to 1 risk reward Black Monday technical analysis",
         "PTJ macro views trend following technical 2025 2026"]
    ),
    (
        "Tepper",
        """You ARE David Tepper. You read the Federal Reserve before the market does. When the government decides something will not fail, you buy it.

Your specialty data focuses on: credit spreads, Treasury yields, delinquency rates, Federal Reserve policy signals.

Find the single best policy-driven trade available RIGHT NOW from the live prices above. Lead with your credit and rates data but trade anything where policy backstop creates asymmetric opportunity. Every asset class is available to you.""",
        ["David Tepper 2009 bank trade Federal Reserve policy reading methodology",
         "Tepper macro views credit Fed policy 2025 2026"]
    ),
    (
        "Andurand",
        """You ARE Pierre Andurand. Physical markets always lead paper markets. You track molecules, not narratives.

Your specialty data focuses on: Cushing storage draws, Hormuz vessel traffic, crack spreads, crude futures positioning, oil-linked currencies.

Find the single best energy or commodity trade available RIGHT NOW from the live prices above. Lead with your physical market data but also trade currencies of oil exporters and importers, shipping stocks, refiners, and any asset where physical reality diverges from paper price. Every asset class is available to you.""",
        ["Pierre Andurand physical commodity flows Hormuz oil 2008 2022 methodology",
         "Andurand Capital oil energy physical market views 2025 2026"]
    )
]

misfit_knowledge_cache = {}
misfit_data_cache = {}
knowledge_refresh_cycles = 8
cycle_count = 0

def send_startup_message():
    send_performance("""🚀 MISFITS SYSTEM ONLINE -- CONTEST MODEL

Five legendary traders. One contest every 15 minutes.

Each Misfit independently scans the full market using their own live data. Every signal uses confirmed live prices -- no hallucination possible.

The scoring system:
Risk reward × conviction × environment weight × Bayesian win rate

Top scoring signal executes. Quarter-Kelly position sizing.
Two signals can execute simultaneously if they do not conflict.

Competing against OmniscientBot on a separate account.
May the best system win.

-- Satis House Consulting""")

def run_cycle():
    global cycle_count, misfit_knowledge_cache, misfit_data_cache
    global daily_start_value, trades_halted_today, recent_signals, orders_this_cycle
    global daily_scorecard_sent

    cycle_count += 1
    orders_this_cycle = 0

    et = pytz.timezone("America/New_York")
    now_et = datetime.now(et)

    if now_et.hour == 9 and now_et.minute < 15 and not daily_scorecard_sent:
        portfolio_state = get_portfolio_state()
        send_performance(format_daily_scorecard(portfolio_state))
        daily_scorecard_sent = True

    if now_et.hour == 9 and now_et.minute >= 15:
        daily_scorecard_sent = False

    if now_et.hour == 9 and now_et.minute < 30:
        daily_start_value = None
        trades_halted_today = False
        cancel_all_orders()

    for key in list(recent_signals.keys()):
        recent_signals[key] += 1
        if recent_signals[key] > DUPLICATE_SIGNAL_BLOCKS:
            del recent_signals[key]

    environment = detect_environment()
    update_misfit_weights(environment)

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

    live_prices = get_all_live_prices()

    if cycle_count % knowledge_refresh_cycles == 1:
        print("Refreshing Misfit knowledge and data...")
        data_funcs = {
            "Soros": get_soros_data,
            "Druckenmiller": get_druckenmiller_data,
            "PTJ": get_ptj_data,
            "Tepper": get_tepper_data,
            "Andurand": get_andurand_data
        }
        for name, task, queries in MISFIT_CONFIGS:
            print(f"  Refreshing {name}...")
            blocks = []
            for q in queries:
                try:
                    results = exa.search_and_contents(q, num_results=2, text={"max_characters": 400})
                    for r in results.results:
                        blocks.append(f"{r.title}: {r.text[:300]}")
                    time.sleep(1)
                except:
                    pass
            misfit_knowledge_cache[name] = "\n\n".join(blocks)
            misfit_data_cache[name] = data_funcs[name]()
            time.sleep(2)

    market_open = is_market_hours()
    market_note = ""
    if not market_open:
        market_note = "\nMARKET CLOSED: Only generate signals for BTC/USD, ETH/USD, or SOL/USD which trade 24/7."
    if is_friday_short_blocked():
        market_note += "\nFRIDAY RULE: No SHORT signals after 2 PM ET."

    signals_scored = {}
    signal_texts = {}
    print(f"Cycle {cycle_count}: running Misfit contest...")

    for name, task, queries in MISFIT_CONFIGS:
        data = misfit_data_cache.get(name, {})
        knowledge = misfit_knowledge_cache.get(name, "")
        weight = misfit_scorecard.get(name, {}).get("weight", 1.0)
        full_task = task + market_note

        raw_text = generate_misfit_signal(name, full_task, data, knowledge, weight, live_prices)
        signal_texts[name] = raw_text
        parsed = parse_misfit_signal(raw_text, live_prices)
        score = score_signal(parsed, name, environment)
        signals_scored[name] = (parsed, score)

        fired = parsed is not None and score >= MIN_SIGNAL_SCORE
        session_stats["misfit_signals"][name]["fired" if fired else "skipped"] += 1
        print(f"  {name}: {'SIGNAL ' + parsed['ticker'] + ' score=' + str(score) if parsed else 'no signal'}")

    winners = run_contest(signals_scored)
    session_stats["total"] += 1

    telegram_brief = f"MISFITS CONTEST -- CYCLE {cycle_count}\n\n"
    for name, (signal, score) in signals_scored.items():
        weight = misfit_scorecard.get(name, {}).get("weight", 1.0)
        if signal:
            telegram_brief += f"{name.upper()} ({weight:.1f}x): {signal['ticker']} {signal['direction'].upper()} @ ${signal['entry']:.2f} score={score:.2f}\n"
        else:
            telegram_brief += f"{name.upper()} ({weight:.1f}x): no signal\n"

    if winners:
        session_stats["execute"] += 1
        save_log()
        telegram_brief += f"\nWINNERS: {', '.join([w[0] for w in winners])}\n"
        for win_name, win_signal, win_score in winners:
            portfolio_value = portfolio_state["portfolio_value"] if portfolio_state else INCEPTION_VALUE
            position_size = kelly_size(win_signal, win_name, portfolio_value)
            if vix >= VIX_REDUCE_THRESHOLD:
                position_size *= 0.5
            result = execute_winner(win_name, win_signal, position_size, portfolio_state, vix)
            perf_msg = format_trade_alert(win_name, win_signal, position_size, win_score)
            send_performance(perf_msg)
            telegram_brief += f"\nEXECUTION ({win_name}): {result}"
    else:
        session_stats["pass"] += 1
        save_log()
        telegram_brief += f"\nNo signal scored above threshold ({MIN_SIGNAL_SCORE}). No trade this cycle."

    send_telegram(telegram_brief)
    print(f"Cycle {cycle_count}: {'EXECUTED ' + str(len(winners)) + ' trades' if winners else 'NO TRADE'}")
    smart_sleep(900)

while True:
    try:
        if cycle_count == 0:
            load_log()
            start_aisstream()
            time.sleep(3)
            send_startup_message()
        run_cycle()
    except Exception as e:
        send_telegram(f"Misfits error: {e}")
        print(f"Error: {e}")
        smart_sleep(900)
