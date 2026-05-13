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

THESIS_FILES = {
    "Soros": "/tmp/soros_thesis.json",
    "Druckenmiller": "/tmp/druckenmiller_thesis.json",
    "PTJ": "/tmp/ptj_thesis.json",
    "Tepper": "/tmp/tepper_thesis.json",
    "Andurand": "/tmp/andurand_thesis.json"
}

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
STOP_PCT = 0.05
TARGET_PCT = 0.15

LEVERAGED_ETFS = {"TQQQ", "SOXL", "TECL", "FAS", "ERX", "TMF", "SPXL", "UPRO"}

TRADEABLE_UNIVERSE = [
    "SOXL", "TECL", "TQQQ", "FAS", "ERX",
    "QQQ", "SPY", "IWM", "XLF",
    "GLD", "SLV", "TLT", "HYG", "UUP",
    "USO", "BNO", "UNG", "XLE", "XOP", "FRO", "VLO",
    "EEM", "EWZ", "TUR", "FXE",
    "BTC-USD", "ETH-USD", "SOL-USD"
]

CRYPTO_TICKERS = {"BTC-USD", "ETH-USD", "SOL-USD"}
CRYPTO_ALPACA_MAP = {
    "BTC-USD": "BTC/USD",
    "ETH-USD": "ETH/USD",
    "SOL-USD": "SOL/USD"
}

live_price_cache = {}
price_cache_timestamps = {}
PRICE_CACHE_TTL = 60

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
    "execute": 0,
    "pass": 0,
    "total": 0,
    "misfit_signals": {
        "Soros": {"fired": 0, "skipped": 0},
        "Druckenmiller": {"fired": 0, "skipped": 0},
        "PTJ": {"fired": 0, "skipped": 0},
        "Tepper": {"fired": 0, "skipped": 0},
        "Andurand": {"fired": 0, "skipped": 0}
    }
}

trade_history = []


# ── LIVE PRICE ENGINE ─────────────────────────────────────────────────────────
# This is the ONLY source of prices in the entire system.
# Claude NEVER provides a price. Claude only provides ticker and direction.
# All prices are fetched from yfinance and locked before any Misfit speaks.

def get_live_price(ticker):
    now = time.time()
    if ticker in live_price_cache and (now - price_cache_timestamps.get(ticker, 0)) < PRICE_CACHE_TTL:
        return live_price_cache[ticker]
    try:
        data = yf.download(ticker, period="2d", interval="1m", progress=False, auto_adjust=True)
        if data.empty:
            data = yf.download(ticker, period="5d", interval="5m", progress=False, auto_adjust=True)
        if not data.empty:
            price = float(data["Close"].squeeze().dropna().iloc[-1])
            live_price_cache[ticker] = price
            price_cache_timestamps[ticker] = now
            return price
    except Exception as e:
        print(f"Price fetch error {ticker}: {e}")
    return None


def fetch_all_live_prices():
    prices = {}
    print("Fetching live prices for all tradeable assets...")
    for ticker in TRADEABLE_UNIVERSE:
        price = get_live_price(ticker)
        if price:
            prices[ticker] = price
        time.sleep(0.15)
    print(f"Live prices confirmed for {len(prices)}/{len(TRADEABLE_UNIVERSE)} assets")
    return prices


def build_price_table(prices):
    lines = [
        "VERIFIED LIVE PRICES -- fetched this second from market data API",
        "You MUST pick a ticker from this list. You MUST NOT invent any price.",
        ""
    ]
    for ticker in TRADEABLE_UNIVERSE:
        if ticker in prices:
            lines.append(f"  {ticker}: ${prices[ticker]:.4f}")
        else:
            lines.append(f"  {ticker}: UNAVAILABLE -- do not use")
    return "\n".join(lines)


def apply_live_price(ticker, direction, prices):
    price = prices.get(ticker)
    if price is None:
        return None
    entry = price
    if direction == "buy":
        stop = round(entry * (1 - STOP_PCT), 4)
        target = round(entry * (1 + TARGET_PCT), 4)
    else:
        stop = round(entry * (1 + STOP_PCT), 4)
        target = round(entry * (1 - TARGET_PCT), 4)
    win_size = abs(target - entry)
    loss_size = abs(entry - stop)
    risk_reward = win_size / loss_size if loss_size > 0 else 3.0
    return {
        "entry": entry,
        "stop": stop,
        "target": target,
        "win_size": win_size,
        "loss_size": loss_size,
        "risk_reward": round(risk_reward, 2)
    }


# ── THESIS PERSISTENCE ────────────────────────────────────────────────────────
# Each Misfit maintains a persistent thesis written after every cycle.
# The thesis is read back at the start of the next cycle.
# Signal flows FROM the thesis, not FROM the news headline.
# This is what separates a macro trader from a reactive trader.
# Thesis resets on container restart and rebuilds within a few cycles. Acceptable.

def read_thesis(name):
    path = THESIS_FILES.get(name)
    if not path:
        return ""
    try:
        if os.path.exists(path):
            with open(path, "r") as f:
                data = json.load(f)
            view = data.get("current_view", "")
            conviction = data.get("conviction", "")
            age = data.get("thesis_age_cycles", 0)
            if view:
                return (
                    f"YOUR PERSISTENT THESIS (built over {age} cycles -- do not abandon without new evidence):\n"
                    f"View: {view}\n"
                    f"Conviction: {conviction}/10\n\n"
                    f"Check new data against this thesis. If data confirms it, raise conviction and signal from it. "
                    f"If data contradicts it, explain why in your THESIS update. "
                    f"Do not flip your thesis on noise. Only flip on meaningful new information."
                )
    except Exception as e:
        print(f"Thesis read error {name}: {e}")
    return ""


def save_thesis(name, raw_output):
    path = THESIS_FILES.get(name)
    if not path:
        return
    try:
        thesis_line = ""
        score_line = ""
        for line in raw_output.strip().split("\n"):
            if line.upper().startswith("THESIS:"):
                thesis_line = line.split(":", 1)[1].strip()
            if line.upper().startswith("SCORE:"):
                score_line = line.split(":", 1)[1].strip()

        if not thesis_line:
            return

        existing_age = 0
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    existing = json.load(f)
                existing_age = existing.get("thesis_age_cycles", 0)
            except:
                pass

        try:
            conviction = float(score_line)
        except:
            conviction = 5.0

        with open(path, "w") as f:
            json.dump({
                "current_view": thesis_line,
                "conviction": conviction,
                "thesis_age_cycles": existing_age + 1,
                "last_updated": datetime.now(pytz.utc).isoformat()
            }, f, indent=2)
        print(f"  {name} thesis saved (age {existing_age + 1}): {thesis_line[:80]}...")
    except Exception as e:
        print(f"Thesis save error {name}: {e}")


# ── PERSISTENT LOG ────────────────────────────────────────────────────────────

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
            for name, sc in data.get("misfit_scorecard", {}).items():
                if name in misfit_scorecard:
                    misfit_scorecard[name].update(sc)
            print(f"Log loaded: {session_stats['total']} cycles, {session_stats['execute']} executions")
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


# ── ENVIRONMENT DETECTION AND WEIGHTS ────────────────────────────────────────

def detect_environment():
    env = {"energy_crisis": False, "credit_crisis": False, "currency_crisis": False, "market_crash": False}
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
    return env


def update_misfit_weights(environment):
    for name in misfit_scorecard:
        sc = misfit_scorecard[name]
        total = sc.get("total", 0)
        correct = sc.get("correct", 0)
        if total >= 10:
            posterior = (correct + 1) / (total + 2)
            sc["weight"] = round(max(0.5, min(2.5, posterior / 0.5)), 2)
        else:
            sc["weight"] = 1.0
    boosts = {
        "energy_crisis": [("Andurand", 2.0)],
        "credit_crisis": [("Tepper", 2.0), ("Druckenmiller", 1.5)],
        "currency_crisis": [("Soros", 2.0)],
        "market_crash": [("PTJ", 2.0), ("Druckenmiller", 1.5)]
    }
    for env_key, names in boosts.items():
        if environment.get(env_key):
            for name, boost in names:
                misfit_scorecard[name]["weight"] = max(misfit_scorecard[name]["weight"], boost)


# ── SPECIALIST DATA FEEDS ─────────────────────────────────────────────────────

def get_fred(series_id):
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
    data = {}
    try:
        usdx = get_fred("DTWEXBGS")
        if usdx: data["dollar_index"] = usdx
        for pair, ticker in [("EUR_USD", "EURUSD=X"), ("GBP_USD", "GBPUSD=X"), ("TUR", "TUR"), ("EWZ", "EWZ")]:
            try:
                px = yf.download(ticker, period="30d", progress=False)["Close"].squeeze()
                data[f"{pair}_30d_pct"] = round(float((px.iloc[-1] - px.iloc[0]) / px.iloc[0] * 100), 2)
            except:
                pass
        try:
            cot = requests.get(
                "https://publicreporting.cftc.gov/resource/jun7-fc8e.json"
                "?$limit=3&$order=report_date_as_yyyy_mm_dd DESC"
                "&$where=contract_market_name=%27EURO FX%27", timeout=10
            ).json()
            if cot:
                net = int(cot[0].get("noncomm_positions_long_all", 0)) - int(cot[0].get("noncomm_positions_short_all", 0))
                data["euro_fx_hedge_fund_net"] = net
        except:
            pass
    except:
        pass
    return data


def get_druckenmiller_data():
    data = {}
    try:
        for series, label in [("BAMLH0A0HYM2", "hy_spread"), ("WALCL", "fed_balance_sheet"), ("DRCCLACBS", "cc_delinquency")]:
            val = get_fred(series)
            if val: data[label] = val
        for asset in ["SPY", "QQQ", "HYG", "TLT", "IWM"]:
            try:
                px = yf.download(asset, period="60d", progress=False)["Close"].squeeze()
                data[f"{asset}_20d_mom"] = round(float((px.iloc[-1] - px.iloc[-20]) / px.iloc[-20] * 100), 2)
            except:
                pass
    except:
        pass
    return data


def get_ptj_data():
    data = {}
    try:
        vix = yf.download("^VIX", period="60d", progress=False)["Close"].squeeze()
        data["vix_current"] = round(float(vix.iloc[-1]), 1)
        data["vix_30d_avg"] = round(float(vix.rolling(30).mean().iloc[-1]), 1)
        data["vix_regime"] = "ELEVATED" if float(vix.iloc[-1]) > float(vix.rolling(30).mean().iloc[-1]) else "NORMAL"
        for ticker in ["SPY", "QQQ", "IWM"]:
            try:
                df = yf.download(ticker, period="200d", progress=False)
                close = df["Close"].squeeze()
                vol = df["Volume"].squeeze()
                sma50 = float(close.rolling(50).mean().iloc[-1])
                price = float(close.iloc[-1])
                data[f"{ticker}_vs_sma50_pct"] = round((price - sma50) / sma50 * 100, 2)
                data[f"{ticker}_volume_ratio"] = round(float(vol.iloc[-1] / vol.rolling(20).mean().iloc[-1]), 2)
            except:
                pass
    except:
        pass
    return data


def get_tepper_data():
    data = {}
    try:
        for series, label in [("BAMLH0A0HYM2", "hy_spread"), ("BAMLC0A0CM", "ig_spread"), ("DRCCLACBS", "cc_delinquency")]:
            val = get_fred(series)
            if val: data[label] = val
        for mat, ticker in [("2Y", "^IRX"), ("10Y", "^TNX"), ("30Y", "^TYX")]:
            try:
                y = yf.download(ticker, period="30d", progress=False)["Close"].squeeze()
                data[f"treasury_{mat}"] = round(float(y.iloc[-1]), 2)
                data[f"treasury_{mat}_30d_change"] = round(float(y.iloc[-1] - y.iloc[0]), 2)
            except:
                pass
        try:
            fed_intel = exa.search_and_contents("Federal Reserve policy credit market 2026", num_results=2, text={"max_characters": 300})
            data["fed_intelligence"] = " | ".join([r.title for r in fed_intel.results])
        except:
            pass
    except:
        pass
    return data


def get_andurand_data():
    data = {}
    try:
        try:
            eia_url = (f"https://api.eia.gov/v2/petroleum/stoc/wstk/data/"
                       f"?frequency=weekly&data[]=value"
                       f"&facets[series][]=W_EPC0_SAX_YCUOK_MBBL"
                       f"&sort[0][column]=period&sort[0][direction]=desc"
                       f"&length=4&api_key={EIA_API_KEY}")
            obs = requests.get(eia_url, timeout=10).json().get("response", {}).get("data", [])
            if obs and len(obs) >= 2:
                data["cushing_stocks_mbbl"] = float(obs[0].get("value", 0))
                data["cushing_draw_mbbl"] = float(obs[0].get("value", 0)) - float(obs[1].get("value", 0))
        except:
            pass
        with hormuz_lock:
            data["vessels_near_hormuz"] = len(hormuz_vessels)
        for ticker, name in [("USO", "wti"), ("BNO", "brent"), ("UNG", "natgas"), ("ERX", "energy_2x"), ("FRO", "tankers")]:
            try:
                px = yf.download(ticker, period="30d", progress=False)["Close"].squeeze()
                data[f"{name}_30d_pct"] = round(float((px.iloc[-1] - px.iloc[0]) / px.iloc[0] * 100), 2)
            except:
                pass
        try:
            cot = requests.get(
                "https://publicreporting.cftc.gov/resource/jun7-fc8e.json"
                "?$limit=3&$order=report_date_as_yyyy_mm_dd DESC"
                "&$where=contract_market_name=%27CRUDE OIL%27", timeout=10
            ).json()
            if cot:
                net = int(cot[0].get("noncomm_positions_long_all", 0)) - int(cot[0].get("noncomm_positions_short_all", 0))
                data["crude_hedge_fund_net"] = net
        except:
            pass
        try:
            hormuz_intel = exa.search_and_contents("Strait Hormuz tanker oil Iran OPEC 2026", num_results=3, text={"max_characters": 400})
            data["hormuz_intelligence"] = " | ".join([r.title for r in hormuz_intel.results])
        except:
            pass
    except:
        pass
    return data


# ── MISFIT SIGNAL GENERATION ──────────────────────────────────────────────────
# Claude outputs ONLY: TICKER, DIRECTION, SCORE, REASON, THESIS.
# Claude NEVER outputs a price. Prices come from the live price engine above.
# THESIS is written to disk after every cycle and read back before the next.
# This is how a real macro trader builds conviction over time instead of
# reacting to every headline from a blank slate.

def generate_misfit_signal(name, persona, specialist_data, knowledge, weight, price_table, market_note, thesis_context=""):
    weight_note = f"\nYour environment weight is {weight:.1f}x -- this is your moment, be aggressive." if weight > 1.0 else ""
    data_str = json.dumps(specialist_data, default=str)[:1500]
    knowledge_str = knowledge[:600] if knowledge else ""
    sc = misfit_scorecard.get(name, {})
    record = f"Your record: {sc.get('correct',0)}/{sc.get('total',0)} correct." if sc.get("total", 0) > 0 else "No closed trades yet."

    thesis_block = f"\n{thesis_context}\n" if thesis_context else ""

    prompt = f"""{persona}{weight_note}
{thesis_block}
{price_table}

YOUR SPECIALIST DATA:
{data_str}

YOUR KNOWLEDGE BASE:
{knowledge_str}

{record}

{market_note}

YOUR TASK:
If you have a persistent thesis above, check whether new data confirms, weakens, or flips it.
Do not abandon a thesis on noise. Only revise it when new evidence is meaningful.
Find the single best trade that flows from your current view of the world.
Pick ONE ticker from the verified price list above.
Pick a direction: BUY or SHORT.
Give a conviction score 1-10 based on how strongly your thesis and data support this trade.
Give one sentence of reasoning from YOUR specific framework.
Write one precise thesis sentence that will be read back to you next cycle.

CRITICAL: You output EXACTLY five lines. Nothing else. No prices. No explanation.

TICKER: [ticker exactly as shown in the price list]
DIRECTION: [BUY or SHORT]
SCORE: [1-10]
REASON: [one sentence from your specific framework and data]
THESIS: [one sentence -- your current market view, updated by what you just saw, persists to next cycle]"""

    msg = client.messages.create(
        model="claude-opus-4-5-20251101",
        max_tokens=150,
        messages=[{"role": "user", "content": prompt}]
    )
    return msg.content[0].text.strip()


def parse_signal_output(raw_text, prices):
    try:
        result = {}
        for line in raw_text.strip().split("\n"):
            if ":" in line:
                key, val = line.split(":", 1)
                result[key.strip().upper()] = val.strip()

        ticker = result.get("TICKER", "").strip().upper()
        direction_raw = result.get("DIRECTION", "").strip().upper()
        score_raw = result.get("SCORE", "5").strip()
        reason = result.get("REASON", "Specialist data signal")

        if not ticker or not direction_raw:
            return None

        if ticker not in prices:
            for t in TRADEABLE_UNIVERSE:
                if t.upper() == ticker:
                    ticker = t
                    break
            if ticker not in prices:
                print(f"Ticker {ticker} not in live prices -- signal rejected")
                return None

        direction = "buy" if "BUY" in direction_raw else "sell" if "SHORT" in direction_raw else None
        if direction is None:
            return None

        try:
            conviction = float(score_raw)
        except:
            conviction = 5.0

        price_data = apply_live_price(ticker, direction, prices)
        if price_data is None:
            return None

        return {
            "ticker": ticker,
            "direction": direction,
            "conviction": conviction,
            "reason": reason,
            "entry": price_data["entry"],
            "stop": price_data["stop"],
            "target": price_data["target"],
            "risk_reward": price_data["risk_reward"],
            "win_size": price_data["win_size"],
            "loss_size": price_data["loss_size"],
            "price_source": "live_yfinance"
        }
    except Exception as e:
        print(f"Signal parse error: {e}")
        return None


def score_signal(signal, name, environment):
    if signal is None:
        return 0.0
    rr = signal.get("risk_reward", 2.0)
    conviction = signal.get("conviction", 5.0) / 10.0
    weight = misfit_scorecard.get(name, {}).get("weight", 1.0)
    sc = misfit_scorecard.get(name, {})
    total = sc.get("total", 0)
    correct = sc.get("correct", 0)
    bayesian = (correct + 1) / (total + 2)
    return round(rr * conviction * weight * bayesian, 3)


def run_contest(signals_scored):
    sorted_signals = sorted(signals_scored.items(), key=lambda x: x[1][1], reverse=True)
    winners = []
    used_tickers = set()
    for name, (signal, score) in sorted_signals:
        if score < MIN_SIGNAL_SCORE:
            continue
        if signal is None:
            continue
        ticker = signal["ticker"]
        if ticker in used_tickers:
            continue
        winners.append((name, signal, score))
        used_tickers.add(ticker)
        if len(winners) >= MAX_ORDERS_PER_CYCLE:
            break
    return winners


def kelly_size(signal, name, portfolio_value):
    sc = misfit_scorecard.get(name, {})
    total = sc.get("total", 0)
    correct = sc.get("correct", 0)
    win_prob = (correct + 1) / (total + 2)
    b = signal["win_size"] / signal["loss_size"] if signal["loss_size"] > 0 else 3.0
    kelly = max(0, (win_prob * b - (1 - win_prob)) / b)
    size = portfolio_value * kelly * QUARTER_KELLY_FRACTION
    return min(size, portfolio_value * MAX_SINGLE_POSITION)


# ── TELEGRAM ──────────────────────────────────────────────────────────────────

def send_telegram(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    for chunk in [message[i:i+4000] for i in range(0, len(message), 4000)]:
        requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": chunk})
        time.sleep(1)


def send_performance(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_PERFORMANCE_TOKEN}/sendMessage"
    for chunk in [message[i:i+4000] for i in range(0, len(message), 4000)]:
        try:
            r = requests.post(url, json={"chat_id": TELEGRAM_PERFORMANCE_CHAT_ID, "text": chunk}, timeout=10)
            print(f"Performance: {r.status_code}")
        except Exception as e:
            print(f"Performance error: {e}")
        time.sleep(1)


def format_trade_alert(name, signal, position_size, score):
    action = "Bought" if signal["direction"] == "buy" else "Sold Short"
    emoji = "🟢" if signal["direction"] == "buy" else "🔴"
    return f"""{emoji} THE MISFITS JUST TRADED

{name} won the contest.
{action} {signal['ticker']} at ${signal['entry']:.4f}
Price source: live market data (yfinance)
Stop loss: ${signal['stop']:.4f} | Target: ${signal['target']:.4f}
Risk reward: {signal['risk_reward']:.1f} to 1
Bet size: ${position_size:,.0f} | Contest score: {score:.2f}

Why: {signal['reason']}

-- Satis House Consulting"""


def format_position_report(portfolio_state, daily_pnl=None):
    pv = portfolio_state["portfolio_value"] if portfolio_state else INCEPTION_VALUE
    net = pv - INCEPTION_VALUE
    net_pct = net / INCEPTION_VALUE * 100
    net_line = f"{'📈' if net >= 0 else '📉'} Net profit since inception: {'+' if net >= 0 else ''}${net:,.0f} ({'+' if net_pct >= 0 else ''}{net_pct:.2f}%)"
    if not portfolio_state or not portfolio_state["positions"]:
        return f"📊 MISFITS PORTFOLIO UPDATE\n\nNo open positions.\nTotal value: ${pv:,.0f}\n{net_line}\n\n-- Satis House Consulting"
    lines = ["📊 MISFITS PORTFOLIO UPDATE\n", f"Total value: ${pv:,.2f}"]
    if daily_pnl is not None:
        lines.append(f"Today: {'📈' if daily_pnl >= 0 else '📉'} {daily_pnl*100:+.2f}%")
    lines.append("\nOpen positions:")
    for symbol, pos in portfolio_state["positions"].items():
        emoji = "✅" if pos["unrealized_pnl"] >= 0 else "⚠️"
        side = "Long" if pos["side"] == "long" else "Short"
        lines.append(f"{emoji} {symbol} ({side}) -- {pos['unrealized_pct']:+.1f}% since entry")
    lines.append(f"\n{net_line}\n\n-- Satis House Consulting")
    return "\n".join(lines)


def format_daily_scorecard(portfolio_state=None):
    total = session_stats["total"]
    if total == 0:
        return "📊 MISFITS SCORECARD\n\nNo cycles yet.\n\n-- Satis House Consulting"
    execute = session_stats["execute"]
    passed = session_stats["pass"]
    net_line = ""
    if portfolio_state:
        pv = portfolio_state["portfolio_value"]
        net = pv - INCEPTION_VALUE
        net_pct = net / INCEPTION_VALUE * 100
        net_line = f"\n{'📈' if net >= 0 else '📉'} Net profit since inception: {'+' if net >= 0 else ''}${net:,.0f} ({net_pct:+.2f}%)"
    lines = [
        "📊 MISFITS DAILY SCORECARD\n",
        f"Total cycles: {total}",
        f"✅ Trades executed: {execute} ({execute/total*100:.0f}%)",
        f"⏭ No qualifying signal: {passed} ({passed/total*100:.0f}%)",
        "\nMisfit contest performance:"
    ]
    for name, votes in session_stats["misfit_signals"].items():
        total_v = votes["fired"] + votes["skipped"]
        rate = votes["fired"] / total_v * 100 if total_v > 0 else 0
        sc = misfit_scorecard.get(name, {})
        record = f"{sc.get('correct',0)}/{sc.get('total',0)}" if sc.get("total", 0) > 0 else "no trades"
        weight = sc.get("weight", 1.0)
        lines.append(f"  {name}: {votes['fired']}/{total_v} signals ({rate:.0f}%) | {record} | {weight:.1f}x")
    lines.append("\nActive theses:")
    for name in THESIS_FILES:
        path = THESIS_FILES[name]
        try:
            if os.path.exists(path):
                with open(path, "r") as f:
                    t = json.load(f)
                age = t.get("thesis_age_cycles", 0)
                view = t.get("current_view", "")[:70]
                lines.append(f"  {name} ({age} cycles): {view}...")
            else:
                lines.append(f"  {name}: no thesis yet")
        except:
            lines.append(f"  {name}: thesis unreadable")
    lines.append(net_line)
    lines.append("\n-- Satis House Consulting")
    return "\n".join(lines)


# ── ALPACA ────────────────────────────────────────────────────────────────────

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


def submit_market_order(symbol, qty, side, notional=None):
    order = {"symbol": symbol, "side": side, "type": "market", "time_in_force": "day"}
    if notional:
        order["notional"] = str(round(notional, 2))
    else:
        order["qty"] = str(qty)
    return alpaca_request("POST", "/v2/orders", order)


def submit_stop_loss(symbol, qty, side, stop_price):
    result = alpaca_request("POST", "/v2/orders", {
        "symbol": symbol, "qty": str(qty), "side": side,
        "type": "stop", "stop_price": str(round(stop_price, 4)),
        "time_in_force": "gtc"
    })
    if not result.get("id"):
        close_position(symbol)
        send_performance(f"⚠️ SAFETY CLOSE\n\nStop loss failed on {symbol}. Position closed to protect capital.\n\n-- Satis House Consulting")
        return False
    return True


def verify_fill(order_id):
    for _ in range(6):
        time.sleep(5)
        try:
            order = alpaca_request("GET", f"/v2/orders/{order_id}")
            status = order.get("status", "")
            if status in ["filled", "partially_filled"]:
                return True
            if status in ["canceled", "rejected", "expired"]:
                return False
        except:
            pass
    return False


def get_portfolio_state():
    try:
        account = get_account()
        positions = get_positions()
        pv = float(account.get("portfolio_value", 100000))
        bp = float(account.get("buying_power", 50000))
        pos_map = {}
        equity_v = crypto_v = leveraged_v = 0.0
        if isinstance(positions, list):
            for pos in positions:
                symbol = pos["symbol"]
                mv = float(pos.get("market_value", 0))
                side = pos.get("side", "long")
                qty = float(pos.get("qty", 0))
                avg = float(pos.get("avg_entry_price", 0))
                cur = float(pos.get("current_price", 0))
                pnl = float(pos.get("unrealized_pl", 0))
                pnl_pct = float(pos.get("unrealized_plpc", 0)) * 100
                pos_map[symbol] = {"side": side, "qty": qty, "market_value": mv, "avg_entry": avg, "current_price": cur, "unrealized_pnl": pnl, "unrealized_pct": pnl_pct}
                if "USD" in symbol: crypto_v += abs(mv)
                elif symbol in LEVERAGED_ETFS: leveraged_v += abs(mv)
                else: equity_v += abs(mv)
        return {"portfolio_value": pv, "buying_power": bp, "positions": pos_map,
                "equity_pct": equity_v / pv if pv > 0 else 0,
                "crypto_pct": crypto_v / pv if pv > 0 else 0,
                "leveraged_pct": leveraged_v / pv if pv > 0 else 0}
    except Exception as e:
        print(f"Portfolio state error: {e}")
        return None


# ── EXECUTION RULES ───────────────────────────────────────────────────────────

def is_market_hours():
    et = pytz.timezone("America/New_York")
    now = datetime.now(et)
    if now.weekday() >= 5: return False
    return now.replace(hour=9, minute=30, second=0, microsecond=0) <= now <= now.replace(hour=16, minute=0, second=0, microsecond=0)


def is_friday_short_blocked():
    et = pytz.timezone("America/New_York")
    now = datetime.now(et)
    return now.weekday() == 4 and now.hour >= FRIDAY_SHORT_CUTOFF_HOUR


def should_close_friday_shorts():
    et = pytz.timezone("America/New_York")
    now = datetime.now(et)
    return now.weekday() == 4 and (now.hour > FRIDAY_SHORT_CLOSE_HOUR or (now.hour == FRIDAY_SHORT_CLOSE_HOUR and now.minute >= FRIDAY_SHORT_CLOSE_MINUTE))


def get_vix():
    try:
        v = yf.download("^VIX", period="1d", interval="1m", progress=False)
        return float(v["Close"].squeeze().dropna().iloc[-1])
    except:
        return 20.0


def check_execution_rules(ticker, direction, portfolio_state, vix):
    global trades_halted_today, daily_start_value, orders_this_cycle
    if orders_this_cycle >= MAX_ORDERS_PER_CYCLE: return False, "Max orders per cycle"
    if trades_halted_today: return False, "Daily loss limit"
    if portfolio_state is None: return False, "Portfolio unavailable"
    pv = portfolio_state["portfolio_value"]
    if daily_start_value is not None:
        if (pv - daily_start_value) / daily_start_value <= -DAILY_LOSS_LIMIT:
            trades_halted_today = True
            send_performance(f"⚡ CIRCUIT BREAKER\n\nPortfolio down {abs((pv-daily_start_value)/daily_start_value)*100:.1f}% today.\nTrading paused until tomorrow.\n\n-- Satis House Consulting")
            return False, "Daily loss limit triggered"
    if vix >= VIX_STOP_THRESHOLD: return False, f"VIX {vix:.0f} too high"
    if direction == "sell" and is_friday_short_blocked(): return False, "Friday short rule"
    if f"{ticker}_{direction}" in recent_signals and recent_signals[f"{ticker}_{direction}"] < DUPLICATE_SIGNAL_BLOCKS: return False, "Duplicate signal"
    positions = portfolio_state["positions"]
    if ticker in positions:
        if positions[ticker]["side"] == "long" and direction == "buy": return False, f"Already long {ticker}"
        if positions[ticker]["side"] == "short" and direction == "sell": return False, f"Already short {ticker}"
    is_crypto = ticker in CRYPTO_TICKERS
    is_leveraged = ticker in LEVERAGED_ETFS
    if is_leveraged and not is_market_hours(): return False, "Leveraged ETF market hours only"
    if not is_crypto and not is_market_hours(): return False, "Market closed"
    if is_leveraged and portfolio_state["leveraged_pct"] >= MAX_LEVERAGED_PCT: return False, "Leveraged cap"
    elif is_crypto and portfolio_state["crypto_pct"] >= MAX_CRYPTO_PCT: return False, "Crypto cap"
    elif not is_crypto and not is_leveraged and portfolio_state["equity_pct"] >= MAX_EQUITY_PCT: return False, "Equity cap"
    return True, "Clear"


def execute_winner(name, signal, position_size, portfolio_state, vix):
    global recent_signals, trade_history, orders_this_cycle
    ticker = signal["ticker"]
    direction = signal["direction"]
    entry_price = signal["entry"]
    stop_price = signal["stop"]
    is_crypto = ticker in CRYPTO_TICKERS

    approved, block_reason = check_execution_rules(ticker, direction, portfolio_state, vix)
    if not approved:
        return f"Blocked: {block_reason}"

    try:
        if is_crypto:
            alpaca_symbol = CRYPTO_ALPACA_MAP.get(ticker, ticker)
            notional = min(position_size, 5000)
            result = submit_market_order(alpaca_symbol, None, direction, notional=notional)
            if result.get("id"):
                orders_this_cycle += 1
                recent_signals[f"{ticker}_{direction}"] = 0
                trade_history.append({"ticker": ticker, "direction": direction, "entry_price": entry_price, "misfit": name, "executed": True, "timestamp": datetime.now(pytz.utc).isoformat()})
                save_log()
                return f"Crypto executed: {alpaca_symbol} ${notional:,.0f} at ~${entry_price:.4f}"
            return f"Crypto order failed: {result.get('message', 'unknown')}"

        existing = portfolio_state["positions"].get(ticker, {})
        if existing:
            ex_side = existing.get("side")
            if (ex_side == "long" and direction == "sell") or (ex_side == "short" and direction == "buy"):
                close_position(ticker)
                time.sleep(2)

        qty = max(1, int(position_size / entry_price))
        order = submit_market_order(ticker, qty, direction)
        order_id = order.get("id")
        if not order_id:
            return f"Order rejected: {order.get('message', 'unknown')}"

        filled = verify_fill(order_id)
        if not filled:
            return f"{ticker} order did not fill"

        orders_this_cycle += 1
        stop_side = "sell" if direction == "buy" else "buy"
        stop_ok = submit_stop_loss(ticker, qty, stop_side, stop_price)
        if stop_ok:
            recent_signals[f"{ticker}_{direction}"] = 0
            trade_history.append({"ticker": ticker, "direction": direction, "entry_price": entry_price, "misfit": name, "executed": True, "timestamp": datetime.now(pytz.utc).isoformat()})
            save_log()
            return f"Executed: {direction.upper()} {qty} {ticker} at ${entry_price:.4f} | Stop: ${stop_price:.4f}"
        return f"{ticker} position closed -- stop loss submission failed"
    except Exception as e:
        return f"Execution error: {e}"


def check_stop_losses(portfolio_state):
    if portfolio_state is None: return
    for symbol, pos in portfolio_state["positions"].items():
        if pos["unrealized_pct"] <= -5:
            try:
                close_position(symbol)
                result = f"+${abs(pos['unrealized_pnl']):,.0f}" if pos["unrealized_pnl"] >= 0 else f"-${abs(pos['unrealized_pnl']):,.0f}"
                send_performance(f"🛑 STOP LOSS\n\nClosed {symbol}\nResult: {result} ({pos['unrealized_pct']:+.1f}%)\nCapital protected.\n\n-- Satis House Consulting")
                misfit_name = next((t.get("misfit", "") for t in reversed(trade_history) if t.get("ticker") == symbol), "")
                if misfit_name and misfit_name in misfit_scorecard:
                    misfit_scorecard[misfit_name]["total"] += 1
                    save_log()
            except Exception as e:
                print(f"Stop loss error {symbol}: {e}")


def close_friday_shorts(portfolio_state):
    if not portfolio_state: return
    for symbol, pos in portfolio_state["positions"].items():
        if pos["side"] == "short":
            try:
                close_position(symbol)
                result = f"+${abs(pos['unrealized_pnl']):,.0f}" if pos["unrealized_pnl"] >= 0 else f"-${abs(pos['unrealized_pnl']):,.0f}"
                send_performance(f"📅 FRIDAY CLOSE\n\nClosed short {symbol} before weekend.\nResult: {result} ({pos['unrealized_pct']:+.1f}%)\n\n-- Satis House Consulting")
            except Exception as e:
                print(f"Friday close error {symbol}: {e}")


def report_positions(portfolio_state):
    if portfolio_state is None: return
    daily_pnl = (portfolio_state["portfolio_value"] - daily_start_value) / daily_start_value if daily_start_value else None
    send_performance(format_position_report(portfolio_state, daily_pnl))


# ── AIS STREAM ────────────────────────────────────────────────────────────────

def start_aisstream():
    def on_message(ws, message):
        try:
            data = json.loads(message)
            if data.get("MessageType") == "PositionReport":
                meta = data.get("MetaData", {})
                pos_data = data.get("Message", {}).get("PositionReport", {})
                with hormuz_lock:
                    hormuz_vessels.append({"name": meta.get("ShipName", "?"), "lat": pos_data.get("Latitude", 0), "lon": pos_data.get("Longitude", 0), "speed": pos_data.get("Sog", 0)})
                    if len(hormuz_vessels) > 100: hormuz_vessels.pop(0)
        except: pass

    def on_open(ws):
        ws.send(json.dumps({"APIKey": AISSTREAM_API_KEY, "MessageType": "Subscribe", "BoundingBoxes": [[[21.0, 55.0], [27.0, 62.0]]], "FilterMessageTypes": ["PositionReport"]}))
        print("AISStream: monitoring Hormuz")

    def run():
        while True:
            try:
                ws = websocket.WebSocketApp("wss://stream.aisstream.io/v0/stream", on_open=on_open, on_message=on_message, on_error=lambda ws, e: print(f"AIS error: {e}"), on_close=lambda ws, *a: None)
                ws.run_forever()
            except:
                pass
            time.sleep(60)

    threading.Thread(target=run, daemon=True).start()


def keepalive():
    try:
        get_account()
        print(f"Keepalive {datetime.now(pytz.utc).strftime('%H:%M:%S')}")
    except: pass


def smart_sleep(seconds):
    elapsed = 0
    while elapsed < seconds:
        chunk = min(480, seconds - elapsed)
        time.sleep(chunk)
        elapsed += chunk
        if elapsed < seconds: keepalive()


# ── MISFIT PERSONAS ───────────────────────────────────────────────────────────

MISFITS = [
    ("Soros",
     """You ARE George Soros. You see reflexivity everywhere -- self-reinforcing feedback loops that eventually collapse. You broke the Bank of England. You find the lie everyone believes and bet against it when defenders run out of ammunition. Your specialist data covers currency stress, sovereign debt, emerging market capital flows, and central bank reserve positions. Trade anything where you see an unsustainable narrative or reflexivity reversal.""",
     ["George Soros reflexivity currency crisis Black Wednesday 1992",
      "Soros Fund Management macro sovereign debt views 2025 2026"]),

    ("Druckenmiller",
     """You ARE Stanley Druckenmiller. You think about what you can lose before what you can make. Your stop is sacred. You concentrate into your single best idea when conviction is high. Your specialist data covers credit spreads, Federal Reserve balance sheet, macro momentum across asset classes, and earnings revisions. Trade the highest conviction asymmetric opportunity you can find anywhere in the market.""",
     ["Stanley Druckenmiller concentration asymmetric bet stop loss methodology",
      "Druckenmiller macro credit Federal Reserve views 2025 2026"]),

    ("PTJ",
     """You ARE Paul Tudor Jones. You called Black Monday 1987. You never trade for less than 5 to 1 risk reward. The chart tells the truth before the fundamentals do. Your specialist data covers VIX regime, volume breakouts, momentum signals, and technical structure. Find the cleanest technical setup with the most defined risk in the market right now.""",
     ["Paul Tudor Jones 5 to 1 risk reward Black Monday technical analysis",
      "PTJ trend following technical momentum views 2025 2026"]),

    ("Tepper",
     """You ARE David Tepper. You made 7 billion dollars in 2009 reading the government's intent before the market did. When the Federal Reserve is on your side you buy everything. Your specialist data covers high yield spreads, Treasury yields, credit delinquency rates, and Federal Reserve policy signals. Find the trade where policy backstop creates asymmetric upside.""",
     ["David Tepper 2009 bank trade Federal Reserve policy reading",
      "Tepper credit Fed policy macro views 2025 2026"]),

    ("Andurand",
     """You ARE Pierre Andurand. Physical markets always lead paper markets. You track molecules not narratives. You called 2008 oil and 2022 Russia Ukraine energy crisis by reading physical flows before they showed in price. Your specialist data covers Cushing storage draws, Hormuz vessel traffic, crude futures positioning, crack spreads, and oil-linked currencies. Find the energy or commodity trade where physical reality diverges from paper price.""",
     ["Pierre Andurand physical commodity flows Hormuz oil 2008 2022",
      "Andurand Capital energy oil physical market views 2025 2026"])
]

misfit_knowledge = {}
misfit_data = {}
knowledge_refresh_cycles = 8
cycle_count = 0


# ── MAIN LOOP ─────────────────────────────────────────────────────────────────

def send_startup_message():
    send_performance("""🚀 MISFITS CONTEST MODEL -- LIVE

Five legendary traders. One contest every 15 minutes.
Thesis persistence layer active -- all five Misfits now build conviction across cycles.

PRICING GUARANTEE:
All prices fetched live from market data API before any Misfit speaks.
Claude outputs ONLY ticker and direction -- never a price.
Entry = live price. Stop = 5% from live. Target = 15% from live.
Zero hallucination possible by architecture.

Competing against OmniscientBot on a separate account.

-- Satis House Consulting""")


def run_cycle():
    global cycle_count, misfit_knowledge, misfit_data
    global daily_start_value, trades_halted_today, recent_signals, orders_this_cycle, daily_scorecard_sent

    cycle_count += 1
    orders_this_cycle = 0

    et = pytz.timezone("America/New_York")
    now_et = datetime.now(et)

    if now_et.hour == 9 and now_et.minute < 15 and not daily_scorecard_sent:
        send_performance(format_daily_scorecard(get_portfolio_state()))
        daily_scorecard_sent = True
    if now_et.hour >= 9 and now_et.minute >= 15:
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
        report_positions(portfolio_state)

    live_prices = fetch_all_live_prices()
    price_table = build_price_table(live_prices)

    if cycle_count % knowledge_refresh_cycles == 1:
        print("Refreshing Misfit knowledge and specialist data...")
        data_funcs = {"Soros": get_soros_data, "Druckenmiller": get_druckenmiller_data, "PTJ": get_ptj_data, "Tepper": get_tepper_data, "Andurand": get_andurand_data}
        for name, persona, queries in MISFITS:
            blocks = []
            for q in queries:
                try:
                    res = exa.search_and_contents(q, num_results=2, text={"max_characters": 400})
                    for r in res.results:
                        blocks.append(f"{r.title}: {r.text[:300]}")
                    time.sleep(1)
                except: pass
            misfit_knowledge[name] = "\n\n".join(blocks)
            misfit_data[name] = data_funcs[name]()
            time.sleep(2)

    market_open = is_market_hours()
    market_note = ""
    if not market_open:
        market_note = "MARKET CLOSED: Only pick BTC-USD, ETH-USD, or SOL-USD which trade 24/7. All other tickers are unavailable."
    if is_friday_short_blocked():
        market_note += " FRIDAY RULE: DIRECTION must be BUY only after 2 PM ET."

    env_active = [k for k, v in environment.items() if v]
    print(f"Cycle {cycle_count} | VIX {vix:.1f} | Env: {env_active or 'standard'} | Running contest...")

    signals_scored = {}
    signal_texts = {}

    for name, persona, queries in MISFITS:
        data = misfit_data.get(name, {})
        knowledge = misfit_knowledge.get(name, "")
        weight = misfit_scorecard.get(name, {}).get("weight", 1.0)

        # Read this Misfit's persistent thesis before they speak
        thesis_context = read_thesis(name)

        raw = generate_misfit_signal(name, persona, data, knowledge, weight, price_table, market_note, thesis_context)

        # Save updated thesis immediately after signal generation
        save_thesis(name, raw)

        signal_texts[name] = raw
        parsed = parse_signal_output(raw, live_prices)
        score = score_signal(parsed, name, environment)
        signals_scored[name] = (parsed, score)

        if parsed and score >= MIN_SIGNAL_SCORE:
            session_stats["misfit_signals"][name]["fired"] += 1
            print(f"  {name}: SIGNAL {parsed['ticker']} {parsed['direction'].upper()} @ ${parsed['entry']:.4f} score={score:.3f}")
        else:
            session_stats["misfit_signals"][name]["skipped"] += 1
            print(f"  {name}: score={score:.3f} -- below threshold")

    winners = run_contest(signals_scored)
    session_stats["total"] += 1

    brief = f"MISFITS CONTEST -- CYCLE {cycle_count}\nEnvironment: {', '.join(env_active) if env_active else 'standard'}\n\n"
    for name, (signal, score) in signals_scored.items():
        weight = misfit_scorecard.get(name, {}).get("weight", 1.0)
        if signal:
            brief += f"{name} ({weight:.1f}x): {signal['ticker']} {signal['direction'].upper()} score={score:.3f}\n"
        else:
            brief += f"{name} ({weight:.1f}x): no valid signal\n"

    if winners:
        session_stats["execute"] += 1
        save_log()
        brief += f"\nWINNERS:"
        for win_name, win_signal, win_score in winners:
            pv = portfolio_state["portfolio_value"] if portfolio_state else INCEPTION_VALUE
            pos_size = kelly_size(win_signal, win_name, pv)
            if vix >= VIX_REDUCE_THRESHOLD:
                pos_size *= 0.5
            result = execute_winner(win_name, win_signal, pos_size, portfolio_state, vix)
            send_performance(format_trade_alert(win_name, win_signal, pos_size, win_score))
            brief += f"\n  {win_name}: {result}"
    else:
        session_stats["pass"] += 1
        save_log()
        brief += f"\nNo signal scored above {MIN_SIGNAL_SCORE}. No trade this cycle."

    send_telegram(brief)
    print(f"Cycle {cycle_count} done: {'EXECUTED' if winners else 'NO TRADE'}")
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
