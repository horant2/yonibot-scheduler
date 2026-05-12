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
MIN_RISK_REWARD = 2.0
MIN_CONTEST_SCORE = 1.5

LEVERAGED_ETFS = {"TQQQ", "SOXL", "TECL", "FAS", "ERX", "TMF", "SPXL", "UPRO"}
ENERGY_ASSETS = {"ERX", "XLE", "USO", "BNO", "UNG", "XOP", "VLO", "FRO"}
TECH_ASSETS = {"TQQQ", "TECL", "QQQ", "SOXL", "XLK"}
FX_ASSETS = {"FXE", "FXB", "FXY", "EEM", "UUP"}
CREDIT_ASSETS = {"HYG", "LQD", "TLT", "IEF", "SHY"}
FINANCIAL_ASSETS = {"FAS", "XLF", "KRE"}

recent_signals = {}
daily_start_value = None
trades_halted_today = False
orders_this_cycle = 0
hormuz_vessels = []
hormuz_lock = threading.Lock()
daily_scorecard_sent = False

misfit_scorecard = {
    "Soros": {"correct": 0, "total": 0, "weight": 1.0, "wins": 0, "losses": 0},
    "Druckenmiller": {"correct": 0, "total": 0, "weight": 1.0, "wins": 0, "losses": 0},
    "PTJ": {"correct": 0, "total": 0, "weight": 1.0, "wins": 0, "losses": 0},
    "Tepper": {"correct": 0, "total": 0, "weight": 1.0, "wins": 0, "losses": 0},
    "Andurand": {"correct": 0, "total": 0, "weight": 1.0, "wins": 0, "losses": 0}
}

session_stats = {
    "execute": 0, "no_trade": 0, "total": 0,
    "contest_wins": {"Soros": 0, "Druckenmiller": 0, "PTJ": 0, "Tepper": 0, "Andurand": 0}
}

trade_history = []

def load_log():
    global session_stats, trade_history, misfit_scorecard
    try:
        if os.path.exists(LOG_FILE):
            with open(LOG_FILE, "r") as f:
                data = json.load(f)
            for key in session_stats:
                if key in data.get("session_stats", {}):
                    saved = data["session_stats"][key]
                    if isinstance(session_stats[key], dict):
                        session_stats[key].update(saved)
                    else:
                        session_stats[key] = saved
            trade_history = data.get("trade_history", [])[-200:]
            for name in misfit_scorecard:
                if name in data.get("misfit_scorecard", {}):
                    misfit_scorecard[name].update(data["misfit_scorecard"][name])
            print(f"Log loaded: {session_stats['total']} cycles, {session_stats['execute']} executed")
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
    sc = misfit_scorecard.get(name, {})
    total = sc.get("total", 0)
    correct = sc.get("correct", 0)
    if total < 10:
        return 1.0
    alpha = correct + 1
    beta = (total - correct) + 1
    posterior = alpha / (alpha + beta)
    return round(max(0.5, min(2.5, posterior / 0.5)), 2)

def detect_environment():
    env = {
        "energy_crisis": False,
        "credit_crisis": False,
        "currency_crisis": False,
        "market_crash": False,
        "tech_bull": False
    }
    try:
        checks = [
            ("USO", "energy_crisis", 0.12, "abs"),
            ("HYG", "credit_crisis", -0.05, "neg"),
            ("SPY", "market_crash", -0.10, "neg"),
            ("UUP", "currency_crisis", 0.05, "abs"),
            ("QQQ", "tech_bull", 0.10, "pos")
        ]
        for ticker, key, threshold, direction in checks:
            try:
                price = yf.download(ticker, period="30d", progress=False)["Close"].squeeze()
                ret = float((price.iloc[-1] - price.iloc[0]) / price.iloc[0])
                if direction == "abs" and abs(ret) > threshold:
                    env[key] = True
                elif direction == "neg" and ret < threshold:
                    env[key] = True
                elif direction == "pos" and ret > threshold:
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
        "energy_crisis": [("Andurand", 2.5)],
        "credit_crisis": [("Tepper", 2.0), ("Druckenmiller", 1.5)],
        "currency_crisis": [("Soros", 2.5)],
        "market_crash": [("PTJ", 2.5), ("Druckenmiller", 1.5)],
        "tech_bull": [("Druckenmiller", 1.5)]
    }
    for env_key, boost_list in boosts.items():
        if environment.get(env_key):
            for name, boost in boost_list:
                current = misfit_scorecard[name]["weight"]
                misfit_scorecard[name]["weight"] = max(current, boost)

def get_fred_data(series_id):
    try:
        r = requests.get(
            "https://api.stlouisfed.org/fred/series/observations",
            params={"series_id": series_id, "api_key": FRED_API_KEY, "file_type": "json", "limit": 5, "sort_order": "desc"},
            timeout=10
        )
        obs = r.json().get("observations", [])
        if obs:
            val = obs[0]["value"]
            return float(val) if val != "." else None
    except:
        pass
    return None

def get_market_context():
    context = {}
    try:
        for ticker in ["SPY", "QQQ", "GLD", "TLT", "HYG", "USO", "EEM", "UUP"]:
            try:
                price = yf.download(ticker, period="60d", progress=False)["Close"].squeeze()
                context[f"{ticker}_price"] = round(float(price.iloc[-1]), 2)
                context[f"{ticker}_5d_pct"] = round(float((price.iloc[-1] - price.iloc[-5]) / price.iloc[-5] * 100), 2)
                context[f"{ticker}_20d_pct"] = round(float((price.iloc[-1] - price.iloc[-20]) / price.iloc[-20] * 100), 2)
                sma20 = float(price.rolling(20).mean().iloc[-1])
                context[f"{ticker}_vs_sma20"] = round((float(price.iloc[-1]) - sma20) / sma20 * 100, 2)
            except:
                pass
        vix = yf.download("^VIX", period="30d", progress=False)["Close"].squeeze()
        context["vix"] = round(float(vix.iloc[-1]), 1)
        context["vix_30d_avg"] = round(float(vix.rolling(30).mean().iloc[-1]), 1)
        t2 = get_fred_data("DGS2")
        t10 = get_fred_data("DGS10")
        if t2 and t10:
            context["yield_2y"] = t2
            context["yield_10y"] = t10
            context["yield_spread"] = round(t10 - t2, 2)
        hy = get_fred_data("BAMLH0A0HYM2")
        if hy:
            context["hy_spread"] = hy
    except Exception as e:
        print(f"Market context error: {e}")
    return context

def get_soros_data():
    data = {}
    try:
        for pair, ticker in [("EUR_USD", "EURUSD=X"), ("GBP_USD", "GBPUSD=X"), ("JPY_USD", "JPY=X"), ("TRY_USD", "TRY=X"), ("BRL_USD", "BRL=X")]:
            try:
                price = yf.download(ticker, period="60d", progress=False)["Close"].squeeze()
                data[f"{pair}_5d"] = round(float((price.iloc[-1] - price.iloc[-5]) / price.iloc[-5] * 100), 2)
                data[f"{pair}_30d"] = round(float((price.iloc[-1] - price.iloc[-30]) / price.iloc[-30] * 100), 2)
                data[f"{pair}_price"] = round(float(price.iloc[-1]), 4)
            except:
                pass
        usdx = get_fred_data("DTWEXBGS")
        if usdx:
            data["dollar_index"] = usdx
        for etf in ["EEM", "TUR", "EWZ", "EWY", "INDA"]:
            try:
                price = yf.download(etf, period="30d", progress=False)["Close"].squeeze()
                data[f"{etf}_30d"] = round(float((price.iloc[-1] - price.iloc[0]) / price.iloc[0] * 100), 2)
            except:
                pass
        try:
            cot = requests.get("https://publicreporting.cftc.gov/resource/jun7-fc8e.json?$limit=3&$order=report_date_as_yyyy_mm_dd DESC&$where=contract_market_name=%27EURO FX%27", timeout=10).json()
            if cot:
                net = int(cot[0].get("noncomm_positions_long_all", 0)) - int(cot[0].get("noncomm_positions_short_all", 0))
                data["euro_fx_hedge_fund_net"] = net
        except:
            pass
        try:
            exa_results = exa.search_and_contents("currency crisis capital flight emerging market central bank intervention 2026", num_results=3, text={"max_characters": 400})
            data["intelligence"] = " | ".join([f"{r.title}: {r.text[:200]}" for r in exa_results.results])
        except:
            pass
    except Exception as e:
        print(f"Soros data error: {e}")
    return data

def get_druckenmiller_data():
    data = {}
    try:
        hy = get_fred_data("BAMLH0A0HYM2")
        ig = get_fred_data("BAMLC0A0CM")
        fed = get_fred_data("WALCL")
        cc = get_fred_data("DRCCLACBS")
        if hy: data["high_yield_spread"] = hy
        if ig: data["ig_spread"] = ig
        if fed: data["fed_balance_sheet_B"] = round(fed / 1000, 1)
        if cc: data["credit_card_delinquency"] = cc
        for asset in ["SPY", "QQQ", "IWM", "HYG", "TLT", "GLD", "DXY"]:
            ticker = "DX-Y.NYB" if asset == "DXY" else asset
            try:
                price = yf.download(ticker, period="60d", progress=False)["Close"].squeeze()
                data[f"{asset}_20d_mom"] = round(float((price.iloc[-1] - price.iloc[-20]) / price.iloc[-20] * 100), 2)
                data[f"{asset}_price"] = round(float(price.iloc[-1]), 2)
            except:
                pass
        try:
            cot = requests.get("https://publicreporting.cftc.gov/resource/jun7-fc8e.json?$limit=3&$order=report_date_as_yyyy_mm_dd DESC&$where=contract_market_name=%27E-MINI S%26P 500%27", timeout=10).json()
            if cot:
                net = int(cot[0].get("noncomm_positions_long_all", 0)) - int(cot[0].get("noncomm_positions_short_all", 0))
                data["sp500_hedge_fund_net"] = net
        except:
            pass
        try:
            exa_results = exa.search_and_contents("macro asymmetric trade credit cycle Federal Reserve balance sheet 2026", num_results=3, text={"max_characters": 400})
            data["intelligence"] = " | ".join([f"{r.title}: {r.text[:200]}" for r in exa_results.results])
        except:
            pass
    except Exception as e:
        print(f"Druckenmiller data error: {e}")
    return data

def get_ptj_data():
    data = {}
    try:
        vix = yf.download("^VIX", period="60d", progress=False)["Close"].squeeze()
        data["vix"] = round(float(vix.iloc[-1]), 1)
        data["vix_30d_avg"] = round(float(vix.rolling(30).mean().iloc[-1]), 1)
        data["vix_regime"] = "ELEVATED" if float(vix.iloc[-1]) > float(vix.rolling(30).mean().iloc[-1]) else "NORMAL"
        for ticker in ["SPY", "QQQ", "IWM", "GLD", "TLT", "USO"]:
            try:
                df = yf.download(ticker, period="200d", progress=False)
                close = df["Close"].squeeze()
                vol = df["Volume"].squeeze()
                price = float(close.iloc[-1])
                sma50 = float(close.rolling(50).mean().iloc[-1])
                sma200 = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else sma50
                vol_ratio = float(vol.iloc[-1] / vol.rolling(20).mean().iloc[-1])
                high_20 = float(close.rolling(20).max().iloc[-1])
                low_20 = float(close.rolling(20).min().iloc[-1])
                data[f"{ticker}_price"] = round(price, 2)
                data[f"{ticker}_vs_sma50"] = round((price - sma50) / sma50 * 100, 2)
                data[f"{ticker}_golden_cross"] = price > sma50 > sma200
                data[f"{ticker}_vol_ratio"] = round(vol_ratio, 2)
                data[f"{ticker}_pct_of_20d_range"] = round((price - low_20) / (high_20 - low_20) * 100, 1) if high_20 > low_20 else 50
            except:
                pass
        try:
            exa_results = exa.search_and_contents("technical breakout momentum setup options flow dark pool 2026", num_results=3, text={"max_characters": 400})
            data["intelligence"] = " | ".join([f"{r.title}: {r.text[:200]}" for r in exa_results.results])
        except:
            pass
    except Exception as e:
        print(f"PTJ data error: {e}")
    return data

def get_tepper_data():
    data = {}
    try:
        for series, label in [
            ("BAMLH0A0HYM2", "hy_spread"),
            ("BAMLC0A0CM", "ig_spread"),
            ("DRCCLACBS", "cc_delinquency"),
            ("DRSFRMACBS", "mortgage_delinquency"),
            ("WALCL", "fed_balance_sheet")
        ]:
            val = get_fred_data(series)
            if val:
                data[label] = val
        for mat, ticker in [("2Y", "^IRX"), ("10Y", "^TNX"), ("30Y", "^TYX")]:
            try:
                y = yf.download(ticker, period="60d", progress=False)["Close"].squeeze()
                data[f"treasury_{mat}"] = round(float(y.iloc[-1]), 2)
                data[f"treasury_{mat}_30d_change"] = round(float(y.iloc[-1] - y.iloc[-30]), 2)
            except:
                pass
        for etf in ["HYG", "LQD", "TLT", "XLF", "KRE"]:
            try:
                price = yf.download(etf, period="30d", progress=False)["Close"].squeeze()
                data[f"{etf}_30d"] = round(float((price.iloc[-1] - price.iloc[0]) / price.iloc[0] * 100), 2)
            except:
                pass
        try:
            cot = requests.get("https://publicreporting.cftc.gov/resource/jun7-fc8e.json?$limit=3&$order=report_date_as_yyyy_mm_dd DESC&$where=contract_market_name=%2710-YEAR T-NOTES%27", timeout=10).json()
            if cot:
                net = int(cot[0].get("noncomm_positions_long_all", 0)) - int(cot[0].get("noncomm_positions_short_all", 0))
                data["treasury_hedge_fund_net"] = net
        except:
            pass
        try:
            exa_results = exa.search_and_contents("Federal Reserve policy credit market high yield sovereign debt 2026", num_results=3, text={"max_characters": 400})
            data["intelligence"] = " | ".join([f"{r.title}: {r.text[:200]}" for r in exa_results.results])
        except:
            pass
    except Exception as e:
        print(f"Tepper data error: {e}")
    return data

def get_andurand_data():
    data = {}
    try:
        try:
            eia_url = f"https://api.eia.gov/v2/petroleum/stoc/wstk/data/?frequency=weekly&data[]=value&facets[series][]=W_EPC0_SAX_YCUOK_MBBL&sort[0][column]=period&sort[0][direction]=desc&length=4&api_key={EIA_API_KEY}"
            r = requests.get(eia_url, timeout=10).json()
            obs = r.get("response", {}).get("data", [])
            if obs and len(obs) >= 2:
                data["cushing_mbbl"] = float(obs[0].get("value", 0))
                data["cushing_draw_mbbl"] = float(obs[0].get("value", 0)) - float(obs[1].get("value", 0))
        except:
            pass
        with hormuz_lock:
            data["vessels_near_hormuz"] = len(hormuz_vessels)
            if hormuz_vessels:
                speeds = [v.get("speed", 0) for v in hormuz_vessels[-10:]]
                data["avg_vessel_speed"] = round(sum(speeds) / len(speeds), 1) if speeds else 0
        for ticker, name in [
            ("USO", "wti"), ("BNO", "brent"), ("UNG", "natgas"),
            ("ERX", "energy_2x"), ("XLE", "energy_sector"),
            ("FRO", "tankers"), ("VLO", "refiners"), ("XOP", "exploration")
        ]:
            try:
                price = yf.download(ticker, period="60d", progress=False)["Close"].squeeze()
                data[f"{name}_price"] = round(float(price.iloc[-1]), 2)
                data[f"{name}_5d"] = round(float((price.iloc[-1] - price.iloc[-5]) / price.iloc[-5] * 100), 2)
                data[f"{name}_30d"] = round(float((price.iloc[-1] - price.iloc[-30]) / price.iloc[-30] * 100), 2)
            except:
                pass
        try:
            cot = requests.get("https://publicreporting.cftc.gov/resource/jun7-fc8e.json?$limit=3&$order=report_date_as_yyyy_mm_dd DESC&$where=contract_market_name=%27CRUDE OIL%27", timeout=10).json()
            if cot:
                net = int(cot[0].get("noncomm_positions_long_all", 0)) - int(cot[0].get("noncomm_positions_short_all", 0))
                data["crude_hedge_fund_net"] = net
        except:
            pass
        for currency, ticker in [("NOK", "NOK=X"), ("CAD", "CAD=X"), ("RUB", "RUB=X")]:
            try:
                fx = yf.download(ticker, period="30d", progress=False)["Close"].squeeze()
                data[f"{currency}_30d"] = round(float((fx.iloc[-1] - fx.iloc[0]) / fx.iloc[0] * 100), 2)
            except:
                pass
        try:
            exa_results = exa.search_and_contents("Strait Hormuz tanker oil OPEC Iran supply disruption physical energy 2026", num_results=4, text={"max_characters": 500})
            data["intelligence"] = " | ".join([f"{r.title}: {r.text[:250]}" for r in exa_results.results])
        except:
            pass
    except Exception as e:
        print(f"Andurand data error: {e}")
    return data

MISFIT_TASKS = {
    "Soros": {
        "persona": """You ARE George Soros. You built your fortune finding lies that markets believe and timing when the defenders run out of ammunition.

Your specialty: currency crises, sovereign stress, reflexivity loops, capital flight, central bank exhaustion.
Your edge: you see unsustainable positions before anyone else.
Your asset universe: ALL assets, but you lead with FX, EM, sovereign bonds.

You MUST find the best trade available right now. If your specialty shows nothing extraordinary, find the best trade anywhere through your reflexivity lens. You never sit idle.""",
        "data_func": get_soros_data,
        "knowledge_queries": [
            "George Soros reflexivity theory currency crisis methodology 2025 2026",
            "emerging market stress capital flight sovereign debt current"
        ]
    },
    "Druckenmiller": {
        "persona": """You ARE Stanley Druckenmiller. You concentrate into your single best idea and size it to matter.

Your specialty: macro credit cycle, Fed policy, asymmetric risk reward, momentum across asset classes.
Your edge: you read credit markets before equity markets move.
Your asset universe: ALL assets. You go wherever the asymmetric opportunity lives.

You MUST find the best trade available right now. State your stop first. If risk reward is not 3:1 or better, find one that is. You never have no opinion.""",
        "data_func": get_druckenmiller_data,
        "knowledge_queries": [
            "Druckenmiller macro concentration asymmetric bet credit cycle 2025 2026",
            "credit spread equity divergence macro opportunity current"
        ]
    },
    "PTJ": {
        "persona": """You ARE Paul Tudor Jones. You never trade without a 5 to 1 risk reward and you read the tape better than anyone.

Your specialty: technical setups, volume breakouts, VIX regime changes, trend following.
Your edge: you see the chart pattern forming before it breaks.
Your asset universe: ALL assets. You follow price and volume wherever they lead.

You MUST find the best trade available right now. Show the entry, the stop below a real technical level, and the target. If one market has no setup, find one that does.""",
        "data_func": get_ptj_data,
        "knowledge_queries": [
            "Paul Tudor Jones technical breakout tape reading momentum 2025 2026",
            "volume breakout technical setup high probability trade current market"
        ]
    },
    "Tepper": {
        "persona": """You ARE David Tepper. You made billions reading government policy before anyone else and betting on what governments cannot afford to let fail.

Your specialty: Federal Reserve policy, credit markets, sovereign support, policy-driven asset pricing.
Your edge: you know when the Fed is your friend before the market does.
Your asset universe: ALL assets. When the Fed pivots everything re-prices. You find what reprices most.

You MUST find the best trade available right now. If no Fed pivot trade exists, find the best credit or rate trade. You always have a view.""",
        "data_func": get_tepper_data,
        "knowledge_queries": [
            "Tepper Federal Reserve policy credit market opportunity 2025 2026",
            "high yield credit spread Treasury yield trade setup current"
        ]
    },
    "Andurand": {
        "persona": """You ARE Pierre Andurand. You track physical commodity flows before they show up in financial prices.

Your specialty: crude oil, natural gas, energy supply chains, tanker markets, refinery margins, geopolitical chokepoints.
Your edge: physical markets lead paper markets always. You read the barrels before the futures.
Your asset universe: ALL assets, but physical energy intelligence is your primary edge. Energy leads everything when supply is disrupted.

You MUST find the best trade available right now. Hormuz is under threat. EIA data is live. CFTC positioning is in your data. The physical market is speaking. Listen to it and find the trade.""",
        "data_func": get_andurand_data,
        "knowledge_queries": [
            "Andurand physical energy commodity flow Hormuz trade methodology 2025 2026",
            "crude oil supply disruption tanker physical market trade current"
        ]
    }
}

def generate_misfit_signal(name, market_context, specific_data, knowledge, weight, market_open, friday_blocked):
    persona = MISFIT_TASKS[name]["persona"]
    weight_note = f"\n\nYour current weight is {weight:.1f}x -- the market environment matches your career defining setup. This is your moment. Size it accordingly." if weight > 1.5 else ""
    market_str = json.dumps(market_context, default=str)[:800]
    data_str = json.dumps(specific_data, default=str)[:1500]
    knowledge_str = knowledge[:600] if knowledge else ""

    restrictions = []
    if not market_open:
        restrictions.append("Equity markets are CLOSED. Only crypto trades execute right now (BTC, ETH, SOL). Generate a crypto signal OR the best equity signal for when markets open.")
    if friday_blocked:
        restrictions.append("FRIDAY RULE: No short positions after 2 PM ET today.")

    restriction_text = "\n".join(restrictions) if restrictions else ""

    prompt = f"""{persona}{weight_note}

BROAD MARKET CONTEXT:
{market_str}

YOUR SPECIFIC DATA:
{data_str}

YOUR KNOWLEDGE BASE:
{knowledge_str}

{restriction_text}

YOUR TASK: You are in a contest against four other legendary traders. The highest scoring signal executes. Generate your BEST trade right now.

Rules:
- You MUST produce a specific trade. Every cycle you compete. Never abstain.
- Include a precise stop loss at a real technical or fundamental level
- Include a realistic target based on your framework
- Minimum 2:1 risk reward or do not submit that trade -- find one with better odds
- Any asset class: equities, ETFs, crypto, FX, bonds, commodities

Output EXACTLY this format and nothing else:
SIGNAL: [ticker symbol]
DIRECTION: [BUY or SHORT]
ENTRY: [specific price]
STOP: [specific price]  
TARGET: [specific price]
CONFIDENCE: [HIGH or MEDIUM]
REASON: [one specific sentence citing your data]"""

    msg = client.messages.create(
        model="claude-opus-4-5-20251101",
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}]
    )
    return msg.content[0].text

def parse_signal(text, name):
    try:
        lines = {}
        for line in text.strip().split("\n"):
            if ":" in line:
                key, val = line.split(":", 1)
                lines[key.strip().upper()] = val.strip()

        ticker = lines.get("SIGNAL", "").upper().strip()
        direction_raw = lines.get("DIRECTION", "BUY").upper()
        direction = "buy" if "BUY" in direction_raw else "sell"

        def parse_price(key):
            val = lines.get(key, "0").replace("$", "").replace(",", "").strip()
            try:
                return float(val)
            except:
                return 0.0

        entry = parse_price("ENTRY")
        stop = parse_price("STOP")
        target = parse_price("TARGET")
        confidence = lines.get("CONFIDENCE", "MEDIUM").upper()
        reason = lines.get("REASON", "Signal from framework analysis")

        if not ticker or entry <= 0 or stop <= 0 or target <= 0:
            return None

        if direction == "buy":
            win_size = target - entry
            loss_size = entry - stop
        else:
            win_size = entry - target
            loss_size = stop - entry

        if loss_size <= 0 or win_size <= 0:
            return None

        risk_reward = win_size / loss_size
        if risk_reward < MIN_RISK_REWARD:
            return None

        weight = misfit_scorecard.get(name, {}).get("weight", 1.0)
        confidence_multiplier = 1.5 if confidence == "HIGH" else 1.0
        bayesian = calculate_bayesian_weight(name)
        contest_score = risk_reward * weight * confidence_multiplier * bayesian

        return {
            "name": name,
            "ticker": ticker,
            "direction": direction,
            "entry": entry,
            "stop": stop,
            "target": target,
            "confidence": confidence,
            "reason": reason,
            "risk_reward": round(risk_reward, 2),
            "win_size": win_size,
            "loss_size": loss_size,
            "weight": weight,
            "contest_score": round(contest_score, 3)
        }
    except Exception as e:
        print(f"Parse error for {name}: {e}")
        return None

def run_contest(signals):
    valid = [s for s in signals if s is not None and s["contest_score"] >= MIN_CONTEST_SCORE]
    if not valid:
        return None, []

    valid.sort(key=lambda x: x["contest_score"], reverse=True)
    winner = valid[0]

    co_signals = []
    for s in valid[1:]:
        if s["ticker"] == winner["ticker"] and s["direction"] == winner["direction"]:
            co_signals.append(s)
        theme_match = False
        if winner["ticker"] in ENERGY_ASSETS and s["ticker"] in ENERGY_ASSETS and s["direction"] == winner["direction"]:
            theme_match = True
        if winner["ticker"] in TECH_ASSETS and s["ticker"] in TECH_ASSETS and s["direction"] == winner["direction"]:
            theme_match = True
        if winner["ticker"] in FX_ASSETS and s["ticker"] in FX_ASSETS and s["direction"] == winner["direction"]:
            theme_match = True
        if theme_match and s not in co_signals:
            co_signals.append(s)

    return winner, co_signals

def calculate_position_size(signal, co_signals, portfolio_value, vix):
    win_probability = 0.45
    sc = misfit_scorecard.get(signal["name"], {})
    total = sc.get("total", 0)
    correct = sc.get("correct", 0)
    if total >= 5:
        win_probability = (correct + 1) / (total + 2)

    b = signal["win_size"] / signal["loss_size"]
    p = win_probability
    q = 1 - p
    kelly = max(0, (p * b - q) / b)

    size_multiplier = 0.25
    if len(co_signals) >= 1:
        size_multiplier = 0.35
    if signal["confidence"] == "HIGH":
        size_multiplier *= 1.2
    if signal["weight"] > 1.5:
        size_multiplier *= 1.3

    quarter_kelly = kelly * size_multiplier
    position_size = min(portfolio_value * quarter_kelly, portfolio_value * MAX_SINGLE_POSITION)
    position_size = max(position_size, portfolio_value * 0.04)

    if vix >= VIX_REDUCE_THRESHOLD:
        position_size *= 0.5

    sizing_note = f"Kelly:{kelly*100:.1f}% x {size_multiplier:.2f} = {quarter_kelly*100:.1f}% | Win prob:{win_probability*100:.0f}% | RR:{signal['risk_reward']:.1f}:1"
    if co_signals:
        sizing_note += f" | {len(co_signals)} co-signal(s) boosted size"

    return round(position_size, 2), sizing_note

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
            print(f"Performance: {r.status_code}")
        except Exception as e:
            print(f"Performance error: {e}")
        time.sleep(1)

def format_trade_performance(signal, co_signals, position_size, sizing_note, result):
    action = "Bought" if signal["direction"] == "buy" else "Sold Short"
    emoji = "🟢" if signal["direction"] == "buy" else "🔴"
    co_text = f"\nCo-signals: {', '.join([s['name'] for s in co_signals])}" if co_signals else ""
    return f"""{emoji} THE MISFITS JUST TRADED

Winner: {signal['name']} (score: {signal['contest_score']:.2f})
{action} {signal['ticker']}
Entry: ~${signal['entry']:.2f} | Stop: ${signal['stop']:.2f} | Target: ${signal['target']:.2f}
Risk reward: {signal['risk_reward']:.1f} to 1
Position: ${position_size:,.0f}{co_text}

Why: {signal['reason']}

{sizing_note}

-- Satis House Consulting"""

def format_position_report(portfolio_state, daily_pnl=None):
    portfolio_value = portfolio_state["portfolio_value"] if portfolio_state else INCEPTION_VALUE
    net_profit = portfolio_value - INCEPTION_VALUE
    net_pct = net_profit / INCEPTION_VALUE * 100
    profit_emoji = "📈" if net_profit >= 0 else "📉"
    net_line = f"{profit_emoji} Net profit since inception: {'+' if net_profit >= 0 else ''}${net_profit:,.0f} ({'+' if net_pct >= 0 else ''}{net_pct:.2f}%)"
    if not portfolio_state or not portfolio_state["positions"]:
        return f"""📊 MISFITS PORTFOLIO UPDATE

No open positions.
Total value: ${portfolio_value:,.0f}
{net_line}

-- Satis House Consulting"""
    lines = ["📊 MISFITS PORTFOLIO UPDATE\n"]
    lines.append(f"Total value: ${portfolio_value:,.2f}")
    if daily_pnl is not None:
        lines.append(f"Today: {'📈' if daily_pnl >= 0 else '📉'} {daily_pnl*100:+.2f}%")
    lines.append("\nOpen positions:")
    for symbol, pos in portfolio_state["positions"].items():
        pnl_emoji = "✅" if pos["unrealized_pnl"] >= 0 else "⚠️"
        direction = "Long" if pos["side"] == "long" else "Short"
        lines.append(f"{pnl_emoji} {symbol} ({direction}) -- {pos['unrealized_pct']:+.1f}% since entry")
    lines.append(f"\n{net_line}")
    lines.append("\n-- Satis House Consulting")
    return "\n".join(lines)

def format_daily_scorecard(portfolio_state=None):
    total = session_stats["total"]
    execute = session_stats["execute"]
    no_trade = session_stats["no_trade"]
    lines = ["📊 MISFITS DAILY SCORECARD\n"]
    lines.append(f"Total cycles: {total}")
    if total > 0:
        lines.append(f"✅ Trades executed: {execute} ({execute/total*100:.0f}%)")
        lines.append(f"⏭ No qualifying signal: {no_trade} ({no_trade/total*100:.0f}%)")
    lines.append(f"\nContest wins:")
    for name, wins in session_stats["contest_wins"].items():
        sc = misfit_scorecard.get(name, {})
        record = f"{sc.get('correct',0)}/{sc.get('total',0)}" if sc.get("total", 0) > 0 else "no closed trades"
        weight = sc.get("weight", 1.0)
        lines.append(f"  {name}: {wins} wins | record: {record} | weight: {weight:.1f}x")
    if portfolio_state:
        pv = portfolio_state["portfolio_value"]
        net = pv - INCEPTION_VALUE
        net_pct = net / INCEPTION_VALUE * 100
        lines.append(f"\n{'📈' if net >= 0 else '📉'} Net profit since inception: {'+' if net >= 0 else ''}${net:,.0f} ({net_pct:+.2f}%)")
    lines.append("\n-- Satis House Consulting")
    return "\n".join(lines)

def alpaca_request(method, endpoint, data=None, params=None):
    headers = {"APCA-API-KEY-ID": ALPACA_API_KEY, "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY, "Content-Type": "application/json"}
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
        result = alpaca_request("POST", "/v2/orders", {"symbol": symbol, "qty": qty, "side": side, "type": "stop", "stop_price": str(round(stop_price, 2)), "time_in_force": "gtc"})
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
    return now.weekday() == 4 and (now.hour > FRIDAY_SHORT_CLOSE_HOUR or (now.hour == FRIDAY_SHORT_CLOSE_HOUR and now.minute >= FRIDAY_SHORT_CLOSE_MINUTE))

def close_friday_shorts(portfolio_state):
    if not portfolio_state or not portfolio_state["positions"]:
        return
    for symbol, pos in portfolio_state["positions"].items():
        if pos["side"] == "short":
            try:
                close_position(symbol)
                result = f"+${abs(pos['unrealized_pnl']):,.0f}" if pos["unrealized_pnl"] >= 0 else f"-${abs(pos['unrealized_pnl']):,.0f}"
                send_performance(f"""📅 FRIDAY RISK MANAGEMENT

Closed short {symbol} before weekend.
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
                position_map[symbol] = {"side": side, "qty": qty, "market_value": market_val, "avg_entry": avg_entry, "current_price": current_price, "unrealized_pnl": unrealized_pnl, "unrealized_pct": unrealized_pct, "pct_of_portfolio": abs(market_val) / portfolio_value if portfolio_value > 0 else 0}
                if "USD" in symbol:
                    crypto_value += abs(market_val)
                elif symbol in LEVERAGED_ETFS:
                    leveraged_value += abs(market_val)
                else:
                    equity_value += abs(market_val)
        return {"portfolio_value": portfolio_value, "buying_power": buying_power, "positions": position_map, "equity_pct": equity_value / portfolio_value if portfolio_value > 0 else 0, "crypto_pct": crypto_value / portfolio_value if portfolio_value > 0 else 0, "leveraged_pct": leveraged_value / portfolio_value if portfolio_value > 0 else 0}
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
            send_performance(f"⚡ CIRCUIT BREAKER\n\nPortfolio down {abs(daily_pnl)*100:.1f}% today.\nTrading paused until tomorrow.\n\n-- Satis House Consulting")
            return False, "Daily loss limit"
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
        return False, "Market closed -- equity signal queued"
    if is_leveraged and portfolio_state["leveraged_pct"] >= MAX_LEVERAGED_PCT:
        return False, "Leveraged cap reached"
    elif is_crypto and portfolio_state["crypto_pct"] >= MAX_CRYPTO_PCT:
        return False, "Crypto cap reached"
    elif not is_crypto and not is_leveraged and portfolio_state["equity_pct"] >= MAX_EQUITY_PCT:
        return False, "Equity cap reached"
    return True, "Clear"

def execute_signal(signal, position_size, portfolio_state, vix):
    global recent_signals, trade_history, orders_this_cycle
    ticker = signal["ticker"]
    direction = signal["direction"]
    stop_target = signal["stop"]

    is_crypto = any(c in ticker.upper() for c in ["BTC", "ETH", "SOL"])
    market_open = is_market_hours()

    if is_crypto:
        crypto_map = {"BTC": "BTC/USD", "BITCOIN": "BTC/USD", "ETH": "ETH/USD", "ETHEREUM": "ETH/USD", "SOL": "SOL/USD"}
        crypto_symbol = crypto_map.get(ticker.upper(), ticker)
        approved, reason = check_execution_rules(crypto_symbol, direction, portfolio_state, vix)
        if approved:
            try:
                notional = min(position_size, 5000)
                alpaca_request("POST", "/v2/orders", {"symbol": crypto_symbol, "notional": str(round(notional, 2)), "side": direction, "type": "market", "time_in_force": "gtc"})
                orders_this_cycle += 1
                recent_signals[f"{crypto_symbol}_{direction}"] = 0
                trade_history.append({"ticker": crypto_symbol, "direction": direction, "entry_price": None, "misfit": signal["name"], "executed": True, "timestamp": datetime.now(pytz.utc).isoformat()})
                save_log()
                return f"Crypto executed: {crypto_symbol} ${notional:,.0f}"
            except Exception as e:
                return f"Crypto failed: {e}"
        return f"Blocked: {reason}"

    if not market_open:
        return "Market closed -- will execute at next open"

    approved, reason = check_execution_rules(ticker, direction, portfolio_state, vix)
    if not approved:
        return f"Blocked: {reason}"

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
            return f"{ticker} did not fill"

        orders_this_cycle += 1
        stop_price = stop_target if stop_target > 0 else (price * 0.95 if direction == "buy" else price * 1.05)
        stop_side = "sell" if direction == "buy" else "buy"
        stop_ok = submit_stop_loss_atomic(ticker, qty, stop_side, stop_price)

        if stop_ok:
            recent_signals[f"{ticker}_{direction}"] = 0
            trade_history.append({"ticker": ticker, "direction": direction, "entry_price": price, "misfit": signal["name"], "executed": True, "timestamp": datetime.now(pytz.utc).isoformat()})
            save_log()
            return f"Executed: {direction.upper()} {qty} {ticker} at ~${price:.2f} | Stop: ${stop_price:.2f}"
        else:
            return f"{ticker} closed -- stop failed"
    except Exception as e:
        return f"Execution failed: {e}"

def update_training_loop(ticker, misfit_name, outcome_pnl_pct):
    trade_was_profitable = outcome_pnl_pct > 0
    if misfit_name in misfit_scorecard:
        misfit_scorecard[misfit_name]["total"] += 1
        if trade_was_profitable:
            misfit_scorecard[misfit_name]["correct"] += 1
            misfit_scorecard[misfit_name]["wins"] = misfit_scorecard[misfit_name].get("wins", 0) + 1
        else:
            misfit_scorecard[misfit_name]["losses"] = misfit_scorecard[misfit_name].get("losses", 0) + 1
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
                result = f"+${abs(pos['unrealized_pnl']):,.0f} profit" if pos["unrealized_pnl"] >= 0 else f"-${abs(pos['unrealized_pnl']):,.0f} loss"
                send_performance(f"""🛑 STOP LOSS TRIGGERED

Closed {symbol}
Result: {result} ({unrealized_pct:+.1f}%)
Capital protected.

-- Satis House Consulting""")
                misfit_name = None
                for trade in reversed(trade_history):
                    if trade.get("ticker") == symbol and trade.get("executed"):
                        misfit_name = trade.get("misfit")
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
                pos = data.get("Message", {}).get("PositionReport", {})
                with hormuz_lock:
                    hormuz_vessels.append({"name": meta.get("ShipName", "Unknown"), "lat": pos.get("Latitude", 0), "lon": pos.get("Longitude", 0), "speed": pos.get("Sog", 0), "timestamp": datetime.now(pytz.utc).isoformat()})
                    if len(hormuz_vessels) > 100:
                        hormuz_vessels.pop(0)
        except:
            pass

    def on_open(ws):
        ws.send(json.dumps({"APIKey": AISSTREAM_API_KEY, "MessageType": "Subscribe", "BoundingBoxes": [[[21.0, 55.0], [27.0, 62.0]]], "FilterMessageTypes": ["PositionReport"]}))
        print("AISStream: monitoring Hormuz")

    def on_error(ws, error):
        print(f"AISStream error: {error}")

    def on_close(ws, *args):
        time.sleep(60)
        start_aisstream()

    def run():
        ws = websocket.WebSocketApp("wss://stream.aisstream.io/v0/stream", on_open=on_open, on_message=on_message, on_error=on_error, on_close=on_close)
        ws.run_forever()

    threading.Thread(target=run, daemon=True).start()

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

misfit_knowledge_cache = {}
misfit_data_cache = {}
knowledge_refresh_cycles = 8
cycle_count = 0

def send_startup_message():
    env = detect_environment()
    active = [k for k, v in env.items() if v]
    env_str = ", ".join(active) if active else "standard conditions"
    weights = {name: misfit_scorecard[name]["weight"] for name in misfit_scorecard}
    weight_str = " | ".join([f"{n}: {w:.1f}x" for n, w in weights.items()])
    send_performance(f"""🚀 THE MISFITS -- CONTEST MODEL ONLINE

Five independent trading minds. One winner per cycle.

Architecture:
- Each Misfit generates their own signal every cycle from their own data
- No bottlenecks. No vetoes. No consensus required.
- Signals scored: Risk/Reward x Environment Weight x Bayesian Win Rate
- Highest score executes. Co-signals boost position size.
- Quarter-Kelly sizing. Five non-negotiable risk rules.

Current environment: {env_str}
Current weights: {weight_str}

Competing against OmniscientBot on separate account.
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

    if cycle_count % knowledge_refresh_cycles == 1:
        print("Refreshing Misfit data and knowledge...")
        for name, config in MISFIT_TASKS.items():
            print(f"  {name}...")
            try:
                misfit_data_cache[name] = config["data_func"]()
            except Exception as e:
                print(f"  Data error for {name}: {e}")
                misfit_data_cache[name] = {}
            blocks = []
            for q in config["knowledge_queries"]:
                try:
                    results = exa.search_and_contents(q, num_results=2, text={"max_characters": 400})
                    for r in results.results:
                        blocks.append(f"{r.title}: {r.text[:300]}")
                    time.sleep(1)
                except:
                    pass
            misfit_knowledge_cache[name] = "\n\n".join(blocks)
            time.sleep(1)

    market_open = is_market_hours()
    friday_blocked = is_friday_short_blocked()
    market_context = get_market_context()

    print(f"Cycle {cycle_count}: running contest...")
    raw_signals = []
    signal_texts = {}

    for name in MISFIT_TASKS:
        try:
            specific_data = misfit_data_cache.get(name, {})
            knowledge = misfit_knowledge_cache.get(name, "")
            weight = misfit_scorecard.get(name, {}).get("weight", 1.0)
            raw_text = generate_misfit_signal(name, market_context, specific_data, knowledge, weight, market_open, friday_blocked)
            signal_texts[name] = raw_text
            parsed = parse_signal(raw_text, name)
            if parsed:
                raw_signals.append(parsed)
                print(f"  {name}: {parsed['ticker']} {parsed['direction'].upper()} RR:{parsed['risk_reward']:.1f} score:{parsed['contest_score']:.2f}")
            else:
                print(f"  {name}: signal rejected (below min RR or parse error)")
            time.sleep(1)
        except Exception as e:
            print(f"  {name} error: {e}")

    winner, co_signals = run_contest(raw_signals)
    session_stats["total"] += 1

    if winner:
        portfolio_value = portfolio_state["portfolio_value"] if portfolio_state else INCEPTION_VALUE
        position_size, sizing_note = calculate_position_size(winner, co_signals, portfolio_value, vix)
        result = execute_signal(winner, position_size, portfolio_state, vix)
        session_stats["execute"] += 1
        session_stats["contest_wins"][winner["name"]] = session_stats["contest_wins"].get(winner["name"], 0) + 1
        save_log()

        perf_msg = format_trade_performance(winner, co_signals, position_size, sizing_note, result)
        send_performance(perf_msg)

        brief = f"MISFITS CONTEST -- CYCLE {cycle_count}\n\n"
        brief += f"🏆 WINNER: {winner['name']} (score: {winner['contest_score']:.2f})\n"
        brief += f"Signal: {winner['ticker']} {winner['direction'].upper()} | RR: {winner['risk_reward']:.1f}:1\n"
        if co_signals:
            brief += f"Co-signals: {', '.join([s['name'] + ' ' + s['ticker'] for s in co_signals])}\n"
        brief += f"\nAll signals this cycle:\n"
        for name, text in signal_texts.items():
            weight = misfit_scorecard.get(name, {}).get("weight", 1.0)
            matching = next((s for s in raw_signals if s["name"] == name), None)
            score_str = f" score:{matching['contest_score']:.2f}" if matching else " (rejected)"
            brief += f"{name} ({weight:.1f}x){score_str}: {text[:120]}\n\n"
        brief += f"Execution: {result}"
        send_telegram(brief)

    else:
        session_stats["no_trade"] += 1
        save_log()

        brief = f"MISFITS CONTEST -- CYCLE {cycle_count} -- NO QUALIFYING SIGNAL\n\n"
        brief += f"Minimum score threshold: {MIN_CONTEST_SCORE} | Minimum RR: {MIN_RISK_REWARD}:1\n\n"
        for name, text in signal_texts.items():
            weight = misfit_scorecard.get(name, {}).get("weight", 1.0)
            matching = next((s for s in raw_signals if s["name"] == name), None)
            score_str = f" score:{matching['contest_score']:.2f}" if matching else " (rejected)"
            brief += f"{name} ({weight:.1f}x){score_str}:\n{text[:150]}\n\n"
        send_telegram(brief)

    print(f"Cycle {cycle_count}: {'EXECUTED ' + winner['name'] if winner else 'NO TRADE'}")
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
