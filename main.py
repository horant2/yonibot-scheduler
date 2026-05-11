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

LEVERAGED_ETFS = {"TQQQ", "SOXL", "TECL", "FAS", "ERX", "TMF", "SPXL", "UPRO"}

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
        "Soros": {"trade": 0, "no_opportunity": 0},
        "Druckenmiller": {"trade": 0, "no_opportunity": 0},
        "PTJ": {"trade": 0, "no_opportunity": 0},
        "Tepper": {"trade": 0, "no_opportunity": 0},
        "Andurand": {"trade": 0, "no_opportunity": 0}
    }
}

trade_history = []

ENERGY_ASSETS = {"ERX", "XLE", "USO", "BNO", "UNG", "XOP", "VLO", "FRO"}
TECH_ASSETS = {"TQQQ", "TECL", "QQQ", "SOXL"}
FINANCIAL_ASSETS = {"FAS", "XLF"}
MACRO_ASSETS = {"TLT", "HYG", "GLD", "SLV", "UUP"}
FX_ASSETS = {"FXE", "FXB", "FXY", "EEM"}
CRYPTO_ASSETS_SET = {"BTC/USD", "ETH/USD", "SOL/USD"}

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
    beta = (total - correct) + 1
    posterior_mean = alpha / (alpha + beta)
    weight = posterior_mean / 0.5
    return round(max(0.5, min(2.5, weight)), 2)

def calculate_kelly_size(win_probability, win_size, loss_size, portfolio_value):
    if loss_size <= 0 or win_size <= 0:
        return portfolio_value * 0.05
    b = win_size / loss_size
    p = win_probability
    q = 1 - p
    kelly = (p * b - q) / b
    kelly = max(0, kelly)
    quarter_kelly = kelly * QUARTER_KELLY_FRACTION
    size = portfolio_value * quarter_kelly
    return min(size, portfolio_value * MAX_SINGLE_POSITION)

def detect_environment():
    env = {"energy_crisis": False, "credit_crisis": False, "currency_crisis": False, "market_crash": False}
    try:
        for ticker, key, threshold, direction in [
            ("USO", "energy_crisis", 0.12, "abs"),
            ("HYG", "credit_crisis", -0.05, "neg"),
            ("SPY", "market_crash", -0.10, "neg"),
            ("UUP", "currency_crisis", 0.05, "abs")
        ]:
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
        bayesian = calculate_bayesian_weight(name)
        misfit_scorecard[name]["weight"] = bayesian

    env_boosts = {
        "energy_crisis": [("Andurand", 2.0)],
        "credit_crisis": [("Tepper", 2.0), ("Druckenmiller", 1.5)],
        "currency_crisis": [("Soros", 2.0)],
        "market_crash": [("PTJ", 2.0), ("Druckenmiller", 1.5)]
    }
    for env_key, boosts in env_boosts.items():
        if environment.get(env_key):
            for name, boost in boosts:
                current = misfit_scorecard[name]["weight"]
                misfit_scorecard[name]["weight"] = max(current, boost)
                print(f"{name} elevated to {misfit_scorecard[name]['weight']:.1f}x -- {env_key}")

def get_fred_data(series_id):
    try:
        r = requests.get("https://api.stlouisfed.org/fred/series/observations",
            params={"series_id": series_id, "api_key": FRED_API_KEY, "file_type": "json", "limit": 5, "sort_order": "desc"},
            timeout=10)
        obs = r.json().get("observations", [])
        if obs:
            val = obs[0]["value"]
            return float(val) if val != "." else None
    except:
        pass
    return None

def get_soros_briefing():
    try:
        data = {}
        usdx = get_fred_data("DTWEXBGS")
        if usdx: data["dollar_index"] = usdx
        for pair, ticker in [("EUR_USD", "EURUSD=X"), ("GBP_USD", "GBPUSD=X"), ("JPY_USD", "JPY=X"), ("EEM", "EEM"), ("TUR", "TUR"), ("EWZ", "EWZ")]:
            try:
                price = yf.download(ticker, period="30d", progress=False)["Close"].squeeze()
                data[f"{pair}_30d_pct"] = round(float((price.iloc[-1] - price.iloc[0]) / price.iloc[0] * 100), 2)
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
            exa_results = exa.search_and_contents("central bank currency intervention capital flight sovereign debt crisis 2026", num_results=3, text={"max_characters": 400})
            data["intelligence"] = " | ".join([r.title for r in exa_results.results])
        except:
            pass
        return data
    except Exception as e:
        return {"error": str(e)}

def get_druckenmiller_briefing():
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
            cot = requests.get("https://publicreporting.cftc.gov/resource/jun7-fc8e.json?$limit=3&$order=report_date_as_yyyy_mm_dd DESC&$where=contract_market_name=%27E-MINI S%26P 500%27", timeout=10).json()
            if cot:
                net = int(cot[0].get("noncomm_positions_long_all", 0)) - int(cot[0].get("noncomm_positions_short_all", 0))
                data["sp500_hedge_fund_net"] = net
        except:
            pass
        return data
    except Exception as e:
        return {"error": str(e)}

def get_ptj_briefing():
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
                sma200 = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else sma50
                price = float(close.iloc[-1])
                vol_ratio = float(vol.iloc[-1] / vol.rolling(20).mean().iloc[-1])
                data[f"{ticker}_vs_sma50_pct"] = round((price - sma50) / sma50 * 100, 2)
                data[f"{ticker}_golden_cross"] = price > sma50 > sma200
                data[f"{ticker}_volume_ratio"] = round(vol_ratio, 2)
            except:
                pass
        return data
    except Exception as e:
        return {"error": str(e)}

def get_tepper_briefing():
    try:
        data = {}
        for series, label in [("BAMLH0A0HYM2", "hy_spread"), ("BAMLC0A0CM", "ig_spread"), ("DRCCLACBS", "cc_delinquency"), ("DRSFRMACBS", "mortgage_delinquency")]:
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
            cot = requests.get("https://publicreporting.cftc.gov/resource/jun7-fc8e.json?$limit=3&$order=report_date_as_yyyy_mm_dd DESC&$where=contract_market_name=%2710-YEAR T-NOTES%27", timeout=10).json()
            if cot:
                net = int(cot[0].get("noncomm_positions_long_all", 0)) - int(cot[0].get("noncomm_positions_short_all", 0))
                data["treasury_hedge_fund_net"] = net
        except:
            pass
        try:
            exa_results = exa.search_and_contents("Federal Reserve policy pivot credit market high yield 2026", num_results=2, text={"max_characters": 300})
            data["fed_intelligence"] = " | ".join([r.title for r in exa_results.results])
        except:
            pass
        return data
    except Exception as e:
        return {"error": str(e)}

def get_andurand_briefing():
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
            if hormuz_vessels:
                speeds = [v.get("speed", 0) for v in hormuz_vessels[-10:]]
                data["avg_vessel_speed_knots"] = round(sum(speeds) / len(speeds), 1) if speeds else 0
        for ticker, name in [("USO", "wti_proxy"), ("BNO", "brent_proxy"), ("UNG", "natgas"), ("ERX", "energy_2x"), ("XLE", "energy_sector"), ("FRO", "tankers"), ("VLO", "refiners")]:
            try:
                price = yf.download(ticker, period="30d", progress=False)["Close"].squeeze()
                data[f"{name}_30d_pct"] = round(float((price.iloc[-1] - price.iloc[0]) / price.iloc[0] * 100), 2)
                data[f"{name}_price"] = round(float(price.iloc[-1]), 2)
            except:
                pass
        try:
            cot = requests.get("https://publicreporting.cftc.gov/resource/jun7-fc8e.json?$limit=3&$order=report_date_as_yyyy_mm_dd DESC&$where=contract_market_name=%27CRUDE OIL%27", timeout=10).json()
            if cot:
                net = int(cot[0].get("noncomm_positions_long_all", 0)) - int(cot[0].get("noncomm_positions_short_all", 0))
                data["crude_hedge_fund_net"] = net
        except:
            pass
        for currency, ticker in [("NOK", "NOK=X"), ("CAD", "CAD=X"), ("SAR", "SAR=X")]:
            try:
                fx = yf.download(ticker, period="30d", progress=False)["Close"].squeeze()
                data[f"{currency}_30d_pct"] = round(float((fx.iloc[-1] - fx.iloc[0]) / fx.iloc[0] * 100), 2)
            except:
                pass
        try:
            exa_results = exa.search_and_contents("Strait Hormuz tanker oil supply Iran OPEC energy 2026", num_results=3, text={"max_characters": 400})
            data["hormuz_intelligence"] = " | ".join([r.title for r in exa_results.results])
        except:
            pass
        return data
    except Exception as e:
        return {"error": str(e)}

def generate_misfit_signal(name, task, briefing_data, knowledge, weight):
    weight_note = f"\nYour environment weight is {weight:.1f}x. This is your moment. Trust your conviction and go big." if weight > 1.0 else ""
    data_str = json.dumps(briefing_data, default=str)[:2000]
    knowledge_str = knowledge[:800] if knowledge else ""

    scorecard = misfit_scorecard.get(name, {})
    total = scorecard.get("total", 0)
    correct = scorecard.get("correct", 0)
    record_str = f"Your track record: {correct}/{total} trades correct." if total > 0 else "No closed trades yet."

    msg = client.messages.create(
        model="claude-opus-4-5-20251101",
        max_tokens=400,
        messages=[{"role": "user", "content": f"""{task}{weight_note}

YOUR LIVE DATA:
{data_str}

YOUR KNOWLEDGE BASE:
{knowledge_str}

{record_str}

YOUR TASK: Generate ONE specific trade opportunity from YOUR data and YOUR framework. Do not react to any other signal. Generate your own independent view.

If you see a genuine opportunity output EXACTLY this format:
SIGNAL: [asset ticker]
DIRECTION: [BUY or SHORT]
ENTRY: [price]
STOP: [price]
TARGET: [price]
CONFIDENCE: [HIGH or MEDIUM]
REASON: [one sentence from your specific framework]

If no genuine opportunity exists in your data right now output exactly:
NO OPPORTUNITY: [one sentence why]

Nothing else. No preamble."""}]
    )
    return msg.content[0].text

def parse_misfit_signal(text):
    try:
        if "NO OPPORTUNITY" in text:
            return None
        lines = text.strip().split("\n")
        result = {}
        for line in lines:
            if ":" in line:
                key, val = line.split(":", 1)
                result[key.strip()] = val.strip()
        if "SIGNAL" not in result or "DIRECTION" not in result:
            return None
        ticker = result["SIGNAL"].strip().upper()
        direction = "buy" if "BUY" in result["DIRECTION"].upper() else "sell"
        entry = float(result.get("ENTRY", "0").replace("$", "").replace(",", "")) if result.get("ENTRY") else 0
        stop = float(result.get("STOP", "0").replace("$", "").replace(",", "")) if result.get("STOP") else 0
        target = float(result.get("TARGET", "0").replace("$", "").replace(",", "")) if result.get("TARGET") else 0
        confidence = result.get("CONFIDENCE", "MEDIUM").upper()
        reason = result.get("REASON", "")
        if entry > 0 and stop > 0 and target > 0:
            if direction == "buy":
                win_size = target - entry
                loss_size = entry - stop
            else:
                win_size = entry - target
                loss_size = stop - entry
            if loss_size <= 0:
                return None
            risk_reward = win_size / loss_size
        else:
            risk_reward = 2.0
            win_size = 1.0
            loss_size = 0.5
        return {
            "ticker": ticker,
            "direction": direction,
            "entry": entry,
            "stop": stop,
            "target": target,
            "confidence": confidence,
            "reason": reason,
            "risk_reward": round(risk_reward, 2),
            "win_size": win_size,
            "loss_size": loss_size
        }
    except Exception as e:
        print(f"Signal parse error: {e}")
        return None

def find_convergence(signals_by_misfit):
    ticker_votes = {}
    theme_votes = {}

    for name, signal in signals_by_misfit.items():
        if signal is None:
            continue
        ticker = signal["ticker"]
        direction = signal["direction"]
        weight = misfit_scorecard.get(name, {}).get("weight", 1.0)
        key = f"{ticker}_{direction}"
        if key not in ticker_votes:
            ticker_votes[key] = {"ticker": ticker, "direction": direction, "weight": 0, "voters": [], "signals": []}
        ticker_votes[key]["weight"] += weight
        ticker_votes[key]["voters"].append(name)
        ticker_votes[key]["signals"].append(signal)

        theme = None
        if ticker in ENERGY_ASSETS:
            theme = f"energy_{direction}"
        elif ticker in TECH_ASSETS:
            theme = f"tech_{direction}"
        elif ticker in MACRO_ASSETS:
            theme = f"macro_{direction}"
        elif ticker in FX_ASSETS:
            theme = f"fx_{direction}"
        elif ticker in FINANCIAL_ASSETS:
            theme = f"financial_{direction}"

        if theme:
            if theme not in theme_votes:
                theme_votes[theme] = {"theme": theme, "direction": direction, "weight": 0, "voters": [], "signals": [], "tickers": []}
            theme_votes[theme]["weight"] += weight
            theme_votes[theme]["voters"].append(name)
            theme_votes[theme]["signals"].append(signal)
            theme_votes[theme]["tickers"].append(ticker)

    best_ticker_key = None
    best_ticker_weight = 0
    for key, data in ticker_votes.items():
        if data["weight"] > best_ticker_weight:
            best_ticker_weight = data["weight"]
            best_ticker_key = key

    best_theme_key = None
    best_theme_weight = 0
    for key, data in theme_votes.items():
        if data["weight"] > best_theme_weight:
            best_theme_weight = data["weight"]
            best_theme_key = key

    if best_ticker_key and best_ticker_weight >= 2.0:
        return "ticker", ticker_votes[best_ticker_key], best_ticker_weight

    if best_theme_key and best_theme_weight >= 2.0:
        theme_data = theme_votes[best_theme_key]
        best_signal = max(theme_data["signals"], key=lambda s: s["risk_reward"])
        theme_data["ticker"] = best_signal["ticker"]
        theme_data["entry"] = best_signal["entry"]
        theme_data["stop"] = best_signal["stop"]
        theme_data["target"] = best_signal["target"]
        theme_data["risk_reward"] = best_signal["risk_reward"]
        theme_data["win_size"] = best_signal["win_size"]
        theme_data["loss_size"] = best_signal["loss_size"]
        theme_data["reason"] = f"Theme convergence: {', '.join([s['reason'] for s in theme_data['signals'][:2]])}"
        return "theme", theme_data, best_theme_weight

    return "none", None, 0

def jane_street_size(convergence_data, portfolio_value, convergence_weight, environment):
    try:
        ticker = convergence_data.get("ticker", "")
        direction = convergence_data.get("direction", "buy")
        win_size = convergence_data.get("win_size", 1.0)
        loss_size = convergence_data.get("loss_size", 0.5)
        risk_reward = convergence_data.get("risk_reward", 2.0)
        voters = convergence_data.get("voters", [])

        total_correct = sum(misfit_scorecard.get(v, {}).get("correct", 0) for v in voters)
        total_trades = sum(misfit_scorecard.get(v, {}).get("total", 1) for v in voters)
        win_probability = (total_correct + 1) / (total_trades + 2)

        raw_kelly = (win_probability * (win_size / loss_size) - (1 - win_probability)) / (win_size / loss_size)
        raw_kelly = max(0, raw_kelly)
        quarter_kelly = raw_kelly * QUARTER_KELLY_FRACTION

        env_active = [k for k, v in environment.items() if v]
        if env_active and any(name in ["Andurand", "Soros", "Tepper"] for name in voters):
            quarter_kelly = min(quarter_kelly * 1.5, MAX_SINGLE_POSITION)

        position_size = min(portfolio_value * quarter_kelly, portfolio_value * MAX_SINGLE_POSITION)
        position_size = max(position_size, portfolio_value * 0.03)

        sizing_note = f"Kelly: {raw_kelly*100:.1f}% | Quarter-Kelly: {quarter_kelly*100:.1f}% | Win prob: {win_probability*100:.0f}% | RR: {risk_reward:.1f}:1"
        return position_size, sizing_note, win_probability
    except Exception as e:
        print(f"Kelly sizing error: {e}")
        return portfolio_value * 0.05, "Default 5%", 0.45

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

def format_convergence_alert(convergence_type, convergence_data, position_size, sizing_note, voters):
    ticker = convergence_data.get("ticker", "")
    direction = convergence_data.get("direction", "buy")
    entry = convergence_data.get("entry", 0)
    stop = convergence_data.get("stop", 0)
    target = convergence_data.get("target", 0)
    rr = convergence_data.get("risk_reward", 0)
    reason = convergence_data.get("reason", "")
    action = "Bought" if direction == "buy" else "Sold Short"
    emoji = "🟢" if direction == "buy" else "🔴"
    conv_label = "Direct convergence" if convergence_type == "ticker" else "Theme convergence"
    return f"""{emoji} THE MISFITS FOUND CONVERGENCE

{action} {ticker}
Entry: ~${entry:.2f} | Stop: ${stop:.2f} | Target: ${target:.2f}
Risk reward: {rr:.1f} to 1
Position size: ${position_size:,.0f}

Why: {reason}

Who converged: {', '.join(voters)}
Type: {conv_label}

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
    lines.append(f"✅ Convergence trades executed: {execute} ({execute/total*100:.0f}%)")
    lines.append(f"⏭ No convergence: {passed} ({passed/total*100:.0f}%)")
    lines.append(f"\nMisfit signal rates:")
    for name, votes in session_stats["misfit_signals"].items():
        total_signals = votes["trade"] + votes["no_opportunity"]
        rate = votes["trade"] / total_signals * 100 if total_signals > 0 else 0
        weight = misfit_scorecard.get(name, {}).get("weight", 1.0)
        sc = misfit_scorecard.get(name, {})
        record = f"{sc.get('correct',0)}/{sc.get('total',0)}" if sc.get("total", 0) > 0 else "no trades"
        lines.append(f"  {name}: {votes['trade']}/{total_signals} signals ({rate:.0f}%) | record: {record} | weight: {weight:.1f}x")
    lines.append(net_line)
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

Closed short position in {symbol} before weekend.
Result: {result} ({pos['unrealized_pct']:+.1f}%)

Short positions do not survive weekends.

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

def execute_convergence_trade(convergence_data, position_size, portfolio_state, vix, voters):
    global recent_signals, trade_history, orders_this_cycle
    ticker = convergence_data.get("ticker", "")
    direction = convergence_data.get("direction", "buy")
    stop_target = convergence_data.get("stop", 0)
    reason = convergence_data.get("reason", "Convergence signal")

    is_crypto = any(c in ticker.upper() for c in ["BTC", "ETH", "SOL"])
    market_open = is_market_hours()

    if is_crypto:
        crypto_map = {"BTC": "BTC/USD", "BITCOIN": "BTC/USD", "ETH": "ETH/USD", "ETHEREUM": "ETH/USD", "SOL": "SOL/USD"}
        crypto_symbol = crypto_map.get(ticker.upper(), ticker)
        approved, block_reason = check_execution_rules(crypto_symbol, direction, portfolio_state, vix)
        if approved:
            try:
                notional = min(position_size, 5000)
                result = alpaca_request("POST", "/v2/orders", {"symbol": crypto_symbol, "notional": str(round(notional, 2)), "side": direction, "type": "market", "time_in_force": "gtc"})
                orders_this_cycle += 1
                recent_signals[f"{crypto_symbol}_{direction}"] = 0
                trade_history.append({"ticker": crypto_symbol, "direction": direction, "entry_price": None, "voters_for": voters, "executed": True, "timestamp": datetime.now(pytz.utc).isoformat()})
                save_log()
                return f"Crypto executed: {crypto_symbol} ${notional:,.0f}"
            except Exception as e:
                return f"Crypto failed: {e}"
        return f"Crypto blocked: {block_reason}"

    if not market_open:
        return "Market closed -- equity signal queued for next open"

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

        price_data = yf.download(ticker, period="1d", interval="1m", progress=False)
        price = float(price_data["Close"].squeeze().iloc[-1])
        qty = max(1, int(position_size / price))

        order = submit_order(ticker, qty, direction)
        order_id = order.get("id", "unknown")
        filled = verify_order_filled(order_id)

        if not filled:
            return f"{ticker} order did not fill"

        orders_this_cycle += 1
        stop_price = price * 0.95 if direction == "buy" else price * 1.05
        if stop_target > 0:
            stop_price = stop_target
        stop_side = "sell" if direction == "buy" else "buy"
        stop_ok = submit_stop_loss_atomic(ticker, qty, stop_side, stop_price)

        if stop_ok:
            recent_signals[f"{ticker}_{direction}"] = 0
            trade_history.append({"ticker": ticker, "direction": direction, "entry_price": price, "voters_for": voters, "executed": True, "timestamp": datetime.now(pytz.utc).isoformat()})
            save_log()
            return f"Executed: {direction.upper()} {qty} {ticker} at ~${price:.2f} | Stop: ${stop_price:.2f}"
        else:
            return f"{ticker} position closed -- stop loss failed"
    except Exception as e:
        return f"Execution failed: {e}"

def update_training_loop(ticker, voters_for, outcome_pnl_pct):
    trade_was_profitable = outcome_pnl_pct > 0
    for name in voters_for:
        clean = name.split("(")[0].strip()
        if clean in misfit_scorecard:
            misfit_scorecard[clean]["total"] += 1
            if trade_was_profitable:
                misfit_scorecard[clean]["correct"] += 1
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

Closed {symbol} position
Result: {result} ({unrealized_pct:+.1f}%)
Capital protected. On to the next signal.

-- Satis House Consulting""")
                voters_for = []
                for trade in reversed(trade_history):
                    if trade.get("ticker") == symbol:
                        voters_for = trade.get("voters_for", [])
                        break
                update_training_loop(symbol, voters_for, unrealized_pct / 100)
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

def run_omniscient_rotation():
    tickers = ["SOXL", "TECL", "TQQQ", "FAS", "ERX", "UUP", "TMF", "BIL"]
    safe = "BIL"
    scores = {}
    prices = {}
    try:
        spy_data = yf.download("SPY", period="220d", interval="1d", progress=False)
        spy_close = spy_data["Close"].squeeze()
        spy_trend = spy_close.iloc[-1] > spy_close.rolling(200).mean().iloc[-1]
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
            if vol == 0 or np.isnan(vol):
                vol = 0.01
            rsi_val = close.diff()
            gain = rsi_val.clip(lower=0).rolling(14).mean()
            loss = -rsi_val.clip(upper=0).rolling(14).mean()
            rsi = float(100 - (100 / (1 + gain / loss)).iloc[-1])
            sma50 = float(close.rolling(50).mean().iloc[-1])
            price = float(close.iloc[-1])
            weighted_mom = (roc_fast * 0.5) + (roc_med * 0.3) + (roc_slow * 0.2)
            risk_adj_mom = weighted_mom / vol
            trend_score = 1.0 if price > sma50 else 0.5
            rsi_penalty = 0.9 if (rsi > 85 or rsi < 30) else 1.0
            scores[ticker] = risk_adj_mom * trend_score * rsi_penalty
            prices[ticker] = price
        except:
            pass
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
        print("AISStream subscribed to Hormuz")

    def on_error(ws, error):
        print(f"AISStream error: {error}")

    def on_close(ws, *args):
        time.sleep(60)
        start_aisstream()

    def run():
        ws = websocket.WebSocketApp("wss://stream.aisstream.io/v0/stream", on_open=on_open, on_message=on_message, on_error=on_error, on_close=on_close)
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
        """You ARE George Soros. Your ONE task: identify currency peg breaks, unsustainable sovereign positions, and reflexivity reversals from your live data.

Look specifically for:
- Currencies being defended at unsustainable levels (reserves depleting)
- Emerging market stress with capital flight
- Reflexivity loops about to reverse
- Central bank ammunition running out

Your asset universe: FX pairs, EM equity ETFs, sovereign bond ETFs, currency options proxies.""",
        ["George Soros Black Wednesday mechanics reflexivity currency crisis",
         "Soros Fund Management macro sovereign currency 2025 2026"]
    ),
    (
        "Druckenmiller",
        """You ARE Stanley Druckenmiller. Your ONE task: identify the single highest conviction asymmetric macro trade from your live credit and momentum data.

Look specifically for:
- Credit spreads diverging from equity prices (credit leads equity)
- Federal Reserve balance sheet inflection points
- Momentum breaks in major asset classes
- Concentration opportunities where risk reward is 3:1 or better

Your asset universe: equities long and short, bonds, leveraged ETFs, credit ETFs.""",
        ["Stanley Druckenmiller concentration asymmetric macro methodology",
         "Druckenmiller macro views credit Federal Reserve 2025 2026"]
    ),
    (
        "PTJ",
        """You ARE Paul Tudor Jones. Your ONE task: find the 5 to 1 or better technical setup from your live volatility and price data.

Look specifically for:
- VIX regime changes creating asymmetric options-like payoffs
- Volume breakouts confirming momentum shifts
- Golden cross or death cross forming
- Oversold bounces with defined stop below recent low

Your asset universe: index ETFs, sector ETFs, volatility plays, trend following across all asset classes.""",
        ["Paul Tudor Jones 5 to 1 risk reward rules Black Monday tape reading",
         "PTJ macro technical analysis trend following 2025 2026"]
    ),
    (
        "Tepper",
        """You ARE David Tepper. Your ONE task: identify trades driven by Federal Reserve policy and credit cycle positioning from your live bond and credit data.

Look specifically for:
- High yield spread compression or blowout
- Treasury yield curve inflection
- Credit delinquency leading indicators turning
- Policy backdrop shifting toward accommodation or tightening

Your asset universe: high yield ETFs, investment grade ETFs, Treasury ETFs, bank stocks, beaten-down equities when Fed pivots.""",
        ["David Tepper 2009 bank trade Federal Reserve reading credit",
         "Tepper macro views Fed policy credit 2025 2026"]
    ),
    (
        "Andurand",
        """You ARE Pierre Andurand. Your ONE task: identify the highest conviction energy trade from your live physical market data.

Look specifically for:
- Cushing storage draws accelerating (bullish crude)
- Hormuz vessel traffic changes (supply disruption risk)
- Crack spread widening (refinery demand signal)
- Hedge fund net positioning diverging from physical reality
- Oil-linked currencies confirming or diverging from crude price

Your asset universe: USO, BNO, ERX, XLE, UNG, XOP, VLO, FRO, oil-exporter currencies, energy sovereign bonds.""",
        ["Pierre Andurand physical commodity flows Hormuz 2008 2022",
         "Andurand Capital oil energy physical market 2025 2026"]
    )
]

misfit_knowledge_cache = {}
misfit_briefing_cache = {}
knowledge_refresh_cycles = 8
cycle_count = 0

def send_startup_message():
    send_performance("""🚀 MISFITS SYSTEM -- NEW ARCHITECTURE ONLINE

No more bottlenecks. No more vetoes.

Each Misfit now independently generates their own trade signal from their own data:

🔵 Soros: currency pegs and reflexivity breaks
🟤 Druckenmiller: credit cycle and macro asymmetry  
📊 PTJ: 5 to 1 technical setups
💚 Tepper: Federal Reserve policy trades
🛢 Andurand: physical energy flows and Hormuz

When two or more Misfits independently converge on the same trade -- that is the signal. Jane Street sizes it using quarter-Kelly. No veto. No bottleneck.

Weights: equal by default. Elevated when the environment matches their career defining trade. Bayesian-updated as trade history builds.

-- Satis House Consulting""")

def run_cycle():
    global cycle_count, misfit_knowledge_cache, misfit_briefing_cache
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
        print("Refreshing Misfit knowledge and briefings...")
        briefing_funcs = {
            "Soros": get_soros_briefing,
            "Druckenmiller": get_druckenmiller_briefing,
            "PTJ": get_ptj_briefing,
            "Tepper": get_tepper_briefing,
            "Andurand": get_andurand_briefing
        }
        for name, task, queries in MISFIT_CONFIGS:
            print(f"  Briefing {name}...")
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
            misfit_briefing_cache[name] = briefing_funcs[name]()
            time.sleep(2)

    rotation_ticker, rotation_score, rotation_summary, _ = run_omniscient_rotation()

    signals_by_misfit = {}
    signal_texts = {}
    market_open = is_market_hours()

    print(f"Cycle {cycle_count}: generating independent signals...")
    for name, task, queries in MISFIT_CONFIGS:
        briefing = misfit_briefing_cache.get(name, {})
        knowledge = misfit_knowledge_cache.get(name, "")
        weight = misfit_scorecard.get(name, {}).get("weight", 1.0)

        market_note = ""
        if not market_open:
            market_note = "\nIMPORTANT: Equity markets are CLOSED. Only generate crypto signals (BTC, ETH, SOL) or NO OPPORTUNITY."
        if is_friday_short_blocked():
            market_note += "\nFRIDAY RULE: No short signals after 2 PM ET."

        full_task = task + market_note

        raw_text = generate_misfit_signal(name, full_task, briefing, knowledge, weight)
        signal_texts[name] = raw_text
        parsed = parse_misfit_signal(raw_text)
        signals_by_misfit[name] = parsed

        has_signal = parsed is not None
        session_stats["misfit_signals"][name]["trade" if has_signal else "no_opportunity"] += 1
        print(f"  {name}: {'SIGNAL: ' + parsed['ticker'] if parsed else 'NO OPPORTUNITY'}")

    convergence_type, convergence_data, convergence_weight = find_convergence(signals_by_misfit)

    if rotation_ticker and rotation_ticker not in ["BIL", "UUP"] and rotation_score > 0.5 and market_open:
        rotation_signal = {
            "ticker": rotation_ticker,
            "direction": "buy",
            "entry": 0,
            "stop": 0,
            "target": 0,
            "confidence": "HIGH",
            "reason": f"OmniscientBot rotation signal (2324% backtest): {rotation_ticker} score {rotation_score:.3f}",
            "risk_reward": 3.0,
            "win_size": 0.15,
            "loss_size": 0.05
        }
        existing_weight = convergence_weight if convergence_data else 0
        rotation_boost = 2.5
        if rotation_boost > existing_weight:
            convergence_type = "rotation"
            convergence_data = rotation_signal
            convergence_data["voters"] = ["OmniscientBot"]
            convergence_weight = rotation_boost
            print(f"  Rotation override: {rotation_ticker} score {rotation_score:.3f}")

    session_stats["total"] += 1

    if convergence_type != "none" and convergence_data:
        voters = convergence_data.get("voters", [v for v in signals_by_misfit if signals_by_misfit[v] and signals_by_misfit[v]["ticker"] == convergence_data.get("ticker")])
        portfolio_value = portfolio_state["portfolio_value"] if portfolio_state else INCEPTION_VALUE
        position_size, sizing_note, win_prob = jane_street_size(convergence_data, portfolio_value, convergence_weight, environment)

        if vix >= VIX_REDUCE_THRESHOLD:
            position_size *= 0.5
            sizing_note += " | VIX reduced"

        convergence_data["voters"] = voters
        trade_result = execute_convergence_trade(convergence_data, position_size, portfolio_state, vix, voters)
        session_stats["execute"] += 1
        save_log()

        performance_msg = format_convergence_alert(convergence_type, convergence_data, position_size, sizing_note, voters)
        send_performance(performance_msg)

        telegram_brief = f"MISFITS CONVERGENCE DETECTED\n"
        telegram_brief += f"Type: {convergence_type} | Weight: {convergence_weight:.1f} | Asset: {convergence_data.get('ticker')}\n\n"
        for name, text in signal_texts.items():
            telegram_brief += f"{name.upper()}:\n{text[:200]}\n\n"
        telegram_brief += f"EXECUTION: {trade_result}"
        send_telegram(telegram_brief)

    else:
        session_stats["pass"] += 1
        save_log()
        no_conv_brief = f"MISFITS CYCLE {cycle_count} -- NO CONVERGENCE\n\n"
        for name, signal in signals_by_misfit.items():
            weight = misfit_scorecard.get(name, {}).get("weight", 1.0)
            if signal:
                no_conv_brief += f"{name.upper()} ({weight:.1f}x): SIGNAL {signal['ticker']} {signal['direction'].upper()} RR:{signal['risk_reward']:.1f}\n"
            else:
                no_conv_brief += f"{name.upper()} ({weight:.1f}x): NO OPPORTUNITY\n"
        no_conv_brief += f"\nRotation: {rotation_ticker} ({rotation_score:.3f})" if rotation_ticker else ""
        send_telegram(no_conv_brief)

    print(f"Cycle {cycle_count} complete: {'EXECUTED' if convergence_type != 'none' else 'NO CONVERGENCE'}")
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
