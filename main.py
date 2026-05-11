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
ROTATION_SIZE = 0.15
MAX_SINGLE_POSITION = 0.15
VIX_REDUCE_THRESHOLD = 35
VIX_STOP_THRESHOLD = 50
MAX_EQUITY_PCT = 0.60
MAX_CRYPTO_PCT = 0.25
MAX_LEVERAGED_PCT = 0.45
DUPLICATE_SIGNAL_BLOCKS = 2

LEVERAGED_ETFS = {"TQQQ", "SOXL", "TECL", "FAS", "ERX", "TMF", "SPXL", "UPRO"}

recent_signals = {}
daily_start_value = None
trades_halted_today = False
orders_this_cycle = 0
hormuz_vessels = []
hormuz_lock = threading.Lock()

misfit_scorecard = {
    "Soros": {"correct": 0, "total": 0, "weight": 1.0},
    "Druckenmiller": {"correct": 0, "total": 0, "weight": 1.0},
    "PTJ": {"correct": 0, "total": 0, "weight": 1.0},
    "Tepper": {"correct": 0, "total": 0, "weight": 1.0},
    "Andurand": {"correct": 0, "total": 0, "weight": 1.0},
    "Jane Street": {"correct": 0, "total": 0, "weight": 1.0}
}

session_stats = {
    "execute": 0,
    "pass": 0,
    "blocked": 0,
    "total": 0,
    "jane_approved": 0,
    "jane_blocked": 0,
    "misfit_votes": {
        "Soros": {"trade": 0, "pass": 0},
        "Druckenmiller": {"trade": 0, "pass": 0},
        "PTJ": {"trade": 0, "pass": 0},
        "Tepper": {"trade": 0, "pass": 0},
        "Andurand": {"trade": 0, "pass": 0}
    }
}

trade_history = []

def load_log():
    global session_stats, trade_history, misfit_scorecard
    try:
        if os.path.exists(LOG_FILE):
            with open(LOG_FILE, "r") as f:
                data = json.load(f)
                saved_stats = data.get("session_stats", {})
                for key in session_stats:
                    if key in saved_stats:
                        if isinstance(session_stats[key], dict):
                            session_stats[key].update(saved_stats.get(key, {}))
                        else:
                            session_stats[key] = saved_stats[key]
                trade_history = data.get("trade_history", [])[-200:]
                saved_scorecard = data.get("misfit_scorecard", {})
                for name in misfit_scorecard:
                    if name in saved_scorecard:
                        misfit_scorecard[name].update(saved_scorecard[name])
                print(f"Log loaded: {session_stats['total']} total cycles, {session_stats['execute']} executions")
    except Exception as e:
        print(f"Log load error: {e}")

def save_log():
    try:
        data = {
            "session_stats": session_stats,
            "trade_history": trade_history[-200:],
            "misfit_scorecard": misfit_scorecard,
            "last_updated": datetime.now(pytz.utc).isoformat()
        }
        with open(LOG_FILE, "w") as f:
            json.dump(data, f, indent=2, default=str)
    except Exception as e:
        print(f"Log save error: {e}")

def log_cycle(verdict, vote_count, weighted_votes, jane_approved, voters_for, voters_against, signal_summary):
    global session_stats
    session_stats["total"] += 1
    if verdict == "EXECUTE":
        session_stats["execute"] += 1
    elif verdict == "BLOCKED":
        session_stats["blocked"] += 1
    else:
        session_stats["pass"] += 1
    if jane_approved:
        session_stats["jane_approved"] += 1
    else:
        session_stats["jane_blocked"] += 1
    for name in voters_for:
        clean_name = name.split("(")[0].strip()
        if clean_name in session_stats["misfit_votes"]:
            session_stats["misfit_votes"][clean_name]["trade"] += 1
    for name in voters_against:
        clean_name = name.split("(")[0].strip()
        if clean_name in session_stats["misfit_votes"]:
            session_stats["misfit_votes"][clean_name]["pass"] += 1
    save_log()

def format_daily_scorecard(portfolio_state=None):
    total = session_stats["total"]
    if total == 0:
        return "No cycles recorded yet."

    execute = session_stats["execute"]
    blocked = session_stats["blocked"]
    passed = session_stats["pass"]
    jane_approved = session_stats["jane_approved"]
    jane_blocked = session_stats["jane_blocked"]

    execute_rate = execute / total * 100
    blocked_rate = blocked / total * 100
    pass_rate = passed / total * 100

    net_profit_line = ""
    if portfolio_state:
        pv = portfolio_state["portfolio_value"]
        net = pv - INCEPTION_VALUE
        net_pct = net / INCEPTION_VALUE * 100
        emoji = "📈" if net >= 0 else "📉"
        net_profit_line = f"\n{emoji} Net profit since inception: {'+' if net >= 0 else ''}${net:,.0f} ({'+' if net_pct >= 0 else ''}{net_pct:.2f}%)"

    lines = [f"📊 MISFITS DAILY SCORECARD\n"]
    lines.append(f"Total cycles run: {total}")
    lines.append(f"✅ Executed: {execute} ({execute_rate:.0f}%)")
    lines.append(f"⏭ Passed: {passed} ({pass_rate:.0f}%)")
    lines.append(f"🛑 Blocked by Jane Street: {blocked} ({blocked_rate:.0f}%)")
    lines.append(f"\nJane Street:")
    lines.append(f"  Approved: {jane_approved} of {total} signals")
    lines.append(f"  Vetoed: {jane_blocked} of {total} signals")

    lines.append(f"\nMisfit voting patterns:")
    for name, votes in session_stats["misfit_votes"].items():
        total_votes = votes["trade"] + votes["pass"]
        if total_votes > 0:
            rate = votes["trade"] / total_votes * 100
            weight = misfit_scorecard.get(name, {}).get("weight", 1.0)
            lines.append(f"  {name}: {votes['trade']}/{total_votes} TRADE votes ({rate:.0f}%) weight={weight:.1f}x")

    if trade_history:
        lines.append(f"\nRecent executions: {len([t for t in trade_history if t.get('executed')])}")

    lines.append(net_profit_line)
    lines.append("\n-- Satis House Consulting")
    return "\n".join(lines)

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

def format_trade_alert(ticker, direction, qty, price, stop_price, size_label, weighted_votes, voters_for, voters_against, reason):
    action = "Bought" if direction == "buy" else "Sold Short"
    total_bet = qty * price
    emoji = "🟢" if direction == "buy" else "🔴"
    against_text = f"\nWho disagreed: {', '.join(voters_against)}" if voters_against else ""
    return f"""{emoji} THE MISFITS JUST TRADED

{action} {qty} shares of {ticker} at ${price:.2f}
Bet size: ${total_bet:,.0f} ({size_label})
Stop loss set at: ${stop_price:.2f}

Why: {reason}

Who agreed: {', '.join(voters_for)}{against_text}
Jane Street approved the math.

-- Satis House Consulting"""

def format_crypto_alert(crypto_symbol, direction, notional, reason, weighted_votes, voters_for, voters_against):
    action = "Bought" if direction == "buy" else "Sold"
    emoji = "🟢" if direction == "buy" else "🔴"
    against_text = f"\nWho disagreed: {', '.join(voters_against)}" if voters_against else ""
    return f"""{emoji} THE MISFITS JUST TRADED CRYPTO

{action} ${notional:,.0f} of {crypto_symbol}
Markets are closed but crypto never sleeps.

Why: {reason}

Who agreed: {', '.join(voters_for)}{against_text}
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
            "type": "stop", "stop_price": str(round(stop_price, 2)), "time_in_force": "gtc"
        })
        if not result.get("id"):
            close_position(symbol)
            send_performance(f"⚠️ SAFETY CLOSE\n\nCould not set stop loss on {symbol}.\nPosition closed to protect capital.\n\n-- Satis House Consulting")
            return False
        return True
    except Exception as e:
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

def detect_environment():
    environment = {"energy_crisis": False, "credit_crisis": False, "currency_crisis": False, "market_crash": False}
    try:
        uso = yf.download("USO", period="30d", progress=False)["Close"].squeeze()
        if len(uso) > 5:
            uso_return = (uso.iloc[-1] - uso.iloc[0]) / uso.iloc[0]
            if abs(uso_return) > 0.15:
                environment["energy_crisis"] = True
        hyg = yf.download("HYG", period="30d", progress=False)["Close"].squeeze()
        if len(hyg) > 5:
            if (hyg.iloc[-1] - hyg.iloc[0]) / hyg.iloc[0] < -0.05:
                environment["credit_crisis"] = True
        spy = yf.download("SPY", period="30d", progress=False)["Close"].squeeze()
        if len(spy) > 5:
            if (spy.iloc[-1] - spy.iloc[0]) / spy.iloc[0] < -0.10:
                environment["market_crash"] = True
    except Exception as e:
        print(f"Environment detection error: {e}")
    return environment

def update_misfit_weights(environment):
    for name in misfit_scorecard:
        misfit_scorecard[name]["weight"] = 1.0
    if environment.get("energy_crisis"):
        misfit_scorecard["Andurand"]["weight"] = 2.0
    if environment.get("credit_crisis"):
        misfit_scorecard["Tepper"]["weight"] = 2.0
        misfit_scorecard["Druckenmiller"]["weight"] = 1.5
    if environment.get("currency_crisis"):
        misfit_scorecard["Soros"]["weight"] = 2.0
    if environment.get("market_crash"):
        misfit_scorecard["PTJ"]["weight"] = 2.0

def get_fred_data(series_id):
    try:
        url = "https://api.stlouisfed.org/fred/series/observations"
        params = {"series_id": series_id, "api_key": FRED_API_KEY, "file_type": "json", "limit": 5, "sort_order": "desc"}
        r = requests.get(url, params=params, timeout=10)
        obs = r.json().get("observations", [])
        if obs:
            val = obs[0]["value"]
            return float(val) if val != "." else None
        return None
    except:
        return None

def get_andurand_data():
    try:
        data = {}
        eia_url = f"https://api.eia.gov/v2/petroleum/stoc/wstk/data/?frequency=weekly&data[]=value&facets[series][]=W_EPC0_SAX_YCUOK_MBBL&sort[0][column]=period&sort[0][direction]=desc&length=4&api_key={EIA_API_KEY}"
        r = requests.get(eia_url, timeout=10).json()
        obs = r.get("response", {}).get("data", [])
        if obs and len(obs) >= 2:
            latest = float(obs[0].get("value", 0))
            prior = float(obs[1].get("value", 0))
            data["cushing_draw_mbbl"] = latest - prior
            data["cushing_stocks_mbbl"] = latest

        with hormuz_lock:
            data["vessels_near_hormuz"] = len(hormuz_vessels)

        energy_tickers = {"USO": "WTI_proxy", "ERX": "Energy_2x", "XLE": "Energy_sector", "FRO": "Tankers"}
        for ticker, name in energy_tickers.items():
            try:
                price = yf.download(ticker, period="30d", progress=False)["Close"].squeeze()
                data[f"{name}_30d_change_pct"] = float((price.iloc[-1] - price.iloc[0]) / price.iloc[0] * 100)
            except:
                pass

        try:
            cot = requests.get("https://publicreporting.cftc.gov/resource/jun7-fc8e.json?$limit=3&$order=report_date_as_yyyy_mm_dd DESC&$where=contract_market_name=%27CRUDE OIL%27", timeout=10).json()
            if cot:
                net = int(cot[0].get("noncomm_positions_long_all", 0)) - int(cot[0].get("noncomm_positions_short_all", 0))
                data["crude_hedge_fund_net"] = net
        except:
            pass

        return data
    except Exception as e:
        print(f"Andurand data error: {e}")
        return {}

def get_tepper_data():
    try:
        data = {}
        hy = get_fred_data("BAMLH0A0HYM2")
        ig = get_fred_data("BAMLC0A0CM")
        cc = get_fred_data("DRCCLACBS")
        if hy: data["high_yield_spread"] = hy
        if ig: data["ig_spread"] = ig
        if cc: data["credit_card_delinquency"] = cc
        for mat, ticker in [("2Y", "^IRX"), ("10Y", "^TNX")]:
            try:
                y = yf.download(ticker, period="30d", progress=False)["Close"].squeeze()
                data[f"treasury_{mat}"] = float(y.iloc[-1])
                data[f"treasury_{mat}_30d_change"] = float(y.iloc[-1] - y.iloc[0])
            except:
                pass
        return data
    except Exception as e:
        print(f"Tepper data error: {e}")
        return {}

def get_soros_data():
    try:
        data = {}
        for pair, ticker in [("EUR_USD", "EURUSD=X"), ("GBP_USD", "GBPUSD=X"), ("EEM", "EEM")]:
            try:
                price = yf.download(ticker, period="30d", progress=False)["Close"].squeeze()
                data[f"{pair}_30d_change_pct"] = float((price.iloc[-1] - price.iloc[0]) / price.iloc[0] * 100)
            except:
                pass
        usdx = get_fred_data("DTWEXBGS")
        if usdx: data["dollar_index"] = usdx
        return data
    except Exception as e:
        print(f"Soros data error: {e}")
        return {}

def get_druckenmiller_data():
    try:
        data = {}
        hy = get_fred_data("BAMLH0A0HYM2")
        fed = get_fred_data("WALCL")
        if hy: data["high_yield_spread"] = hy
        if fed: data["fed_balance_sheet_billions"] = fed / 1000
        for asset in ["SPY", "QQQ", "HYG", "TLT"]:
            try:
                price = yf.download(asset, period="60d", progress=False)["Close"].squeeze()
                data[f"{asset}_momentum_20d"] = float((price.iloc[-1] - price.iloc[-20]) / price.iloc[-20] * 100)
            except:
                pass
        return data
    except Exception as e:
        print(f"Druckenmiller data error: {e}")
        return {}

def get_ptj_data():
    try:
        data = {}
        vix = yf.download("^VIX", period="60d", progress=False)["Close"].squeeze()
        data["vix_current"] = float(vix.iloc[-1])
        data["vix_30d_avg"] = float(vix.rolling(30).mean().iloc[-1])
        data["vix_regime"] = "ELEVATED" if float(vix.iloc[-1]) > float(vix.rolling(30).mean().iloc[-1]) else "NORMAL"
        for ticker in ["SPY", "QQQ"]:
            try:
                df = yf.download(ticker, period="60d", progress=False)
                close = df["Close"].squeeze()
                vol = df["Volume"].squeeze()
                data[f"{ticker}_volume_ratio"] = float(vol.iloc[-1] / vol.rolling(20).mean().iloc[-1])
                data[f"{ticker}_vs_sma50"] = float((close.iloc[-1] - close.rolling(50).mean().iloc[-1]) / close.rolling(50).mean().iloc[-1] * 100)
            except:
                pass
        return data
    except Exception as e:
        print(f"PTJ data error: {e}")
        return {}

def start_aisstream():
    def on_message(ws, message):
        try:
            data = json.loads(message)
            if data.get("MessageType") == "PositionReport":
                meta = data.get("MetaData", {})
                pos = data.get("Message", {}).get("PositionReport", {})
                with hormuz_lock:
                    hormuz_vessels.append({
                        "name": meta.get("ShipName", "Unknown"),
                        "lat": pos.get("Latitude", 0),
                        "lon": pos.get("Longitude", 0),
                        "speed": pos.get("Sog", 0),
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
        print("AISStream closed -- reconnecting in 60s")
        time.sleep(60)
        start_aisstream()

    def run():
        ws = websocket.WebSocketApp("wss://stream.aisstream.io/v0/stream",
            on_open=on_open, on_message=on_message, on_error=on_error, on_close=on_close)
        ws.run_forever()

    threading.Thread(target=run, daemon=True).start()
    print("AISStream monitoring Hormuz")

def check_execution_rules(ticker, direction, position_size, portfolio_state, vix):
    global trades_halted_today, daily_start_value, orders_this_cycle
    if orders_this_cycle >= MAX_ORDERS_PER_CYCLE:
        return False, "Max orders per cycle"
    if trades_halted_today:
        return False, "Daily loss limit hit"
    if portfolio_state is None:
        return False, "Portfolio unavailable"
    portfolio_value = portfolio_state["portfolio_value"]
    if daily_start_value is not None:
        daily_pnl = (portfolio_value - daily_start_value) / daily_start_value
        if daily_pnl <= -DAILY_LOSS_LIMIT:
            trades_halted_today = True
            send_performance(f"⚡ CIRCUIT BREAKER\n\nPortfolio down {abs(daily_pnl)*100:.1f}% today.\nAll trading paused until tomorrow.\n\n-- Satis House Consulting")
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
        return False, "Market closed"
    if is_leveraged and portfolio_state["leveraged_pct"] >= MAX_LEVERAGED_PCT:
        return False, "Leveraged cap reached"
    elif is_crypto and portfolio_state["crypto_pct"] >= MAX_CRYPTO_PCT:
        return False, "Crypto cap reached"
    elif not is_crypto and not is_leveraged and portfolio_state["equity_pct"] >= MAX_EQUITY_PCT:
        return False, "Equity cap reached"
    return True, "Clear"

def get_position_size(weighted_votes, portfolio_value, vix, high_conviction_rotation=False):
    if high_conviction_rotation:
        base_pct = ROTATION_SIZE
        label = "15% of portfolio -- rotation conviction"
    elif weighted_votes >= 8.0:
        base_pct = 0.10
        label = "10% of portfolio -- maximum conviction"
    elif weighted_votes >= 6.0:
        base_pct = 0.08
        label = "8% of portfolio -- high conviction"
    else:
        base_pct = 0.05
        label = "5% of portfolio -- base conviction"
    if vix >= VIX_REDUCE_THRESHOLD:
        base_pct *= 0.5
        label = f"reduced to {base_pct*100:.0f}% -- VIX at {vix:.0f}"
    size = min(portfolio_value * base_pct, portfolio_value * MAX_SINGLE_POSITION)
    return size, label

def update_training_loop(ticker, voters_for, voters_against, outcome_pnl_pct):
    trade_was_profitable = outcome_pnl_pct > 0
    for name in voters_for:
        clean = name.split("(")[0].strip()
        if clean in misfit_scorecard:
            misfit_scorecard[clean]["total"] += 1
            if trade_was_profitable:
                misfit_scorecard[clean]["correct"] += 1
    for name in voters_against:
        clean = name.split("(")[0].strip()
        if clean in misfit_scorecard:
            misfit_scorecard[clean]["total"] += 1
            if not trade_was_profitable:
                misfit_scorecard[clean]["correct"] += 1
    save_log()

def build_scorecard_context():
    lines = []
    for name, scores in misfit_scorecard.items():
        total = scores["total"]
        weight = scores["weight"]
        if total > 0:
            win_rate = scores["correct"] / total * 100
            lines.append(f"{name}: {win_rate:.0f}% win rate ({scores['correct']}/{total}) weight={weight:.1f}x")
        else:
            lines.append(f"{name}: no trades yet weight={weight:.1f}x")
    return "MISFIT TRACK RECORD:\n" + "\n".join(lines) if lines else ""

def calculate_weighted_votes(verdicts):
    weighted_total = 0
    voters_for = []
    voters_against = []
    for name, verdict in verdicts:
        weight = misfit_scorecard.get(name, {}).get("weight", 1.0)
        if "VOTE: TRADE" in verdict:
            weighted_total += weight
            voters_for.append(f"{name}({weight:.1f}x)")
        else:
            voters_against.append(name)
    return weighted_total, voters_for, voters_against

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
    for ticker in ["TQQQ", "SOXL", "TECL", "FAS", "ERX", "XLE", "USO", "BNO",
                   "UNG", "GLD", "SLV", "TLT", "HYG", "QQQ", "SPY", "IWM", "EEM",
                   "FXE", "FXB", "UUP", "VLO", "FRO", "XOP"]:
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
        if any(word in line.lower() for word in ["because", "thesis", "confirmed", "why", "momentum", "breakout", "oversold", "overbought", "physical", "crisis", "pivot"]):
            return line.strip()
    return lines[0].strip() if lines else "Multiple indicators confirmed"

def execute_trade(signal, weighted_votes, verdicts, portfolio_state, vix, rotation_ticker=None, high_conviction_rotation=False):
    global recent_signals, trade_history, orders_this_cycle
    if portfolio_state is None:
        return "Portfolio unavailable"
    portfolio_value = portfolio_state["portfolio_value"]
    position_size, size_label = get_position_size(weighted_votes, portfolio_value, vix, high_conviction_rotation)
    direction = extract_direction(signal)
    reason = extract_reason(signal)
    market_open = is_market_hours()
    results = []
    voters_for = [name for name, verdict in verdicts if "VOTE: TRADE" in verdict]
    voters_against = [name for name, verdict in verdicts if "VOTE: PASS" in verdict]

    crypto_asset = extract_crypto(signal)
    if crypto_asset and orders_this_cycle < MAX_ORDERS_PER_CYCLE:
        crypto_map = {"BTC": "BTC/USD", "BITCOIN": "BTC/USD", "ETH": "ETH/USD", "ETHEREUM": "ETH/USD", "SOL": "SOL/USD", "SOLANA": "SOL/USD"}
        crypto_symbol = crypto_map.get(crypto_asset.upper())
        if crypto_symbol:
            approved, block_reason = check_execution_rules(crypto_symbol, direction, position_size, portfolio_state, vix)
            if approved:
                try:
                    notional = min(position_size, 5000)
                    result = alpaca_request("POST", "/v2/orders", {"symbol": crypto_symbol, "notional": str(round(notional, 2)), "side": direction, "type": "market", "time_in_force": "gtc"})
                    orders_this_cycle += 1
                    send_performance(format_crypto_alert(crypto_symbol, direction, notional, reason, weighted_votes, voters_for, voters_against))
                    recent_signals[f"{crypto_symbol}_{direction}"] = 0
                    trade_history.append({"ticker": crypto_symbol, "direction": direction, "entry_price": None, "voters_for": voters_for, "voters_against": voters_against, "executed": True, "timestamp": datetime.now(pytz.utc).isoformat()})
                    results.append(f"Crypto: {crypto_symbol}")
                except Exception as e:
                    results.append(f"Crypto failed: {e}")
            else:
                results.append(f"Crypto blocked: {block_reason}")

    if market_open and orders_this_cycle < MAX_ORDERS_PER_CYCLE:
        ticker = extract_ticker(signal, rotation_ticker)
        if ticker:
            approved, block_reason = check_execution_rules(ticker, direction, position_size, portfolio_state, vix)
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
                        results.append(f"{ticker} order did not fill")
                    else:
                        orders_this_cycle += 1
                        stop_price = price * 0.95 if direction == "buy" else price * 1.05
                        stop_side = "sell" if direction == "buy" else "buy"
                        stop_ok = submit_stop_loss_atomic(ticker, qty, stop_side, stop_price)
                        if stop_ok:
                            send_performance(format_trade_alert(ticker, direction, qty, price, stop_price, size_label, weighted_votes, voters_for, voters_against, reason))
                            recent_signals[f"{ticker}_{direction}"] = 0
                            trade_history.append({"ticker": ticker, "direction": direction, "entry_price": price, "voters_for": voters_for, "voters_against": voters_against, "executed": True, "timestamp": datetime.now(pytz.utc).isoformat()})
                            results.append(f"Equity: {ticker} filled and protected")
                        else:
                            results.append(f"Equity: {ticker} -- stop failed, position closed")
                except Exception as e:
                    results.append(f"Equity failed: {e}")
            else:
                results.append(f"Equity blocked: {block_reason}")
    return "\n".join(results) if results else "No trades executed"

def ask_misfit(name, persona, signal, specific_data="", knowledge=""):
    scorecard = build_scorecard_context()
    weight = misfit_scorecard.get(name, {}).get("weight", 1.0)
    weight_note = f"\nYour environment weight is {weight:.1f}x -- this is YOUR moment, trust your conviction." if weight > 1.0 else ""
    data_context = f"\n\nYOUR MARKET DATA:\n{specific_data}" if specific_data else ""
    knowledge_ctx = f"\n\nKNOWLEDGE:\n{knowledge}" if knowledge else ""
    score_ctx = f"\n\nTRACK RECORD:\n{scorecard}" if scorecard else ""
    msg = client.messages.create(
        model="claude-opus-4-5-20251101", max_tokens=400,
        messages=[{"role": "user", "content": f"""{persona}{weight_note}{data_context}{knowledge_ctx}{score_ctx}

YoniBot signal: {signal}

Use your data. If you see a better trade across any asset class say so. 2-3 sentences. Brutal and direct. End with VOTE: TRADE or VOTE: PASS."""}]
    )
    return msg.content[0].text

def ask_jane_street(signal, verdicts, weighted_votes):
    debate = "\n\n".join([f"{n}:\n{v}" for n, v in verdicts])
    scorecard = build_scorecard_context()
    msg = client.messages.create(
        model="claude-opus-4-5-20251101", max_tokens=300,
        messages=[{"role": "user", "content": f"""You ARE Jane Street quant engine. Position sizer and risk gatekeeper only.

TRACK RECORD: {scorecard}
WEIGHTED VOTES: {weighted_votes:.1f} (threshold 3.0)
SIGNAL: {signal}
DEBATE: {debate}

Rules: APPROVE if stop loss is defined AND risk reward is 2 to 1 or better AND weighted votes above 3.0. BLOCK only if downside is undefined or risk reward is negative.

Output: win probability, risk reward, Kelly fraction, position size recommendation, one sentence. End with VETO: BLOCKED or VETO: APPROVED."""}]
    )
    return msg.content[0].text

MISFITS = [
    ("Soros",
     "You ARE George Soros. Find the hidden peg. Where is the lie everyone believes and when do the defenders run out of ammunition? Asset universe: FX, sovereign bonds, EM currencies, currency options.",
     ["George Soros Black Wednesday 1992 mechanics reflexivity",
      "Soros Fund Management macro sovereign currency views 2025 2026"]),
    ("Druckenmiller",
     "You ARE Stanley Druckenmiller. Stop first, target second, size third. Concentrate into your best idea. Asset universe: equities long and short, bonds, macro, leveraged ETFs.",
     ["Stanley Druckenmiller concentration asymmetric bet methodology",
      "Druckenmiller macro views credit Federal Reserve 2025 2026"]),
    ("PTJ",
     "You ARE Paul Tudor Jones. Only 5 to 1 or better. Asset universe: equity indices, options, volatility, trend following across all asset classes.",
     ["Paul Tudor Jones 5 to 1 risk reward rules Black Monday",
      "PTJ macro views technical analysis trend following 2025 2026"]),
    ("Tepper",
     "You ARE David Tepper. Read the Federal Reserve before the market does. Asset universe: equities, high yield, investment grade, sovereign bonds, bank stocks.",
     ["David Tepper 2009 bank trade Federal Reserve reading",
      "Tepper macro views credit Fed policy 2025 2026"]),
    ("Andurand",
     "You ARE Pierre Andurand. Physical markets lead paper always. Asset universe: energy equities, energy ETFs, oil-linked currencies, sovereign bonds of oil exporters, shipping stocks, refiners.",
     ["Pierre Andurand physical commodity flows Hormuz 2008 2022",
      "Andurand Capital oil energy views 2025 2026"]),
]

misfit_knowledge_cache = {}
misfit_data_cache = {}
knowledge_refresh_cycles = 8
cycle_count = 0
daily_scorecard_sent = False

def send_startup_message():
    send_performance("""🚀 MISFITS SYSTEM ONLINE

The Misfits are watching the markets.

Daily scorecard fires every morning at 9 AM ET showing:
- How many signals were executed, passed, and blocked
- Which Misfit voted correctly
- Jane Street approval rate
- Net profit since inception

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
        scorecard_msg = format_daily_scorecard(portfolio_state)
        send_performance(scorecard_msg)
        daily_scorecard_sent = True
        print("Daily scorecard sent")

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
        print("Refreshing Misfit knowledge and data...")
        data_funcs = {
            "Soros": get_soros_data,
            "Druckenmiller": get_druckenmiller_data,
            "PTJ": get_ptj_data,
            "Tepper": get_tepper_data,
            "Andurand": get_andurand_data
        }
        for name, persona, queries in MISFITS:
            print(f"  Refreshing {name}...")
            blocks = []
            for q in queries:
                try:
                    results = exa.search_and_contents(q, num_results=2, text={"max_characters": 400})
                    for r in results.results:
                        blocks.append(f"SOURCE: {r.title}\n{r.text[:300]}")
                    time.sleep(1)
                except:
                    pass
            misfit_knowledge_cache[name] = "\n\n".join(blocks)
            misfit_data_cache[name] = data_funcs[name]()
            time.sleep(2)

    market_open = is_market_hours()
    rotation_ticker, rotation_score, rotation_summary, _ = run_omniscient_rotation()
    high_conviction_rotation = rotation_ticker not in ["BIL", "UUP", None] and rotation_score > 0.5 and market_open

    scorecard = build_scorecard_context()
    env_active = [k for k, v in environment.items() if v]
    portfolio_summary = ""
    if portfolio_state:
        portfolio_summary = f"""
PORTFOLIO: ${portfolio_state['portfolio_value']:,.2f}
Equity: {portfolio_state['equity_pct']*100:.1f}% | Crypto: {portfolio_state['crypto_pct']*100:.1f}% | Leveraged: {portfolio_state['leveraged_pct']*100:.1f}%
Positions: {list(portfolio_state['positions'].keys()) or 'None'}
VIX: {vix:.1f}
Active environments: {', '.join(env_active) if env_active else 'Standard'}
Session stats: {session_stats['execute']} executed / {session_stats['pass']} passed / {session_stats['blocked']} blocked"""

    andurand_summary = ""
    if misfit_data_cache.get("Andurand"):
        a = misfit_data_cache["Andurand"]
        vessels = a.get("vessels_near_hormuz", 0)
        cushing = a.get("cushing_draw_mbbl", 0)
        andurand_summary = f"\nANDURAND INTEL: {vessels} vessels near Hormuz | Cushing draw: {cushing:.1f}M bbls"

    yoni_push = f"\nHIGH CONVICTION ROTATION: {rotation_ticker} score {rotation_score:.3f}" if high_conviction_rotation else ""
    friday_note = "\nFRIDAY: No new short signals." if is_friday_short_blocked() else ""
    weekend_note = "" if market_open else "\nMARKET CLOSED. Crypto signals only."

    context = f"""MARKET: {'OPEN' if market_open else 'CLOSED'}
{portfolio_summary}
{scorecard}
{andurand_summary}

OMNISCIENT ROTATION:
{rotation_summary or 'Unavailable'}
{yoni_push}"""

    yoni = client.messages.create(
        model="claude-opus-4-5-20251101", max_tokens=1024,
        messages=[{"role": "user", "content": f"""You are YoniBot, autonomous trading intelligence for Satis House Consulting.

{context}
{weekend_note}
{friday_note}

Scan ALL asset classes: equities, leveraged ETFs, crypto, FX, bonds, commodities, energy, shipping.
Generate the single best trade signal. Include explicit stop loss and target for risk reward calculation.
Market closed means crypto only. No shorts on Fridays after 2 PM ET. Say NO SIGNAL only if nothing qualifies.
Output: Asset, Direction, Entry, Stop, Target, Risk Reward, Why. Max 300 words."""}]
    )
    signal = yoni.content[0].text

    verdicts = []
    for name, persona, queries in MISFITS:
        knowledge = misfit_knowledge_cache.get(name, "")
        specific_data = json.dumps(misfit_data_cache.get(name, {}), default=str)[:1000]
        verdict = ask_misfit(name, persona, signal, specific_data, knowledge)
        verdicts.append((name, verdict))

    weighted_votes, voters_for, voters_against = calculate_weighted_votes(verdicts)
    jane = ask_jane_street(signal, verdicts, weighted_votes)
    approved = "VETO: APPROVED" in jane
    majority = weighted_votes >= 3.0

    if majority and approved:
        trade_result = execute_trade(signal, weighted_votes, verdicts, portfolio_state, vix,
                                     rotation_ticker if high_conviction_rotation else None,
                                     high_conviction_rotation)
        verdict_line = f"VERDICT: EXECUTE -- weighted votes {weighted_votes:.1f}\n{trade_result}"
        log_cycle("EXECUTE", 0, weighted_votes, approved, voters_for, voters_against, signal[:100])
    elif not approved:
        verdict_line = f"VERDICT: BLOCKED -- Jane Street vetoed (weighted votes: {weighted_votes:.1f})"
        log_cycle("BLOCKED", 0, weighted_votes, approved, voters_for, voters_against, signal[:100])
    else:
        verdict_line = f"VERDICT: PASS -- weighted votes {weighted_votes:.1f} below threshold"
        log_cycle("PASS", 0, weighted_votes, approved, voters_for, voters_against, signal[:100])

    send_telegram(f"YONIBOT SIGNAL\n{signal}")
    time.sleep(2)
    debate_msg = "THE MISFITS DEBATE\n"
    for name, verdict in verdicts:
        weight = misfit_scorecard.get(name, {}).get("weight", 1.0)
        debate_msg += f"\n{name.upper()} ({weight:.1f}x):\n{verdict}\n"
    send_telegram(debate_msg)
    time.sleep(2)
    send_telegram(f"JANE STREET:\n{jane}\n\n{verdict_line}")
    print(f"Cycle {cycle_count}: {verdict_line[:80]}")

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
