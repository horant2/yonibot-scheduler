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

INCEPTION_VALUE = 100000

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
exa = Exa(api_key=EXA_API_KEY)

MAX_ORDERS_PER_CYCLE = 2
DAILY_LOSS_LIMIT = 0.03
FRIDAY_SHORT_CUTOFF_HOUR = 14
FRIDAY_SHORT_CLOSE_HOUR = 15
FRIDAY_SHORT_CLOSE_MINUTE = 30

CONVICTION_SIZING = {5: 0.10, 4: 0.08, 3: 0.05}
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

trade_history = []

def detect_environment():
    environment = {
        "energy_crisis": False,
        "credit_crisis": False,
        "currency_crisis": False,
        "fed_pivot": False,
        "market_crash": False,
        "tech_breakout": False
    }
    try:
        uso = yf.download("USO", period="30d", progress=False)["Close"].squeeze()
        uso_return = (uso.iloc[-1] - uso.iloc[0]) / uso.iloc[0]
        if uso_return > 0.15 or uso_return < -0.15:
            environment["energy_crisis"] = True

        hyg = yf.download("HYG", period="30d", progress=False)["Close"].squeeze()
        hyg_return = (hyg.iloc[-1] - hyg.iloc[0]) / hyg.iloc[0]
        if hyg_return < -0.05:
            environment["credit_crisis"] = True

        uup = yf.download("UUP", period="30d", progress=False)["Close"].squeeze()
        uup_return = (uup.iloc[-1] - uup.iloc[0]) / uup.iloc[0]
        if abs(uup_return) > 0.05:
            environment["currency_crisis"] = True

        spy = yf.download("SPY", period="30d", progress=False)["Close"].squeeze()
        spy_return = (spy.iloc[-1] - spy.iloc[0]) / spy.iloc[0]
        if spy_return < -0.10:
            environment["market_crash"] = True

        qqq = yf.download("QQQ", period="30d", progress=False)["Close"].squeeze()
        qqq_return = (qqq.iloc[-1] - qqq.iloc[0]) / qqq.iloc[0]
        if qqq_return > 0.10:
            environment["tech_breakout"] = True
    except Exception as e:
        print(f"Environment detection error: {e}")

    return environment

def update_misfit_weights(environment):
    global misfit_scorecard
    for name in misfit_scorecard:
        misfit_scorecard[name]["weight"] = 1.0

    if environment["energy_crisis"]:
        misfit_scorecard["Andurand"]["weight"] = 2.0
        print("ANDURAND elevated to 2.0x -- energy crisis detected")

    if environment["credit_crisis"]:
        misfit_scorecard["Tepper"]["weight"] = 2.0
        misfit_scorecard["Druckenmiller"]["weight"] = 1.5
        print("TEPPER elevated to 2.0x -- credit crisis detected")

    if environment["currency_crisis"]:
        misfit_scorecard["Soros"]["weight"] = 2.0
        print("SOROS elevated to 2.0x -- currency crisis detected")

    if environment["market_crash"]:
        misfit_scorecard["PTJ"]["weight"] = 2.0
        misfit_scorecard["Druckenmiller"]["weight"] = 1.5
        print("PTJ elevated to 2.0x -- market crash detected")

    if environment["fed_pivot"]:
        misfit_scorecard["Tepper"]["weight"] = 2.0
        print("TEPPER elevated to 2.0x -- Fed pivot detected")

def get_fred_data(series_id):
    try:
        url = f"https://api.stlouisfed.org/fred/series/observations"
        params = {
            "series_id": series_id,
            "api_key": FRED_API_KEY,
            "file_type": "json",
            "limit": 10,
            "sort_order": "desc"
        }
        r = requests.get(url, params=params, timeout=10)
        data = r.json()
        obs = data.get("observations", [])
        if obs:
            latest = obs[0]
            return float(latest["value"]) if latest["value"] != "." else None
        return None
    except Exception as e:
        print(f"FRED error {series_id}: {e}")
        return None

def get_soros_data():
    try:
        data = {}
        usdx = get_fred_data("DTWEXBGS")
        if usdx:
            data["dollar_index"] = usdx

        eur_usd = yf.download("EURUSD=X", period="30d", progress=False)["Close"].squeeze()
        gbp_usd = yf.download("GBPUSD=X", period="30d", progress=False)["Close"].squeeze()
        jpy_usd = yf.download("JPY=X", period="30d", progress=False)["Close"].squeeze()
        em_fx = yf.download("EEM", period="30d", progress=False)["Close"].squeeze()

        data["eur_usd_30d_change"] = float((eur_usd.iloc[-1] - eur_usd.iloc[0]) / eur_usd.iloc[0] * 100)
        data["gbp_usd_30d_change"] = float((gbp_usd.iloc[-1] - gbp_usd.iloc[0]) / gbp_usd.iloc[0] * 100)
        data["em_equities_30d_change"] = float((em_fx.iloc[-1] - em_fx.iloc[0]) / em_fx.iloc[0] * 100)

        try:
            cot_url = "https://publicreporting.cftc.gov/resource/jun7-fc8e.json?$limit=5&$order=report_date_as_yyyy_mm_dd DESC&$where=contract_market_name=%27EURO FX%27"
            cot_data = requests.get(cot_url, timeout=10).json()
            if cot_data:
                latest = cot_data[0]
                net_noncom = int(latest.get("noncomm_positions_long_all", 0)) - int(latest.get("noncomm_positions_short_all", 0))
                data["euro_fx_hedge_fund_net"] = net_noncom
        except:
            pass

        sovereign_etfs = {
            "Italy": "ITLY",
            "Brazil": "EWZ",
            "Turkey": "TUR",
            "South Africa": "EZA"
        }
        stress = []
        for country, etf in sovereign_etfs.items():
            try:
                price = yf.download(etf, period="30d", progress=False)["Close"].squeeze()
                chg = float((price.iloc[-1] - price.iloc[0]) / price.iloc[0] * 100)
                if chg < -10:
                    stress.append(f"{country}: {chg:.1f}%")
            except:
                pass
        if stress:
            data["sovereign_stress"] = stress

        exa_data = exa.search_and_contents(
            "central bank currency intervention capital flight sovereign debt crisis 2026",
            num_results=3, text={"max_characters": 400}
        )
        data["currency_intelligence"] = "\n".join([f"{r.title}: {r.text[:300]}" for r in exa_data.results])

        return data
    except Exception as e:
        print(f"Soros data error: {e}")
        return {}

def get_druckenmiller_data():
    try:
        data = {}
        hyg_spread = get_fred_data("BAMLH0A0HYM2")
        if hyg_spread:
            data["high_yield_spread_pct"] = hyg_spread

        fed_assets = get_fred_data("WALCL")
        if fed_assets:
            data["fed_balance_sheet_billions"] = fed_assets / 1000

        credit_delinquency = get_fred_data("DRCCLACBS")
        if credit_delinquency:
            data["credit_card_delinquency_pct"] = credit_delinquency

        commercial_lending = get_fred_data("TOTCI")
        if commercial_lending:
            data["commercial_lending_billions"] = commercial_lending

        assets = ["SPY", "QQQ", "IWM", "HYG", "TLT", "GLD"]
        for asset in assets:
            try:
                price = yf.download(asset, period="60d", progress=False)["Close"].squeeze()
                data[f"{asset}_momentum"] = float((price.iloc[-1] - price.iloc[-20]) / price.iloc[-20] * 100)
            except:
                pass

        try:
            cot_url = "https://publicreporting.cftc.gov/resource/jun7-fc8e.json?$limit=5&$order=report_date_as_yyyy_mm_dd DESC&$where=contract_market_name=%27E-MINI S%26P 500%27"
            cot_data = requests.get(cot_url, timeout=10).json()
            if cot_data:
                latest = cot_data[0]
                net = int(latest.get("noncomm_positions_long_all", 0)) - int(latest.get("noncomm_positions_short_all", 0))
                data["sp500_hedge_fund_net_position"] = net
        except:
            pass

        exa_data = exa.search_and_contents(
            "earnings revisions credit cycle Federal Reserve balance sheet macro 2026",
            num_results=3, text={"max_characters": 400}
        )
        data["macro_intelligence"] = "\n".join([f"{r.title}: {r.text[:300]}" for r in exa_data.results])

        return data
    except Exception as e:
        print(f"Druckenmiller data error: {e}")
        return {}

def get_ptj_data():
    try:
        data = {}
        vix_data = yf.download("^VIX", period="60d", progress=False)["Close"].squeeze()
        vix9d = yf.download("^VIX9D", period="5d", progress=False)["Close"].squeeze()
        vix3m = yf.download("^VIX3M", period="5d", progress=False)["Close"].squeeze()

        data["vix_current"] = float(vix_data.iloc[-1])
        data["vix_30d_avg"] = float(vix_data.rolling(30).mean().iloc[-1])

        try:
            if len(vix9d) > 0 and len(vix3m) > 0:
                data["vix_term_structure"] = "CONTANGO" if float(vix3m.iloc[-1]) > float(vix9d.iloc[-1]) else "BACKWARDATION"
                data["vix9d"] = float(vix9d.iloc[-1])
                data["vix3m"] = float(vix3m.iloc[-1])
        except:
            pass

        for ticker in ["SPY", "QQQ", "IWM"]:
            try:
                df = yf.download(ticker, period="60d", progress=False)
                close = df["Close"].squeeze()
                volume = df["Volume"].squeeze()
                avg_vol = float(volume.rolling(20).mean().iloc[-1])
                current_vol = float(volume.iloc[-1])
                data[f"{ticker}_volume_ratio"] = current_vol / avg_vol if avg_vol > 0 else 1.0

                sma50 = float(close.rolling(50).mean().iloc[-1])
                sma200 = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else sma50
                price = float(close.iloc[-1])
                data[f"{ticker}_vs_sma50"] = (price - sma50) / sma50 * 100
                data[f"{ticker}_golden_cross"] = price > sma50 > sma200
            except:
                pass

        pcr_proxy = yf.download("UVXY", period="30d", progress=False)["Close"].squeeze()
        data["fear_proxy_30d_change"] = float((pcr_proxy.iloc[-1] - pcr_proxy.iloc[0]) / pcr_proxy.iloc[0] * 100)

        exa_data = exa.search_and_contents(
            "options flow gamma squeeze technical breakout market structure 2026",
            num_results=3, text={"max_characters": 400}
        )
        data["technical_intelligence"] = "\n".join([f"{r.title}: {r.text[:300]}" for r in exa_data.results])

        return data
    except Exception as e:
        print(f"PTJ data error: {e}")
        return {}

def get_tepper_data():
    try:
        data = {}
        hyg_spread = get_fred_data("BAMLH0A0HYM2")
        ig_spread = get_fred_data("BAMLC0A0CM")
        if hyg_spread:
            data["high_yield_spread"] = hyg_spread
        if ig_spread:
            data["investment_grade_spread"] = ig_spread
        if hyg_spread and ig_spread:
            data["hy_ig_differential"] = hyg_spread - ig_spread

        credit_card_del = get_fred_data("DRCCLACBS")
        mortgage_del = get_fred_data("DRSFRMACBS")
        if credit_card_del:
            data["credit_card_delinquency"] = credit_card_del
        if mortgage_del:
            data["mortgage_delinquency"] = mortgage_del

        for maturity, ticker in [("2Y", "^IRX"), ("10Y", "^TNX"), ("30Y", "^TYX")]:
            try:
                y = yf.download(ticker, period="30d", progress=False)["Close"].squeeze()
                data[f"treasury_{maturity}"] = float(y.iloc[-1])
                data[f"treasury_{maturity}_30d_change"] = float(y.iloc[-1] - y.iloc[0])
            except:
                pass

        try:
            cot_url = "https://publicreporting.cftc.gov/resource/jun7-fc8e.json?$limit=5&$order=report_date_as_yyyy_mm_dd DESC&$where=contract_market_name=%2710-YEAR T-NOTES%27"
            cot_data = requests.get(cot_url, timeout=10).json()
            if cot_data:
                latest = cot_data[0]
                net = int(latest.get("noncomm_positions_long_all", 0)) - int(latest.get("noncomm_positions_short_all", 0))
                data["treasury_hedge_fund_net_position"] = net
        except:
            pass

        exa_data = exa.search_and_contents(
            "Federal Reserve policy pivot credit market high yield bonds 2026",
            num_results=3, text={"max_characters": 400}
        )
        data["fed_intelligence"] = "\n".join([f"{r.title}: {r.text[:300]}" for r in exa_data.results])

        return data
    except Exception as e:
        print(f"Tepper data error: {e}")
        return {}

def get_eia_data():
    try:
        data = {}
        base = "https://api.eia.gov/v2"
        headers = {"X-Params": json.dumps({"api_key": EIA_API_KEY})}

        endpoints = {
            "cushing_stocks": f"{base}/petroleum/stoc/wstk/data/?frequency=weekly&data[]=value&facets[series][]=W_EPC0_SAX_YCUOK_MBBL&sort[0][column]=period&sort[0][direction]=desc&length=4&api_key={EIA_API_KEY}",
            "refinery_utilization": f"{base}/petroleum/pnp/wiup/data/?frequency=weekly&data[]=value&facets[series][]=WCRFPUS2&sort[0][column]=period&sort[0][direction]=desc&length=4&api_key={EIA_API_KEY}",
            "crude_production": f"{base}/petroleum/crd/crpdn/data/?frequency=weekly&data[]=value&facets[series][]=WCRFPUS2&sort[0][column]=period&sort[0][direction]=desc&length=4&api_key={EIA_API_KEY}",
            "spr_stocks": f"{base}/petroleum/stoc/wstk/data/?frequency=weekly&data[]=value&facets[series][]=W_EPC0_SAX_YSPR_MBBL&sort[0][column]=period&sort[0][direction]=desc&length=4&api_key={EIA_API_KEY}"
        }

        for name, url in endpoints.items():
            try:
                r = requests.get(url, timeout=10)
                result = r.json()
                obs = result.get("response", {}).get("data", [])
                if obs and len(obs) >= 2:
                    latest = float(obs[0].get("value", 0))
                    prior = float(obs[1].get("value", 0))
                    change = latest - prior
                    data[name] = {"latest": latest, "prior": prior, "change": change}
            except Exception as e:
                print(f"EIA endpoint error {name}: {e}")

        crack_spread = {}
        try:
            uso = yf.download("USO", period="5d", progress=False)["Close"].squeeze().iloc[-1]
            rbob = yf.download("UGA", period="5d", progress=False)["Close"].squeeze().iloc[-1]
            heat = yf.download("UHN", period="5d", progress=False)["Close"].squeeze().iloc[-1]
            data["gasoline_crack_proxy"] = float(rbob - uso)
            data["heating_oil_crack_proxy"] = float(heat - uso)
        except:
            pass

        return data
    except Exception as e:
        print(f"EIA data error: {e}")
        return {}

def get_hormuz_intelligence():
    global hormuz_vessels
    try:
        vessel_snapshot = []
        with hormuz_lock:
            vessel_snapshot = list(hormuz_vessels[-20:])

        exa_data = exa.search_and_contents(
            "Strait of Hormuz tanker shipping Iran blockade oil vessel 2026",
            num_results=5, text={"max_characters": 500}
        )
        hormuz_news = "\n".join([f"{r.title}: {r.text[:400]}" for r in exa_data.results])

        opec_data = exa.search_and_contents(
            "OPEC production quota compliance Saudi Arabia oil output 2026",
            num_results=3, text={"max_characters": 400}
        )
        opec_news = "\n".join([f"{r.title}: {r.text[:300]}" for r in opec_data.results])

        return {
            "recent_vessels_near_hormuz": vessel_snapshot,
            "vessel_count": len(vessel_snapshot),
            "hormuz_intelligence": hormuz_news,
            "opec_intelligence": opec_news
        }
    except Exception as e:
        print(f"Hormuz intelligence error: {e}")
        return {}

def get_andurand_data():
    try:
        data = {}
        eia = get_eia_data()
        data["eia"] = eia

        hormuz = get_hormuz_intelligence()
        data["hormuz"] = hormuz

        energy_tickers = {
            "WTI_proxy": "USO",
            "Brent_proxy": "BNO",
            "Natural_gas": "UNG",
            "Energy_sector": "XLE",
            "Energy_2x": "ERX",
            "Refiners": "VLO",
            "Tankers": "FRO",
            "LNG": "TELL"
        }
        for name, ticker in energy_tickers.items():
            try:
                price = yf.download(ticker, period="30d", progress=False)["Close"].squeeze()
                data[f"{name}_price"] = float(price.iloc[-1])
                data[f"{name}_30d_change"] = float((price.iloc[-1] - price.iloc[0]) / price.iloc[0] * 100)
            except:
                pass

        try:
            cot_url = "https://publicreporting.cftc.gov/resource/jun7-fc8e.json?$limit=5&$order=report_date_as_yyyy_mm_dd DESC&$where=contract_market_name=%27CRUDE OIL%27"
            cot_data = requests.get(cot_url, timeout=10).json()
            if cot_data:
                latest = cot_data[0]
                net = int(latest.get("noncomm_positions_long_all", 0)) - int(latest.get("noncomm_positions_short_all", 0))
                data["crude_hedge_fund_net_position"] = net
        except:
            pass

        oil_currencies = {
            "Saudi_riyal": "SAR=X",
            "Norwegian_krone": "NOK=X",
            "Canadian_dollar": "CAD=X",
            "Russian_ruble": "RUB=X"
        }
        for name, ticker in oil_currencies.items():
            try:
                fx = yf.download(ticker, period="30d", progress=False)["Close"].squeeze()
                data[f"{name}_30d_change"] = float((fx.iloc[-1] - fx.iloc[0]) / fx.iloc[0] * 100)
            except:
                pass

        return data
    except Exception as e:
        print(f"Andurand data error: {e}")
        return {}

def start_aisstream():
    def on_message(ws, message):
        global hormuz_vessels
        try:
            data = json.loads(message)
            msg_type = data.get("MessageType", "")
            if msg_type == "PositionReport":
                metadata = data.get("MetaData", {})
                vessel_name = metadata.get("ShipName", "Unknown")
                mmsi = metadata.get("MMSI", "")
                lat = data.get("Message", {}).get("PositionReport", {}).get("Latitude", 0)
                lon = data.get("Message", {}).get("PositionReport", {}).get("Longitude", 0)
                speed = data.get("Message", {}).get("PositionReport", {}).get("Sog", 0)
                with hormuz_lock:
                    hormuz_vessels.append({
                        "name": vessel_name,
                        "mmsi": mmsi,
                        "lat": lat,
                        "lon": lon,
                        "speed": speed,
                        "timestamp": datetime.now(pytz.utc).isoformat()
                    })
                    if len(hormuz_vessels) > 100:
                        hormuz_vessels = hormuz_vessels[-100:]
        except Exception as e:
            print(f"AIS message error: {e}")

    def on_open(ws):
        subscribe = {
            "APIKey": AISSTREAM_API_KEY,
            "MessageType": "Subscribe",
            "BoundingBoxes": [
                [
                    [21.0, 55.0],
                    [27.0, 62.0]
                ]
            ],
            "FilterMessageTypes": ["PositionReport"]
        }
        ws.send(json.dumps(subscribe))
        print("AISStream subscribed to Hormuz bounding box")

    def on_error(ws, error):
        print(f"AISStream error: {error}")

    def on_close(ws, close_status_code, close_msg):
        print("AISStream connection closed -- reconnecting in 60s")
        time.sleep(60)
        start_aisstream()

    def run_ws():
        ws = websocket.WebSocketApp(
            "wss://stream.aisstream.io/v0/stream",
            on_open=on_open,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close
        )
        ws.run_forever()

    thread = threading.Thread(target=run_ws, daemon=True)
    thread.start()
    print("AISStream thread started -- monitoring Hormuz")

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
        return False, "Max orders per cycle"
    if trades_halted_today:
        return False, "Daily loss limit hit"
    if portfolio_state is None:
        return False, "Portfolio state unavailable"
    portfolio_value = portfolio_state["portfolio_value"]
    if daily_start_value is not None:
        daily_pnl = (portfolio_value - daily_start_value) / daily_start_value
        if daily_pnl <= -DAILY_LOSS_LIMIT:
            trades_halted_today = True
            send_performance(f"⚡ CIRCUIT BREAKER\n\nPortfolio down {abs(daily_pnl)*100:.1f}% today.\nAll trading paused until tomorrow.\n\n-- Satis House Consulting")
            return False, "Daily loss limit triggered"
    if vix >= VIX_STOP_THRESHOLD:
        return False, f"VIX {vix:.0f} -- above danger threshold"
    if direction == "sell" and is_friday_short_blocked():
        return False, "Friday short rule"
    signal_key = f"{ticker}_{direction}"
    if signal_key in recent_signals and recent_signals[signal_key] < DUPLICATE_SIGNAL_BLOCKS:
        return False, "Duplicate signal"
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
        return False, "Leveraged ETF market hours only"
    if not is_crypto and not is_market_hours():
        return False, "Market closed"
    if is_leveraged and portfolio_state["leveraged_pct"] >= MAX_LEVERAGED_PCT:
        return False, "Leveraged allocation cap"
    elif is_crypto and portfolio_state["crypto_pct"] >= MAX_CRYPTO_PCT:
        return False, "Crypto allocation cap"
    elif not is_crypto and not is_leveraged and portfolio_state["equity_pct"] >= MAX_EQUITY_PCT:
        return False, "Equity allocation cap"
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
        base_pct = base_pct * 0.5
        label = f"reduced to {base_pct*100:.0f}% -- VIX at {vix:.0f}"
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
        weight = scores["weight"]
        if total > 0:
            win_rate = scores["correct"] / total * 100
            lines.append(f"{name}: {win_rate:.0f}% win rate ({scores['correct']}/{total}) weight={weight:.1f}x")
        else:
            lines.append(f"{name}: No trades yet -- weight={weight:.1f}x")
    return "MISFIT TRACK RECORD AND WEIGHTS:\n" + "\n".join(lines) if lines else ""

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
    for ticker in ["TQQQ", "SOXL", "TECL", "FAS", "ERX", "ERX", "XLE", "USO", "BNO",
                   "UNG", "GLD", "SLV", "TLT", "HYG", "QQQ", "SPY", "IWM", "EEM",
                   "FXE", "FXB", "FXY", "UUP", "VLO", "FRO", "XOP"]:
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
                    trade_history.append({"ticker": crypto_symbol, "direction": direction, "entry_price": None, "voters_for": voters_for, "voters_against": voters_against})
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
                            trade_history.append({"ticker": ticker, "direction": direction, "entry_price": price, "voters_for": voters_for, "voters_against": voters_against})
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
    weight_note = f"\nYour current environment weight is {weight:.1f}x -- {'this is YOUR setup, trust your conviction' if weight > 1.0 else 'standard weight'}." if weight > 1.0 else ""
    data_context = f"\n\nYOUR SPECIFIC MARKET DATA:\n{specific_data}" if specific_data else ""
    knowledge_ctx = f"\n\nDEEP KNOWLEDGE:\n{knowledge}" if knowledge else ""
    score_ctx = f"\n\nTRACK RECORD:\n{scorecard}" if scorecard else ""

    msg = client.messages.create(
        model="claude-opus-4-5-20251101", max_tokens=500,
        messages=[{"role": "user", "content": f"""{persona}{weight_note}{data_context}{knowledge_ctx}{score_ctx}

YoniBot signal:
{signal}

You have your own market data above. Use it. Generate your own view on the best trade right now across ANY asset class -- equities long or short, options, crypto, foreign exchange, sovereign bonds, commodities, energy. If you see a better trade than what YoniBot flagged, say so explicitly.

2-3 sentences. Brutal and direct. End with VOTE: TRADE or VOTE: PASS on its own line."""}]
    )
    return msg.content[0].text

def ask_jane_street(signal, verdicts, all_misfit_data):
    debate = "\n\n".join([f"{n}:\n{v}" for n, v in verdicts])
    scorecard = build_scorecard_context()
    data_summary = json.dumps({k: str(v)[:200] for k, v in all_misfit_data.items()}, indent=2)[:2000]

    msg = client.messages.create(
        model="claude-opus-4-5-20251101", max_tokens=500,
        messages=[{"role": "user", "content": f"""You ARE Jane Street quant engine. Pure math. No narrative.

TRACK RECORD:
{scorecard}

ALL MISFIT DATA SUMMARY:
{data_summary}

SIGNAL:
{signal}

DEBATE:
{debate}

Calculate edge, Kelly size, max drawdown. Consider weighted votes. If any Misfit identified a better trade than the primary signal, evaluate that too.

End with VETO: BLOCKED or VETO: APPROVED on its own line."""}]
    )
    return msg.content[0].text

MISFITS = [
    ("Soros",
     "You ARE George Soros. Find the hidden peg. Where is the lie everyone believes and when do the defenders run out of ammunition? Your asset universe: foreign exchange, sovereign bonds, emerging market currencies, currency options, cross-currency positions. You look for reflexivity -- where the narrative is reinforcing an unsustainable position.",
     ["George Soros Black Wednesday 1992 pound sterling ERM mechanics",
      "George Soros reflexivity theory currency crisis 2024 2025",
      "Soros Fund Management macro sovereign debt currency views 2025 2026",
      "George Soros emerging market currency crisis capital flows"]),
    ("Druckenmiller",
     "You ARE Stanley Druckenmiller. State your stop first, target second, conviction size third. Your asset universe: equities long and short, bonds, macro themes, leveraged ETFs, credit. You concentrate into your best ideas and press winners.",
     ["Stanley Druckenmiller Deutsche Mark Soros 1992 concentration",
      "Stanley Druckenmiller philosophy asymmetric bet position sizing",
      "Stanley Druckenmiller macro views credit cycle Federal Reserve 2025 2026",
      "Druckenmiller risk management stop loss methodology"]),
    ("PTJ",
     "You ARE Paul Tudor Jones. Never trade without a 5 to 1 risk reward. Your asset universe: equity indices, options, volatility trades, technical breakouts, trend following across all asset classes. You use charts and tape reading to find the setup.",
     ["Paul Tudor Jones Black Monday 1987 prediction tape reading",
      "Paul Tudor Jones 5 to 1 risk reward rules trading",
      "Paul Tudor Jones macro views technical analysis 2025 2026",
      "PTJ Tudor Investment volatility options trend following"]),
    ("Tepper",
     "You ARE David Tepper. Read the Federal Reserve before the market does. Your asset universe: equities, high yield bonds, investment grade credit, sovereign bonds, bank stocks. You buy when the government makes it clear it will not let things fail.",
     ["David Tepper 2009 bank trade Appaloosa billions",
      "David Tepper Federal Reserve policy reading strategy",
      "David Tepper macro views credit Fed policy 2025 2026",
      "Tepper credit equity sovereign bonds macro methodology"]),
    ("Andurand",
     "You ARE Pierre Andurand. Physical markets lead paper markets always. Your asset universe: energy equities, energy ETFs long and short, energy options, oil-linked currencies, sovereign bonds of oil exporters and importers, shipping stocks, refinery stocks. You read physical flows before they show in price.",
     ["Pierre Andurand 2008 2022 oil trade physical flows",
      "Pierre Andurand tanker market commodity flows methodology",
      "Andurand Capital oil energy views 2025 2026",
      "Pierre Andurand Hormuz geopolitical energy supply disruption"]),
]

misfit_knowledge_cache = {}
misfit_data_cache = {}
knowledge_refresh_cycles = 8
cycle_count = 0

def send_startup_message():
    environment = detect_environment()
    active_env = [k for k, v in environment.items() if v]
    env_str = ", ".join(active_env) if active_env else "standard conditions"

    send_performance(f"""🚀 MISFITS SYSTEM ONLINE

The Misfits are watching the markets.
Current environment detected: {env_str}

Each Misfit now has their own data universe:
🔵 Soros: FX positioning, sovereign stress, currency flows
🟤 Druckenmiller: credit spreads, Fed balance sheet, macro cycle
📊 PTJ: VIX structure, volume, technical breakouts
💚 Tepper: Treasury auction data, credit delinquency, Fed signals
🛢 Andurand: EIA storage draws, Hormuz vessel tracking, crack spreads

Asset universe: equities, bonds, crypto, FX, commodities, options proxies

Five rules protecting capital. Weighted votes when environment matches career trades.

-- Satis House Consulting""")

def run_cycle():
    global cycle_count, misfit_knowledge_cache, misfit_data_cache
    global daily_start_value, trades_halted_today, recent_signals, orders_this_cycle

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
        print("Refreshing knowledge and data for all Misfits...")
        data_funcs = {
            "Soros": get_soros_data,
            "Druckenmiller": get_druckenmiller_data,
            "PTJ": get_ptj_data,
            "Tepper": get_tepper_data,
            "Andurand": get_andurand_data
        }
        for name, persona, queries in MISFITS:
            print(f"  Refreshing {name}...")
            misfit_knowledge_cache[name] = "\n\n".join([
                f"SOURCE: {r.title}\n{r.text[:400]}"
                for q in queries
                for r in (exa.search_and_contents(q, num_results=2, text={"max_characters": 400}).results
                          if True else [])
            ])
            misfit_data_cache[name] = data_funcs[name]()
            time.sleep(2)

    market_open = is_market_hours()
    rotation_ticker, rotation_score, rotation_summary, _ = run_omniscient_rotation()
    high_conviction_rotation = rotation_ticker not in ["BIL", "UUP", None] and rotation_score > 0.5 and market_open

    scorecard = build_scorecard_context()
    portfolio_summary = ""
    if portfolio_state:
        env_active = [k for k, v in environment.items() if v]
        portfolio_summary = f"""
PORTFOLIO: ${portfolio_state['portfolio_value']:,.2f}
Equity: {portfolio_state['equity_pct']*100:.1f}% | Crypto: {portfolio_state['crypto_pct']*100:.1f}% | Leveraged: {portfolio_state['leveraged_pct']*100:.1f}%
Positions: {list(portfolio_state['positions'].keys()) or 'None'}
VIX: {vix:.1f}
Active environments: {', '.join(env_active) if env_active else 'Standard'}"""

    yoni_push = f"\nHIGH CONVICTION ROTATION: {rotation_ticker} score {rotation_score:.3f}" if high_conviction_rotation else ""
    friday_note = "\nFRIDAY: No new short signals." if is_friday_short_blocked() else ""
    weekend_note = "" if market_open else "\nMARKET CLOSED. Crypto signals only."

    andurand_summary = ""
    if misfit_data_cache.get("Andurand"):
        a = misfit_data_cache["Andurand"]
        hormuz = a.get("hormuz", {})
        eia = a.get("eia", {})
        vessel_count = hormuz.get("vessel_count", 0)
        andurand_summary = f"\nANDURAND PHYSICAL INTEL: {vessel_count} vessels tracked near Hormuz"
        if eia.get("cushing_stocks"):
            cs = eia["cushing_stocks"]
            andurand_summary += f" | Cushing draw: {cs.get('change', 0):.1f}M bbls"

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

Scan ALL asset classes: equities long and short, leveraged ETFs, crypto, foreign exchange, sovereign bonds, commodities, energy, shipping, refiners.

Generate the single best trade signal right now based on all available data. Consider the active environment and which Misfit's framework best applies.

If market closed, crypto signals only.
No short signals on Fridays after 2 PM ET.
Say NO SIGNAL only if nothing genuinely qualifies.

Output: Asset, Direction, Entry, Stop, Target, Why. Max 300 words."""}]
    )
    signal = yoni.content[0].text

    verdicts = []
    all_misfit_data = {}
    for name, persona, queries in MISFITS:
        knowledge = misfit_knowledge_cache.get(name, "")
        specific_data = json.dumps(misfit_data_cache.get(name, {}), default=str)[:1500]
        all_misfit_data[name] = misfit_data_cache.get(name, {})
        verdict = ask_misfit(name, persona, signal, specific_data, knowledge)
        verdicts.append((name, verdict))

    weighted_votes, voters_for, voters_against = calculate_weighted_votes(verdicts)
    jane = ask_jane_street(signal, verdicts, all_misfit_data)
    approved = "VETO: APPROVED" in jane
    majority = weighted_votes >= 3.0

    if majority and approved:
        trade_result = execute_trade(signal, weighted_votes, verdicts, portfolio_state, vix,
                                     rotation_ticker if high_conviction_rotation else None,
                                     high_conviction_rotation)
        verdict_line = f"VERDICT: EXECUTE -- weighted votes {weighted_votes:.1f}. Jane Street approved.\n{trade_result}"
    elif not approved:
        verdict_line = f"VERDICT: BLOCKED -- Jane Street vetoed (weighted votes: {weighted_votes:.1f})"
    else:
        verdict_line = f"VERDICT: PASS -- weighted votes {weighted_votes:.1f} below threshold"

    send_telegram(f"YONIBOT SIGNAL\n{signal}")
    time.sleep(2)
    debate_msg = "THE MISFITS DEBATE\n"
    for name, verdict in verdicts:
        weight = misfit_scorecard.get(name, {}).get("weight", 1.0)
        debate_msg += f"\n{name.upper()} ({weight:.1f}x):\n{verdict}\n"
    send_telegram(debate_msg)
    time.sleep(2)
    send_telegram(f"JANE STREET:\n{jane}\n\n{verdict_line}")
    print(f"Brief sent. {verdict_line}")

    smart_sleep(900)

while True:
    try:
        if cycle_count == 0:
            start_aisstream()
            time.sleep(3)
            send_startup_message()
        run_cycle()
    except Exception as e:
        send_telegram(f"Misfits error: {e}")
        print(f"Error: {e}")
        smart_sleep(900)
