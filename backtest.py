import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

TICKERS = ["SOXL", "TECL", "TQQQ", "FAS", "ERX", "UUP", "TMF"]
SAFE = "BIL"
CONFIDENCE_THRESHOLD = 0.10
TARGET_VOL = 0.80
START_DATE = "2019-01-01"
END_DATE = datetime.now().strftime("%Y-%m-%d")
INITIAL_CAPITAL = 100000

def calc_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = -delta.clip(upper=0).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def download_all_data():
    print("Downloading historical data...")
    all_tickers = TICKERS + [SAFE, "SPY", "QQQ", "BTC-USD"]
    data = {}
    for ticker in all_tickers:
        try:
            df = yf.download(ticker, start=START_DATE, end=END_DATE,
                           progress=False, auto_adjust=True)
            if len(df) > 100:
                data[ticker] = df["Close"].squeeze()
                print(f"  {ticker}: {len(df)} days")
        except Exception as e:
            print(f"  {ticker}: failed -- {e}")
    return data

def score_assets(data, date_idx, all_dates):
    scores = {}
    current_date = all_dates[date_idx]

    spy_series = data.get("SPY")
    spy_trend = True
    if spy_series is not None:
        spy_hist = spy_series[spy_series.index <= current_date]
        if len(spy_hist) >= 200:
            spy_trend = float(spy_hist.iloc[-1]) > float(spy_hist.rolling(200).mean().iloc[-1])

    for ticker in TICKERS:
        if ticker not in data:
            continue
        try:
            series = data[ticker]
            hist = series[series.index <= current_date]
            if len(hist) < 65:
                continue

            close = hist
            roc_fast = float((close.iloc[-1] - close.iloc[-10]) / close.iloc[-10])
            roc_med = float((close.iloc[-1] - close.iloc[-22]) / close.iloc[-22])
            roc_slow = float((close.iloc[-1] - close.iloc[-64]) / close.iloc[-64])
            vol = float(close.pct_change().rolling(21).std().iloc[-1])
            rsi = float(calc_rsi(close).iloc[-1])
            sma50 = float(close.rolling(50).mean().iloc[-1])
            price = float(close.iloc[-1])

            if vol == 0 or np.isnan(vol):
                vol = 0.01

            weighted_mom = (roc_fast * 0.5) + (roc_med * 0.3) + (roc_slow * 0.2)
            risk_adj_mom = weighted_mom / vol
            trend_score = 1.0 if price > sma50 else 0.5
            rsi_penalty = 0.9 if (rsi > 85 or rsi < 30) else 1.0
            final_score = risk_adj_mom * trend_score * rsi_penalty
            scores[ticker] = final_score
        except:
            pass

    if not scores:
        return SAFE, 0, spy_trend

    sorted_assets = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    best_ticker = sorted_assets[0][0]
    best_score = sorted_assets[0][1]

    if not spy_trend:
        uup_score = scores.get("UUP", -999)
        if uup_score > 0 and uup_score > best_score:
            best_ticker = "UUP"
            best_score = uup_score
        elif best_score < 0:
            best_ticker = SAFE
            best_score = 0

    if best_score <= 0:
        best_ticker = SAFE

    return best_ticker, best_score, spy_trend

def calc_target_weight(data, ticker, date):
    try:
        series = data[ticker]
        hist = series[series.index <= date]
        if len(hist) < 21:
            return 1.0
        rets = hist.pct_change().dropna().tail(20)
        curr_vol = float(np.std(rets) * np.sqrt(252))
        if curr_vol > 0:
            weight = TARGET_VOL / curr_vol
        else:
            weight = 1.0
        return min(1.0, weight)
    except:
        return 1.0

def run_omniscient_backtest(data):
    print("\nRunning OmniscientBot backtest...")
    all_dates = sorted(set().union(*[set(s.index) for s in data.values()]))
    all_dates = [d for d in all_dates if d >= pd.Timestamp(START_DATE)]
    rebalance_dates = all_dates[::5]

    portfolio_value = INITIAL_CAPITAL
    current_holding = SAFE
    current_weight = 1.0
    trade_log = []
    daily_values = []

    for i, date in enumerate(all_dates):
        if current_holding != SAFE and current_holding in data:
            series = data[current_holding]
            hist = series[series.index <= date]
            if len(hist) >= 2:
                daily_return = float((hist.iloc[-1] - hist.iloc[-2]) / hist.iloc[-2])
                portfolio_value *= (1 + daily_return * current_weight)

        daily_values.append({"date": date, "value": portfolio_value, "holding": current_holding})

        if date in rebalance_dates and i > 64:
            date_idx = all_dates.index(date)
            best_ticker, best_score, spy_trend = score_assets(data, date_idx, all_dates)
            should_rotate = False

            if current_holding == SAFE:
                if best_score > 0.02:
                    should_rotate = True
            elif current_holding != best_ticker:
                curr_series = data.get(current_holding)
                if curr_series is not None:
                    curr_hist = curr_series[curr_series.index <= date]
                    if len(curr_hist) >= 65:
                        curr_close = curr_hist
                        curr_roc_fast = float((curr_close.iloc[-1] - curr_close.iloc[-10]) / curr_close.iloc[-10])
                        curr_roc_med = float((curr_close.iloc[-1] - curr_close.iloc[-22]) / curr_close.iloc[-22])
                        curr_roc_slow = float((curr_close.iloc[-1] - curr_close.iloc[-64]) / curr_close.iloc[-64])
                        curr_vol = float(curr_close.pct_change().rolling(21).std().iloc[-1]) or 0.01
                        curr_score = ((curr_roc_fast * 0.5) + (curr_roc_med * 0.3) + (curr_roc_slow * 0.2)) / curr_vol
                        if best_score > curr_score * (1 + CONFIDENCE_THRESHOLD):
                            should_rotate = True
                        elif curr_score < -0.02:
                            best_ticker = SAFE
                            should_rotate = True

            if should_rotate:
                new_weight = calc_target_weight(data, best_ticker, date) if best_ticker != SAFE else 1.0
                trade_log.append({
                    "date": date.strftime("%Y-%m-%d"),
                    "from": current_holding,
                    "to": best_ticker,
                    "score": round(best_score, 3),
                    "weight": round(new_weight, 2),
                    "portfolio_value": round(portfolio_value, 2)
                })
                current_holding = best_ticker
                current_weight = new_weight

    return pd.DataFrame(daily_values), trade_log

def run_spy_benchmark(data):
    spy = data.get("SPY")
    if spy is None:
        return None
    spy_hist = spy[spy.index >= pd.Timestamp(START_DATE)]
    portfolio = INITIAL_CAPITAL * (spy_hist / spy_hist.iloc[0])
    return portfolio

def run_misfit_proxy_backtest(data):
    print("Running Misfit proxy backtest (momentum + macro)...")
    all_dates = sorted(set().union(*[set(s.index) for s in data.values()]))
    all_dates = [d for d in all_dates if d >= pd.Timestamp(START_DATE)]

    assets = ["SPY", "QQQ", "GLD", "TLT", "USO", "EEM"]
    portfolio_value = INITIAL_CAPITAL
    current_holding = "SPY"
    daily_values = []

    for i, date in enumerate(all_dates):
        if current_holding in data:
            series = data[current_holding]
            hist = series[series.index <= date]
            if len(hist) >= 2:
                daily_return = float((hist.iloc[-1] - hist.iloc[-2]) / hist.iloc[-2])
                portfolio_value *= (1 + daily_return)

        daily_values.append({"date": date, "value": portfolio_value})

        if i % 20 == 0 and i > 60:
            scores = {}
            for asset in assets:
                if asset not in data:
                    continue
                try:
                    series = data[asset]
                    hist = series[series.index <= date]
                    if len(hist) < 65:
                        continue
                    roc = float((hist.iloc[-1] - hist.iloc[-20]) / hist.iloc[-20])
                    vol = float(hist.pct_change().rolling(20).std().iloc[-1]) or 0.01
                    scores[asset] = roc / vol
                except:
                    pass

            if scores:
                best = max(scores, key=scores.get)
                if scores[best] > 0:
                    current_holding = best

    return pd.DataFrame(daily_values)

def calculate_metrics(values_series, label):
    values = pd.Series(values_series)
    total_return = (values.iloc[-1] - values.iloc[0]) / values.iloc[0] * 100
    years = len(values) / 252
    cagr = ((values.iloc[-1] / values.iloc[0]) ** (1/years) - 1) * 100
    daily_returns = values.pct_change().dropna()
    sharpe = float(daily_returns.mean() / daily_returns.std() * np.sqrt(252)) if daily_returns.std() > 0 else 0
    rolling_max = values.cummax()
    drawdown = (values - rolling_max) / rolling_max
    max_drawdown = float(drawdown.min() * 100)
    win_rate = float((daily_returns > 0).mean() * 100)

    print(f"\n{'='*50}")
    print(f"{label}")
    print(f"{'='*50}")
    print(f"Total Return:    {total_return:>10.1f}%")
    print(f"CAGR:            {cagr:>10.1f}%")
    print(f"Sharpe Ratio:    {sharpe:>10.2f}")
    print(f"Max Drawdown:    {max_drawdown:>10.1f}%")
    print(f"Win Rate:        {win_rate:>10.1f}%")
    print(f"Final Value:     ${values.iloc[-1]:>10,.0f}")

    return {
        "label": label,
        "total_return": total_return,
        "cagr": cagr,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
        "final_value": float(values.iloc[-1])
    }

def print_trade_log(trade_log, n=20):
    print(f"\nLast {n} OmniscientBot Rotations:")
    print(f"{'Date':<12} {'From':<8} {'To':<8} {'Score':<8} {'Weight':<8} {'Portfolio':>12}")
    print("-" * 60)
    for trade in trade_log[-n:]:
        print(f"{trade['date']:<12} {trade['from']:<8} {trade['to']:<8} {trade['score']:<8} {trade['weight']:<8} ${trade['portfolio_value']:>10,.0f}")

def print_holding_summary(trade_log):
    holding_counts = {}
    for trade in trade_log:
        ticker = trade["to"]
        holding_counts[ticker] = holding_counts.get(ticker, 0) + 1
    total = sum(holding_counts.values())
    print(f"\nOmniscientBot Rotation Summary ({len(trade_log)} total rotations):")
    for ticker, count in sorted(holding_counts.items(), key=lambda x: x[1], reverse=True):
        pct = count / total * 100
        print(f"  {ticker:<8}: {count:>4} rotations ({pct:.1f}%)")

def main():
    print(f"SATIS HOUSE CONSULTING -- STRATEGY BACKTEST")
    print(f"Period: {START_DATE} to {END_DATE}")
    print(f"Initial Capital: ${INITIAL_CAPITAL:,.0f}")
    print("="*60)

    data = download_all_data()
    omniscient_df, trade_log = run_omniscient_backtest(data)
    spy_series = run_spy_benchmark(data)
    misfit_df = run_misfit_proxy_backtest(data)

    results = []
    results.append(calculate_metrics(omniscient_df["value"].values, "OMNISCIENTBOT -- TheOmniscientParadox Rotation"))

    if spy_series is not None:
        results.append(calculate_metrics(spy_series.values, "SPY BUY AND HOLD BENCHMARK"))

    results.append(calculate_metrics(misfit_df["value"].values, "MISFITS PROXY -- Multi-Asset Momentum"))

    print("\n" + "="*60)
    print("HEAD TO HEAD COMPARISON")
    print("="*60)
    print(f"{'Strategy':<45} {'Return':>10} {'CAGR':>8} {'Sharpe':>8} {'MaxDD':>8}")
    print("-"*80)
    for r in results:
        print(f"{r['label'][:44]:<45} {r['total_return']:>9.1f}% {r['cagr']:>7.1f}% {r['sharpe']:>8.2f} {r['max_drawdown']:>7.1f}%")

    print_trade_log(trade_log)
    print_holding_summary(trade_log)

    print(f"\nBACKTEST COMPLETE")
    print(f"OmniscientBot final: ${omniscient_df['value'].iloc[-1]:,.0f}")
    if spy_series is not None:
        print(f"SPY benchmark final: ${spy_series.iloc[-1]:,.0f}")
    print(f"Misfits proxy final: ${misfit_df['value'].iloc[-1]:,.0f}")

if __name__ == "__main__":
    main()
