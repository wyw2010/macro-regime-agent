#!/usr/bin/env python3
"""
Macro Regime Alert Agent

Daily macro regime detection for a systematic fund equally invested in
momentum, growth, and value factors. Pulls indicators from FRED, uses
Claude to classify the regime, and emails a digest with factor impact.

Usage:
    python macro_regime_agent.py              # fetch, analyze, and email
    python macro_regime_agent.py --dry-run    # print to console instead of emailing
"""

import argparse
import json
import os
import smtplib
import sys
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import anthropic
import requests
from dotenv import load_dotenv

load_dotenv(override=True)

# ---------------------------------------------------------------------------
# FRED data layer
# ---------------------------------------------------------------------------

FRED_SERIES = {
    # --- Rates & Monetary Policy ---
    "FEDFUNDS": "Fed Funds Rate",
    "DGS10": "10Y Treasury Yield",
    "DGS2": "2Y Treasury Yield",
    "DFII10": "10Y Real Yield (TIPS)",
    "T10Y2Y": "10Y-2Y Treasury Spread",
    # --- Inflation ---
    "T10YIE": "10Y Breakeven Inflation",
    "CPIAUCSL": "CPI (Index)",
    "PCEPILFE": "Core PCE (Index)",
    # --- Growth & Activity ---
    "MANEMP": "ISM Mfg Employment (PMI proxy)",
    "ICSA": "Initial Jobless Claims",
    "UNRATE": "Unemployment Rate",
    "A191RL1Q225SBEA": "Real GDP Growth (QoQ ann.)",
    # --- Risk Sentiment & Credit ---
    "VIXCLS": "VIX",
    "BAMLH0A0HYM2": "High Yield OAS Spread",
    "BAMLC0A0CM": "IG Corporate Spread",
    # --- Equity & Market ---
    "SP500": "S&P 500",
    # --- Dollar & Commodities ---
    "DTWEXBGS": "Trade Weighted USD Index",
    "DCOILWTICO": "WTI Crude Oil",
}
# Note: Bloomberg AGG is not on FRED — IG Corporate Spread serves as a bond market stress proxy.


def fetch_fred_series(series_id: str, api_key: str, lookback_days: int | None = None, retries: int = 3) -> list[dict]:
    """Fetch recent observations for a single FRED series with retry logic.

    If lookback_days is None, fetches all available history.
    """
    import time
    start = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d") if lookback_days else "1900-01-01"
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "sort_order": "desc",
        "observation_start": start,
    }
    for attempt in range(retries):
        resp = requests.get(url, params=params, timeout=15)
        if resp.status_code == 200:
            observations = resp.json().get("observations", [])
            return [o for o in observations if o["value"] != "."]
        if attempt < retries - 1:
            time.sleep(2 ** attempt)  # 1s, 2s backoff
    raise RuntimeError(f"FRED returned {resp.status_code} for {series_id} after {retries} attempts")


CHANGE_PERIODS = [
    ("1d", 1),
    ("1w", 5),
    ("1m", 25),
    ("3m", 80),
    ("1y", 350),
]


def fetch_all_indicators(api_key: str) -> tuple[dict, dict]:
    """Fetch all FRED indicators and compute current value + changes across multiple periods.

    Returns:
        (indicators, raw_series) where raw_series maps label -> list of (date_str, float_value)
        sorted ascending by date, for use in correlation computation.
    """
    indicators = {}
    raw_series = {}
    for series_id, label in FRED_SERIES.items():
        try:
            obs = fetch_fred_series(series_id, api_key, lookback_days=None)  # all-time for correlations
            if not obs:
                entry = {"current": "N/A", "date": "N/A"}
                for key, _ in CHANGE_PERIODS:
                    entry[f"{key}_abs"] = "N/A"
                    entry[f"{key}_pct"] = "N/A"
                indicators[label] = entry
                raw_series[label] = []
                continue

            # Store raw series ascending by date
            raw_series[label] = [
                (o["date"], float(o["value"])) for o in reversed(obs)
            ]

            current = float(obs[0]["value"])
            current_date = obs[0]["date"]

            # Find historical values for each period
            period_vals = {key: None for key, _ in CHANGE_PERIODS}
            for o in obs[1:]:  # skip current
                days_ago = (datetime.now() - datetime.strptime(o["date"], "%Y-%m-%d")).days
                for key, min_days in CHANGE_PERIODS:
                    if days_ago >= min_days and period_vals[key] is None:
                        period_vals[key] = float(o["value"])

            def fmt_change(old, new):
                if old is None:
                    return {"abs": "N/A", "pct": "N/A"}
                diff = new - old
                pct = (diff / abs(old) * 100) if old != 0 else 0.0
                return {"abs": f"{diff:+.2f}", "pct": f"{pct:+.1f}%"}

            entry = {
                "current": f"{current:.2f}",
                "date": current_date,
            }
            for key, _ in CHANGE_PERIODS:
                chg = fmt_change(period_vals[key], current)
                entry[f"{key}_abs"] = chg["abs"]
                entry[f"{key}_pct"] = chg["pct"]
            indicators[label] = entry
        except Exception as e:
            # Sanitize error message to never leak API keys
            err_msg = str(e)
            if api_key in err_msg:
                err_msg = err_msg.replace(api_key, "***")
            entry = {"current": "Unavailable", "date": "N/A"}
            for key, _ in CHANGE_PERIODS:
                entry[f"{key}_abs"] = "N/A"
                entry[f"{key}_pct"] = "N/A"
            indicators[label] = entry
            raw_series[label] = []
            print(f"  WARNING: Failed to fetch {label}: {err_msg}")

    return indicators, raw_series


def _compute_corr_single(raw_series: dict, sp500_label: str, lookback_days: int | None = None) -> dict:
    """Compute correlation of each indicator's weekly changes with S&P 500 weekly returns.

    Args:
        lookback_days: If set, only use data from the last N days. None = all available.

    Returns dict mapping label -> {"corr": float, "abs_corr": float, "n_obs": int}
    """
    from statistics import correlation, StatisticsError

    sp500_data = raw_series.get(sp500_label, [])
    if len(sp500_data) < 10:
        return {}

    # Apply lookback filter
    if lookback_days is not None:
        cutoff = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
        sp500_data = [(d, v) for d, v in sp500_data if d >= cutoff]

    sp500_by_date = {d: v for d, v in sp500_data}
    sp500_dates = sorted(sp500_by_date.keys())

    results = {}
    for label, series in raw_series.items():
        if label == sp500_label or len(series) < 10:
            continue

        series_filtered = [(d, v) for d, v in series if d >= cutoff] if lookback_days else series
        series_by_date = {d: v for d, v in series_filtered}

        # Detect if this is a low-frequency series (monthly/quarterly) by checking
        # average gap between observations. If avg gap > 7 days, forward-fill.
        sorted_obs_dates = sorted(series_by_date.keys())
        if len(sorted_obs_dates) >= 2:
            first = datetime.strptime(sorted_obs_dates[0], "%Y-%m-%d")
            last = datetime.strptime(sorted_obs_dates[-1], "%Y-%m-%d")
            avg_gap = (last - first).days / max(len(sorted_obs_dates) - 1, 1)
        else:
            avg_gap = 999

        if avg_gap > 7:  # monthly, quarterly, etc. — forward-fill to daily
            filled = {}
            last_val = None
            all_dates = sorted(set(sp500_dates) | set(sorted_obs_dates))
            for d in all_dates:
                if d in series_by_date:
                    last_val = series_by_date[d]
                if last_val is not None:
                    filled[d] = last_val
            series_by_date = filled

        # Compute weekly (5-day) changes on overlapping dates
        common_dates = sorted(set(sp500_dates) & set(series_by_date.keys()))
        if len(common_dates) < 15:
            continue

        sp500_changes = []
        indicator_changes = []
        step = 5
        for i in range(step, len(common_dates), step):
            d_now = common_dates[i]
            d_prev = common_dates[i - step]
            sp_now, sp_prev = sp500_by_date.get(d_now), sp500_by_date.get(d_prev)
            ind_now, ind_prev = series_by_date.get(d_now), series_by_date.get(d_prev)

            if all(v is not None and v != 0 for v in [sp_now, sp_prev, ind_now, ind_prev]):
                sp500_changes.append((sp_now - sp_prev) / abs(sp_prev))
                indicator_changes.append((ind_now - ind_prev) / abs(ind_prev))

        if len(sp500_changes) < 8:
            continue

        try:
            corr = correlation(indicator_changes, sp500_changes)
            results[label] = {
                "corr": round(corr, 3),
                "abs_corr": round(abs(corr), 3),
                "n_obs": len(sp500_changes),
            }
        except (StatisticsError, ZeroDivisionError):
            continue

    return results


def compute_correlations(raw_series: dict, sp500_label: str = "S&P 500") -> dict:
    """Compute dual-timeframe correlations: trailing 1Y and all-time.

    Returns dict mapping label -> {
        "corr_1y", "abs_corr_1y", "n_obs_1y",
        "corr_all", "abs_corr_all", "n_obs_all",
    } sorted by abs_corr_1y descending (primary ranking on recent data).
    """
    corr_1y = _compute_corr_single(raw_series, sp500_label, lookback_days=365)
    corr_all = _compute_corr_single(raw_series, sp500_label, lookback_days=None)

    # Merge into unified dict
    all_labels = set(corr_1y.keys()) | set(corr_all.keys())
    results = {}
    for label in all_labels:
        r1y = corr_1y.get(label, {})
        rall = corr_all.get(label, {})
        results[label] = {
            "corr_1y": r1y.get("corr", None),
            "abs_corr_1y": r1y.get("abs_corr", 0.0),
            "n_obs_1y": r1y.get("n_obs", 0),
            "corr_all": rall.get("corr", None),
            "abs_corr_all": rall.get("abs_corr", 0.0),
            "n_obs_all": rall.get("n_obs", 0),
        }

    # Sort by 1Y absolute correlation descending (primary ranking)
    results = dict(sorted(results.items(), key=lambda x: x[1]["abs_corr_1y"], reverse=True))
    return results


# ---------------------------------------------------------------------------
# Claude analysis layer
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a macro regime analyst for a systematic equity fund that is equally \
weighted across three factors: Momentum, Growth, and Value.

Your job is to:
1. Classify the current macro regime and assess implications for each factor.
2. Using the signal correlation rankings provided, predict the near-term S&P 500 regime.

REGIME FRAMEWORK:
- Reflation: Rising growth + rising inflation → Value tailwind, Growth headwind
- Goldilocks: Rising growth + falling inflation → Momentum & Growth tailwind
- Stagflation: Falling growth + rising inflation → Headwind for all, especially Momentum
- Deflation/Slowdown: Falling growth + falling inflation → Growth tailwind, Value headwind

SIGNAL-BASED PREDICTION:
Use the correlation rankings to weight signals by importance. Signals with higher \
|correlation| to S&P 500 should carry more weight in your prediction. Consider:
- Are the highest-correlated signals (VIX, HY spreads, etc.) flashing risk-on or risk-off?
- Is there convergence or divergence among top signals?
- What do the 1D/1W changes in top signals suggest about near-term direction?

Respond in this EXACT JSON structure (no markdown fencing):
{
  "regime": "<Reflation|Goldilocks|Stagflation|Deflation>",
  "confidence": "<High|Medium|Low>",
  "regime_summary": "<2-3 sentence summary of current regime>",
  "transition_signals": "<2-3 sentences on any early warning signs of regime change>",
  "factor_scorecard": {
    "momentum": {"rating": "<Tailwind|Neutral|Headwind>", "rationale": "<1-2 sentences>"},
    "growth": {"rating": "<Tailwind|Neutral|Headwind>", "rationale": "<1-2 sentences>"},
    "value": {"rating": "<Tailwind|Neutral|Headwind>", "rationale": "<1-2 sentences>"}
  },
  "risks_watchlist": "<3-4 bullet points of key risks or things to monitor, separated by newlines>",
  "sp500_prediction": {
    "direction": "<Bullish|Bearish|Neutral>",
    "confidence": "<High|Medium|Low>",
    "timeframe": "1-4 weeks",
    "rationale": "<3-5 sentences explaining the prediction, referencing the top correlated signals by name and their recent moves. Be specific about which signals agree/disagree and what that implies.>"
  }
}
"""


def build_analysis_prompt(indicators: dict, correlations: dict) -> str:
    """Format indicator data into a user prompt for Claude."""
    period_labels = ["1D", "1W", "1M", "3M", "1Y"]
    period_keys = ["1d", "1w", "1m", "3m", "1y"]

    lines = ["Here are the latest macro indicators with absolute change and (percent change):\n"]
    header = f"{'Indicator':<35} {'Current':>10}"
    for pl in period_labels:
        header += f" {pl:>16}"
    lines.append(header)
    lines.append("-" * 120)

    for label, data in indicators.items():
        row = f"{label:<35} {data['current']:>10}"
        for pk in period_keys:
            abs_val = data.get(f"{pk}_abs", "N/A")
            pct_val = data.get(f"{pk}_pct", "N/A")
            if abs_val == "N/A":
                row += f" {'N/A':>16}"
            else:
                row += f" {abs_val + ' (' + pct_val + ')':>16}"
        lines.append(row)

    if correlations:
        lines.append("\n\nSignal Ranking (weekly correlation with S&P 500 returns):")
        lines.append(f"{'Rank':<6} {'Indicator':<35} {'1Y Corr':>9} {'(n)':>6}  {'All-Time Corr':>14} {'(n)':>6}")
        lines.append("-" * 85)
        for rank, (label, info) in enumerate(correlations.items(), 1):
            c1y = f"{info['corr_1y']:+.3f}" if info['corr_1y'] is not None else "  N/A"
            call = f"{info['corr_all']:+.3f}" if info['corr_all'] is not None else "  N/A"
            lines.append(
                f"{rank:<6} {label:<35} {c1y:>9} {info['n_obs_1y']:>5}  {call:>14} {info['n_obs_all']:>5}"
            )
        lines.append("\nRanked by trailing 1Y |correlation|. All-time shown for context.")

    lines.append("\nClassify the current macro regime and provide factor impact assessment. "
                 "Pay special attention to the highest-correlated signals when forming your view.")
    return "\n".join(lines)


def analyze_regime(indicators: dict, correlations: dict) -> dict:
    """Call Claude API to classify regime and assess factor impacts."""
    api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    client = anthropic.Anthropic(api_key=api_key)
    prompt = build_analysis_prompt(indicators, correlations)

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2048,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = message.content[0].text
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"raw_response": raw, "error": "Failed to parse JSON from Claude response"}


# ---------------------------------------------------------------------------
# Email layer
# ---------------------------------------------------------------------------

RATING_COLORS = {
    "Tailwind": "#22c55e",
    "Neutral": "#eab308",
    "Headwind": "#ef4444",
}


def build_email_html(analysis: dict, indicators: dict, correlations: dict) -> str:
    """Build an HTML email from the analysis results."""
    date_str = datetime.now().strftime("%B %d, %Y")

    if "error" in analysis:
        return f"<html><body><h2>Macro Regime Agent Error</h2><pre>{analysis.get('raw_response', analysis['error'])}</pre></body></html>"

    regime = analysis.get("regime", "Unknown")
    confidence = analysis.get("confidence", "Unknown")
    summary = analysis.get("regime_summary", "")
    transitions = analysis.get("transition_signals", "")
    scorecard = analysis.get("factor_scorecard", {})
    risks = analysis.get("risks_watchlist", "")
    prediction = analysis.get("sp500_prediction", {})

    # Factor scorecard rows
    factor_rows = ""
    for factor in ["momentum", "growth", "value"]:
        info = scorecard.get(factor, {})
        rating = info.get("rating", "N/A")
        color = RATING_COLORS.get(rating, "#6b7280")
        rationale = info.get("rationale", "")
        factor_rows += f"""
        <tr>
            <td style="padding:10px 12px;font-weight:600;text-transform:capitalize;">{factor}</td>
            <td style="padding:10px 12px;color:{color};font-weight:700;">{rating}</td>
            <td style="padding:10px 12px;color:#374151;line-height:1.4;">{rationale}</td>
        </tr>"""

    # Indicator rows
    indicator_rows = ""
    period_keys = ["1d", "1w", "1m", "3m", "1y"]

    def color_change(val):
        if val == "N/A":
            return "#6b7280"
        return "#22c55e" if val.startswith("+") else "#ef4444" if val.startswith("-") else "#6b7280"

    for label, data in indicators.items():
        cells = f'<td style="padding:6px 8px;">{label}</td>'
        cells += f'<td style="padding:6px 8px;text-align:right;">{data["current"]}</td>'
        for pk in period_keys:
            abs_val = data.get(f"{pk}_abs", "N/A")
            pct_val = data.get(f"{pk}_pct", "N/A")
            color = color_change(abs_val)
            if abs_val == "N/A":
                cells += f'<td style="padding:6px 8px;text-align:right;color:#6b7280;">N/A</td>'
            else:
                cells += (
                    f'<td style="padding:6px 8px;text-align:right;color:{color};">'
                    f'{abs_val}<br><span style="font-size:11px;opacity:0.8;">{pct_val}</span></td>'
                )
        indicator_rows += f"<tr>{cells}</tr>"

    # Signal ranking rows
    signal_rows = ""

    def _strength(abs_corr):
        if abs_corr is None or abs_corr == 0.0:
            return ("N/A", "#6b7280")
        if abs_corr >= 0.5:
            return ("Strong", "#ef4444")
        if abs_corr >= 0.3:
            return ("Moderate", "#f59e0b")
        return ("Weak", "#6b7280")

    def _fmt_corr(val):
        return f"{val:+.3f}" if val is not None else "N/A"

    for rank, (label, info) in enumerate(correlations.items(), 1):
        c1y = info["corr_1y"]
        ac1y = info["abs_corr_1y"]
        n1y = info["n_obs_1y"]
        call = info["corr_all"]
        acall = info["abs_corr_all"]
        nall = info["n_obs_all"]

        strength_1y, color_1y = _strength(ac1y)
        strength_all, color_all = _strength(acall)
        bar_width = min((ac1y or 0) * 100, 100)

        signal_rows += f"""
        <tr>
            <td style="padding:6px 8px;text-align:center;font-weight:600;">{rank}</td>
            <td style="padding:6px 8px;">{label}</td>
            <td style="padding:6px 8px;text-align:right;font-family:monospace;color:{color_1y};">{_fmt_corr(c1y)}</td>
            <td style="padding:6px 8px;width:90px;">
                <div style="background:#e5e7eb;border-radius:4px;height:14px;width:100%;">
                    <div style="background:{color_1y};border-radius:4px;height:14px;width:{bar_width}%;"></div>
                </div>
            </td>
            <td style="padding:6px 8px;text-align:center;color:#6b7280;font-size:12px;">{n1y}</td>
            <td style="padding:6px 8px;text-align:right;font-family:monospace;color:{color_all};">{_fmt_corr(call)}</td>
            <td style="padding:6px 8px;text-align:center;color:#6b7280;font-size:12px;">{nall}</td>
        </tr>"""

    # Format risks as list items
    risk_items = ""
    for line in risks.split("\n"):
        line = line.strip().lstrip("•-* ")
        if line:
            risk_items += f"<li style='margin-bottom:4px;'>{line}</li>"

    html = f"""
    <html>
    <head>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
    </head>
    <body style="font-family:'Times New Roman',Times,serif;max-width:720px;margin:0 auto;color:#1f2937;padding:16px;font-size:18px;line-height:1.5;">

    <h1 style="font-size:28px;margin-bottom:4px;">Macro Regime Daily Digest</h1>
    <p style="color:#6b7280;margin-top:0;font-size:16px;">{date_str}</p>
    <p style="color:#9ca3af;margin-top:-8px;font-size:13px;">Agent created by Will Wu (willwu@stern.nyu.edu)</p>

    <div style="background:#f0f9ff;border-left:4px solid #3b82f6;padding:16px;margin:20px 0;border-radius:4px;">
        <h2 style="margin:0 0 8px 0;font-size:20px;">
            Regime: {regime}
            <span style="font-weight:400;color:#6b7280;font-size:16px;">({confidence} confidence)</span>
        </h2>
        <p style="margin:0;color:#374151;font-size:17px;line-height:1.5;">{summary}</p>
    </div>

    <h3 style="font-size:21px;">Factor Scorecard</h3>
    <table style="width:100%;border-collapse:collapse;margin-bottom:20px;font-size:16px;">
        <tr style="background:#f3f4f6;">
            <th style="padding:10px 12px;text-align:left;">Factor</th>
            <th style="padding:10px 12px;text-align:left;">Rating</th>
            <th style="padding:10px 12px;text-align:left;">Rationale</th>
        </tr>
        {factor_rows}
    </table>

    <h3 style="font-size:21px;">Transition Signals</h3>
    <p style="color:#374151;background:#fffbeb;padding:14px;border-radius:4px;border-left:4px solid #f59e0b;font-size:16px;line-height:1.5;">
        {transitions}
    </p>

    <h3 style="font-size:21px;">Key Indicators</h3>
    <div style="overflow-x:auto;-webkit-overflow-scrolling:touch;margin-bottom:20px;">
    <table style="min-width:600px;width:100%;border-collapse:collapse;font-size:15px;">
        <tr style="background:#f3f4f6;">
            <th style="padding:8px 10px;text-align:left;white-space:nowrap;">Indicator</th>
            <th style="padding:8px 10px;text-align:right;">Current</th>
            <th style="padding:8px 10px;text-align:right;">1D</th>
            <th style="padding:8px 10px;text-align:right;">1W</th>
            <th style="padding:8px 10px;text-align:right;">1M</th>
            <th style="padding:8px 10px;text-align:right;">3M</th>
            <th style="padding:8px 10px;text-align:right;">1Y</th>
        </tr>
        {indicator_rows}
    </table>
    </div>

    <h3 style="font-size:21px;">Signal Ranking</h3>
    <p style="color:#6b7280;font-size:14px;margin-bottom:8px;">Weekly correlation with S&P 500 returns. Ranked by trailing 1Y; all-time shown for context.</p>
    <div style="overflow-x:auto;-webkit-overflow-scrolling:touch;margin-bottom:20px;">
    <table style="min-width:550px;width:100%;border-collapse:collapse;font-size:15px;">
        <tr style="background:#f3f4f6;">
            <th style="padding:8px 10px;text-align:center;">#</th>
            <th style="padding:8px 10px;text-align:left;white-space:nowrap;">Indicator</th>
            <th style="padding:8px 10px;text-align:right;">1Y Corr</th>
            <th style="padding:8px 10px;text-align:center;">Strength</th>
            <th style="padding:8px 10px;text-align:center;font-size:13px;">n</th>
            <th style="padding:8px 10px;text-align:right;">All-Time</th>
            <th style="padding:8px 10px;text-align:center;font-size:13px;">n</th>
        </tr>
        {signal_rows}
    </table>
    </div>

    <h3 style="font-size:21px;">Risks & Watchlist</h3>
    <ul style="color:#374151;">{risk_items}</ul>

    {_build_prediction_box(prediction)}

    <hr style="border:none;border-top:1px solid #e5e7eb;margin:24px 0;">
    <p style="color:#9ca3af;font-size:12px;">
        Generated by Macro Regime Agent &middot; Data from FRED &middot; Analysis by Claude
    </p>
    </body>
    </html>
    """
    # Minify HTML to stay under Gmail's ~102KB clipping threshold
    import re
    html = re.sub(r'>\s+<', '><', html)  # remove whitespace between tags
    html = re.sub(r'\s{2,}', ' ', html)  # collapse multiple spaces
    return html


def _build_prediction_box(prediction: dict) -> str:
    """Build the S&P 500 regime prediction box HTML."""
    if not prediction:
        return ""

    direction = prediction.get("direction", "Unknown")
    confidence = prediction.get("confidence", "Unknown")
    timeframe = prediction.get("timeframe", "1-4 weeks")
    rationale = prediction.get("rationale", "")

    # Colors and icons by direction
    dir_styles = {
        "Bullish": {"bg": "#ecfdf5", "border": "#10b981", "color": "#065f46", "icon": "&#9650;"},
        "Bearish": {"bg": "#fef2f2", "border": "#ef4444", "color": "#991b1b", "icon": "&#9660;"},
        "Neutral": {"bg": "#fffbeb", "border": "#f59e0b", "color": "#92400e", "icon": "&#9670;"},
    }
    style = dir_styles.get(direction, dir_styles["Neutral"])

    return f"""
    <div style="margin:24px 0;border:2px solid {style['border']};border-radius:8px;overflow:hidden;">
        <div style="background:{style['bg']};padding:16px 20px;border-bottom:1px solid {style['border']};">
            <h3 style="margin:0;font-size:18px;color:{style['color']};">
                {style['icon']} S&P 500 Signal-Based Outlook: {direction}
            </h3>
            <p style="margin:4px 0 0 0;color:{style['color']};font-size:13px;">
                Confidence: {confidence} &middot; Timeframe: {timeframe}
            </p>
        </div>
        <div style="padding:16px 20px;background:white;">
            <p style="margin:0;color:#374151;line-height:1.6;">{rationale}</p>
        </div>
        <div style="padding:8px 20px;background:#f9fafb;border-top:1px solid #e5e7eb;">
            <p style="margin:0;color:#9ca3af;font-size:11px;font-style:italic;">
                Based on correlation-weighted analysis of macro signals. Not investment advice.
            </p>
        </div>
    </div>"""


def _load_subscribers() -> list[str]:
    """Load subscriber emails from subscribers.json if it exists."""
    subscribers_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "subscribers.json")
    if not os.path.exists(subscribers_path):
        return []
    try:
        with open(subscribers_path) as f:
            data = json.load(f)
        return [e.strip() for e in data.get("subscribers", []) if e.strip()]
    except (json.JSONDecodeError, IOError) as e:
        print(f"  WARNING: Could not read subscribers.json: {e}")
        return []


def send_email(html: str, subject: str):
    """Send the HTML email via SMTP."""
    smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    email_from = os.getenv("EMAIL_FROM")
    email_password = os.getenv("EMAIL_PASSWORD")
    email_to_raw = os.getenv("EMAIL_TO", "")

    if not all([email_from, email_password]):
        print("ERROR: EMAIL_FROM and EMAIL_PASSWORD must be set in .env")
        sys.exit(1)

    # Merge EMAIL_TO env var with subscribers.json (deduplicated)
    env_recipients = [addr.strip() for addr in email_to_raw.split(",") if addr.strip()]
    file_recipients = _load_subscribers()
    recipients = list(dict.fromkeys(env_recipients + file_recipients))  # dedup, preserving order

    if not recipients:
        print("ERROR: No recipients found in EMAIL_TO or subscribers.json")
        sys.exit(1)

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = email_from
    msg["To"] = ", ".join(recipients)
    msg.attach(MIMEText(html, "html"))

    with smtplib.SMTP(smtp_server, smtp_port) as server:
        server.starttls()
        server.login(email_from, email_password)
        server.sendmail(email_from, recipients, msg.as_string())
    print(f"  Sent to {len(recipients)} recipient(s): {', '.join(recipients)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Macro Regime Alert Agent")
    parser.add_argument("--dry-run", action="store_true", help="Print analysis to console instead of emailing")
    args = parser.parse_args()

    fred_key = os.getenv("FRED_API_KEY")
    if not fred_key:
        print("ERROR: FRED_API_KEY not set. Copy .env.example to .env and fill in your keys.")
        sys.exit(1)

    if not os.getenv("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set.")
        sys.exit(1)

    # Step 1: Fetch macro data
    print("Fetching FRED indicators...")
    indicators, raw_series = fetch_all_indicators(fred_key)
    for label, data in indicators.items():
        chgs = "  ".join(
            f"{pk}: {data.get(f'{pk}_abs', 'N/A')} ({data.get(f'{pk}_pct', 'N/A')})"
            for pk in ["1d", "1w", "1m", "3m", "1y"]
        )
        print(f"  {label:<35} {data['current']:>10}  {chgs}")

    # Step 1b: Compute correlations with S&P 500
    print("\nComputing signal correlations with S&P 500 (1Y + all-time)...")
    correlations = compute_correlations(raw_series)
    for rank, (label, info) in enumerate(correlations.items(), 1):
        c1y = f"{info['corr_1y']:+.3f}" if info['corr_1y'] is not None else "  N/A"
        call = f"{info['corr_all']:+.3f}" if info['corr_all'] is not None else "  N/A"
        print(f"  #{rank:<3} {label:<35}  1Y: {c1y} (n={info['n_obs_1y']})  All: {call} (n={info['n_obs_all']})")

    # Step 2: Analyze with Claude
    print("\nAnalyzing macro regime with Claude...")
    analysis = analyze_regime(indicators, correlations)

    if "error" in analysis:
        print(f"WARNING: {analysis['error']}")
        print(analysis.get("raw_response", ""))
    else:
        print(f"\n  Regime:     {analysis['regime']} ({analysis['confidence']} confidence)")
        scorecard = analysis.get("factor_scorecard", {})
        for f in ["momentum", "growth", "value"]:
            info = scorecard.get(f, {})
            print(f"  {f.capitalize():<12} {info.get('rating', 'N/A')}")
        pred = analysis.get("sp500_prediction", {})
        if pred:
            print(f"\n  S&P 500 Outlook: {pred.get('direction', '?')} ({pred.get('confidence', '?')} confidence, {pred.get('timeframe', '?')})")

    # Step 3: Build email and send (or print)
    date_str = datetime.now().strftime("%Y-%m-%d")
    regime_name = analysis.get("regime", "Unknown")
    subject = f"Macro Regime Alert [{date_str}]: {regime_name}"
    html = build_email_html(analysis, indicators, correlations)

    if args.dry_run:
        print(f"\n{'='*60}")
        print(f"Subject: {subject}")
        print(f"{'='*60}")
        # Print a plain-text summary for dry-run
        print(json.dumps(analysis, indent=2))
        print("\n(HTML email built successfully — use without --dry-run to send)")
    else:
        print("\nSending email...")
        send_email(html, subject)
        print("Email sent successfully!")


if __name__ == "__main__":
    main()
