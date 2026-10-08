"""Touch Grass: which hours are worth being outside, forecast locally with TabPFN.

One file, one run:
  1. pull ~70 days of hourly weather for one place from Open-Meteo (free, no key)
  2. turn it into a plain table: hour, weekday, and what the weather was 1/2/3/7 days ago
  3. TabPFN v2 (open weights, CPU, no account) predicts temperature, rain, wind, cloud
     for the target day (today if you run in the morning, tomorrow if you run at night)
  4. score every daylight hour, pick the best windows, write one offline index.html

Honesty check built in: before predicting the target day, the same pipeline predicts
the last few fully observed days from the days before them and reports its mean
absolute error next to the dumbest baseline ("same hour the day before").
If TabPFN does not beat that on a variable, the page says so.

Usage:
  python touch_grass.py                               # Paris, auto: today before 18h, else tomorrow
  python touch_grass.py --day tomorrow
  python touch_grass.py --lat 45.76 --lon 4.84 --name Lyon
  python touch_grass.py --csv data/paris_hourly.csv   # offline, reuse a saved pull
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

os.environ.setdefault("TABPFN_NO_BROWSER", "1")  # v2 weights are ungated; never pop a login

VARS = ["temperature_2m", "precipitation", "wind_speed_10m", "cloud_cover"]
LABELS = {"temperature_2m": "Temperature (°C)", "precipitation": "Rain (mm/h)", "wind_speed_10m": "Wind (km/h)", "cloud_cover": "Cloud (%)"}
TRAIN_DAYS = 41  # ponytail: TabPFN v2 caps CPU fits at 1000 rows; 41*24 = 984. GPU → raise it.
HOLDOUT_DAYS = 3  # how many past days the honesty check re-predicts
N_ESTIMATORS = 4  # TabPFN ensemble size; 4 ≈ half the CPU time of the default 8, same story
PAST_DAYS = 70  # Open-Meteo forecast endpoint keeps ~71 past days of hourly values
MODEL_CARD = "TabPFN-v2 regressor, Prior Labs License v1.1 (Apache-2.0 + attribution)"


# ----------------------------------------------------------------------------- data
def fetch(lat: float, lon: float, tz: str) -> dict:
    q = urllib.parse.urlencode(
        {
            "latitude": lat,
            "longitude": lon,
            "hourly": ",".join(VARS + ["is_day"]),
            "past_days": PAST_DAYS,
            "forecast_days": 2,  # today + tomorrow; only is_day (sunrise/sunset) is read from them
            "timezone": tz,
        }
    )
    with urllib.request.urlopen(f"https://api.open-meteo.com/v1/forecast?{q}", timeout=60) as r:
        return json.load(r)["hourly"]


def save_csv(h: dict, path: Path) -> None:
    keys = ["time"] + VARS + ["is_day"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(keys)
        for i in range(len(h["time"])):
            w.writerow([h[k][i] for k in keys])


def load_csv(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    h = {"time": [r["time"] for r in rows]}
    for k in VARS + ["is_day"]:
        h[k] = [None if r[k] in ("", "None") else float(r[k]) for r in rows]
    return h


def to_days(h: dict) -> list[dict]:
    """Drop leading nulls, then group hourly arrays into complete 24-hour days."""
    first = next(i for i, v in enumerate(h[VARS[0]]) if v is not None)
    while h["time"][first][11:13] != "00":
        first += 1
    days = []
    for s in range(first, len(h["time"]) - 23, 24):
        day = {"date": h["time"][s][:10], "is_day": np.array(h["is_day"][s : s + 24], float)}
        for v in VARS:
            day[v] = np.array(h[v][s : s + 24], float)
        days.append(day)
    return days


# ------------------------------------------------------------------------- features
def lag_days(base: int) -> tuple[int, ...]:
    """base = 1 when the target is today (yesterday is fully observed), 2 when it is tomorrow."""
    return (base, base + 1, base + 2, 7)


def features(days: list[dict], d: int, base: int) -> np.ndarray:
    """24 rows for day index d: hour, weekday, and each VAR at each lag (all observed before day d)."""
    dow = dt.date.fromisoformat(days[d]["date"]).weekday()
    cols = [np.arange(24), np.full(24, dow)]
    for v in VARS:
        for back in lag_days(base):
            cols.append(days[d - back][v])
    return np.column_stack(cols)


def dataset(days: list[dict], first: int, last: int, base: int) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    X = np.vstack([features(days, d, base) for d in range(first, last + 1)])
    Y = {v: np.concatenate([days[d][v] for d in range(first, last + 1)]) for v in VARS}
    return X, Y


# ---------------------------------------------------------------------------- model
def predict_day(days: list[dict], target: int, base: int, last_observed: int) -> dict[str, dict[str, np.ndarray]]:
    """One TabPFN regressor per variable, trained on days ending at `last_observed`, predicting `target`."""
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion

    first = max(7, last_observed - TRAIN_DAYS + 1)
    X, Y = dataset(days, first, last_observed, base)
    Xq = features(days, target, base)
    out = {}
    for v in VARS:
        reg = TabPFNRegressor.create_default_for_version(ModelVersion.V2, device="cpu", n_estimators=N_ESTIMATORS)
        reg.fit(X, Y[v])
        o = reg.predict(Xq, output_type="main", quantiles=[0.1, 0.9])
        pt, lo, hi = np.asarray(o["mean"]), np.asarray(o["quantiles"][0]), np.asarray(o["quantiles"][1])
        if v != "temperature_2m":
            top = 100.0 if v == "cloud_cover" else None
            pt, lo, hi = (np.clip(a, 0, top) for a in (pt, lo, hi))
        out[v] = {"pt": pt, "lo": lo, "hi": hi}
    return out


def mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


def holdout(days: list[dict], last_observed: int, base: int) -> dict[str, dict[str, float]]:
    """Re-predict the last HOLDOUT_DAYS observed days; MAE of TabPFN vs 'same hour, `base` days earlier'."""
    tab = {v: [] for v in VARS}
    per = {v: [] for v in VARS}
    for t in range(last_observed - HOLDOUT_DAYS + 1, last_observed + 1):
        p = predict_day(days, t, base, t - base)
        for v in VARS:
            tab[v].append(mae(p[v]["pt"], days[t][v]))
            per[v].append(mae(days[t - base][v], days[t][v]))
    return {v: {"tabpfn": float(np.mean(tab[v])), "persist": float(np.mean(per[v]))} for v in VARS}


# ---------------------------------------------------------------------------- score
def comfort(t: float, rain: float, wind: float, cloud: float) -> float:
    """0-100. Transparent rule applied AFTER prediction, so the model never sees it."""
    s = 100.0
    s -= 4.0 * abs(t - 18.0)  # 18 C is the sweet spot; each degree off costs 4
    s -= 25.0 * rain  # 1 mm/h of rain is a real reason to stay in
    s -= 1.2 * max(0.0, wind - 15.0)  # calm up to 15 km/h, then it bites
    s -= 0.15 * cloud  # full overcast costs 15
    return max(0.0, min(100.0, s))


def best_windows(score: np.ndarray, is_day: np.ndarray, width: int = 2, n: int = 3) -> list[tuple[int, float]]:
    """Top-n non-overlapping daylight windows of `width` hours by mean score."""
    cands = []
    for s in range(0, 24 - width + 1):
        if is_day[s : s + width].all():
            cands.append((s, float(score[s : s + width].mean())))
    cands.sort(key=lambda x: -x[1])
    picked: list[tuple[int, float]] = []
    for s, m in cands:
        if all(abs(s - p) >= width for p, _ in picked):
            picked.append((s, m))
        if len(picked) == n:
            break
    return sorted(picked)


# ----------------------------------------------------------------------------- html
def svg_chart(pred: dict, windows: list[tuple[int, float]], is_day: np.ndarray) -> str:
    W, H, L, R, T, B = 760, 300, 48, 20, 20, 44
    pw, ph = W - L - R, H - T - B
    x = lambda h: L + h * pw / 23  # noqa: E731
    t = pred["temperature_2m"]
    tmin, tmax = float(np.floor(t["lo"].min() - 1)), float(np.ceil(t["hi"].max() + 1))
    y = lambda v: T + (tmax - v) * ph / (tmax - tmin)  # noqa: E731
    parts = [f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" aria-label="Forecast temperature, rain and best windows">']
    for h in range(24):
        if not is_day[h]:
            parts.append(f'<rect x="{x(h) - pw / 46:.1f}" y="{T}" width="{pw / 23:.1f}" height="{ph}" fill="#2B2725" opacity="0.06"/>')
    for s, _ in windows:
        parts.append(f'<rect x="{x(s) - pw / 46:.1f}" y="{T}" width="{2 * pw / 23:.1f}" height="{ph}" fill="#9E2B25" opacity="0.14"/>')
    band = " ".join(f"{x(h):.1f},{y(t['hi'][h]):.1f}" for h in range(24)) + " " + " ".join(
        f"{x(h):.1f},{y(t['lo'][h]):.1f}" for h in reversed(range(24))
    )
    parts.append(f'<polygon points="{band}" fill="#2B2725" opacity="0.10"/>')
    line = " ".join(f"{x(h):.1f},{y(t['pt'][h]):.1f}" for h in range(24))
    parts.append(f'<polyline points="{line}" fill="none" stroke="#2B2725" stroke-width="2"/>')
    rain = pred["precipitation"]["pt"]
    for h in range(24):
        bh = min(ph, rain[h] / 2.0 * ph)  # 2 mm/h = full height
        if bh > 0.5:
            parts.append(f'<rect x="{x(h) - 5:.1f}" y="{T + ph - bh:.1f}" width="10" height="{bh:.1f}" fill="#4A6FA5" opacity="0.75"/>')
    for v in range(int(tmin), int(tmax) + 1, 2):
        parts.append(f'<text x="{L - 8}" y="{y(v) + 4:.1f}" font-size="11" text-anchor="end" fill="#2B2725">{v}°</text>')
    for h in range(0, 24, 3):
        parts.append(f'<text x="{x(h):.1f}" y="{H - 24}" font-size="11" text-anchor="middle" fill="#2B2725">{h:02d}h</text>')
    parts.append(f'<text x="{L}" y="{H - 6}" font-size="11" fill="#2B2725" opacity="0.7">line: predicted °C (band 10–90%) · blue bars: predicted rain mm/h · red: best windows · grey: night</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def render(name: str, date: str, pred: dict, score: np.ndarray, windows: list[tuple[int, float]], is_day: np.ndarray, ho: dict, base: int) -> str:
    rows = "".join(
        f"<tr><td>{h:02d}:00</td><td>{pred['temperature_2m']['pt'][h]:.0f}°</td><td>{pred['precipitation']['pt'][h]:.1f}</td>"
        f"<td>{pred['wind_speed_10m']['pt'][h]:.0f}</td><td>{pred['cloud_cover']['pt'][h]:.0f}%</td><td>{score[h]:.0f}</td></tr>"
        for h in range(24)
        if is_day[h]
    )
    win = "".join(f"<li><b>{s:02d}:00–{s + 2:02d}:00</b> · score {m:.0f}</li>" for s, m in windows) or "<li>No daylight window scored above zero. Stay in, read a book.</li>"
    hrows = "".join(
        f"<tr><td>{LABELS[v]}</td><td>{ho[v]['tabpfn']:.2f}</td><td>{ho[v]['persist']:.2f}</td><td>{'yes' if ho[v]['tabpfn'] < ho[v]['persist'] else 'no'}</td></tr>"
        for v in VARS
    )
    baseline = "same hour yesterday" if base == 1 else f"same hour {base} days earlier"
    return f"""<!doctype html>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Touch Grass · {name} · {date}</title>
<style>
:root{{--bg:#F5EFE6;--ink:#2B2725;--red:#9E2B25}}
body{{margin:0;background:var(--bg);color:var(--ink);font:16px/1.5 system-ui,sans-serif}}
main{{max-width:800px;margin:0 auto;padding:32px 16px 48px}}
h1{{font-family:Georgia,'Times New Roman',serif;font-weight:400;font-size:2rem;margin:0 0 4px}}
h2{{font-family:Georgia,serif;font-weight:400;font-size:1.25rem;margin:32px 0 8px;border-top:1px solid rgba(43,39,37,.2);padding-top:16px}}
.sub{{opacity:.7;margin:0 0 24px}}
ul.win{{font-size:1.15rem;padding-left:20px}} ul.win b{{color:var(--red)}}
table{{border-collapse:collapse;width:100%;font-size:.95rem}} td,th{{text-align:right;padding:4px 8px;border-bottom:1px solid rgba(43,39,37,.12)}} td:first-child,th:first-child{{text-align:left}}
small{{opacity:.7}}
</style>
<main>
<h1>Touch Grass</h1>
<p class="sub">{name} · {date} · forecast on this laptop, no account, no cloud. Model: {MODEL_CARD}.</p>
<h2>Best hours to be outside</h2>
<ul class="win">{win}</ul>
{svg_chart(pred, windows, is_day)}
<h2>Daylight hours, predicted</h2>
<table><tr><th>Hour</th><th>Temp</th><th>Rain</th><th>Wind</th><th>Cloud</th><th>Score</th></tr>{rows}</table>
<p><small>Score = 100 − 4·|temp−18| − 25·rain − 1.2·max(0, wind−15) − 0.15·cloud. The model predicts the weather; this rule is applied afterwards and is yours to change.</small></p>
<h2>Was it worth running a model?</h2>
<p>Same pipeline, re-run on the last {HOLDOUT_DAYS} fully observed days, each predicted only from the days before it. Mean absolute error against what actually happened, next to the dumbest baseline: "{baseline}".</p>
<table><tr><th>Variable</th><th>TabPFN</th><th>{baseline.capitalize()}</th><th>Model better?</th></tr>{hrows}</table>
<p><small>Inputs per hour: hour of day, weekday, and each variable {", ".join(str(d) for d in lag_days(base)[:-1])} and 7 days earlier. Training window: {TRAIN_DAYS} days, {N_ESTIMATORS} ensemble members, CPU. Data: Open-Meteo. Built for the DEV Hacktoberfest Open-Source AI Challenge, Week 1 "Touch Grass". — Yvoo Lab</small></p>
</main>
"""


# ----------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lat", type=float, default=48.8566)
    ap.add_argument("--lon", type=float, default=2.3522)
    ap.add_argument("--name", default="Paris")
    ap.add_argument("--tz", default="Europe/Paris")
    ap.add_argument("--day", choices=["auto", "today", "tomorrow"], default="auto")
    ap.add_argument("--csv", help="reuse a saved pull instead of calling Open-Meteo")
    ap.add_argument("--out", default="docs/index.html")
    a = ap.parse_args()

    if a.csv:
        h = load_csv(Path(a.csv))
    else:
        h = fetch(a.lat, a.lon, a.tz)
        Path("data").mkdir(exist_ok=True)
        save_csv(h, Path("data") / f"{a.name.lower()}_hourly.csv")
    days = to_days(h)
    today = len(days) - 2  # the pull ends with today + tomorrow
    day = a.day if a.day != "auto" else ("tomorrow" if dt.datetime.now().hour >= 18 else "today")
    base = 1 if day == "today" else 2
    target = today + base - 1
    last_observed = today - 1  # yesterday is the last day whose 24 hours are all behind us
    assert last_observed - HOLDOUT_DAYS - 7 >= 1, "need at least ~12 full days of history"

    ho = holdout(days, last_observed, base)
    pred = predict_day(days, target, base, last_observed)
    score = np.array([comfort(*(pred[v]["pt"][hh] for v in VARS)) for hh in range(24)])
    is_day = days[target]["is_day"] > 0
    windows = best_windows(score, is_day)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(a.name, days[target]["date"], pred, score, windows, is_day, ho, base), encoding="utf-8")
    print(f"{a.name} {days[target]['date']} ({day})  best windows: " + ", ".join(f"{s:02d}-{s + 2:02d}h ({m:.0f})" for s, m in windows))
    for v in VARS:
        print(f"  holdout MAE {v:<16} tabpfn {ho[v]['tabpfn']:.2f}  persistence {ho[v]['persist']:.2f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
