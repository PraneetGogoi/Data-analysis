# ⚡ SILVER EDGERUNNERS — MARKET ANALYTICS PROJECT

A 10-year silver price (XAG/USD) analytics project featuring three interactive
dashboard themes built with vanilla HTML, CSS, and Chart.js.

---

## 📁 FOLDER STRUCTURE

```
SILVER_EDGERUNNERS_PROJECT/
│
├── dashboards/
│   ├── 01_pink_anime_dashboard.html     ← Kawaii pink anime theme (✨ Silver Moon)
│   ├── 02_cyberpunk_dashboard.html      ← Dark cyberpunk / Night City theme
│   └── 03_edgerunners_dashboard.html    ← Cyberpunk Edgerunners (yellow + magenta)
│
├── dataset/
│   └── silver_prices_10y.csv            ← Raw silver price data (2016–2026)
│
├── notebook/
│   └── model.ipynb                      ← Jupyter notebook with EDA & visualizations
│
├── assets/
│   └── edgerunners_reference.png        ← UI theme reference image
│
└── README.md                            ← This file
```

---

## 📊 DATASET

**File:** `dataset/silver_prices_10y.csv`  
**Range:** January 2016 — January 2026  
**Rows:** ~2,607 trading days  

| Column         | Description                        |
|----------------|------------------------------------|
| Date           | Trading date                       |
| Open           | Opening price (USD)                |
| High           | Daily high (USD)                   |
| Low            | Daily low (USD)                    |
| Close          | Closing price (USD)                |
| Adj Close      | Adjusted closing price             |
| Volume         | Daily trading volume               |
| Daily_Return   | % change from previous close       |
| MA_20          | 20-day moving average              |
| MA_50          | 50-day moving average              |
| MA_200         | 200-day moving average             |
| Volatility_20  | 20-day rolling volatility          |
| Year           | Year                               |
| Month          | Month (1–12)                       |
| Day_of_Week    | Day of week (0=Mon)                |
| Quarter        | Quarter (1–4)                      |

**Key Stats:**
- All-Time High: $35.36 (Dec 25, 2025)
- All-Time Low:  $12.44 (Apr 04, 2016)
- Best Year:     +48.9% (2024)
- Worst Year:    -15.7% (2018)
- Total Return:  +123% over 10 years

---

## 🖥️ DASHBOARDS

All dashboards are **standalone HTML files** — just open in any modern browser.
No server, no installation needed. The dataset is embedded directly in each file.

### 01 — Pink Anime (Silver Moon ✨)
Soft kawaii aesthetic with floating sparkles, pastel pinks and lavenders,
Pacifico + Quicksand fonts, animated card reveals.

### 02 — Cyberpunk (Night City)
Dark terminal aesthetic with red neon `#FF003C`, scanlines, CRT scan beam,
glitch animations, Bebas Neue + Share Tech Mono fonts.

### 03 — Edgerunners (Reference Match)
Directly matched to the Cyberpunk Edgerunners anime artwork:
electric yellow `#FFE600` + hot magenta `#FF2D78`, Black Han Sans title font,
custom crosshair cursor with ring, animated iris loader, yellow border treatment.

---

## 📈 CHARTS INCLUDED (all 3 dashboards)

1. Price + Moving Averages (MA20 / MA50 / MA200) — with 1Y/3Y/5Y/All filter
2. Annual Return Matrix (2016–2025)
3. Monthly Seasonality — average return by month
4. 20-Day Rolling Volatility
5. Daily Return Distribution histogram
6. Drawdown from All-Time High
7. Year-over-Year Performance Overlay (2020–2025)
8. Average Volume by Year
9. Monthly Hi-Lo Price Range (last 3 years)
10. Feature Correlation Heatmap

---

## 🔧 TECH STACK

- **HTML5 / CSS3 / Vanilla JavaScript** — no frameworks
- **Chart.js 4.4.1** — all charts (CDN)
- **Google Fonts** — via CDN
- Data embedded as JSON inside each HTML file

---

*Built with Claude Sonnet · Night City Exchange · v2.077*
