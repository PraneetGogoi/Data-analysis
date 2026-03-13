# ⚡ SILVER MOON — MARKET ANALYTICS PROJECT

A 10-year silver price (XAG/USD) analytics project featuring interactive
dashboard themes built with vanilla HTML, CSS, and Chart.js.

<img width="1710" height="986" alt="Screenshot 2026-03-14 at 12 57 51 AM" src="https://github.com/user-attachments/assets/cef5fe85-1c9a-4cf1-bdf6-b2bbd5421803" />


---

## 📁 FOLDER STRUCTURE

```
SILVER_EDGERUNNERS_PROJECT/
│
├── dashboards/
│   ├── 01_pink_anime_dashboard.html     ← Kawaii pink anime theme (✨ Silver Moon)
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
