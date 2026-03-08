---

# 🎓 Global Placement Analytics & Prediction

An end-to-end data science project that analyzes global student placement trends and provides a real-time, animated dashboard for placement probability and salary prediction.

---

## 🚀 Overview

This project bridges the gap between historical data analysis and predictive modeling. It features a custom-built **Employability Index** and utilizes **Random Forest** algorithms to help students understand their market value based on academic and practical credentials.

### Key Visualizations

* **Dynamic Placement Predictor:** Interactive sidebar to simulate student profiles.
* **Animated Market Trends:** Real-time updating charts for salary distributions by country and industry.
* **Feature Importance:** Visual breakdown of what drives placement success (CGPA vs. Internships vs. Skills).

---

## 🛠️ Features & Tech Stack

### 🧠 Machine Learning Model

* **Classification:** Predicting `Placement Status` using `RandomForestClassifier`.
* **Regression:** Predicting `Expected Salary` for placed candidates using `RandomForestRegressor`.
* **Feature Engineering:** * $Academic Strength = \frac{CGPA}{1 + (Backlogs \times 0.5)}$
* $Practical Score = Internships \times Quality Score$
* $Employability Index = (Academic \times 0.35) + (Practical \times 0.35) + (Skills \times 0.30)$



### 📊 Dashboard (Interactive & Animated)

* **Framework:** Plotly Dash
* **Animations:** Fluid transitions for bar charts and scatter plots using `transition_duration`.
* **Responsiveness:** Grid-based layout for mobile and desktop viewing.

---

## 📁 Repository Structure

```text
├── data/
│   └── global_placement.csv      # Dataset containing 10,000+ student records
├── notebooks/
│   └── model.ipynb               # Exploratory Data Analysis & Model Training
├── app_animated.py               # Main Dash application with animations
├── requirements.txt              # Project dependencies
└── README.md                     # Project documentation

```

---

## ⚡ Getting Started

1. **Clone the repo:**
```bash
git clone https://github.com/PraneetGogoi/Data-analysis.git
cd Data-analysis

```


2. **Install Dependencies:**
```bash
pip install -r requirements.txt

```


3. **Run the Dashboard:**
```bash
python app_animated.py

```


*The dashboard will be available at `http://127.0.0.1:8050/*`

---

## 📈 Dashboard Preview

The dashboard includes several key sections:

1. **KPI Row:** Real-time counters for Global Placement Rate and Median Salary.
2. **Predictor Gauge:** An animated gauge showing your % chance of being placed.
3. **Market Comparison:** Side-by-side animated analysis of Country-wise salary vs. CGPA correlation.

![Uploading Screenshot 2026-03-08 at 11.42.04 AM.png…]()


---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://www.google.com/search?q=https://github.com/PraneetGogoi/Data-analysis/issues).

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.

---
