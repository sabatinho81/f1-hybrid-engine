# 🏎️ F1 Hybrid Intelligence & Prediction Engine

A state-of-the-art machine learning framework and interactive Streamlit dashboard designed to forecast Formula 1 qualifying grids, podium probabilities, and grand prix race outcomes. Built with an optimized **XGBoost Classifier** and a **5,000-iteration Monte Carlo simulation engine**, this project bridges historical telemetry, current constructor performance, and real-time weather dynamics to decode the art of race strategy.

---

## 🌟 Why This Project Stands Out

Formula 1 prediction is notoriously difficult because a single variable—a safety car, a sudden downpour, or a bad grid slot—can completely rewrite the podium. Traditional statistical models rely on static point totals, missing the chaotic variance of race day. 

This engine solves that problem through a **hybrid architecture**:
1. **Deterministic Pre-Qualifying Modeling:** Uses a balanced multi-factor pace score to project single-lap performance before sessions even begin.
2. **Probabilistic Monte Carlo Simulation:** Rather than spitting out a single rigid guess, it simulates the race $5,000$ times with randomized noise, mapping out true probabilistic odds for every driver.
3. **Zero Data-Leakage Walk-Forward Validation:** Rigorously back-tested against historical Grand Prix data to continuously optimize hyperparameter performance and minimize real-world error.

---

## 🧠 Under the Hood: How Everything Is Calculated

The engine operates on a three-tier computational pipeline designed to balance historical talent with current car performance.

### 1. The Pre-Qualifying Pace Score (The Starting Grid)
Before a wheel turns on track, the engine estimates a driver's single-lap pace using a weighted mathematical blend of **Recent Form** and **Track History**:

$$\text{Pace Score} = (0.4 \times \text{Recent Form}) + (0.6 \times \text{Track Average})$$

* **Recent Form ($40\%$ Weight):** The driver’s rolling average finishing position over their last 3 races. This captures immediate car upgrades and current momentum.
* **Track History ($60\%$ Weight):** The driver's historical expanding average finish at that *specific circuit*. Optimization loops proved that aerodynamic tracks dictate long-term performance far more reliably than short-term streaks.
* *Note:* A lower Pace Score equals a faster expected grid position.

### 2. Machine Learning Classification (XGBoost)
Once the starting grid is established, an **XGBoost Classifier** evaluates each driver's probability of scoring a podium finish ($\text{Position} \le 3$). 
* **Optimized Parameters:** Tuned via extensive walk-forward cross-validation across modern seasons ($\text{max-depth}=3$, $\text{learning-rate}=0.03$, $\text{n-estimators}=100$).
* **Features Analyzed:** Starting grid position (~55% model importance), rolling recent form, historical track averages, and encoded constructor team strengths.

### 3. The Monte Carlo Race Engine ($N = 5,000$)
To transform machine learning probabilities into concrete race expectations, the model runs 5,000 virtual race simulations per weekend:
* **Weather Variance Injection:** Pulls live meteorological data via Open-Meteo. If precipitation is detected, a dynamic performance variance modifier is injected into the simulation matrix to simulate wet-weather chaos and strategic errors.
* **Expected Points Calculation ($E[\text{Points}]$):** Computes the weighted average points haul across all simulations using the official FIA scoring system:

$$E[\text{Points}] = \sum_{k=1}^{10} P(\text{Finish} = \text{Position}_k) \cdot \text{Points}(\text{Position}_k)$$

---

## 🚀 Key Features

* **Dual-Mode Grid Ingestion:** Seamlessly toggle between algorithmic pre-qualifying predictions or live, official session data pulled directly from the FastF1 API.
* **Dynamic Championship Tracking:** Instantly projects how simulation results will alter both the Drivers' (WDC) and Constructors' (WCC) World Championships.
* **Live Circuit Weather Integration:** Automatically detects track temperature and wet/dry status to adjust simulation volatility in real-time.
* **Professional Team Branding:** Features clean, standardized constructor names and metrics tailored for an executive-level dashboard experience.

---

## 📥 Comprehensive Installation Guide

Follow these step-by-step instructions to set up, configure, and run the F1 Prediction Engine locally on your machine.

### Step 1: Clone the Repository

```bash
git clone [https://github.com/YOUR_USERNAME/f1-hybrid-engine.git](https://github.com/YOUR_USERNAME/f1-hybrid-engine.git)
cd f1-hybrid-engine
```

### Step 2: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```


### Step 3: Run the Streamlit Dashboard

```bash
streamlit run app.py
```


