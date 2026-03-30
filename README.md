# Market Regimes and Trading Signals

A quantitative research project exploring how market regime classification can be used to dynamically adjust portfolio exposure and evaluate its impact on risk and return relative to a passive benchmark.

---

## Overview

This project develops a regime-aware investment framework that:

- classifies market conditions using price and macroeconomic features  
- maps those regimes to portfolio exposure decisions  
- evaluates performance against a buy-and-hold benchmark  
- separates signal generation from portfolio construction  

The core idea is to distinguish between:

- market understanding (regime classification)  
- portfolio decision-making (exposure mapping)  

---

## Motivation

Financial markets exhibit different regimes over time, such as trending periods, volatility spikes, and drawdowns. A static investment strategy may not respond effectively to these changing conditions.

This project investigates whether adjusting portfolio exposure based on identified regimes can improve portfolio characteristics, particularly in terms of risk management.

---

## Methodology

### Feature Engineering

Market conditions are described using a set of features, including:

- moving averages to capture trend  
- realized volatility  
- implied volatility (VIX)  
- drawdowns  
- interest rate proxies  

---

### Regime Classification

Each observation is classified into one of four regimes:

- Risk-on: strong trend, low volatility  
- Recovery: improving conditions with elevated volatility  
- Stress: unstable or deteriorating conditions  
- Risk-off: severe stress or drawdown  

---

### Signal Construction

Regimes are mapped to portfolio exposure levels. The strategy does not predict returns directly but adjusts risk exposure based on the current regime.

Example mapping:

- Risk-on: full exposure  
- Recovery: moderately high exposure  
- Stress: reduced exposure  
- Risk-off: minimal or zero exposure  

---

### Backtesting

The backtest is designed to reflect realistic trading conditions:

- signals are lagged by one day to avoid look-ahead bias  
- transaction costs are applied based on turnover  
- performance is compared to a buy-and-hold benchmark  
- the dataset is divided into train, validation, and test periods  

---

### Performance Evaluation

Performance is evaluated using standard metrics:

- total return and CAGR  
- annualized volatility  
- Sharpe ratio  
- Sortino ratio  
- maximum drawdown  
- Calmar ratio  

---

## Results

The initial implementation demonstrated that:

- the regime classification is economically meaningful  
- the baseline exposure mapping was overly conservative  
- adjusting exposure levels significantly improved performance  

However, relative to a buy-and-hold benchmark:

- the strategy reduces volatility and drawdowns  
- but sacrifices upside participation  
- resulting in lower total return and lower Sharpe ratio  

This suggests that the framework functions effectively as a risk management overlay, but requires further refinement to compete on a risk-adjusted basis.

---

## Key Insights

- regime classification and portfolio construction should be treated as separate components  
- performance is highly sensitive to how regimes are mapped to exposure  
- reducing risk does not guarantee improved risk-adjusted returns  
- systematic experimentation is required to isolate model vs portfolio effects  

---

## Limitations

- single-asset focus (SPY)  
- rule-based regime definitions  
- fixed parameter thresholds  
- no leverage or multi-asset diversification  
- simplified transaction cost model  

---

## Future Work

Potential extensions include:

- optimizing exposure mappings using systematic methods  
- extending to multi-asset portfolios  
- incorporating machine learning techniques for regime detection  
- modelling transaction costs in more detail  
- introducing probabilistic or dynamic regime transitions  

---

## Project Structure
market-regimes-trading-signals/
│
├── src/
│ ├── config.py
│ ├── data.py
│ ├── features.py
│ ├── regimes.py
│ ├── signals.py
│ ├── backtest.py
│ ├── metrics.py
│
├── notebooks/
│ └── main.ipynb
│
├── data/
├── results/
│ ├── tables/
│
└── README.md


---

## How to Run

1. Clone the repository:

   git clone <repository-url>  
   cd market-regimes-trading-signals  

2. Install dependencies:

   pip install -r requirements.txt  

3. Run the notebook:

   jupyter notebook notebooks/main.ipynb  

---

## Summary

This project demonstrates a structured approach to quantitative research, combining:

- feature engineering  
- regime classification  
- signal construction  
- realistic backtesting  
- risk-adjusted performance evaluation  

It highlights that improving a strategy often depends as much on portfolio construction as on the underlying model.