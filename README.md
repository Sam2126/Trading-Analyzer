# Trading-Performance-Analyzer

## 📌 Overview

This project analyzes the relationship between **Bitcoin market sentiment (Fear & Greed Index)** and **trader performance** using historical trading data from Hyperliquid.

The goal is to uncover behavioral patterns, evaluate trading outcomes across sentiment zones, and derive actionable trading strategies.

---

## 📂 Dataset Used

### 1. Bitcoin Sentiment Dataset

* Columns: `date`, `classification`
* Categories:

  * Extreme Fear
  * Fear
  * Neutral
  * Greed
  * Extreme Greed

### 2. Historical Trader Data (Hyperliquid)

* Columns include:

  * Account
  * Coin
  * Side (BUY/SELL)
  * Execution Price
  * Size
  * Timestamp
  * Closed PnL
  * Leverage
  * Direction

---

## ⚙️ Methodology

1. **Data Cleaning**

   * Converted timestamps to date format
   * Handled missing and invalid PnL values
   * Standardized sentiment labels

2. **Data Merging**

   * Merged datasets on `date`

3. **Feature Engineering**

   * Created `is_win` column
   * Normalized trade direction (Long/Short)

4. **Analysis Performed**

   * Average PnL by sentiment
   * Win rate by sentiment
   * Trade volume analysis
   * Trader and coin performance
   * Sentiment lag effect (1-day delay)

5. **Visualization**

   * Bar charts (PnL, volume)
   * Win rate comparison
   * PnL distribution (box plot)
   * Monthly heatmap

---

## 🖥️ Project Files

```
PRIMETRADE/
│
├── analysis.py                  # Main analysis script
├── index.html                  # Dashboard (visual presentation)
├── trader_sentiment_analysis.png # Generated visualization
├── historical_data.csv         # Trade dataset
├── fear_greed_index.csv        # Sentiment dataset
└── README.md                   # Project documentation
```

---


## 🚀 How to Run

1. Install dependencies:

```
pip install pandas numpy matplotlib
```

2. Run analysis:

```
python analysis.py
```

3. Open dashboard:

```
Open index.html in browser
```

---

## 🧠 Conclusion

Market sentiment plays a significant role in trading outcomes.
However, **risk management and leverage control** are stronger predictors of long-term success than sentiment alone.

---

## 👤 Author

**Samarth Khandelwal**

---
