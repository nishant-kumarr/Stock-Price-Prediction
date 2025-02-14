# Stock Price Direction Prediction using LSTM

## Overview
Predicting stock price movement is a highly challenging task due to the volatility and unpredictability of financial markets. This project serves as an early-stage exploration into Quantitative Finance (Quant Finance) by leveraging Long Short-Term Memory (LSTM) networks—a powerful deep learning approach for sequential data—to analyze trends and fluctuations in stock prices. By integrating advanced feature engineering, market volatility management, and interpretability tools, this project aims to provide a structured approach to understanding AI-driven financial modeling.

## Why This Project Matters in Quantitative Finance
Quantitative Finance is the backbone of modern financial markets, driving algorithmic trading, risk management, and investment strategies. Leading quant firms like Renaissance Technologies, Citadel, and Two Sigma collectively manage over $100 billion in assets, generating multi-billion-dollar annual revenues by leveraging machine learning and statistical models to execute trades at high frequencies.

This project is a small-scale implementation designed to understand how deep learning can be applied to market prediction—a field where over 80% of trades on major exchanges are algorithmic. While not at the scale of professional quant firms, this approach offers insights into how AI-driven trading models operate and adapt to market fluctuations.

## Why LSTM for Stock Prediction?
Stock price movements depend on historical patterns, market fluctuations, and investor sentiment. Unlike traditional machine learning models, LSTMs capture temporal dependencies and long-range correlations, making them ideal for analyzing market behavior. This helps:

- Prevent critical information loss in sequential data.
- Reduce noise interference from short-term fluctuations.
- Enhance prediction of short-term volatility and long-term trends.

## How to Use

### Cloning the Repository
Clone the project repository to your local machine to access all necessary files and scripts.

### Installing Dependencies
Ensure all required libraries are installed by running:

```bash
pip install -r requirements.txt
```

### Key dependencies include:
- `yfinance`: Fetches historical stock data.
- `pandas_ta`: Computes technical indicators for enhanced feature engineering.
- `keras-self-attention`: Implements self-attention mechanisms for improved feature importance recognition.
- `imbalanced-learn`: Handles imbalanced datasets using resampling techniques.
- `shap`: Provides interpretability to model predictions.
- `seaborn`: Generates insightful financial visualizations.

### Downloading Stock Data
```python
import yfinance as yf
ticker = 'AAPL'
data = yf.download(ticker, start='2020-01-01', end='2023-01-01')
```

## Advanced Data Preprocessing

### Handling Market Noise & Missing Values
Financial data often has missing values. This dataset undergoes rigorous cleaning, using trend-based interpolation to retain time-series integrity.

### Feature Scaling for Market Volatility Management
`MinMaxScaler` ensures uniform feature scaling, reducing the impact of extreme market fluctuations.

### Technical Indicators for Enhanced Predictive Power
- `Moving Averages`, `Bollinger Bands`, `RSI`, `MACD`, and other key indicators provide signals for trend reversals and momentum shifts.
- `Principal Component Analysis (PCA)` : Reduces dimensionality, retaining key market signals while eliminating redundant data.

## Handling Market Imbalance & Rare Events
Financial markets exhibit class imbalance, where sharp price movements (crashes, spikes) occur less frequently but have high impact. This model addresses this challenge by:

- `Synthetic Minority Over-sampling Technique (SMOTE)` : Generates synthetic samples for underrepresented price movement categories, improving the model's ability to detect rare but critical events.
- `Class Weighting` : Dynamically adjusts weights to prevent prediction bias toward the majority trend.

## Training the LSTM Model
Stock prices are influenced by historical patterns, investor sentiment, and external market factors. The model architecture is designed to maximize insight extraction:

- `Bidirectional LSTM Layers` : Learn from past and future trends simultaneously, improving market trend forecasting.
- `Self-Attention Mechanism` : Highlights key moments in price movement, ensuring critical market signals are not diluted.
- `Dropout Layers & Regularization` : Reduce overfitting, ensuring the model generalizes well to unseen data.
- `Adam Optimizer & Early Stopping` : Adaptive optimization improves learning efficiency, preventing excessive training cycles.

## Evaluating Market Prediction Performance
To ensure robustness, the model undergoes rigorous evaluation:

- `Confusion Matrix & Classification Report` : Measures precision, recall, and F1-score.
- `ROC Curve & AUC Score` : Assesses classification confidence in stock price movements.
- `SHAP (SHapley Additive Explanations)` : Ensures transparency by revealing which factors drive stock price predictions.

## Working Mechanism
1. `Data Acquisition`: Fetches historical stock data from Yahoo Finance.
2. `Preprocessing & Feature Engineering` : Cleans, normalizes, and enhances data with financial indicators.
3. `Dimensionality Reduction & Balancing` : Applies PCA and SMOTE to refine dataset quality.
4. `Model Training` : LSTM-based deep learning model is fine-tuned for stock price prediction.
5. `Performance Evaluation` : Assesses the model’s reliability using multiple evaluation metrics.
6. `Interpretability Analysis` : SHAP visualizations reveal influential stock price movement factors.

## Files & Components
- `stock_price_prediction.ipynb`: Jupyter Notebook containing data preprocessing, model training, and evaluation.
- `AAPL_historical_data.csv`: Raw stock data downloaded from Yahoo Finance.
- `AAPL_historical_data_output.csv`: Preprocessed stock data optimized for modeling.
- `requirements.txt`: List of dependencies for seamless execution.

## Cutting-Edge Technologies & Features
- `Python & TensorFlow` : Core deep learning framework for stock prediction.
- `LSTM & Self-Attention` : Enhances time-series forecasting accuracy by capturing intricate dependencies.
- `SMOTE` : Balances classes, improving rare event detection.
- `Yahoo Finance API` : Fetches real-time and historical stock data for market analysis.
- `Seaborn & SHAP` : Provides interactive financial visualizations and interpretability.
- `PCA` : Ensures the model focuses on meaningful price movement patterns.

## Why This Project is a First Step into Quantitative Finance
While this project is not a full-scale trading model, it serves as an introductory exploration into how deep learning can support Quantitative Finance applications. Unlike traditional models, this approach:

- Detects trend reversals and momentum shifts with LSTMs and self-attention.
- Handles market fluctuations more robustly, reducing prediction noise.
- Improves interpretability of AI-driven financial models, crucial for real-world algorithmic trading applications.

## Future Directions
Quantitative finance is a rapidly evolving field. Potential improvements include:

- Integrating real-time trading algorithms.
- Expanding to multi-asset portfolios and macroeconomic indicators.
- Refining risk assessment by incorporating advanced volatility models.

## Final Thoughts
This project provides a foundation for understanding AI-driven market analysis, making stock price movement forecasting more precise, explainable, and actionable. While it is an early-stage model, it reflects key principles used in Quantitative Finance, demonstrating how AI can enhance market prediction strategies.
