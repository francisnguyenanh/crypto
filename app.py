from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import time
from textblob import TextBlob
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score
import xgboost as xgb
import json

app = Flask(__name__)

def get_limit_for_interval(interval):
    """Chọn limit tối ưu cho từng interval để phân tích đúng khung thời gian."""
    interval_limits = {
        '15m': 1000,  # ~10.4 ngày
        '30m': 1000,  # ~20.8 ngày
        '1h': 1000,   # ~41.7 ngày
        '4h': 1000,   # ~166.7 ngày (~5.5 tháng)
        '1d': 1500,   # ~4.1 năm
        '1w': 200     # ~4 năm
    }
    return interval_limits.get(interval, 1000)  # Mặc định 1000 nếu interval không xác định

def get_binance_data(symbol, interval='1h', limit=1000):
    """Lấy dữ liệu lịch sử từ Binance API."""
    url = "https://api.binance.com/api/v3/klines"
    params = {
        'symbol': symbol,
        'interval': interval,
        'limit': limit
    }
    response = requests.get(url, params=params)
    data = response.json()
    
    df = pd.DataFrame(data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base_volume', 'taker_buy_quote_volume', 'ignore'
    ])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['close'] = df['close'].astype(float)
    df['open'] = df['open'].astype(float)
    df['high'] = df['high'].astype(float)
    df['low'] = df['low'].astype(float)
    df['volume'] = df['volume'].astype(float)
    return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

def get_sentiment(coin):
    """Phân tích tâm lý từ dữ liệu giả lập (thay thế cho dữ liệu X)."""
    mock_posts = {
        'BTC': ["Bitcoin bullish!", "BTC may correct soon.", "Institutional buying in BTC!"],
        'ETH': ["Ethereum upgrade success!", "ETH DeFi is booming!", "ETH gas fees high."],
        'XRP': ["XRP legal win!", "Bearish on XRP.", "XRP stable."],
        'ADA': ["Cardano smart contracts live!", "ADA slow development.", "ADA staking rewards."],
        'SOL': ["Solana fastest blockchain!", "SOL network issues.", "SOL ecosystem growing."],
        'MATIC': ["Polygon scaling!", "MATIC new partnerships!", "Layer 2 adoption."],
        'LINK': ["Chainlink oracles critical!", "LINK price feeds stable.", "LINK adoption growing."],
        'XLM': ["Stellar for payments!", "XLM banking partnerships.", "XLM undervalued."],
        'SUI': ["SUI breakout!", "SUI new tech promising.", "SUI early adoption."]
    }
    posts = mock_posts.get(coin, ["Neutral sentiment."])
    scores = [TextBlob(post).sentiment.polarity for post in posts]
    return sum(scores) / len(scores)

def calculate_indicators(df, coin, interval='1h'):
    """Tính toán chỉ báo kỹ thuật, điều chỉnh window theo interval."""
    # Điều chỉnh window dựa trên interval
    rsi_window = 7 if interval in ['15m', '30m'] else 14
    sma_short_window = 5 if interval in ['15m', '30m'] else 10
    sma_long_window = 20 if interval in ['15m', '30m'] else 50 if interval in ['1h', '4h'] else 100

    # SMA
    df['sma_short'] = df['close'].rolling(window=sma_short_window).mean()
    df['sma_long'] = df['close'].rolling(window=sma_long_window).mean()
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_window).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Lagged RSI
    df['rsi_lag1'] = df['rsi'].shift(1)
    df['rsi_lag2'] = df['rsi'].shift(2)
    
    # Volume SMA và xu hướng
    df['volume_sma'] = df['volume'].rolling(window=10).mean()
    df['volume_trend'] = np.where(df['volume_sma'] > df['volume_sma'].shift(1), 1, 0)
    
    # Volume Spike
    df['volume_spike'] = np.where(df['volume'] > df['volume_sma'] * 1.5, 1, 0)
    
    # OBV
    df['obv'] = (np.sign(df['close'].diff()) * df['volume']).cumsum()
    df['obv'] = df['obv'].fillna(0)
    
    # Phân kỳ OBV-Giá
    df['obv_price_divergence'] = 0
    df.loc[(df['close'].diff() > 0) & (df['obv'].diff() < 0), 'obv_price_divergence'] = -1  # Bearish
    df.loc[(df['close'].diff() < 0) & (df['obv'].diff() > 0), 'obv_price_divergence'] = 1   # Bullish
    
    # Phân kỳ RSI-Giá
    df['rsi_price_divergence'] = 0
    df.loc[(df['close'].diff() > 0) & (df['rsi'] > 70) & (df['rsi'].diff() < 0), 'rsi_price_divergence'] = -1
    df.loc[(df['close'].diff() < 0) & (df['rsi'] < 30) & (df['rsi'].diff() > 0), 'rsi_price_divergence'] = 1
    
    # MACD
    df['ema12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema26'] = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = df['ema12'] - df['ema26']
    df['signal_line'] = df['macd'].ewm(span=9, adjust=False).mean()
    
    # Bollinger Bands
    df['bb_mid'] = df['close'].rolling(window=20).mean()
    df['bb_std'] = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_mid'] + 2 * df['bb_std']
    df['bb_lower'] = df['bb_mid'] - 2 * df['bb_std']
    
    # Support và Resistance
    df['support'] = df['low'].rolling(window=sma_long_window).min()
    df['resistance'] = df['high'].rolling(window=sma_long_window).max()
    
    # Momentum
    df['momentum'] = df['close'].pct_change(periods=5) * 100
    
    # Lagged Price Change
    df['price_change_lag1'] = df['close'].pct_change(periods=1) * 100
    
    # Stochastic Oscillator
    df['lowest_low'] = df['low'].rolling(window=14).min()
    df['highest_high'] = df['high'].rolling(window=14).max()
    df['stoch_k'] = 100 * (df['close'] - df['lowest_low']) / (df['highest_high'] - df['lowest_low'])
    df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()
    
    # Williams %R
    df['williams_r'] = -100 * (df['highest_high'] - df['close']) / (df['highest_high'] - df['lowest_low'])
    
    # Average True Range (ATR)
    df['tr1'] = df['high'] - df['low']
    df['tr2'] = abs(df['high'] - df['close'].shift())
    df['tr3'] = abs(df['low'] - df['close'].shift())
    df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
    df['atr'] = df['tr'].rolling(window=14).mean()
    
    # Price Rate of Change (ROC)
    df['roc'] = ((df['close'] - df['close'].shift(12)) / df['close'].shift(12)) * 100
    
    # Commodity Channel Index (CCI)
    df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
    df['sma_tp'] = df['typical_price'].rolling(window=20).mean()
    df['mad'] = df['typical_price'].rolling(window=20).apply(lambda x: abs(x - x.mean()).mean())
    df['cci'] = (df['typical_price'] - df['sma_tp']) / (0.015 * df['mad'])
    
    # Average Directional Index (ADX)
    df['plus_dm'] = df['high'].diff()
    df['minus_dm'] = -df['low'].diff()
    df['plus_dm'] = df['plus_dm'].where(df['plus_dm'] > df['minus_dm'], 0).where(df['plus_dm'] > 0, 0)
    df['minus_dm'] = df['minus_dm'].where(df['minus_dm'] > df['plus_dm'], 0).where(df['minus_dm'] > 0, 0)
    df['tr14'] = df['tr'].rolling(window=14).sum()
    df['plus_di'] = 100 * (df['plus_dm'].rolling(window=14).sum() / df['tr14'])
    df['minus_di'] = 100 * (df['minus_dm'].rolling(window=14).sum() / df['tr14'])
    df['dx'] = 100 * abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di'])
    df['adx'] = df['dx'].rolling(window=14).mean()
    
    # Ichimoku Cloud
    df['tenkan_sen'] = (df['high'].rolling(window=9).max() + df['low'].rolling(window=9).min()) / 2
    df['kijun_sen'] = (df['high'].rolling(window=26).max() + df['low'].rolling(window=26).min()) / 2
    df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(26)
    df['senkou_span_b'] = ((df['high'].rolling(window=52).max() + df['low'].rolling(window=52).min()) / 2).shift(26)
    df['chikou_span'] = df['close'].shift(-26)
    
    # Sentiment
    df['sentiment'] = get_sentiment(coin)
    
    return df

def backtest_strategy(df):
    """Backtest chiến lược dựa trên total_score."""
    df = df.copy()
    df['returns'] = df['close'].pct_change()
    df['strategy_returns'] = 0.0
    position = 0
    
    for i in range(1, len(df)):
        if df['total_score'].iloc[i] >= 8 and position <= 0:
            position = 1
            df.loc[df.index[i], 'strategy_returns'] = df['returns'].iloc[i]
        elif df['total_score'].iloc[i] <= -8 and position >= 0:
            position = -1
            df.loc[df.index[i], 'strategy_returns'] = -df['returns'].iloc[i]
        elif position != 0:
            df.loc[df.index[i], 'strategy_returns'] = df['returns'].iloc[i] * position
    
    df['cumulative_returns'] = (1 + df['strategy_returns']).cumprod() - 1
    total_return = df['cumulative_returns'].iloc[-1] if len(df['cumulative_returns']) > 0 else 0
    sharpe_ratio = (df['strategy_returns'].mean() / df['strategy_returns'].std()) * np.sqrt(252) if df['strategy_returns'].std() != 0 else 0
    return {'total_return': total_return, 'sharpe_ratio': sharpe_ratio}

def train_trend_model(df):
    """Huấn luyện mô hình ensemble với các đặc trưng bổ sung."""
    features = ['sma_short', 'sma_long', 'rsi', 'rsi_lag1', 'rsi_lag2', 'volume_sma', 'volume_trend', 
                'volume_spike', 'obv', 'obv_price_divergence', 'rsi_price_divergence', 'macd', 
                'signal_line', 'bb_upper', 'bb_lower', 'support', 'resistance', 'momentum', 
                'price_change_lag1', 'stoch_k', 'stoch_d', 'williams_r', 'atr', 'roc', 'cci', 
                'adx', 'tenkan_sen', 'kijun_sen', 'senkou_span_a', 'senkou_span_b']
    
    df['price_change'] = (df['close'].shift(-1) > df['close']).astype(int)
    
    train_df = df[features + ['price_change']].dropna()
    if len(train_df) < 50:
        return None, None, None
    
    X = train_df[features]
    y = train_df['price_change']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {
        'logistic': LogisticRegression(max_iter=1000, random_state=42),
        'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'svm': SVC(probability=True, random_state=42),
        'xgboost': xgb.XGBClassifier(random_state=42, eval_metric='logloss')
    }
    
    trained_models = {}
    accuracies = {}
    
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        trained_models[name] = model
        y_pred = model.predict(X_test_scaled)
        accuracies[name] = accuracy_score(y_test, y_pred)
    
    ensemble = VotingClassifier(estimators=[
        ('lr', models['logistic']),
        ('rf', models['random_forest']),
        ('svm', models['svm']),
        ('xgb', models['xgboost'])
    ], voting='soft')
    
    ensemble.fit(X_train_scaled, y_train)
    ensemble_pred = ensemble.predict(X_test_scaled)
    ensemble_accuracy = accuracy_score(y_test, ensemble_pred)
    
    if ensemble_accuracy >= max(accuracies.values()):
        final_model = ensemble
        final_accuracy = ensemble_accuracy
        model_type = "Ensemble"
    else:
        best_model_name = max(accuracies, key=accuracies.get)
        final_model = trained_models[best_model_name]
        final_accuracy = accuracies[best_model_name]
        model_type = best_model_name.title()
    
    feature_importance = None
    if model_type.lower() in ['random_forest', 'xgboost']:
        feature_importance = final_model.feature_importances_
        feature_importance = dict(zip(features, feature_importance))
    elif model_type == "Ensemble":
        rf_model = trained_models['random_forest']
        feature_importance = dict(zip(features, rf_model.feature_importances_))
    
    return final_model, scaler, {'accuracy': final_accuracy, 'type': model_type, 
                                'all_accuracies': accuracies, 'feature_importance': feature_importance}

def predict_trend(df, model, scaler, model_info=None):
    """Dự đoán xu hướng giá và độ tin cậy."""
    features = ['sma_short', 'sma_long', 'rsi', 'rsi_lag1', 'rsi_lag2', 'volume_sma', 'volume_trend', 
                'volume_spike', 'obv', 'obv_price_divergence', 'rsi_price_divergence', 'macd', 
                'signal_line', 'bb_upper', 'bb_lower', 'support', 'resistance', 'momentum', 
                'price_change_lag1', 'stoch_k', 'stoch_d', 'williams_r', 'atr', 'roc', 'cci', 
                'adx', 'tenkan_sen', 'kijun_sen', 'senkou_span_a', 'senkou_span_b']
    latest = df[features].iloc[-1:].dropna()
    
    if latest.empty or model is None:
        return "Unknown", 0.0, None
    
    X_scaled = scaler.transform(latest)
    prob = model.predict_proba(X_scaled)[0]
    prediction = "Up" if prob[1] > prob[0] else "Down"
    confidence = max(prob) * 100
    
    return prediction, confidence, model_info

def generate_signals(df, interval='1h'):
    """Tạo tín hiệu mua/bán với bộ lọc nhiễu và tín hiệu kết hợp."""
    df['sma_signal'] = 0
    df['rsi_signal'] = 0
    df['volume_signal'] = 0
    df['obv_signal'] = 0
    df['macd_signal'] = 0
    df['bb_signal'] = 0
    df['sr_signal'] = 0
    df['momentum_signal'] = 0
    df['sentiment_signal'] = 0
    df['stoch_signal'] = 0
    df['williams_signal'] = 0
    df['roc_signal'] = 0
    df['cci_signal'] = 0
    df['adx_signal'] = 0
    df['divergence_signal'] = 0
    df['combined_signal'] = 0
    df['ichimoku_signal'] = 0
    
    # Bộ lọc nhiễu: Yêu cầu tín hiệu được xác nhận trong 2 cây nến liên tiếp
    confirm_candles = 2 if interval in ['15m', '30m'] else 1
    
    for i in range(confirm_candles, len(df)):
        # SMA Signal
        if (df['sma_short'].iloc[i] > df['sma_long'].iloc[i] and 
            df['sma_short'].iloc[i-1] <= df['sma_long'].iloc[i-1]):
            df.loc[df.index[i], 'sma_signal'] = 1
        elif (df['sma_short'].iloc[i] < df['sma_long'].iloc[i] and 
              df['sma_short'].iloc[i-1] >= df['sma_long'].iloc[i-1]):
            df.loc[df.index[i], 'sma_signal'] = -1
        
        # RSI Signal (động với ngưỡng điều chỉnh)
        rsi_overbought = 75 if interval in ['15m', '30m'] else 70
        rsi_oversold = 25 if interval in ['15m', '30m'] else 30
        if df['rsi'].iloc[i] < rsi_oversold and all(df['rsi'].iloc[i-j] < rsi_oversold for j in range(confirm_candles)):
            df.loc[df.index[i], 'rsi_signal'] = 1
        elif df['rsi'].iloc[i] > rsi_overbought and all(df['rsi'].iloc[i-j] > rsi_overbought for j in range(confirm_candles)):
            df.loc[df.index[i], 'rsi_signal'] = -1
        elif rsi_oversold <= df['rsi'].iloc[i] <= rsi_overbought:
            df.loc[df.index[i], 'rsi_signal'] = df['rsi_signal'].iloc[i-1]
        
        # Volume Signal (yêu cầu volume_spike)
        if df['volume_trend'].iloc[i] == 1 and df['volume_spike'].iloc[i] == 1:
            if df['sma_signal'].iloc[i] > 0:
                df.loc[df.index[i], 'volume_signal'] = 1
            elif df['sma_signal'].iloc[i] < 0:
                df.loc[df.index[i], 'volume_signal'] = -1
        
        # OBV Signal (yêu cầu xác nhận volume_spike)
        if (df['obv'].iloc[i] > df['obv'].iloc[i-1] and 
            df['close'].iloc[i] > df['close'].iloc[i-1] and 
            df['volume_spike'].iloc[i] == 1):
            df.loc[df.index[i], 'obv_signal'] = 1
        elif (df['obv'].iloc[i] < df['obv'].iloc[i-1] and 
              df['close'].iloc[i] < df['close'].iloc[i-1] and 
              df['volume_spike'].iloc[i] == 1):
            df.loc[df.index[i], 'obv_signal'] = -1
        
        # MACD Signal
        if (df['macd'].iloc[i] > df['signal_line'].iloc[i] and 
            df['macd'].iloc[i-1] <= df['signal_line'].iloc[i-1]):
            df.loc[df.index[i], 'macd_signal'] = 1
        elif (df['macd'].iloc[i] < df['signal_line'].iloc[i] and 
              df['macd'].iloc[i-1] >= df['signal_line'].iloc[i-1]):
            df.loc[df.index[i], 'macd_signal'] = -1
        
        # Bollinger Bands Signal
        if df['close'].iloc[i] <= df['bb_lower'].iloc[i]:
            df.loc[df.index[i], 'bb_signal'] = 1
        elif df['close'].iloc[i] >= df['bb_upper'].iloc[i]:
            df.loc[df.index[i], 'bb_signal'] = -1
        
        # Support/Resistance Signal
        price_range = df['resistance'].iloc[i] - df['support'].iloc[i]
        if price_range > 0:
            if abs(df['close'].iloc[i] - df['support'].iloc[i]) / price_range < 0.1:
                df.loc[df.index[i], 'sr_signal'] = 1
            elif abs(df['close'].iloc[i] - df['resistance'].iloc[i]) / price_range < 0.1:
                df.loc[df.index[i], 'sr_signal'] = -1
        
        # Momentum Signal
        if df['momentum'].iloc[i] > 0:
            df.loc[df.index[i], 'momentum_signal'] = 1
        elif df['momentum'].iloc[i] < 0:
            df.loc[df.index[i], 'momentum_signal'] = -1
        
        # Sentiment Signal
        if df['sentiment'].iloc[i] > 0.3:
            df.loc[df.index[i], 'sentiment_signal'] = 1
        elif df['sentiment'].iloc[i] < -0.3:
            df.loc[df.index[i], 'sentiment_signal'] = -1
        
        # Stochastic Signal
        if df['stoch_k'].iloc[i] < 20 and df['stoch_d'].iloc[i] < 20:
            df.loc[df.index[i], 'stoch_signal'] = 1
        elif df['stoch_k'].iloc[i] > 80 and df['stoch_d'].iloc[i] > 80:
            df.loc[df.index[i], 'stoch_signal'] = -1
        
        # Williams %R Signal
        if df['williams_r'].iloc[i] < -80:
            df.loc[df.index[i], 'williams_signal'] = 1
        elif df['williams_r'].iloc[i] > -20:
            df.loc[df.index[i], 'williams_signal'] = -1
        
        # ROC Signal
        if df['roc'].iloc[i] > 0:
            df.loc[df.index[i], 'roc_signal'] = 1
        elif df['roc'].iloc[i] < 0:
            df.loc[df.index[i], 'roc_signal'] = -1
        
        # CCI Signal
        if df['cci'].iloc[i] < -100:
            df.loc[df.index[i], 'cci_signal'] = 1
        elif df['cci'].iloc[i] > 100:
            df.loc[df.index[i], 'cci_signal'] = -1
        
        # ADX Signal (đi ngang khi ADX < 25)
        if df['adx'].iloc[i] > 25 and df['plus_di'].iloc[i] > df['minus_di'].iloc[i]:
            df.loc[df.index[i], 'adx_signal'] = 1
        elif df['adx'].iloc[i] > 25 and df['plus_di'].iloc[i] < df['minus_di'].iloc[i]:
            df.loc[df.index[i], 'adx_signal'] = -1
        elif df['adx'].iloc[i] < 25:
            df.loc[df.index[i], 'adx_signal'] = 0  # Thị trường đi ngang
        
        # Divergence Signal
        if df['obv_price_divergence'].iloc[i] == 1 or df['rsi_price_divergence'].iloc[i] == 1:
            df.loc[df.index[i], 'divergence_signal'] = 1
        elif df['obv_price_divergence'].iloc[i] == -1 or df['rsi_price_divergence'].iloc[i] == -1:
            df.loc[df.index[i], 'divergence_signal'] = -1
        
        # Combined Signal (OBV, RSI, VOL)
        if (df['rsi_signal'].iloc[i] == 1 and 
            df['obv_signal'].iloc[i] == 1 and 
            df['volume_signal'].iloc[i] == 1):
            df.loc[df.index[i], 'combined_signal'] = 2  # Mạnh hơn tín hiệu riêng lẻ
        elif (df['rsi_signal'].iloc[i] == -1 and 
              df['obv_signal'].iloc[i] == -1 and 
              df['volume_signal'].iloc[i] == -1):
            df.loc[df.index[i], 'combined_signal'] = -2
    
        # Ichimoku Cloud Signal
        if (df['close'].iloc[i] > df['senkou_span_a'].iloc[i] and 
            df['close'].iloc[i] > df['senkou_span_b'].iloc[i] and 
            df['tenkan_sen'].iloc[i] > df['kijun_sen'].iloc[i]):
            df.loc[df.index[i], 'ichimoku_signal'] = 1
        elif (df['close'].iloc[i] < df['senkou_span_a'].iloc[i] and 
              df['close'].iloc[i] < df['senkou_span_b'].iloc[i] and 
              df['tenkan_sen'].iloc[i] < df['kijun_sen'].iloc[i]):
            df.loc[df.index[i], 'ichimoku_signal'] = -1
    
    # Tổng hợp điểm số, ưu tiên combined_signal và divergence_signal
    df['total_score'] = (df['sma_signal'] + 
                         df['rsi_signal'] + 
                         df['volume_signal'] + 
                         df['obv_signal'] + 
                         df['macd_signal'] + 
                         df['bb_signal'] + 
                         df['sr_signal'] + 
                         df['momentum_signal'] + 
                         df['sentiment_signal'] * 1.5 +  # Tăng trọng số cho sentiment
                         df['stoch_signal'] + 
                         df['williams_signal'] + 
                         df['roc_signal'] + 
                         df['cci_signal'] + 
                         df['adx_signal'] + 
                         df['divergence_signal'] * 1.5 +  # Trọng số cao hơn cho phân kỳ
                         df['combined_signal'] * 2 +      # Trọng số cao hơn cho tín hiệu kết hợp
                         df['ichimoku_signal'])
    
    return df

def calculate_risk_reward(current_price, take_profit_pct, stop_loss_pct, recommendation, signal_score, trend_prediction, confidence, atr=None):
    """Tính toán rủi ro/lợi nhuận dựa trên ATR."""
    if atr:
        take_profit_pct = min(take_profit_pct, atr * 2 / current_price * 100)  # Giới hạn TP dựa trên ATR
        stop_loss_pct = min(stop_loss_pct, atr / current_price * 100)           # Giới hạn SL dựa trên ATR
    
    take_profit_price = current_price * (1 + take_profit_pct / 100)
    stop_loss_price = current_price * (1 - stop_loss_pct / 100)
    
    potential_gain = take_profit_pct
    potential_loss = stop_loss_pct
    rr_ratio = round(potential_gain / potential_loss, 2) if potential_loss > 0 else float('inf')
    
    adjusted_recommendation = recommendation
    color = "#28a745"  # Green
    
    # Điều chỉnh khuyến nghị dựa trên tín hiệu và thị trường đi ngang
    if recommendation == "HOLD":
        if signal_score >= 2 and confidence >= 60:
            adjusted_recommendation = "HOLD/WEAK BUY (có thể mua ít)"
            color = "#20c997"  # Teal
        elif signal_score <= -2 and confidence >= 60:
            adjusted_recommendation = "HOLD/WEAK SELL (có thể bán ít)"
            color = "#fd7e14"  # Orange
        else:
            adjusted_recommendation = "HOLD/WAIT (chờ tín hiệu rõ ràng hơn)"
            color = "#6c757d"  # Gray
    
    elif rr_ratio < 1.5:
        if recommendation in ["STRONG BUY", "BUY"]:
            adjusted_recommendation = "HOLD - Tỷ lệ rủi ro/lợi nhuận không tốt"
            color = "#ffc107"  # Yellow
    elif rr_ratio >= 2.0 and confidence > 70:
        if recommendation == "BUY":
            adjusted_recommendation = "STRONG BUY - Tỷ lệ rủi ro/lợi nhuận xuất sắc"
            color = "#28a745"  # Green
    elif confidence < 60:
        if recommendation in ["STRONG BUY", "BUY"]:
            adjusted_recommendation = "HOLD - Độ tin cậy thấp, nên chờ"
            color = "#ffc107"  # Yellow
    
    if recommendation in ["SELL", "STRONG SELL"]:
        color = "#dc3545"  # Red
        if confidence < 60:
            adjusted_recommendation = "HOLD - Hướng không rõ ràng"
            color = "#ffc107"  # Yellow
    
    # Cảnh báo đảo chiều nếu có tín hiệu phân kỳ
    advice = ""
    if signal_score >= 8 and 'divergence_signal' in df.columns and df['divergence_signal'].iloc[-1] == -1:
        advice = "CẢNH BÁO: Phân kỳ giảm giá (Bearish Divergence) - Có thể đảo chiều giảm."
    elif signal_score <= -8 and 'divergence_signal' in df.columns and df['divergence_signal'].iloc[-1] == 1:
        advice = "CẢNH BÁO: Phân kỳ tăng giá (Bullish Divergence) - Có thể đảo chiều tăng."
    elif "STRONG BUY" in adjusted_recommendation:
        advice = "Tín hiệu mua mạnh. Nếu chưa có thì nên mua. Nếu đã có thì có thể tăng thêm position."
    elif "BUY" in adjusted_recommendation and "WEAK" not in adjusted_recommendation:
        advice = "Tín hiệu mua tích cực. Nếu chưa có thì có thể mua. Nếu đã có thì giữ."
    elif "WEAK BUY" in adjusted_recommendation:
        advice = "Tín hiệu mua yếu. Nếu chưa có thì có thể mua ít để test. Nếu đã có thì giữ."
    elif "STRONG SELL" in adjusted_recommendation:
        advice = "Tín hiệu bán mạnh. Nếu đã có thì nên bán. Nếu chưa có thì tuyệt đối không mua."
    elif "SELL" in adjusted_recommendation and "WEAK" not in adjusted_recommendation:
        advice = "Tín hiệu bán. Nếu đã có thì có thể bán bớt. Nếu chưa có thì không nên mua."
    elif "WEAK SELL" in adjusted_recommendation:
        advice = "Tín hiệu bán yếu. Nếu đã có thì có thể bán bớt hoặc giữ. Nếu chưa có thì chờ."
    elif "WAIT" in adjusted_recommendation:
        advice = "Thị trường đi ngang, không có tín hiệu rõ ràng. Nên chờ để có tín hiệu tốt hơn."
    else:
        advice = "Tín hiệu trung tính. Nếu đã có thì giữ. Nếu chưa có thì chờ tín hiệu rõ ràng hơn."
    
    return {
        'take_profit_price': float(round(take_profit_price, 2)),
        'stop_loss_price': float(round(stop_loss_price, 2)),
        'take_profit_pct': float(take_profit_pct),
        'stop_loss_pct': float(stop_loss_pct),
        'ratio': float(rr_ratio),
        'adjusted_recommendation': adjusted_recommendation,
        'advice': advice,
        'color': color
    }

@app.route('/get_settings', methods=['GET'])
def get_settings():
    import os
    try:
        if not os.path.exists('user_settings.txt'):
            settings = {
                'symbol': 'BTCJPY',
                'interval': '1h',
                'investment_amount': 1000000,
                'take_profit': 3.0,
                'stop_loss': 1.5
            }
        else:
            with open('user_settings.txt', 'r', encoding='utf-8') as f:
                settings = json.load(f)
        return jsonify(settings)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/save_settings', methods=['POST'])
def save_settings():
    try:
        data = request.get_json()
        settings = {
            'symbol': data.get('symbol', 'BTCJPY'),
            'interval': data.get('interval', '1h'),
            'investment_amount': data.get('investment_amount', 1000000),
            'take_profit': data.get('take_profit', 3.0),
            'stop_loss': data.get('stop_loss', 1.5)
        }
        with open('user_settings.txt', 'w', encoding='utf-8') as f:
            json.dump(settings, f, ensure_ascii=False, indent=2)
        return jsonify({'success': True, 'settings': settings})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    global df  # Lưu df để sử dụng trong calculate_risk_reward
    try:
        data = request.get_json()
    except Exception as e:
        print(f"Error at get_json: {str(e)}")
        return jsonify({'error': f'get_json: {str(e)}'}), 500

    try:
        symbol = data['symbol']
        interval = data.get('interval', '1h')
        take_profit_pct = data.get('take_profit', 5.0)
        stop_loss_pct = data.get('stop_loss', 3.0)
        investment_amount = data.get('investment_amount', 0.0)
        limit = get_limit_for_interval(interval)
    except Exception as e:
        print(f"Error at parse input: {str(e)}")
        return jsonify({'error': f'parse input: {str(e)}'}), 500

    try:
        coin = symbol.replace('JPY', '')
        df = get_binance_data(symbol, interval=interval, limit=limit)
    except Exception as e:
        print(f"Error at get_binance_data: {str(e)}")
        return jsonify({'error': f'get_binance_data: {str(e)}'}), 500

    try:
        df = calculate_indicators(df, coin, interval)
    except Exception as e:
        print(f"Error at calculate_indicators: {str(e)}")
        return jsonify({'error': f'calculate_indicators: {str(e)}'}), 500

    try:
        df = generate_signals(df, interval)
    except Exception as e:
        print(f"Error at generate_signals: {str(e)}")
        return jsonify({'error': f'generate_signals: {str(e)}'}), 500

    try:
        model, scaler, model_info = train_trend_model(df)
    except Exception as e:
        print(f"Error at train_trend_model: {str(e)}")
        return jsonify({'error': f'train_trend_model: {str(e)}'}), 500

    try:
        trend, confidence, _ = predict_trend(df, model, scaler, model_info)
    except Exception as e:
        print(f"Error at predict_trend: {str(e)}")
        return jsonify({'error': f'predict_trend: {str(e)}'}), 500

    try:
        latest = df.iloc[-1]
        current_price = latest['close']
        total_score = latest['total_score']
        if total_score >= 8:
            recommendation = "STRONG BUY"
        elif total_score >= 6:
            recommendation = "BUY"
        elif total_score <= -8:
            recommendation = "STRONG SELL"
        elif total_score <= -6:
            recommendation = "SELL"
        else:
            recommendation = "HOLD"
    except Exception as e:
        print(f"Error at get latest/score/recommendation: {str(e)}")
        return jsonify({'error': f'get latest/score/recommendation: {str(e)}'}), 500

    try:
        risk_reward = calculate_risk_reward(
            current_price, take_profit_pct, stop_loss_pct, 
            recommendation, total_score, trend, confidence, latest.get('atr')
        )
    except Exception as e:
        print(f"Error at calculate_risk_reward: {str(e)}")
        return jsonify({'error': f'calculate_risk_reward: {str(e)}'}), 500

    try:
        backtest_results = backtest_strategy(df)
    except Exception as e:
        print(f"Error at backtest_strategy: {str(e)}")
        backtest_results = {'total_return': 0, 'sharpe_ratio': 0}

    try:
        investment_details = None
        if investment_amount > 0:
            coin_amount = investment_amount / current_price
            profit_amount = investment_amount * (take_profit_pct / 100)
            loss_amount = investment_amount * (stop_loss_pct / 100)
            investment_details = {
                'investment_amount': float(investment_amount),
                'coin_amount': float(coin_amount),
                'profit_amount': float(profit_amount),
                'loss_amount': float(loss_amount),
                'profit_price': float(current_price * (1 + take_profit_pct / 100)),
                'loss_price': float(current_price * (1 - stop_loss_pct / 100))
            }
    except Exception as e:
        print(f"Error at investment_details: {str(e)}")
        return jsonify({'error': f'investment_details: {str(e)}'}), 500

    try:
        chart_data = {
            'timestamps': df['timestamp'].dt.strftime('%Y-%m-%d %H:%M').tail(50).tolist(),
            'prices': [float(x) for x in df['close'].tail(50).tolist()],
            'volumes': [float(x) for x in df['volume'].tail(50).tolist()],
            'obv': [float(x) for x in df['obv'].tail(50).tolist()],
            'sma_short': [float(x) for x in df['sma_short'].tail(50).tolist()],
            'sma_long': [float(x) for x in df['sma_long'].tail(50).tolist()],
            'bb_upper': [float(x) for x in df['bb_upper'].tail(50).tolist()],
            'bb_lower': [float(x) for x in df['bb_lower'].tail(50).tolist()],
            'rsi': [float(x) for x in df['rsi'].tail(50).tolist()],
            'tenkan_sen': [float(x) for x in df['tenkan_sen'].tail(50).tolist()],
            'kijun_sen': [float(x) for x in df['kijun_sen'].tail(50).tolist()],
            'senkou_span_a': [float(x) for x in df['senkou_span_a'].tail(50).tolist()],
            'senkou_span_b': [float(x) for x in df['senkou_span_b'].tail(50).tolist()]
        }
    except Exception as e:
        print(f"Error at chart_data: {str(e)}")
        return jsonify({'error': f'chart_data: {str(e)}'}), 500

    def to_native(obj):
        if isinstance(obj, dict):
            return {k: to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [to_native(x) for x in obj]
        elif isinstance(obj, tuple):
            return tuple(to_native(x) for x in obj)
        elif isinstance(obj, (np.floating, float)):
            return float(obj)
        elif isinstance(obj, (np.integer, int)):
            return int(obj)
        else:
            return obj

    try:
        safe_chart_data = to_native(chart_data)
        safe_risk_reward = to_native(risk_reward)
        safe_investment_details = to_native(investment_details) if investment_details else None
        def safe_float(val, ndigits=2, default=0.0):
            try:
                if pd.isna(val) or val is None:
                    return default
                return float(round(float(val), ndigits))
            except:
                return default

        safe_indicators = to_native({
            'rsi': safe_float(latest.get('rsi'), 2),
            'rsi_lag1': safe_float(latest.get('rsi_lag1'), 2),
            'macd': safe_float(latest.get('macd'), 4),
            'signal_line': safe_float(latest.get('signal_line'), 4),
            'bb_upper': safe_float(latest.get('bb_upper'), 2),
            'bb_lower': safe_float(latest.get('bb_lower'), 2),
            'support': safe_float(latest.get('support'), 2),
            'resistance': safe_float(latest.get('resistance'), 2),
            'momentum': safe_float(latest.get('momentum'), 2),
            'obv': safe_float(latest.get('obv'), 2),
            'stoch_k': safe_float(latest.get('stoch_k'), 2),
            'stoch_d': safe_float(latest.get('stoch_d'), 2),
            'williams_r': safe_float(latest.get('williams_r'), 2),
            'roc': safe_float(latest.get('roc'), 2),
            'cci': safe_float(latest.get('cci'), 2),
            'adx': safe_float(latest.get('adx'), 2),
            'sentiment': safe_float(latest.get('sentiment'), 2),
            'volume_trend': 'Increasing' if latest.get('volume_trend', 0) == 1 else 'Stable/Decreasing',
            'tenkan_sen': safe_float(latest.get('tenkan_sen'), 2),
            'kijun_sen': safe_float(latest.get('kijun_sen'), 2),
            'senkou_span_a': safe_float(latest.get('senkou_span_a'), 2),
            'senkou_span_b': safe_float(latest.get('senkou_span_b'), 2)
        })
    except Exception as e:
        print(f"Error at indicators/safe_float: {str(e)}")
        return jsonify({'error': f'indicators/safe_float: {str(e)}'}), 500

    try:
        analysis_result = {
            'symbol': str(symbol),
            'current_price': float(round(float(current_price), 2)),
            'recommendation': str(recommendation),
            'trend_prediction': str(trend),
            'confidence': float(round(float(confidence), 2)),
            'signal_score': int(total_score),
            'risk_reward': safe_risk_reward,
            'investment_details': safe_investment_details,
            'indicators': safe_indicators,
            'model_info': to_native(model_info) if model_info else {'accuracy': 0, 'type': 'Unknown'},
            'chart_data': safe_chart_data,
            'last_timestamp': str(latest['timestamp']) if 'timestamp' in latest else None,
            'backtest_results': to_native(backtest_results)
        }
        analysis_result = to_native(analysis_result)
        return jsonify(analysis_result)
    except Exception as e:
        print(f"Error at jsonify/return: {str(e)}")
        return jsonify({'error': f'jsonify/return: {str(e)}'}), 500

@app.route('/suggest_best_pairs', methods=['POST'])
def suggest_best_pairs():
    try:
        data = request.get_json()
        interval = data.get('interval', '1h')
        take_profit_pct = data.get('take_profit', 5.0)
        stop_loss_pct = data.get('stop_loss', 3.0)
        investment_amount = data.get('investment_amount', 0.0)
        limit = get_limit_for_interval(interval)

        pairs = [
            'BTCJPY', 'ETHJPY', 'XRPJPY', 'ADAJPY', 'SOLJPY',
            'MATICJPY', 'LINKJPY', 'XLMJPY', 'SUIJPY'
        ]

        suggestions = []

        for symbol in pairs:
            try:
                coin = symbol.replace('JPY', '')
                df = get_binance_data(symbol, interval=interval, limit=limit)
                df = calculate_indicators(df, coin, interval)
                df = generate_signals(df, interval)

                model, scaler, model_info = train_trend_model(df)
                trend, confidence, _ = predict_trend(df, model, scaler, model_info)

                latest = df.iloc[-1]
                current_price = latest['close']
                total_score = latest['total_score']

                if total_score >= 8:
                    recommendation = "STRONG BUY"
                elif total_score >= 6:
                    recommendation = "BUY"
                elif total_score <= -8:
                    recommendation = "STRONG SELL"
                elif total_score <= -6:
                    recommendation = "SELL"
                else:
                    recommendation = "HOLD"

                risk_reward = calculate_risk_reward(
                    current_price, take_profit_pct, stop_loss_pct, 
                    recommendation, total_score, trend, confidence, latest.get('atr')
                )

                composite_score = 0
                if recommendation in ["STRONG BUY", "BUY"]:
                    composite_score = total_score + confidence/10
                    if risk_reward['ratio'] >= 2.0:
                        composite_score += 5
                    elif risk_reward['ratio'] >= 1.5:
                        composite_score += 2
                    if latest.get('combined_signal', 0) == 2:
                        composite_score += 3  # Bonus cho tín hiệu kết hợp mạnh
                    if latest.get('divergence_signal', 0) == 1:
                        composite_score += 2  # Bonus cho phân kỳ tăng
                elif recommendation == "HOLD" and total_score >= 2:
                    composite_score = total_score + confidence/20
                else:
                    composite_score = -100

                backtest_results = backtest_strategy(df)

                suggestion = {
                    'symbol': symbol,
                    'current_price': round(current_price, 2),
                    'recommendation': recommendation,
                    'signal_score': int(total_score),
                    'confidence': round(confidence, 2),
                    'trend_prediction': trend,
                    'composite_score': composite_score,
                    'risk_reward': risk_reward,
                    'model_accuracy': round(model_info['accuracy'] * 100, 1) if model_info else 0,
                    'backtest_results': backtest_results
                }

                suggestions.append(suggestion)

            except Exception as e:
                print(f"Error analyzing {symbol}: {str(e)}")
                continue
        
        suggestions.sort(key=lambda x: x['composite_score'], reverse=True)
        top_suggestions = suggestions[:2]
        
        if len(top_suggestions) < 2:
            return jsonify({'error': 'Không đủ cặp tiền tệ có tín hiệu tích cực để đề xuất'}), 400
        
        result = {
            'suggestions': top_suggestions,
            'take_profit': take_profit_pct,
            'stop_loss': stop_loss_pct,
            'investment_amount': investment_amount,
            'analysis_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)