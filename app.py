import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import time
from textblob import TextBlob  # For mock sentiment analysis
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import json

def get_binance_data(symbol, interval='1h', limit=500):
    """Fetch historical klines from Binance public API."""
    url = "https://api.binance.com/api/v3/klines"
    params = {
        'symbol': symbol,
        'interval': interval,
        'limit': limit
    }
    response = requests.get(url, params=params)
    data = response.json()
    
    # Convert to DataFrame
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

def get_mock_sentiment(coin):
    """Mock sentiment analysis for the given coin (simulating X posts)."""
    mock_posts = {
        'XRP': ["XRP to the moon!", "Bearish on XRP, dumping soon", "XRP looks stable"],
        'SUI': ["SUI is breaking out!", "Not sure about SUI", "SUI has potential"],
        'XLM': ["XLM undervalued!", "XLM is dead", "XLM steady growth"]
    }
    posts = mock_posts.get(coin, ["Neutral comment"])
    scores = [TextBlob(post).sentiment.polarity for post in posts]
    avg_score = sum(scores) / len(scores)
    return avg_score

def get_real_sentiment(coin):
    """Placeholder for real X API sentiment analysis."""
    # Requires X Premium API access (e.g., tweepy or python-x)
    # Example implementation (commented out):
    """
    import tweepy
    client = tweepy.Client(bearer_token='your_bearer_token')
    tweets = client.search_recent_tweets(query=f"{coin} crypto -is:retweet", max_results=10)
    scores = [TextBlob(tweet.text).sentiment.polarity for tweet in tweets.data]
    return sum(scores) / len(scores) if scores else 0
    """
    return get_mock_sentiment(coin)  # Fallback to mock sentiment

def calculate_indicators(df, coin):
    """Calculate technical indicators and sentiment."""
    # SMA
    df['sma_short'] = df['close'].rolling(window=10).mean()
    df['sma_long'] = df['close'].rolling(window=50).mean()
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Lagged RSI
    df['rsi_lag1'] = df['rsi'].shift(1)
    df['rsi_lag2'] = df['rsi'].shift(2)
    
    # Volume SMA and trend
    df['volume_sma'] = df['volume'].rolling(window=10).mean()
    df['volume_trend'] = np.where(df['volume_sma'] > df['volume_sma'].shift(1), 1, 0)
    
    # On-Balance Volume (OBV)
    df['obv'] = (np.sign(df['close'].diff()) * df['volume']).cumsum()
    df['obv'] = df['obv'].fillna(0)
    
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
    
    # Support and Resistance
    df['support'] = df['low'].rolling(window=50).min()
    df['resistance'] = df['high'].rolling(window=50).max()
    
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
    
    # Sentiment (mocked or real)
    df['sentiment'] = get_real_sentiment(coin)
    
    return df

def train_trend_model(df):
    """Train ensemble models with XGBoost to predict price trend."""
    features = ['sma_short', 'sma_long', 'rsi', 'rsi_lag1', 'rsi_lag2', 'volume_sma', 'volume_trend', 
                'obv', 'macd', 'signal_line', 'bb_upper', 'bb_lower', 'support', 'resistance', 
                'momentum', 'price_change_lag1', 'stoch_k', 'stoch_d', 'williams_r', 'atr', 'roc', 
                'cci', 'adx']
    df['price_change'] = (df['close'].shift(-1) > df['close']).astype(int)  # 1 for up, 0 for down
    
    # Drop rows with NaN values
    train_df = df[features + ['price_change']].dropna()
    if len(train_df) < 30:
        return None, None, None
    
    X = train_df[features]
    y = train_df['price_change']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train models
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
        print(f"{name.title()} - Test Accuracy: {accuracies[name]:.3f}")
    
    # Ensemble model
    ensemble = VotingClassifier(
        estimators=[
            ('lr', models['logistic']),
            ('rf', models['random_forest']),
            ('svm', models['svm']),
            ('xgb', models['xgboost'])
        ],
        voting='soft'
    )
    
    ensemble.fit(X_train_scaled, y_train)
    ensemble_pred = ensemble.predict(X_test_scaled)
    ensemble_accuracy = accuracy_score(y_test, ensemble_pred)
    print(f"Ensemble Model - Test Accuracy: {ensemble_accuracy:.3f}")
    
    # Select best model
    if ensemble_accuracy >= max(accuracies.values()):
        final_model = ensemble
        final_accuracy = ensemble_accuracy
        model_type = "Ensemble"
    else:
        best_model_name = max(accuracies, key=accuracies.get)
        final_model = trained_models[best_model_name]
        final_accuracy = accuracies[best_model_name]
        model_type = best_model_name.title()
    
    # Feature importance (for Random Forest or XGBoost)
    feature_importance = None
    if model_type.lower() in ['random_forest', 'xgboost']:
        feature_importance = final_model.feature_importances_ if model_type.lower() == 'xgboost' else final_model.feature_importances_
        feature_importance = dict(zip(features, feature_importance))
    
    return final_model, scaler, {'accuracy': final_accuracy, 'type': model_type, 'feature_importance': feature_importance}

def predict_trend(df, model, scaler, model_info=None):
    """Predict price trend and confidence score."""
    features = ['sma_short', 'sma_long', 'rsi', 'rsi_lag1', 'rsi_lag2', 'volume_sma', 'volume_trend', 
                'obv', 'macd', 'signal_line', 'bb_upper', 'bb_lower', 'support', 'resistance', 
                'momentum', 'price_change_lag1', 'stoch_k', 'stoch_d', 'williams_r', 'atr', 'roc', 
                'cci', 'adx']
    latest = df[features].iloc[-1:].dropna()
    
    if latest.empty or model is None:
        return "Unknown", 0.0, None
    
    X_scaled = scaler.transform(latest)
    prob = model.predict_proba(X_scaled)[0]
    prediction = "Up" if prob[1] > prob[0] else "Down"
    confidence = max(prob) * 100
    
    return prediction, confidence, model_info

def generate_signals(df):
    """Generate buy/sell signals based on indicators and sentiment."""
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
    
    for i in range(1, len(df)):
        # SMA Signal
        if df['sma_short'].iloc[i] > df['sma_long'].iloc[i] and df['sma_short'].iloc[i-1] <= df['sma_long'].iloc[i-1]:
            df['sma_signal'].iloc[i] = 1
        elif df['sma_short'].iloc[i] < df['sma_long'].iloc[i] and df['sma_short'].iloc[i-1] >= df['sma_long'].iloc[i-1]:
            df['sma_signal'].iloc[i] = -1
            
        # RSI Signal
        if df['rsi'].iloc[i] < 30:
            df['rsi_signal'].iloc[i] = 1
        elif df['rsi'].iloc[i] > 70:
            df['rsi_signal'].iloc[i] = -1
        elif 30 <= df['rsi'].iloc[i] <= 70:
            df['rsi_signal'].iloc[i] = 0 if df['rsi_signal'].iloc[i-1] == 0 else df['rsi_signal'].iloc[i-1]
            
        # Volume Signal
        if df['volume_trend'].iloc[i] == 1:
            df['volume_signal'].iloc[i] = 1 if df['sma_signal'].iloc[i] > 0 else -1 if df['sma_signal'].iloc[i] < 0 else 0
            
        # OBV Signal
        if df['obv'].iloc[i] > df['obv'].iloc[i-1] and df['close'].iloc[i] > df['close'].iloc[i-1]:
            df['obv_signal'].iloc[i] = 1
        elif df['obv'].iloc[i] < df['obv'].iloc[i-1] and df['close'].iloc[i] < df['close'].iloc[i-1]:
            df['obv_signal'].iloc[i] = -1
            
        # MACD Signal
        if df['macd'].iloc[i] > df['signal_line'].iloc[i] and df['macd'].iloc[i-1] <= df['signal_line'].iloc[i-1]:
            df['macd_signal'].iloc[i] = 1
        elif df['macd'].iloc[i] < df['signal_line'].iloc[i] and df['macd'].iloc[i-1] >= df['signal_line'].iloc[i-1]:
            df['macd_signal'].iloc[i] = -1
            
        # Bollinger Bands Signal
        if df['close'].iloc[i] <= df['bb_lower'].iloc[i]:
            df['bb_signal'].iloc[i] = 1
        elif df['close'].iloc[i] >= df['bb_upper'].iloc[i]:
            df['bb_signal'].iloc[i] = -1
            
        # Support/Resistance Signal
        price_range = df['resistance'].iloc[i] - df['support'].iloc[i]
        if price_range > 0:
            if abs(df['close'].iloc[i] - df['support'].iloc[i]) / price_range < 0.1:
                df['sr_signal'].iloc[i] = 1
            elif abs(df['close'].iloc[i] - df['resistance'].iloc[i]) / price_range < 0.1:
                df['sr_signal'].iloc[i] = -1
                
        # Momentum Signal
        if df['momentum'].iloc[i] > 0:
            df['momentum_signal'].iloc[i] = 1
        elif df['momentum'].iloc[i] < 0:
            df['momentum_signal'].iloc[i] = -1
            
        # Sentiment Signal
        if df['sentiment'].iloc[i] > 0.2:
            df['sentiment_signal'].iloc[i] = 1
        elif df['sentiment'].iloc[i] < -0.2:
            df['sentiment_signal'].iloc[i] = -1
            
        # Stochastic Signal
        if df['stoch_k'].iloc[i] < 20 and df['stoch_d'].iloc[i] < 20:
            df['stoch_signal'].iloc[i] = 1
        elif df['stoch_k'].iloc[i] > 80 and df['stoch_d'].iloc[i] > 80:
            df['stoch_signal'].iloc[i] = -1
            
        # Williams %R Signal
        if df['williams_r'].iloc[i] < -80:
            df['williams_signal'].iloc[i] = 1
        elif df['williams_r'].iloc[i] > -20:
            df['williams_signal'].iloc[i] = -1
            
        # ROC Signal
        if df['roc'].iloc[i] > 0:
            df['roc_signal'].iloc[i] = 1
        elif df['roc'].iloc[i] < 0:
            df['roc_signal'].iloc[i] = -1
            
        # CCI Signal
        if df['cci'].iloc[i] < -100:
            df['cci_signal'].iloc[i] = 1
        elif df['cci'].iloc[i] > 100:
            df['cci_signal'].iloc[i] = -1
            
        # ADX Signal
        if df['adx'].iloc[i] > 25 and df['plus_di'].iloc[i] > df['minus_di'].iloc[i]:
            df['adx_signal'].iloc[i] = 1
        elif df['adx'].iloc[i] > 25 and df['plus_di'].iloc[i] < df['minus_di'].iloc[i]:
            df['adx_signal'].iloc[i] = -1
    
    # Calculate total score
    df['total_score'] = (df['sma_signal'] + df['rsi_signal'] + df['volume_signal'] + 
                         df['obv_signal'] + df['macd_signal'] + df['bb_signal'] + 
                         df['sr_signal'] + df['momentum_signal'] + df['sentiment_signal'] + 
                         df['stoch_signal'] + df['williams_signal'] + df['roc_signal'] + 
                         df['cci_signal'] + df['adx_signal'])
    
    return df

def analyze_pair(symbol):
    """Analyze a trading pair and return recommendation with trend prediction."""
    coin = symbol.replace('JPY', '')
    try:
        df = get_binance_data(symbol)
        df = calculate_indicators(df, coin)
        df = generate_signals(df)
        
        # Train model
        model, scaler, model_info = train_trend_model(df)
        trend, confidence, model_info = predict_trend(df, model, scaler, model_info)
        
        latest = df.iloc[-1]
        price = latest['close']
        rsi = latest['rsi']
        volume_sma = latest['volume_sma']
        obv = latest['obv']
        macd = latest['macd']
        signal_line = latest['signal_line']
        bb_upper = latest['bb_upper']
        bb_lower = latest['bb_lower']
        support = latest['support']
        resistance = latest['resistance']
        momentum = latest['momentum']
        sentiment = latest['sentiment']
        stoch_k = latest['stoch_k']
        stoch_d = latest['stoch_d']
        williams_r = latest['williams_r']
        roc = latest['roc']
        cci = latest['cci']
        adx = latest['adx']
        total_score = latest['total_score']
        
        print(f"\nAnalysis for {symbol}:")
        print(f"Current Price: {price:.2f} JPY")
        print(f"RSI: {rsi:.2f} (Lag1: {latest['rsi_lag1']:.2f}, Lag2: {latest['rsi_lag2']:.2f})")
        print(f"Volume SMA (10-period): {volume_sma:.2f}")
        print(f"Volume Trend: {'Increasing' if latest['volume_trend'] == 1 else 'Stable/Decreasing'}")
        print(f"OBV: {obv:.2f}")
        print(f"MACD: {macd:.4f}, Signal Line: {signal_line:.4f}")
        print(f"Bollinger Bands: Lower={bb_lower:.2f}, Upper={bb_upper:.2f}")
        print(f"Support: {support:.2f}, Resistance: {resistance:.2f}")
        print(f"Momentum (5-period): {momentum:.2f}%")
        print(f"Stochastic K: {stoch_k:.2f}, D: {stoch_d:.2f}")
        print(f"Williams %R: {williams_r:.2f}")
        print(f"ROC: {roc:.2f}%")
        print(f"CCI: {cci:.2f}")
        print(f"ADX: {adx:.2f}")
        print(f"Market Sentiment: {sentiment:.2f} ({'Positive' if sentiment > 0.2 else 'Negative' if sentiment < -0.2 else 'Neutral'})")
        print(f"Signal Score: {total_score:.0f}")
        
        if model_info:
            print(f"\nModel Performance:")
            print(f"Model Type: {model_info['type']}")
            print(f"Model Accuracy: {model_info['accuracy']:.3f} ({model_info['accuracy']*100:.1f}%)")
            if model_info['feature_importance']:
                print("\nFeature Importance:")
                for feat, importance in sorted(model_info['feature_importance'].items(), key=lambda x: x[1], reverse=True):
                    print(f"  {feat}: {importance:.4f}")
        
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
        
        print(f"\nRecommendation: {recommendation}")
        print(f"Trend Prediction: {trend} with {confidence:.2f}% confidence")
        print(f"Based on SMA, RSI, Volume, OBV, MACD, Bollinger Bands, Support/Resistance, Momentum, Sentiment, Stochastic, Williams %R, ROC, CCI, ADX")
        
        return recommendation, trend, confidence, df
    except Exception as e:
        print(f"Error analyzing {symbol}: {str(e)}")
        return None, None, None, None

def plot_chart(df, symbol):
    """Generate a chart for price, volume SMA, and OBV."""
    timestamps = df['timestamp'].dt.strftime('%Y-%m-%d %H:%M').tail(50).tolist()
    prices = df['close'].tail(50).tolist()
    volume_sma = df['volume_sma'].tail(50).tolist()
    obv = df['obv'].tail(50).tolist()
    
    chart_config = {
        "type": "line",
        "data": {
            "labels": timestamps,
            "datasets": [
                {
                    "label": "Price (JPY)",
                    "data": prices,
                    "borderColor": "rgba(75, 192, 192, 1)",
                    "yAxisID": "y",
                    "fill": False
                },
                {
                    "label": "Volume SMA",
                    "data": volume_sma,
                    "borderColor": "rgba(255, 99, 132, 1)",
                    "yAxisID": "y1",
                    "fill": False
                },
                {
                    "label": "OBV",
                    "data": obv,
                    "borderColor": "rgba(54, 162, 235, 1)",
                    "yAxisID": "y2",
                    "fill": False
                }
            ]
        },
        "options": {
            "responsive": True,
            "scales": {
                "y": {
                    "type": "linear",
                    "position": "left",
                    "title": {"display": True, "text": "Price (JPY)"}
                },
                "y1": {
                    "type": "linear",
                    "position": "right",
                    "title": {"display": True, "text": "Volume SMA"},
                    "grid": {"drawOnChartArea": False}
                },
                "y2": {
                    "type": "linear",
                    "position": "right",
                    "title": {"display": True, "text": "OBV"},
                    "grid": {"drawOnChartArea": False}
                },
                "x": {
                    "title": {"display": True, "text": "Time"}
                }
            }
        }
    }
    
    print("\nChart generated (Price, Volume SMA, OBV):")
    print(f"```chartjs\n{json.dumps(chart_config, indent=2)}\n```")

def main():
    pairs = ['XRPJPY', 'SUIJPY', 'XLMJPY']
    print("Available trading pairs:")
    for idx, pair in enumerate(pairs, 1):
        print(f"{idx}. {pair}")
    selected_idx = input("Select a trading pair by number (e.g., 1): ").strip()
    if not selected_idx.isdigit() or not (1 <= int(selected_idx) <= len(pairs)):
        print("Invalid selection. Please choose a valid number.")
        return
    selected_pair = pairs[int(selected_idx) - 1]
    recommendation, trend, confidence, df = analyze_pair(selected_pair)
    
    if recommendation and input("\nWould you like to see a chart of price, volume, and OBV? (yes/no): ").lower() == 'yes':
        plot_chart(df, selected_pair)

if __name__ == "__main__":
    main()