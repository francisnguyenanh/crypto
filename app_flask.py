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
    """Mock sentiment analysis for the given coin."""
    import random
    mock_posts = {
        'BTC': ["Bitcoin bullish trend!", "BTC consolidating", "Institutional buying BTC"],
        'ETH': ["Ethereum 2.0 upgrade!", "ETH DeFi growth", "Smart contracts booming"],
        'XRP': ["XRP to the moon!", "Bearish on XRP, dumping soon", "XRP looks stable"],
        'ADA': ["Cardano smart contracts!", "ADA staking rewards", "Slow development"],
        'SOL': ["Solana fast transactions!", "SOL ecosystem growing", "Network congestion issues"],
        'MATIC': ["Polygon scaling solution!", "MATIC partnerships", "Layer 2 adoption"],
        'LINK': ["Chainlink oracles essential!", "LINK price feeds", "Oracle network"],
        'XLM': ["Stellar cross-border payments!", "XLM partnership with banks", "Remittance solution"],
        'SUI': ["SUI is breaking out!", "New blockchain technology", "Object-centric design"]
    }
    posts = mock_posts.get(coin, ["Neutral comment"])
    scores = [TextBlob(post).sentiment.polarity for post in posts]
    avg_score = sum(scores) / len(scores)
    return avg_score

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
    
    # Additional technical indicators
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
    
    # Sentiment
    df['sentiment'] = get_mock_sentiment(coin)
    
    return df

def train_trend_model(df):
    """Train ensemble models with XGBoost to predict price trend."""
    features = ['sma_short', 'sma_long', 'rsi', 'rsi_lag1', 'rsi_lag2', 'volume_sma', 'volume_trend', 
                'obv', 'macd', 'signal_line', 'bb_upper', 'bb_lower', 'support', 'resistance', 
                'momentum', 'price_change_lag1', 'stoch_k', 'stoch_d', 'williams_r', 'atr', 'roc', 
                'cci', 'adx']
    df['price_change'] = (df['close'].shift(-1) > df['close']).astype(int)
    
    train_df = df[features + ['price_change']].dropna()
    if len(train_df) < 30:
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
        'svm': SVC(probability=True, random_state=42)
    }
    
    # Add XGBoost if available
    try:
        models['xgboost'] = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
    except:
        pass  # Skip XGBoost if not installed
    
    trained_models = {}
    accuracies = {}
    
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        trained_models[name] = model
        y_pred = model.predict(X_test_scaled)
        accuracies[name] = accuracy_score(y_test, y_pred)
        
        # Cross-validation score
        try:
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
            #print(f"{name.title()} - Test Accuracy: {accuracies[name]:.3f}, CV Score: {cv_scores.mean():.3f}")
        except:
            print(f"ERROR 2: {name.title()} - Test Accuracy: {accuracies[name]:.3f}")
    
    # Create ensemble model
    ensemble_models = [('lr', models['logistic']), ('rf', models['random_forest']), ('svm', models['svm'])]
    if 'xgboost' in models:
        ensemble_models.append(('xgb', models['xgboost']))
    
    ensemble = VotingClassifier(estimators=ensemble_models, voting='soft')
    
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
    
    # Feature importance (for Random Forest or XGBoost)
    feature_importance = None
    if model_type.lower() in ['random_forest', 'xgboost']:
        try:
            feature_importance = final_model.feature_importances_
            feature_importance = dict(zip(features, feature_importance))
        except:
            pass
    elif model_type == "Ensemble":
        # Get feature importance from Random Forest in ensemble
        try:
            rf_model = trained_models['random_forest']
            feature_importance = dict(zip(features, rf_model.feature_importances_))
        except:
            pass
    
    return final_model, scaler, {'accuracy': final_accuracy, 'type': model_type, 
                                 'all_accuracies': accuracies, 'feature_importance': feature_importance}

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
        if (df['sma_short'].iloc[i] > df['sma_long'].iloc[i] and 
            df['sma_short'].iloc[i-1] <= df['sma_long'].iloc[i-1]):
            df.loc[df.index[i], 'sma_signal'] = 1
        elif (df['sma_short'].iloc[i] < df['sma_long'].iloc[i] and 
              df['sma_short'].iloc[i-1] >= df['sma_long'].iloc[i-1]):
            df.loc[df.index[i], 'sma_signal'] = -1
        # RSI Signal (improved logic)
        if df['rsi'].iloc[i] < 30:
            df.loc[df.index[i], 'rsi_signal'] = 1
        elif df['rsi'].iloc[i] > 70:
            df.loc[df.index[i], 'rsi_signal'] = -1
        elif 30 <= df['rsi'].iloc[i] <= 70:
            prev = df['rsi_signal'].iloc[i-1]
            df.loc[df.index[i], 'rsi_signal'] = 0 if prev == 0 else prev
        # Volume Signal
        if df['volume_trend'].iloc[i] == 1:
            if df['sma_signal'].iloc[i] > 0:
                df.loc[df.index[i], 'volume_signal'] = 1
            elif df['sma_signal'].iloc[i] < 0:
                df.loc[df.index[i], 'volume_signal'] = -1
            else:
                df.loc[df.index[i], 'volume_signal'] = 0
        # OBV Signal
        if df['obv'].iloc[i] > df['obv'].iloc[i-1] and df['close'].iloc[i] > df['close'].iloc[i-1]:
            df.loc[df.index[i], 'obv_signal'] = 1
        elif df['obv'].iloc[i] < df['obv'].iloc[i-1] and df['close'].iloc[i] < df['close'].iloc[i-1]:
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
        if df['sentiment'].iloc[i] > 0.2:
            df.loc[df.index[i], 'sentiment_signal'] = 1
        elif df['sentiment'].iloc[i] < -0.2:
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
        # ADX Signal
        if df['adx'].iloc[i] > 25 and df['plus_di'].iloc[i] > df['minus_di'].iloc[i]:
            df.loc[df.index[i], 'adx_signal'] = 1
        elif df['adx'].iloc[i] > 25 and df['plus_di'].iloc[i] < df['minus_di'].iloc[i]:
            df.loc[df.index[i], 'adx_signal'] = -1
    
    # Calculate total score
    df['total_score'] = (df['sma_signal'] + df['rsi_signal'] + df['volume_signal'] + 
                         df['obv_signal'] + df['macd_signal'] + df['bb_signal'] + 
                         df['sr_signal'] + df['momentum_signal'] + df['sentiment_signal'] + 
                         df['stoch_signal'] + df['williams_signal'] + df['roc_signal'] + 
                         df['cci_signal'] + df['adx_signal'])
    
    return df

def calculate_risk_reward(current_price, take_profit_pct, stop_loss_pct, recommendation, signal_score, trend_prediction, confidence):
    """Calculate risk/reward analysis and adjust recommendation."""
    take_profit_price = current_price * (1 + take_profit_pct / 100)
    stop_loss_price = current_price * (1 - stop_loss_pct / 100)
    
    # Risk/Reward ratio
    potential_gain = take_profit_pct
    potential_loss = stop_loss_pct
    rr_ratio = round(potential_gain / potential_loss, 2)
    
    # Adjust recommendation based on risk/reward and confidence
    adjusted_recommendation = recommendation
    color = "#28a745"  # green
    
    # More detailed recommendation logic
    if recommendation == "HOLD":
        if signal_score >= 2:  # Slightly positive but not strong enough for BUY
            adjusted_recommendation = "HOLD/WEAK BUY (có thể mua ít)"
            color = "#20c997"  # teal
        elif signal_score <= -2:  # Slightly negative but not strong enough for SELL
            adjusted_recommendation = "HOLD/WEAK SELL (có thể bán ít)"
            color = "#fd7e14"  # orange
        else:  # Neutral signal
            adjusted_recommendation = "HOLD/WAIT (chờ tín hiệu rõ ràng hơn)"
            color = "#6c757d"  # gray
    
    # More conservative approach based on risk management
    elif rr_ratio < 1.5:  # Poor risk/reward ratio
        if recommendation in ["STRONG BUY", "BUY"]:
            adjusted_recommendation = "HOLD - Tỷ lệ rủi ro/lợi nhuận không tốt"
            color = "#ffc107"  # yellow
    elif rr_ratio >= 2.0 and confidence > 70:  # Good risk/reward with high confidence
        if recommendation == "BUY":
            adjusted_recommendation = "STRONG BUY - Tỷ lệ rủi ro/lợi nhuận xuất sắc"
            color = "#28a745"  # green
    elif confidence < 60:  # Low confidence
        if recommendation in ["STRONG BUY", "BUY"]:
            adjusted_recommendation = "HOLD - Độ tin cậy thấp, nên chờ"
            color = "#ffc107"  # yellow
    
    # For sell signals
    if recommendation in ["SELL", "STRONG SELL"]:
        color = "#dc3545"  # red
        if confidence < 60:
            adjusted_recommendation = "HOLD - Hướng không rõ ràng"
            color = "#ffc107"  # yellow
    
    # Add specific advice based on recommendation
    advice = ""
    if "STRONG BUY" in adjusted_recommendation:
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
        'take_profit_price': round(take_profit_price, 2),
        'stop_loss_price': round(stop_loss_price, 2),
        'take_profit_pct': take_profit_pct,
        'stop_loss_pct': stop_loss_pct,
        'ratio': rr_ratio,
        'adjusted_recommendation': adjusted_recommendation,
        'advice': advice,
        'color': color
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.get_json()
        symbol = data['symbol']
        take_profit_pct = data.get('take_profit', 5.0)
        stop_loss_pct = data.get('stop_loss', 3.0)
        investment_amount = data.get('investment_amount', 0.0)
        
        coin = symbol.replace('JPY', '')
        df = get_binance_data(symbol)
        df = calculate_indicators(df, coin)
        df = generate_signals(df)
        
        # Train model
        model, scaler, model_info = train_trend_model(df)
        trend, confidence, _ = predict_trend(df, model, scaler, model_info)
        
        latest = df.iloc[-1]
        current_price = latest['close']
        
        # Determine recommendation (updated thresholds for more signals)
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
        
        # Calculate risk/reward analysis
        risk_reward = calculate_risk_reward(
            current_price, take_profit_pct, stop_loss_pct, 
            recommendation, total_score, trend, confidence
        )
        
        # Calculate investment details if amount provided
        investment_details = None
        if investment_amount > 0:
            coin_amount = investment_amount / current_price
            profit_amount = investment_amount * (take_profit_pct / 100)
            loss_amount = investment_amount * (stop_loss_pct / 100)
            
            investment_details = {
                'investment_amount': investment_amount,
                'coin_amount': coin_amount,
                'profit_amount': profit_amount,
                'loss_amount': loss_amount,
                'profit_price': current_price * (1 + take_profit_pct / 100),
                'loss_price': current_price * (1 - stop_loss_pct / 100)
            }
        
        # Prepare chart data (add OBV to chart)
        chart_data = {
            'timestamps': df['timestamp'].dt.strftime('%Y-%m-%d %H:%M').tail(50).tolist(),
            'prices': df['close'].tail(50).tolist(),
            'volumes': df['volume'].tail(50).tolist(),
            'obv': df['obv'].tail(50).tolist(),
            'sma_short': df['sma_short'].tail(50).tolist(),
            'sma_long': df['sma_long'].tail(50).tolist(),
            'bb_upper': df['bb_upper'].tail(50).tolist(),
            'bb_lower': df['bb_lower'].tail(50).tolist(),
            'rsi': df['rsi'].tail(50).tolist()
        }
        
        def pyfloat(val):
            if isinstance(val, (np.floating,)):
                return float(val)
            return val

        def pyfloat_list(lst):
            return [pyfloat(x) for x in lst]

        safe_chart_data = {k: pyfloat_list(v) if isinstance(v, list) else v for k, v in chart_data.items()}
        safe_risk_reward = {k: pyfloat(v) if isinstance(v, (float, np.floating)) else v for k, v in risk_reward.items()}
        safe_investment_details = {k: pyfloat(v) if isinstance(v, (float, np.floating)) else v for k, v in investment_details.items()} if investment_details else None
        safe_indicators = {k: pyfloat(v) if isinstance(v, (float, np.floating)) else v for k, v in {
            'rsi': round(latest['rsi'], 2),
            'rsi_lag1': round(latest['rsi_lag1'], 2) if not pd.isna(latest['rsi_lag1']) else 0,
            'macd': round(latest['macd'], 4),
            'signal_line': round(latest['signal_line'], 4),
            'bb_upper': round(latest['bb_upper'], 2),
            'bb_lower': round(latest['bb_lower'], 2),
            'support': round(latest['support'], 2),
            'resistance': round(latest['resistance'], 2),
            'momentum': round(latest['momentum'], 2),
            'obv': round(latest['obv'], 2),
            'stoch_k': round(latest['stoch_k'], 2),
            'stoch_d': round(latest['stoch_d'], 2),
            'williams_r': round(latest['williams_r'], 2),
            'roc': round(latest['roc'], 2),
            'cci': round(latest['cci'], 2),
            'adx': round(latest['adx'], 2) if not pd.isna(latest['adx']) else 0,
            'sentiment': round(latest['sentiment'], 2),
            'volume_trend': 'Increasing' if latest['volume_trend'] == 1 else 'Stable/Decreasing'
        }.items()}

        analysis_result = {
            'symbol': symbol,
            'current_price': pyfloat(round(current_price, 2)),
            'recommendation': recommendation,
            'trend_prediction': trend,
            'confidence': pyfloat(round(confidence, 2)),
            'signal_score': int(total_score),
            'risk_reward': safe_risk_reward,
            'investment_details': safe_investment_details,
            'indicators': safe_indicators,
            'model_info': model_info if model_info else {'accuracy': 0, 'type': 'Unknown'},
            'chart_data': safe_chart_data,
            'last_timestamp': str(latest['timestamp']) if 'timestamp' in latest else None
        }
        
        return jsonify(analysis_result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/suggest_best_pairs', methods=['POST'])
def suggest_best_pairs():
    try:
        data = request.get_json()
        take_profit_pct = data.get('take_profit', 5.0)
        stop_loss_pct = data.get('stop_loss', 3.0)
        investment_amount = data.get('investment_amount', 0.0)
        
        # List of all available trading pairs
        pairs = [
            'BTCJPY', 'ETHJPY', 'XRPJPY', 'ADAJPY', 'SOLJPY',
            'MATICJPY', 'LINKJPY', 'XLMJPY', 'SUIJPY'
        ]
        
        suggestions = []
        
        for symbol in pairs:
            try:
                coin = symbol.replace('JPY', '')
                df = get_binance_data(symbol)
                df = calculate_indicators(df, coin)
                df = generate_signals(df)
                
                # Train model
                model, scaler, model_info = train_trend_model(df)
                trend, confidence, _ = predict_trend(df, model, scaler, model_info)
                
                latest = df.iloc[-1]
                current_price = latest['close']
                total_score = latest['total_score']
                
                # Determine recommendation
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
                
                # Calculate risk/reward analysis
                risk_reward = calculate_risk_reward(
                    current_price, take_profit_pct, stop_loss_pct, 
                    recommendation, total_score, trend, confidence
                )
                
                # Calculate composite score for ranking
                # Prioritize BUY signals with high confidence and good risk/reward
                composite_score = 0
                if recommendation in ["STRONG BUY", "BUY"]:
                    composite_score = total_score + confidence/10
                    # Bonus for good risk/reward ratio
                    if risk_reward['ratio'] >= 2.0:
                        composite_score += 5
                    elif risk_reward['ratio'] >= 1.5:
                        composite_score += 2
                elif recommendation == "HOLD" and total_score >= 2:
                    composite_score = total_score + confidence/20  # Lower score for HOLD
                else:
                    composite_score = -100  # Exclude SELL signals
                
                suggestion = {
                    'symbol': symbol,
                    'current_price': round(current_price, 2),
                    'recommendation': recommendation,
                    'signal_score': int(total_score),
                    'confidence': round(confidence, 2),
                    'trend_prediction': trend,
                    'composite_score': composite_score,
                    'risk_reward': risk_reward,
                    'model_accuracy': round(model_info['accuracy'] * 100, 1) if model_info else 0
                }
                
                suggestions.append(suggestion)
                
            except Exception as e:
                print(f"Error analyzing {symbol}: {str(e)}")
                continue
        
        # Sort by composite score and take top 2
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
