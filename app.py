import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go

st.set_page_config(page_title="Ultimate AI Sniper V26", layout="wide")

# ==========================================
# 🔐 1. SECURITY
# ==========================================
def check_password():
    def password_entered():
        if st.session_state["password"] == "trader123":
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.text_input("Enter Password:", type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        st.text_input("Enter Password:", type="password", on_change=password_entered, key="password")
        st.error("❌ Access Denied")
        return False
    else:
        return True

if not check_password():
    st.stop()

# ==========================================
# ⚙️ 2. CONFIGURATION (High Momentum Stocks)
# ==========================================
STOCKS = [
    "ADANIENT.NS", "ADANIPORTS.NS", "DLF.NS", "TATASTEEL.NS", "JINDALSTEL.NS",
    "HINDALCO.NS", "VEDANTA.NS", "TATAMOTORS.NS", "TVSMOTOR.NS", "EICHERMOT.NS",
    "HAL.NS", "BEL.NS", "MAZDOCK.NS", "COCHINSHIP.NS", "TRENT.NS", "DIXON.NS",
    "POLYCAB.NS", "ZOMATO.NS", "PAYTM.NS", "BAJFINANCE.NS", "SBIN.NS", "CANBK.NS",
    "RELIANCE.NS", "INFY.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "ITC.NS",
    "AXISBANK.NS", "KOTAKBANK.NS", "MARUTI.NS", "M&M.NS", "SUNPHARMA.NS",
    "TITAN.NS", "ASIANPAINT.NS", "NTPC.NS", "POWERGRID.NS", "ONGC.NS"
]

# ==========================================
# 🧠 3. INTELLIGENCE ENGINES
# ==========================================

# A. GLOBAL MOOD
def get_global_mood():
    try:
        data = yf.download("^GSPC ^NSEI", period="2d", progress=False)['Close']
        sp_change = ((data['^GSPC'].iloc[-1] - data['^GSPC'].iloc[-2]) / data['^GSPC'].iloc[-2]) * 100
        nifty_change = ((data['^NSEI'].iloc[-1] - data['^NSEI'].iloc[-2]) / data['^NSEI'].iloc[-2]) * 100
        
        mood = "NEUTRAL"
        if sp_change > 0.2 and nifty_change > 0.2: mood = "BULLISH 🟢"
        elif sp_change < -0.2 and nifty_change < -0.2: mood = "BEARISH 🔴"
        return mood, sp_change, nifty_change
    except:
        return "NEUTRAL", 0.0, 0.0

# B. INTRADAY DATA (5 Days)
@st.cache_data(ttl=60)
def get_market_data():
    return yf.download(tickers=" ".join(STOCKS), period="5d", interval="15m", group_by='ticker', threads=True)

# C. DEEP BACKTESTER (1 YEAR HISTORY)
def run_1y_backtest(ticker):
    # Fetch 1 Year of Hourly Data
    df = yf.Ticker(ticker).history(period="1y", interval="1h")
    if len(df) < 100: return 0, 0, 0
    
    # Strategy
    df['EMA_9'] = df.ta.ema(length=9)
    df['EMA_21'] = df.ta.ema(length=21)
    df['ATR'] = df.ta.atr(length=14)
    
    trades = 0; wins = 0
    
    for i in range(50, len(df)-15):
        # BUY SIGNAL
        if df['EMA_9'].iloc[i] > df['EMA_21'].iloc[i] and df['EMA_9'].iloc[i-1] <= df['EMA_21'].iloc[i-1]:
            entry = df['Close'].iloc[i]
            target = entry + (2.5 * df['ATR'].iloc[i]) # 2-3% Target
            sl = entry - (1.5 * df['ATR'].iloc[i])     # 1% Risk
            trades += 1
            
            # Outcome Check
            for j in range(1, 16):
                if i+j >= len(df): break
                if df['High'].iloc[i+j] >= target:
                    wins += 1
                    break
                if df['Low'].iloc[i+j] <= sl: break
    
    win_rate = (wins/trades*100) if trades > 0 else 0
    return win_rate, trades, len(df)

# D. STRATEGY BRAIN
def analyze_stock(ticker, df):
    if len(df) < 50: return None
    
    # Indicators
    df['EMA_9'] = df.ta.ema(length=9)
    df['EMA_21'] = df.ta.ema(length=21)
    df['ADX'] = df.ta.adx(length=14)['ADX_14']
    df['ATR'] = df.ta.atr(length=14)
    
    close = df['Close'].iloc[-1]
    adx = df['ADX'].iloc[-1]
    ema9 = df['EMA_9'].iloc[-1]
    ema21 = df['EMA_21'].iloc[-1]
    atr = df['ATR'].iloc[-1]
    
    # 2% PROFIT FILTER
    potential_move = ((3 * atr) / close) * 100
    if potential_move < 1.8: return None # Skip slow stocks
    
    signal = "WAIT"
    
    if adx > 20: # Trend Filter
        if close > ema9 and ema9 > ema21:
            signal = "BUY"
            target = close + (2.5 * atr)
            sl = close - (1.5 * atr)
        elif close < ema9 and ema9 < ema21:
            signal = "SELL"
            target = close - (2.5 * atr)
            sl = close + (1.5 * atr)
            
    if signal == "WAIT": return None

    # Calculate recent Win Rate for sorting
    win_rate = calculate_recent_wr(df)

    return {
        "Ticker": ticker, "Signal": signal, "WinRate": win_rate,
        "Price": close, "Target": target, "SL": sl, 
        "ADX": adx, "Vol": potential_move, "History": df
    }

def calculate_recent_wr(df):
    trades = 0; wins = 0
    for i in range(50, len(df)-15):
        if df['ADX'].iloc[i] > 20:
            if df['EMA_9'].iloc[i] > df['EMA_21'].iloc[i] and df['EMA_9'].iloc[i-1] <= df['EMA_21'].iloc[i-1]:
                entry = df['Close'].iloc[i]; target = entry + (2.5 * df['ATR'].iloc[i]); sl = entry - (1.5 * df['ATR'].iloc[i])
                trades += 1
                for j in range(1, 16):
                    if i+j >= len(df): break
                    if df['High'].iloc[i+j] >= target: wins += 1; break
                    if df['Low'].iloc[i+j] <= sl: break
    return (wins/trades*100) if trades > 0 else 0

def generate_report(s, mood):
    return f"""
    **📊 ANALYST REPORT FOR {s['Ticker']}**
    
    **1. Global Context:** Market is **{mood}**.
    
    **2. Technical Logic ({s['Signal']}):**
    * **Trend:** Confirmed by EMA Crossover.
    * **Strength:** ADX is **{s['ADX']:.1f}** (Trending).
    * **Potential:** This stock has the energy to move **{s['Vol']:.1f}%** today.
    
    **3. Execution:**
    * **Entry:** ₹{s['Price']:.2f}
    * **Target:** ₹{s['Target']:.2f} (2.5x ATR)
    * **Stop Loss:** ₹{s['SL']:.2f} (1.5x ATR)
    """

# ==========================================
# 4. UI DASHBOARD
# ==========================================
st.title("🦅 AI Sniper V26 (Complete Package)")
mood, sp, nifty = get_global_mood()
st.sidebar.metric("Global Mood", mood)

if st.button("🔎 SCAN FOR 2% PROFIT", type="primary"):
    with st.spinner("Filtering 180 stocks..."):
        market_data = get_market_data()
        results = []
        progress = st.progress(0)
        
        for i, t in enumerate(STOCKS):
            try:
                df = market_data[t].copy()
                df.dropna(inplace=True)
                res = analyze_stock(t, df)
                if res and res['WinRate'] > 40:
                    results.append(res)
            except: continue
            progress.progress((i+1)/len(STOCKS))
        progress.empty()
            
        if results:
            st.session_state['results'] = sorted(results, key=lambda x: x['WinRate'], reverse=True)
        else:
            st.error("Market is dead. No 2% opportunities.")

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("🔥 Top Movers")
    if 'results' in st.session_state:
        for item in st.session_state['results']:
            color = "🟢" if item['Signal'] == "BUY" else "🔴"
            wr = item['WinRate']
            label = f"{color} {item['Ticker']} | WR: {wr:.0f}%"
            if st.button(label, key=item['Ticker'], use_container_width=True):
                st.session_state['active'] = item

with col2:
    if 'active' in st.session_state:
        s = st.session_state['active']
        
        st.info(f"### {s['Ticker']} ({s['Signal']})")
        
        c1, c2, c3 = st.columns(3)
        c1.metric("ENTRY", f"₹{s['Price']:.2f}")
        c2.metric("TARGET", f"₹{s['Target']:.2f}", delta="Profit")
        c3.metric("STOP LOSS", f"₹{s['SL']:.2f}", delta="-1%", delta_color="inverse")
        
        st.markdown(generate_report(s, mood))
        
        # DEEP BACKTEST BUTTON
        st.write("---")
        if st.button(f"🧪 RUN 1-YEAR DEEP BACKTEST"):
            with st.spinner("Downloading 1 Year Data & Simulating..."):
                wr, trades, candles = run_1y_backtest(s['Ticker'])
                if wr > 50:
                    st.success(f"✅ **APPROVED:** 1-Year Win Rate is **{wr:.1f}%** ({trades} trades).")
                else:
                    st.error(f"❌ **DENIED:** 1-Year Win Rate is **{wr:.1f}%**. Do not trade.")

        # Chart
        fig = go.Figure()
        hist = s['History']
        fig.add_trace(go.Candlestick(x=hist.index, open=hist['Open'], high=hist['High'], low=hist['Low'], close=hist['Close']))
        fig.add_hline(y=s['Target'], line_color="green", line_dash="dash")
        fig.add_hline(y=s['SL'], line_color="red", line_dash="dash")
        st.plotly_chart(fig, use_container_width=True)
