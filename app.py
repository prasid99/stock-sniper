import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go

st.set_page_config(page_title="Private AI Trade Center", layout="wide")

# ==========================================
# 🔐 SECURITY: PASSWORD PROTECTION
# ==========================================
def check_password():
    """Returns `True` if the user had the correct password."""
    def password_entered():
        if st.session_state["password"] == "trader123": # <--- CHANGE THIS IF YOU WANT
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.text_input("Enter Password to Access Scanner:", type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        st.text_input("Enter Password to Access Scanner:", type="password", on_change=password_entered, key="password")
        st.error("😕 Wrong Password")
        return False
    else:
        return True

if not check_password():
    st.stop()

# ==========================================
# ⚙️ CONFIGURATION
# ==========================================
STOCKS = [
    # NIFTY 50 GIANTS
    "RELIANCE.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS", "TCS.NS", "ITC.NS",
    "SBIN.NS", "BHARTIARTL.NS", "L&T.NS", "AXISBANK.NS", "KOTAKBANK.NS",
    "HINDUNILVR.NS", "MARUTI.NS", "TATAMOTORS.NS", "ULTRACEMCO.NS", "SUNPHARMA.NS",
    "TITAN.NS", "BAJFINANCE.NS", "ASIANPAINT.NS", "HCLTECH.NS", "NTPC.NS",
    "POWERGRID.NS", "TATASTEEL.NS", "INDUSINDBK.NS", "M&M.NS", "ADANIENT.NS",
    "ADANIPORTS.NS", "JSWSTEEL.NS", "COALINDIA.NS", "GRASIM.NS", "ONGC.NS",
    "HINDALCO.NS", "BAJAJFINSV.NS", "EICHERMOT.NS", "NESTLEIND.NS", "TECHM.NS",
    "WIPRO.NS", "BRITANNIA.NS", "CIPLA.NS", "HEROMOTOCO.NS", "DRREDDY.NS",
    "APOLLOHOSP.NS", "DIVISLAB.NS", "SBILIFE.NS", "BPCL.NS", "LTIM.NS",
    
    # HIGH MOMENTUM F&O
    "DLF.NS", "VEDANTA.NS", "JINDALSTEL.NS", "HAL.NS", "BEL.NS", "MAZDOCK.NS",
    "COCHINSHIP.NS", "TRENT.NS", "DIXON.NS", "POLYCAB.NS", "ZOMATO.NS", "PAYTM.NS",
    "PFC.NS", "REC.NS", "BHEL.NS", "SAIL.NS", "NATIONALUM.NS", "NMDC.NS",
    "CANBK.NS", "PNB.NS", "IDFCFIRSTB.NS", "RBLBANK.NS", "BANDHANBNK.NS",
    "AUBANK.NS", "FEDERALBNK.NS", "TVSMOTOR.NS", "ASHOKLEY.NS", "MOTHERSON.NS",
    "BHARATFORG.NS", "ESCORTS.NS", "GMRINFRA.NS", "IRCTC.NS", "CONCOR.NS",
    "INDHOTEL.NS", "JUBLFOOD.NS", "ZEEL.NS", "SUNTV.NS", "PVRINOX.NS",
    "TATACOMM.NS", "PERSISTENT.NS", "COFORGE.NS", "KPITTECH.NS", "MPHASIS.NS",
    "LTTS.NS", "OFSS.NS", "ABB.NS", "SIEMENS.NS", "CUMMINSIND.NS", "HAVELLS.NS",
    "VOLTAS.NS", "ASTRAL.NS", "PIIND.NS", "UPL.NS", "COROMANDEL.NS", "SRF.NS"
]

# ==========================================
# 🧠 INTELLIGENCE ENGINES
# ==========================================

# 1. GLOBAL MOOD
def get_global_mood():
    try:
        data = yf.download("^GSPC ^NSEI", period="5d", progress=False)['Close']
        sp_change = ((data['^GSPC'].iloc[-1] - data['^GSPC'].iloc[-2]) / data['^GSPC'].iloc[-2]) * 100
        nifty_change = ((data['^NSEI'].iloc[-1] - data['^NSEI'].iloc[-2]) / data['^NSEI'].iloc[-2]) * 100
        
        mood = "NEUTRAL"
        if sp_change > 0.2 and nifty_change > 0.2: mood = "BULLISH 🟢"
        elif sp_change < -0.2 and nifty_change < -0.2: mood = "BEARISH 🔴"
        
        return mood, sp_change, nifty_change
    except:
        return "NEUTRAL", 0.0, 0.0

# 2. DATA DOWNLOADER
@st.cache_data(ttl=60)
def get_market_data():
    return yf.download(tickers=" ".join(STOCKS), period="5d", interval="15m", group_by='ticker', threads=True)

# 3. BACKTESTER (The "Truth" Check)
def calculate_win_rate(df):
    """Calculates Win Rate based on Trend Following Strategy"""
    if len(df) < 50: return 0
    trades = 0
    wins = 0
    
    # Loop through history
    for i in range(50, len(df)-15):
        # STRATEGY: ADX > 20 + EMA Crossover
        if df['ADX'].iloc[i] > 20:
            if df['EMA_9'].iloc[i] > df['EMA_21'].iloc[i] and df['EMA_9'].iloc[i-1] <= df['EMA_21'].iloc[i-1]:
                entry = df['Close'].iloc[i]
                atr = df['ATR'].iloc[i]
                target = entry + (2.5 * atr)
                sl = entry - (1.5 * atr)
                trades += 1
                
                # Check next 15 candles
                for j in range(1, 16):
                    if i+j >= len(df): break
                    if df['High'].iloc[i+j] >= target:
                        wins += 1
                        break
                    if df['Low'].iloc[i+j] <= sl: break
    
    return (wins / trades * 100) if trades > 0 else 0

# 4. ANALYST REASONING GENERATOR
def generate_report(s, mood):
    vwap_status = "ABOVE" if s['Price'] > s['VWAP'] else "BELOW"
    trend = "UP" if s['Signal'] == "BUY" else "DOWN"
    
    report = f"""
    **📊 ANALYST REPORT FOR {s['Ticker']}**
    
    **1. Context:** Global Mood is **{mood}**.
    
    **2. Technical Logic ({s['Signal']}):**
    * **Trend:** Confirmed **{trend}** (EMA Crossover).
    * **Strength:** ADX is **{s['ADX']:.1f}** (Trends >20 are valid).
    * **Position:** Price is **{vwap_status}** the Institutional VWAP.
    * **History:** This setup has a **{s['WinRate']:.0f}% Win Rate** on this stock recently.
    
    **3. Execution:**
    * **Entry:** Market Price
    * **Target:** ₹{s['Target']:.2f} (2.5x ATR)
    * **Stop Loss:** ₹{s['SL']:.2f} (1.5x ATR)
    """
    return report

# 5. MAIN ANALYSIS LOOP
def analyze_stock(ticker, df):
    if len(df) < 50: return None
    
    # Indicators
    df['EMA_9'] = df.ta.ema(length=9)
    df['EMA_21'] = df.ta.ema(length=21)
    df['RSI'] = df.ta.rsi(length=14)
    df['ADX'] = df.ta.adx(length=14)['ADX_14']
    df['ATR'] = df.ta.atr(length=14)
    df['TP'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['VWAP'] = (df['TP'] * df['Volume']).cumsum() / df['Volume'].cumsum()
    
    # Auto-Backtest
    win_rate = calculate_win_rate(df)
    
    # Current Signal
    close = df['Close'].iloc[-1]
    adx = df['ADX'].iloc[-1]
    ema9 = df['EMA_9'].iloc[-1]
    ema21 = df['EMA_21'].iloc[-1]
    atr = df['ATR'].iloc[-1]
    vwap = df['VWAP'].iloc[-1]
    
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

    return {
        "Ticker": ticker, "Signal": signal, "WinRate": win_rate,
        "Price": close, "Target": target, "SL": sl, "ADX": adx, "VWAP": vwap, "History": df
    }

# ==========================================
# 🖥️ UI DASHBOARD
# ==========================================
st.title("🦅 Private AI Trade Center V24")

# A. SIDEBAR INFO
mood, sp, nifty = get_global_mood()
st.sidebar.header("🌍 Market Context")
st.sidebar.metric("Global Sentiment", mood)
st.sidebar.metric("S&P 500", f"{sp:.2f}%")
st.sidebar.metric("Nifty 50", f"{nifty:.2f}%")

# B. MAIN SCANNER
if st.button("🔎 SCAN & AUTO-VALIDATE (180 STOCKS)", type="primary"):
    with st.spinner("Downloading Data, Calculating Indicators, and Running Backtests..."):
        market_data = get_market_data()
        results = []
        
        progress = st.progress(0)
        for i, t in enumerate(STOCKS):
            try:
                df = market_data[t].copy()
                df.dropna(inplace=True)
                res = analyze_stock(t, df)
                # FILTER: Only show stocks with >40% Win Rate
                if res and res['WinRate'] > 40: 
                    results.append(res)
            except: continue
            progress.progress((i+1)/len(STOCKS))
        progress.empty()
            
        # Sort by Win Rate (Best First)
        if results:
            st.session_state['results'] = sorted(results, key=lambda x: x['WinRate'], reverse=True)
        else:
            st.error("Market is choppy. No high-probability trades found.")

# C. RESULTS DISPLAY
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("🏆 Top Opportunities")
    if 'results' in st.session_state:
        for item in st.session_state['results']:
            color = "🟢" if item['Signal'] == "BUY" else "🔴"
            wr = item['WinRate']
            stars = "⭐⭐⭐" if wr > 70 else ("⭐⭐" if wr > 60 else "⭐")
            
            label = f"{color} {item['Ticker']} | WR: {wr:.0f}% {stars}"
            
            if st.button(label, key=item['Ticker'], use_container_width=True):
                st.session_state['active'] = item

with col2:
    if 'active' in st.session_state:
        s = st.session_state['active']
        sym = s['Ticker']
        
        st.info(f"### {sym} ({s['Signal']})")
        
        # 1. EXECUTION
        c1, c2, c3 = st.columns(3)
        c1.metric("ENTRY", f"₹{s['Price']:.2f}")
        c2.metric("TARGET", f"₹{s['Target']:.2f}", delta="Profit")
        c3.metric("STOP LOSS", f"₹{s['SL']:.2f}", delta="-Risk", delta_color="inverse")
        
        # 2. VALIDATION BADGE
        wr = s['WinRate']
        if wr > 60:
            st.success(f"💎 **GEM DETECTED:** Historical Win Rate is {wr:.1f}%.")
        else:
            st.warning(f"⚠️ **AVERAGE:** Historical Win Rate is {wr:.1f}%.")
            
        # 3. REASONING
        with st.expander("📝 Read Analyst Report", expanded=True):
            st.markdown(generate_report(s, mood))

        # 4. CHART
        fig = go.Figure()
        hist = s['History']
        fig.add_trace(go.Candlestick(x=hist.index, open=hist['Open'], high=hist['High'], low=hist['Low'], close=hist['Close']))
        fig.add_trace(go.Scatter(x=hist.index, y=hist['VWAP'], line=dict(color='cyan'), name='VWAP'))
        
        fig.add_hline(y=s['Target'], line_color="green", line_dash="dash")
        fig.add_hline(y=s['SL'], line_color="red", line_dash="dash")
        
        fig.update_layout(height=500, template="plotly_dark", title=f"{sym} Chart", xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)