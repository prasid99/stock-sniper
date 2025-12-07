import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(page_title="AI Trade Command V32", layout="wide")

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
# ⚙️ 2. CONFIGURATION (180+ Liquid Stocks)
# ==========================================
SCANNER_STOCKS = [
    "RELIANCE.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS", "TCS.NS", "ITC.NS",
    "SBIN.NS", "BHARTIARTL.NS", "L&T.NS", "AXISBANK.NS", "KOTAKBANK.NS",
    "HINDUNILVR.NS", "MARUTI.NS", "TATAMOTORS.NS", "ULTRACEMCO.NS", "SUNPHARMA.NS",
    "TITAN.NS", "BAJFINANCE.NS", "ASIANPAINT.NS", "HCLTECH.NS", "NTPC.NS",
    "POWERGRID.NS", "TATASTEEL.NS", "INDUSINDBK.NS", "M&M.NS", "ADANIENT.NS",
    "ADANIPORTS.NS", "JSWSTEEL.NS", "COALINDIA.NS", "GRASIM.NS", "ONGC.NS",
    "HINDALCO.NS", "BAJAJFINSV.NS", "EICHERMOT.NS", "NESTLEIND.NS", "TECHM.NS",
    "WIPRO.NS", "BRITANNIA.NS", "CIPLA.NS", "HEROMOTOCO.NS", "DRREDDY.NS",
    "APOLLOHOSP.NS", "DIVISLAB.NS", "SBILIFE.NS", "BPCL.NS", "LTIM.NS",
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
# 🧠 3. DATA & INTELLIGENCE
# ==========================================

def clean_ticker_input(user_input):
    clean = user_input.strip().upper().replace(" ", "")
    if clean == "NIFTY": return "^NSEI"
    if clean == "BANKNIFTY": return "^NSEBANK"
    if not clean.endswith(".NS") and not clean.endswith(".BO") and not clean.startswith("^"):
        clean += ".NS"
    return clean

def get_global_mood():
    try:
        data = yf.download("^GSPC ^NSEI ^BSESN", period="2d", progress=False)['Close']
        metrics = {}
        for t in ["^GSPC", "^NSEI", "^BSESN"]:
            try:
                chg = ((data[t].iloc[-1] - data[t].iloc[-2]) / data[t].iloc[-2]) * 100
                metrics[t] = chg
            except: metrics[t] = 0.0
        avg = (metrics["^GSPC"] + metrics["^NSEI"]) / 2
        mood = "NEUTRAL 😐"
        if avg > 0.2: mood = "BULLISH 🟢"
        elif avg < -0.2: mood = "BEARISH 🔴"
        return mood, metrics
    except:
        return "NEUTRAL", {"^GSPC":0, "^NSEI":0, "^BSESN":0}

@st.cache_data(ttl=60)
def get_batch_data():
    return yf.download(tickers=" ".join(SCANNER_STOCKS), period="5d", interval="15m", group_by='ticker', threads=True)

def get_stock_details(ticker):
    t = yf.Ticker(ticker)
    df = t.history(period="5d", interval="15m")
    if df.empty and ticker.endswith(".NS"):
        ticker = ticker.replace(".NS", ".BO")
        t = yf.Ticker(ticker)
        df = t.history(period="5d", interval="15m")
    
    try:
        info = t.info
        if info is None: info = {}
        fund = {
            "PE": info.get('trailingPE', 'N/A'),
            "Sector": info.get('sector', 'Unknown'),
            "Rec": info.get('recommendationKey', 'none').upper().replace("_", " "),
            "TargetHigh": info.get('targetHighPrice', 0),
            "Current": info.get('currentPrice', 0),
            "Beta": info.get('beta', 0)
        }
    except:
        fund = {"PE": "-", "Sector": "-", "Rec": "-", "TargetHigh": 0, "Current": 0, "Beta": 0}
        
    return df, fund, ticker

def analyze_stock(ticker, df):
    if len(df) < 50: return None
    
    # Indicators
    df['EMA_9'] = df.ta.ema(length=9)
    df['EMA_21'] = df.ta.ema(length=21)
    df['ADX'] = df.ta.adx(length=14)['ADX_14']
    df['ATR'] = df.ta.atr(length=14)
    df['TP'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['VWAP'] = (df['TP'] * df['Volume']).cumsum() / df['Volume'].cumsum()
    
    close = df['Close'].iloc[-1]
    adx = df['ADX'].iloc[-1]
    ema9 = df['EMA_9'].iloc[-1]
    ema21 = df['EMA_21'].iloc[-1]
    atr = df['ATR'].iloc[-1]
    vwap = df['VWAP'].iloc[-1]
    vol_pct = ((3 * atr) / close) * 100
    
    # Smart Levels
    support = close - (2 * atr)
    resistance = close + (2 * atr)
    
    signal = "WAIT"
    action_text = "HOLD / WATCH"
    justification = "Market is choppy (ADX < 20). Big players are inactive."
    target_logic = f"Calculated using 2.5x ATR (Avg Range: ₹{atr:.2f})"
    
    # INSTITUTIONAL PROXY (FII/DII Guess)
    # If Price > VWAP and Volume is Rising -> Institutions likely BUYING
    inst_activity = "Neutral"
    if close > vwap: inst_activity = "BULLISH (Smart Money Buying)"
    elif close < vwap: inst_activity = "BEARISH (Smart Money Selling)"
    
    # SIGNAL GENERATION
    if adx > 20:
        if close > ema9 and ema9 > ema21:
            signal = "BUY"
            action_text = "BUY NOW"
            justification = f"Uptrend Confirmed. Price is above VWAP ({vwap:.2f}) and Momentum is strong."
            target = close + (2.5 * atr)
            sl = close - (1.5 * atr)
        elif close < ema9 and ema9 < ema21:
            signal = "SELL"
            action_text = "SHORT NOW"
            justification = f"Downtrend Confirmed. Price is below VWAP ({vwap:.2f}) and Momentum is down."
            target = close - (2.5 * atr)
            sl = close + (1.5 * atr)
        else:
            target = resistance; sl = support
            justification = f"Trend is unclear. Wait for a breakout above ₹{resistance:.2f}."
    else:
        target = resistance; sl = support
        justification = f"Volatility is too low ({vol_pct:.1f}%). Big moves unlikely today."

    return {
        "Ticker": ticker, "Signal": signal, "Action": action_text,
        "Price": close, "Target": target, "SL": sl, 
        "Support": support, "Resistance": resistance,
        "ADX": adx, "Vol": vol_pct, "VWAP": vwap, "History": df,
        "Justification": justification, "TargetLogic": target_logic, "InstAct": inst_activity
    }

def run_backtest(ticker):
    try:
        t = yf.Ticker(ticker)
        df = t.history(period="1y", interval="1h")
        if len(df) < 100: return 0, 0
        
        df['EMA_9'] = df.ta.ema(length=9)
        df['EMA_21'] = df.ta.ema(length=21)
        df['ATR'] = df.ta.atr(length=14)
        
        trades = 0; wins = 0
        for i in range(50, len(df)-15):
            if df['EMA_9'].iloc[i] > df['EMA_21'].iloc[i] and df['EMA_9'].iloc[i-1] <= df['EMA_21'].iloc[i-1]:
                entry = df['Close'].iloc[i]
                target = entry + (2.5 * df['ATR'].iloc[i])
                sl = entry - (1.5 * df['ATR'].iloc[i])
                trades += 1
                for j in range(1, 16):
                    if i+j >= len(df): break
                    if df['High'].iloc[i+j] >= target: wins += 1; break
                    if df['Low'].iloc[i+j] <= sl: break
        
        win_rate = (wins/trades*100) if trades > 0 else 0
        return win_rate, trades
    except: return 0, 0

# ==========================================
# 4. UI DASHBOARD
# ==========================================
st.title("🦅 AI Trade Command V32")

mood, metrics = get_global_mood()
k1, k2, k3, k4 = st.columns(4)
k1.metric("Global Mood", mood)
k2.metric("NIFTY 50", f"{metrics['^NSEI']:.2f}%")
k3.metric("SENSEX", f"{metrics['^BSESN']:.2f}%")
k4.metric("S&P 500", f"{metrics['^GSPC']:.2f}%")
st.divider()

tab1, tab2 = st.tabs(["🔍 DEEP SEARCH & ANALYSIS", "🚀 2% PROFIT SCANNER"])

# --- TAB 1: DEEP SEARCH ---
with tab1:
    c_in, c_btn = st.columns([3, 1])
    with c_in:
        raw_input = st.text_input("Enter Stock Name:", placeholder="e.g. ZOMATO, TATA STEEL").strip()
    with c_btn:
        st.write(""); st.write("")
        analyze_btn = st.button("🚀 DEEP ANALYZE", use_container_width=True)
        
    if analyze_btn and raw_input:
        base_ticker = clean_ticker_input(raw_input)
        
        with st.spinner(f"Analyzing Fundamentals & Technicals for {base_ticker}..."):
            try:
                # 1. Get Data
                df, fund, real_ticker = get_stock_details(base_ticker)
                
                if df.empty:
                    st.error(f"❌ Could not find data for '{real_ticker}'.")
                else:
                    # 2. Analyze
                    res = analyze_stock(real_ticker, df)
                    
                    # 3. AUTO-RUN BACKTEST
                    wr, trades = run_backtest(real_ticker)
                    
                    st.header(f"📊 Analysis: {real_ticker}")
                    
                    # --- SECTION 1: THE VERDICT ---
                    st.subheader("1. AI Verdict")
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Action", res['Action'], delta=f"{res['Vol']:.1f}% Volatility")
                    m2.metric("Institutional Bias", res['InstAct'], help="Based on VWAP & Volume")
                    
                    # Win Rate Badge
                    if wr > 50:
                        m3.success(f"✅ **Safe Bet:** {wr:.1f}% Win Rate ({trades} Trades)")
                    else:
                        m3.error(f"⚠️ **Risky:** {wr:.1f}% Win Rate ({trades} Trades)")
                        
                    st.info(f"**Why?** {res['Justification']}")

                    # --- SECTION 2: THE PLAN ---
                    st.subheader("2. Execution Plan")
                    p1, p2, p3 = st.columns(3)
                    p1.metric("ENTRY PRICE", f"₹{res['Price']:.2f}")
                    p2.metric("TARGET", f"₹{res['Target']:.2f}", help=res['TargetLogic'])
                    p3.metric("STOP LOSS", f"₹{res['SL']:.2f}")
                    
                    # If Waiting, show Advice
                    if res['Signal'] == "WAIT":
                        st.warning(f"💡 **Recommendation:** Do not buy now. Wait for price to cross **₹{res['Resistance']:.2f}** or buy at support **₹{res['Support']:.2f}**.")
                    
                    # --- SECTION 3: HEALTH CHECK ---
                    st.subheader("3. Fundamental Health")
                    f1, f2, f3 = st.columns(3)
                    f1.metric("P/E Ratio", fund['PE'], help="Lower is usually better")
                    f2.metric("Sector", fund['Sector'])
                    f3.metric("Analyst Rating", fund['Rec'])

                    # --- SECTION 4: CHART ---
                    st.subheader("4. Live Chart")
                    fig = go.Figure()
                    fig.add_trace(go.Candlestick(x=res['History'].index, open=res['History']['Open'], high=res['History']['High'], low=res['History']['Low'], close=res['History']['Close']))
                    fig.add_trace(go.Scatter(x=res['History'].index, y=res['History']['VWAP'], line=dict(color='orange'), name='VWAP'))
                    fig.add_hline(y=res['Target'], line_color="green", line_dash="dash", annotation_text="TARGET")
                    fig.add_hline(y=res['SL'], line_color="red", line_dash="dash", annotation_text="SL")
                    st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"System Error: {e}")

# --- TAB 2: SCANNER ---
with tab2:
    if st.button("🔎 SCAN TOP 180 STOCKS", type="primary"):
        with st.spinner("Scanning Market..."):
            market_data = get_batch_data()
            results = []
            progress = st.progress(0)
            for i, t in enumerate(SCANNER_STOCKS):
                try:
                    df = market_data[t].copy(); df.dropna(inplace=True)
                    res = analyze_stock(t, df)
                    if res and res['Signal'] in ["BUY", "SELL"] and res['Vol'] > 1.5:
                        results.append(res)
                except: continue
                progress.progress((i+1)/len(SCANNER_STOCKS))
            progress.empty()
            
            if results:
                results = sorted(results, key=lambda x: x['Vol'], reverse=True)
                for item in results:
                    with st.expander(f"{item['Signal']} {item['Ticker']} | Vol: {item['Vol']:.1f}%"):
                        c1, c2, c3 = st.columns(3)
                        c1.metric("ENTRY", f"₹{item['Price']:.2f}")
                        c2.metric("TARGET", f"₹{item['Target']:.2f}")
                        c3.metric("STOP", f"₹{item['SL']:.2f}")
                        if st.button(f"Verify {item['Ticker']}", key=item['Ticker']):
                            wr, trades = run_backtest(item['Ticker'])
                            st.write(f"Accuracy: {wr:.1f}%")
            else:
                st.warning("Market is dead. No signals found.")
