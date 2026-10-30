import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import sqlite3, os, time
from datetime import datetime, date
from streamlit_autorefresh import st_autorefresh

APP_DIR=os.path.dirname(os.path.abspath(__file__))
DB=os.path.join(APP_DIR,'data','paper_trading.db')
os.makedirs(os.path.dirname(DB), exist_ok=True)
DEFAULT_SYMBOLS='RELIANCE,TCS,INFY,HDFCBANK,ICICIBANK,SBIN,ITC,LT,AXISBANK,BAJFINANCE,TATAMOTORS,MARUTI,SUNPHARMA,TATASTEEL,ONGC,NTPC,POWERGRID,ADANIENT,COALINDIA,HINDALCO,JSWSTEEL,TECHM,WIPRO,HCLTECH,INDUSINDBK,ULTRACEMCO,ASIANPAINT,TITAN,GRASIM,NESTLEIND'

def conn():
    c=sqlite3.connect(DB,check_same_thread=False)
    c.execute('''CREATE TABLE IF NOT EXISTS trades(id INTEGER PRIMARY KEY AUTOINCREMENT, symbol TEXT, side TEXT, entry REAL, stop REAL, target REAL, qty INTEGER, opened TEXT, status TEXT, exit REAL, closed TEXT, pnl REAL, reason TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS signals(id INTEGER PRIMARY KEY AUTOINCREMENT, symbol TEXT, side TEXT, price REAL, time TEXT, reason TEXT, UNIQUE(symbol,side,time))''')
    # Backward-compatible migration for existing V3 paper-trading databases.
    columns={row[1] for row in c.execute('PRAGMA table_info(trades)').fetchall()}
    if 'last_bar' not in columns:
        c.execute('ALTER TABLE trades ADD COLUMN last_bar TEXT')
    c.commit(); return c

def get_bars(symbol):
    try:
        df=yf.download(symbol+'.NS',period='5d',interval='5m',auto_adjust=True,progress=False,threads=False,timeout=12)
        if df is None or df.empty: return pd.DataFrame()
        if isinstance(df.columns,pd.MultiIndex): df.columns=df.columns.get_level_values(0)
        df=df.rename(columns={str(x).title():str(x).title() for x in df.columns})
        required=['Open','High','Low','Close','Volume']
        if not all(x in df.columns for x in required): return pd.DataFrame()
        df=df[required].dropna()
        # Exclude current incomplete candle: public providers can publish partial bars.
        if len(df)>1: df=df.iloc[:-1]
        return df
    except Exception:
        return pd.DataFrame()

def rsi(series,period=14):
    d=series.diff(); gain=d.clip(lower=0).ewm(alpha=1/period,adjust=False).mean(); loss=(-d.clip(upper=0)).ewm(alpha=1/period,adjust=False).mean()
    rs=gain/loss.replace(0,np.nan); return 100-(100/(1+rs))

def evaluate(df):
    if len(df)<35: return None
    x=df.copy(); x['ema9']=x.Close.ewm(span=9,adjust=False).mean(); x['ema15']=x.Close.ewm(span=15,adjust=False).mean(); x['rsi']=rsi(x.Close)
    # Intraday VWAP using date groups; bars are provider local-time indexed.
    idx=pd.to_datetime(x.index)
    dates=idx.date
    tp=(x.High+x.Low+x.Close)/3; vol=x.Volume.replace(0,np.nan)
    x['vwap']=(tp*vol).groupby(dates).cumsum()/vol.groupby(dates).cumsum()
    x['volavg']=x.Volume.shift(1).rolling(10).mean()
    a=x.iloc[-2]; b=x.iloc[-1]
    if pd.isna(b.rsi) or pd.isna(b.volavg) or b.volavg<=0: return None
    bullish_engulf=(b.Close>b.Open and a.Close<a.Open and b.Close>=a.Open and b.Open<=a.Close)
    bearish_engulf=(b.Close<b.Open and a.Close>a.Open and b.Close<=a.Open and b.Open>=a.Close)
    rng=max(float(b.High-b.Low),1e-9); body=abs(float(b.Close-b.Open))
    bull_hammer=(min(b.Open,b.Close)-b.Low)>=body*1.5 and (b.High-max(b.Open,b.Close))<=body
    bear_star=(b.High-max(b.Open,b.Close))>=body*1.5 and (min(b.Open,b.Close)-b.Low)<=body
    volok=float(b.Volume)>=1.5*float(b.volavg)
    cross_up=float(a.ema9)<=float(a.ema15) and float(b.ema9)>float(b.ema15)
    cross_dn=float(a.ema9)>=float(a.ema15) and float(b.ema9)<float(b.ema15)
    if cross_up and b.Close>b.vwap and 40<=b.rsi<=70 and volok and (bullish_engulf or bull_hammer or (body/rng>.85 and b.Close>b.Open)):
        return {'side':'BUY','price':float(b.Close),'stop':float(b.Low),'reason':'EMA9 crossed above EMA15; above VWAP; RSI/volume/candle confirmed'}
    if cross_dn and b.Close<b.vwap and 30<=b.rsi<=60 and volok and (bearish_engulf or bear_star or (body/rng>.85 and b.Close<b.Open)):
        return {'side':'SELL','price':float(b.Close),'stop':float(b.High),'reason':'EMA9 crossed below EMA15; below VWAP; RSI/volume/candle confirmed'}
    return None

def scan(symbols, capital_risk, max_positions, max_trades, daily_profit, daily_loss):
    c=conn(); now=datetime.now().isoformat(timespec='seconds'); status=[]; fresh=[]
    day=date.today().isoformat()
    closed=pd.read_sql_query('SELECT pnl FROM trades WHERE status="CLOSED" AND closed LIKE ?',c,params=(day+'%',))
    daypnl=float(closed.pnl.sum()) if not closed.empty else 0.0
    opens=pd.read_sql_query('SELECT * FROM trades WHERE status="OPEN"',c)
    n_today=c.execute('SELECT COUNT(*) FROM trades WHERE opened LIKE ?', (day+'%',)).fetchone()[0]
    halted=daypnl>=daily_profit or daypnl<=-daily_loss or n_today>=max_trades
    for sym in symbols:
        df=get_bars(sym)
        if df.empty:
            status.append({'Symbol':sym,'Status':'No data','Last price':None,'Signal':'—'}); continue
        price=float(df.Close.iloc[-1]); sig=evaluate(df); status.append({'Symbol':sym,'Status':'OK','Last price':round(price,2),'Signal':sig['side'] if sig else '—'})
        # Manage open paper trades on each scan, including stop/target checks.
        current=c.execute('SELECT id,side,entry,stop,target,qty,last_bar FROM trades WHERE symbol=? AND status="OPEN"',(sym,)).fetchall()
        bar_time=str(df.index[-1])
        ema9=float(df.Close.ewm(span=9,adjust=False).mean().iloc[-1])
        # Small 0.10% buffer below/above the yellow 9 EMA to reduce noise-triggered exits.
        buffer=ema9*0.001
        for tid,side,entry,stop,target,qty,last_bar in current:
            # Evaluate a completed candle only once. The newly trailed stop becomes
            # active from the next completed candle, avoiding same-bar false hits.
            if last_bar == bar_time:
                continue
            last=df.iloc[-1]; exit_price=None; reason=''
            if side=='BUY' and float(last.Low)<=stop: exit_price=stop; reason='Trailing/initial stop-loss'
            elif side=='BUY' and float(last.High)>=target: exit_price=target; reason='Target'
            elif side=='SELL' and float(last.High)>=stop: exit_price=stop; reason='Trailing/initial stop-loss'
            elif side=='SELL' and float(last.Low)<=target: exit_price=target; reason='Target'
            if exit_price is not None:
                pnl=(exit_price-entry)*qty*(1 if side=='BUY' else -1)
                c.execute('UPDATE trades SET status="CLOSED",exit=?,closed=?,pnl=?,reason=?,last_bar=? WHERE id=?',(exit_price,now,pnl,now+' '+reason,bar_time,tid))
            else:
                # One-way trailing: BUY stop can only rise; SELL stop can only fall.
                if side=='BUY':
                    candidate=ema9-buffer
                    new_stop=max(float(stop),candidate) if candidate < float(last.Close) else float(stop)
                else:
                    candidate=ema9+buffer
                    new_stop=min(float(stop),candidate) if candidate > float(last.Close) else float(stop)
                c.execute('UPDATE trades SET stop=?,last_bar=? WHERE id=?',(new_stop,bar_time,tid))
            c.commit()
        if sig and not halted:
            # avoid duplicate entry on same latest bar and same symbol; max active positions enforced
            bar_time=str(df.index[-1]); dup=c.execute('SELECT 1 FROM signals WHERE symbol=? AND side=? AND time=?',(sym,sig['side'],bar_time)).fetchone()
            opens_n=c.execute('SELECT COUNT(*) FROM trades WHERE status="OPEN"').fetchone()[0]
            if not dup and opens_n<max_positions and n_today<max_trades:
                risk_per_share=abs(sig['price']-sig['stop'])
                qty=int(capital_risk/risk_per_share) if risk_per_share>0 else 0
                if qty>0:
                    target=sig['price']+2.5*risk_per_share if sig['side']=='BUY' else sig['price']-2.5*risk_per_share
                    c.execute('INSERT INTO trades(symbol,side,entry,stop,target,qty,opened,status,reason,last_bar) VALUES(?,?,?,?,?,?,?,"OPEN",?,?)',(sym,sig['side'],sig['price'],sig['stop'],target,qty,now,sig['reason'],bar_time))
                    c.execute('INSERT OR IGNORE INTO signals(symbol,side,price,time,reason) VALUES(?,?,?,?,?)',(sym,sig['side'],sig['price'],bar_time,sig['reason'])); c.commit(); fresh.append(f"{sym} {sig['side']} @ {sig['price']:.2f}"); n_today+=1
    c.close(); return pd.DataFrame(status),fresh

st.set_page_config(page_title='SmartStock V3 — No API Key',page_icon='📈',layout='wide')
st.title('📈 SmartStock V4 — 9 EMA Trailing Paper Trading')
st.caption('Public Yahoo Finance data • 9 EMA one-way trailing SL • No Angel One API key • Paper trades only')
with st.sidebar:
    st.header('Scanner settings')
    refresh=st.slider('Refresh interval (seconds)',30,300,60,step=30)
    symbols_raw=st.text_area('NSE symbols (comma separated)',DEFAULT_SYMBOLS,height=180)
    symbols=list(dict.fromkeys(s.strip().upper().replace('.NS','') for s in symbols_raw.split(',') if s.strip()))
    risk=st.number_input('Max risk per trade (₹)',min_value=100,max_value=10000,value=1500,step=100)
    maxpos=st.selectbox('Max open positions',[1,2],index=1)
    maxtrades=st.selectbox('Max trades per day',[2,4,6],index=2)
    profit_limit=st.number_input('Daily profit limit (₹)',min_value=500,value=14000,step=500)
    loss_limit=st.number_input('Daily loss limit (₹)',min_value=500,value=4500,step=500)
    st.warning('Keep the PC and internet on. Data availability/rate limits depend on the public provider.')
st_autorefresh(interval=refresh*1000,key='auto_refresh')
if not symbols: st.error('Enter at least one NSE symbol.'); st.stop()
if st.button('Scan now',type='primary') or 'last_scan' not in st.session_state or time.time()-st.session_state.get('last_scan',0)>=refresh:
    with st.spinner(f'Scanning {len(symbols)} symbols using public market data…'):
        scan_df,new_trades=scan(symbols,risk,maxpos,maxtrades,profit_limit,loss_limit)
        st.session_state.scan_df=scan_df; st.session_state.new_trades=new_trades; st.session_state.last_scan=time.time()
scan_df=st.session_state.get('scan_df',pd.DataFrame()); new_trades=st.session_state.get('new_trades',[])
if new_trades: st.success('New paper trade(s): '+', '.join(new_trades))
c=conn(); opens=pd.read_sql_query('SELECT symbol,side,entry,stop,target,qty,opened FROM trades WHERE status="OPEN" ORDER BY opened DESC',c); alltrades=pd.read_sql_query('SELECT symbol,side,entry,exit,qty,pnl,status,reason,opened,closed FROM trades ORDER BY id DESC LIMIT 200',c)
closed=pd.read_sql_query('SELECT pnl FROM trades WHERE status="CLOSED" AND closed LIKE ?',c,params=(date.today().isoformat()+'%',)); daypnl=float(closed.pnl.sum()) if not closed.empty else 0.0
m1,m2,m3,m4=st.columns(4); m1.metric('Symbols configured',len(symbols)); m2.metric('Open paper positions',len(opens)); m3.metric('Today realised P&L',f'₹{daypnl:,.2f}'); m4.metric('Last scan',datetime.fromtimestamp(st.session_state.get('last_scan',time.time())).strftime('%H:%M:%S'))
if daypnl>=profit_limit: st.warning('Daily profit limit reached. New entries will be blocked on the next scan; open trades are not auto-flattened.')
if daypnl<=-loss_limit: st.error('Daily loss limit reached. New entries will be blocked on the next scan; open trades are not auto-flattened.')
t1,t2,t3=st.tabs(['Scanner','Open positions','Trade history'])
with t1:
    st.info('Automatic refresh runs while this Streamlit page is open. Public data can be delayed, missing or rate-limited. This is not an exchange-grade feed.')
    if not scan_df.empty: st.dataframe(scan_df,use_container_width=True,hide_index=True)
with t2:
    st.dataframe(opens,use_container_width=True,hide_index=True)
    if not opens.empty:
        tid=st.selectbox('Select position ID to close manually',pd.read_sql_query('SELECT id,symbol,side,entry FROM trades WHERE status="OPEN"',c).apply(lambda r:f"{r.id} | {r.symbol} | {r.side} | {r.entry}",axis=1).tolist())
        if st.button('Close selected paper position'):
            id_=int(tid.split('|')[0].strip()); row=c.execute('SELECT symbol,side,entry,qty FROM trades WHERE id=? AND status="OPEN"',(id_,)).fetchone()
            if row:
                sym,side,entry,qty=row; pxrow=scan_df[scan_df.Symbol==sym] if not scan_df.empty else pd.DataFrame()
                if not pxrow.empty and pd.notna(pxrow.iloc[0]['Last price']):
                    ex=float(pxrow.iloc[0]['Last price']); pnl=(ex-entry)*qty*(1 if side=='BUY' else -1); c.execute('UPDATE trades SET status="CLOSED",exit=?,closed=?,pnl=?,reason=? WHERE id=?',(ex,datetime.now().isoformat(timespec='seconds'),pnl,'Manual close',id_)); c.commit(); st.success(f'Closed at {ex:.2f}, P&L ₹{pnl:.2f}'); st.rerun()
with t3: st.dataframe(alltrades,use_container_width=True,hide_index=True)
c.close()
st.caption('V4: Trailing SL is recalculated after each newly completed 5-minute candle: BUY = 9 EMA − 0.10% buffer (only moves up); SELL = 9 EMA + 0.10% buffer (only moves down). The updated stop applies from the next completed candle. Strategy signals are experimental; public-source prices may be delayed. No real orders are sent.')
