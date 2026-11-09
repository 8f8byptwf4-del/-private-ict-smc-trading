from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
import secrets
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from datetime import datetime, timezone
from dotenv import load_dotenv
import os, time, requests, pandas as pd, numpy as np

load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env'))
app = FastAPI(title='Private AI ICT SMC Signal API', version='1.1.0', docs_url=None, redoc_url=None)
security = HTTPBasic()

def require_login(credentials: HTTPBasicCredentials = Depends(security)):
    expected_user = os.getenv('APP_USERNAME', 'owner')
    expected_password = os.getenv('APP_PASSWORD', '')
    user_ok = secrets.compare_digest(credentials.username.encode(), expected_user.encode())
    pass_ok = bool(expected_password) and secrets.compare_digest(credentials.password.encode(), expected_password.encode())
    if not (user_ok and pass_ok):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='Login required', headers={'WWW-Authenticate': 'Basic'})
    return credentials.username
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = {}
SYMBOLS = {'BTCUSDT':'BTCUSDT', 'XAUUSD':'XAU/USD', 'EURUSD':'EUR/USD'}


def parse_binance(rows):
    df=pd.DataFrame(rows, columns=['t','open','high','low','close','volume','ct','q','n','tb','tq','ig'])
    df['time']=pd.to_datetime(df['t'],unit='ms',utc=True)
    for c in ['open','high','low','close','volume']: df[c]=pd.to_numeric(df[c])
    return df[['time','open','high','low','close','volume']]


def fetch_btc():
    base='https://api.binance.com/api/v3/klines'
    def get(interval,limit):
        r=requests.get(base,params={'symbol':'BTCUSDT','interval':interval,'limit':limit},timeout=12); r.raise_for_status(); return parse_binance(r.json())
    return {'1m':get('1m',1000),'15m':get('15m',500),'1h':get('1h',500)}


def fetch_twelvedata(symbol):
    key=os.getenv('TWELVEDATA_API_KEY','').strip()
    if not key: raise RuntimeError('Missing TWELVEDATA_API_KEY. Add it to backend/.env to enable XAUUSD and EURUSD live data.')
    def get(interval,outputsize):
        r=requests.get('https://api.twelvedata.com/time_series',params={'symbol':SYMBOLS[symbol],'interval':interval,'outputsize':outputsize,'apikey':key,'timezone':'UTC'},timeout=15)
        r.raise_for_status(); data=r.json()
        if 'values' not in data: raise RuntimeError(data.get('message','Twelve Data returned no candle data'))
        df=pd.DataFrame(data['values']).rename(columns={'datetime':'time'}); df['time']=pd.to_datetime(df['time'],utc=True)
        for c in ['open','high','low','close']: df[c]=pd.to_numeric(df[c])
        df['volume']=pd.to_numeric(df.get('volume',0),errors='coerce').fillna(0)
        return df[['time','open','high','low','close','volume']].sort_values('time').reset_index(drop=True)
    return {'1m':get('1min',500),'15m':get('15min',500),'1h':get('1h',500)}


def get_data(symbol, force=False):
    if symbol not in SYMBOLS: raise HTTPException(404,'Unsupported market')
    now=time.time(); cached=CACHE.get(symbol)
    if not force and cached and now-cached['at']<20: return cached['frames'], cached['source'], cached['at']
    try:
        frames=fetch_btc() if symbol=='BTCUSDT' else fetch_twelvedata(symbol)
        CACHE[symbol]={'frames':frames,'source':'Binance public REST' if symbol=='BTCUSDT' else 'Twelve Data REST','at':now}
        return frames,CACHE[symbol]['source'],now
    except Exception as e:
        if cached and now-cached['at']<180: return cached['frames'],cached['source']+' (cached; feed refresh failed)',cached['at']
        raise HTTPException(503,str(e))


def resample(df, tf):
    rule={'5m':'5min','4h':'4h'}[tf]
    return df.set_index('time').resample(rule).agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna().reset_index()


def pivots(df, n=3):
    highs=[]; lows=[]
    for i in range(n, len(df)-n):
        if df.high.iloc[i] == df.high.iloc[i-n:i+n+1].max(): highs.append((i,float(df.high.iloc[i])))
        if df.low.iloc[i] == df.low.iloc[i-n:i+n+1].min(): lows.append((i,float(df.low.iloc[i])))
    return highs,lows


def analyze(raw):
    frames={'1m':raw['1m'],'5m':resample(raw['1m'],'5m'),'15m':raw['15m'],'1h':raw['1h'],'4h':resample(raw['1h'],'4h')}
    f4,f1,f15,f5,fm=[frames[x] for x in ['4h','1h','15m','5m','1m']]
    if min(len(f4),len(f1),len(f15),len(f5),len(fm))<8: raise HTTPException(503,'Not enough candles to analyze all timeframes yet')
    price=float(fm.close.iloc[-1]); checks=[]; score_buy=0; score_sell=0
    def bias(frame):
        c=frame.close; ema=float(c.ewm(span=min(20,max(5,len(c)//2)),adjust=False).mean().iloc[-1]); return 'BULLISH' if c.iloc[-1]>ema and ema>float(c.iloc[-4]) else 'BEARISH' if c.iloc[-1]<ema and ema<float(c.iloc[-4]) else 'NEUTRAL'
    b4,b1,b15= bias(f4),bias(f1),bias(f15)
    for label,b,w in [('4H trend',b4,10),('1H institutional bias',b1,20),('15M structure bias',b15,15)]:
        if b=='BULLISH': score_buy+=w
        elif b=='BEARISH': score_sell+=w
        checks.append({'concept':label,'status':b,'detail':'EMA slope and close alignment; objective trend proxy'})
    # 5m swing structure / BOS
    h,l=pivots(f5,2); prev_high=next((v for i,v in reversed(h) if i<len(f5)-2),None); prev_low=next((v for i,v in reversed(l) if i<len(f5)-2),None)
    bos_up=prev_high is not None and float(f5.close.iloc[-1])>prev_high
    bos_dn=prev_low is not None and float(f5.close.iloc[-1])<prev_low
    if bos_up: score_buy+=15
    if bos_dn: score_sell+=15
    checks.append({'concept':'5M BOS / CHoCH','status':'Bullish BOS' if bos_up else 'Bearish BOS' if bos_dn else 'Not confirmed','detail':'Close beyond last confirmed 5m swing'})
    # sweep: recent bar pierces prior range then closes back inside
    recent=f5.iloc[-2]; prior=f5.iloc[-12:-2]
    sweep_low=float(recent.low)<float(prior.low.min()) and float(recent.close)>float(prior.low.min())
    sweep_high=float(recent.high)>float(prior.high.max()) and float(recent.close)<float(prior.high.max())
    if sweep_low: score_buy+=20
    if sweep_high: score_sell+=20
    checks.append({'concept':'Liquidity sweep','status':'SSL sweep' if sweep_low else 'BSL sweep' if sweep_high else 'Not detected','detail':'Wick through recent 10-bar extreme with close back inside'})
    # 3-candle FVG
    a,b,c=f5.iloc[-3],f5.iloc[-2],f5.iloc[-1]
    bull_fvg=float(c.low)>float(a.high); bear_fvg=float(c.high)<float(a.low)
    if bull_fvg: score_buy+=10
    if bear_fvg: score_sell+=10
    checks.append({'concept':'Fair Value Gap','status':'Bullish FVG' if bull_fvg else 'Bearish FVG' if bear_fvg else 'None','detail':'Three-candle non-overlap; latest completed candles'})
    # displacement body relative to recent median
    body=abs(float(b.close)-float(b.open)); med=float((f5.close-f5.open).abs().iloc[-20:-2].median()) if len(f5)>22 else 0
    disp=med>0 and body>1.5*med
    disp_up=disp and b.close>b.open; disp_dn=disp and b.close<b.open
    if disp_up: score_buy+=10
    if disp_dn: score_sell+=10
    checks.append({'concept':'Displacement','status':'Bullish' if disp_up else 'Bearish' if disp_dn else 'Not confirmed','detail':'Middle candle body > 1.5x recent median body'})
    # premium/discount 15m dealing range
    rng=f15.iloc[-20:]; mid=(float(rng.high.max())+float(rng.low.min()))/2
    loc='Discount' if price<mid else 'Premium'
    if loc=='Discount': score_buy+=5
    else: score_sell+=5
    checks.append({'concept':'Premium / Discount','status':loc,'detail':'Position within recent 20-bar 15m range'})
    # 1m trigger: last completed close direction and micro break
    m1=fm.iloc[-2]; mprev=fm.iloc[-5:-2]
    trig_buy=float(m1.close)>float(m1.open) and float(m1.close)>float(mprev.high.max())
    trig_sell=float(m1.close)<float(m1.open) and float(m1.close)<float(mprev.low.min())
    if trig_buy: score_buy+=10
    if trig_sell: score_sell+=10
    checks.append({'concept':'1M precision trigger','status':'Bullish trigger' if trig_buy else 'Bearish trigger' if trig_sell else 'Not confirmed','detail':'Completed 1m candle close beyond preceding micro range'})
    side='WAIT'; score=max(score_buy,score_sell); direction='BUY' if score_buy>score_sell else 'SELL'
    if score>=65 and abs(score_buy-score_sell)>=20 and ((direction=='BUY' and trig_buy and b1=='BULLISH') or (direction=='SELL' and trig_sell and b1=='BEARISH')): side=direction
    # structural stop from 5m recent swing/range with minimum volatility buffer
    atr=float((f5.high-f5.low).rolling(14).mean().iloc[-1]); entry=price; sl=tp1=tp2=tp3=None; rr=None
    if side!='WAIT':
        if side=='BUY':
            structural=float(f5.low.iloc[-8:].min()); sl=min(structural,entry-0.8*atr); risk=entry-sl
            if risk<=0: side='WAIT'
            else: tp1=entry+1.5*risk; tp2=entry+2*risk; tp3=entry+3*risk
        else:
            structural=float(f5.high.iloc[-8:].max()); sl=max(structural,entry+0.8*atr); risk=sl-entry
            if risk<=0: side='WAIT'
            else: tp1=entry-1.5*risk; tp2=entry-2*risk; tp3=entry-3*risk
        if side!='WAIT': rr='1:1.5 / 1:2 / 1:3'
    reason='No high-confluence setup. Wait for 1H alignment and a confirmed 1M trigger.' if side=='WAIT' else f'{side} confluence: 1H bias aligned with 5M liquidity/structure and 1M trigger. Rule-based ICT/SMC proxy; validate spread and execution.'
    return {'symbol':None,'decision':side,'score':int(score),'scores':{'buy':score_buy,'sell':score_sell},'bias':{'4h':b4,'1h':b1,'15m':b15},'price':price,'entry':entry if side!='WAIT' else None,'sl':sl if side!='WAIT' else None,'tp1':tp1,'tp2':tp2,'tp3':tp3,'rr':rr,'reason':reason,'checks':checks,'candles':{tf:[{'time':int(x.time.timestamp()),'open':float(x.open),'high':float(x.high),'low':float(x.low),'close':float(x.close)} for x in frames[tf].tail(300).itertuples()] for tf in ['1m','5m','15m','1h','4h']},'as_of':datetime.now(timezone.utc).isoformat()}

@app.get('/api/health')
def health(_user: str = Depends(require_login)): return {'status':'ok','service':'ICT/SMC signal-only API'}

@app.get('/api/scan/{symbol}')
def scan(symbol:str, refresh:bool=False, _user: str = Depends(require_login)):
    symbol=symbol.upper(); raw,source,at=get_data(symbol,refresh)
    result=analyze(raw); result['symbol']=symbol; result['data_source']=source; result['data_age_seconds']=round(time.time()-at,1); result['data_status']='LIVE' if time.time()-at<90 else 'STALE'
    return result

@app.get('/api/markets')
def markets(_user: str = Depends(require_login)):
    out=[]
    for s in SYMBOLS:
        try:
            raw,source,at=get_data(s)
            latest=raw['1m'].iloc[-1]
            out.append({'symbol':s,'price':float(latest.close),'source':source,'status':'LIVE' if time.time()-at<90 else 'STALE','as_of':latest.time.isoformat()})
        except Exception as e: out.append({'symbol':s,'price':None,'source':'—','status':'UNAVAILABLE','error':str(e)})
    return out

@app.get('/')
def index(_user: str = Depends(require_login)): return FileResponse(os.path.join(ROOT,'frontend','index.html'))
# Dashboard is served only through the authenticated '/' route; do not expose frontend files publicly.
