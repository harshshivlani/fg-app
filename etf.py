#data manipulation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import openpyxl

#filter warnings for final presentation
import warnings
warnings.filterwarnings("ignore")
import edhec_risk_kit as erk
import yfinance as yf
import ipywidgets as widgets
from ipywidgets import interact, interact_manual
from datetime import date
import time
import datetime
import investpy
import plotly
import plotly.graph_objects as go
import plotly.express as px
from bs4 import BeautifulSoup 
import csv
from plotly.subplots import make_subplots
from pandas.tseries import offsets


#notebook formatting
from IPython.core.display import display, HTML

def drawdowns2020(data):
    return_series = pd.DataFrame(data.pct_change().dropna()[str(date.today().year):])
    wealth_index = 1000*(1+return_series).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index - previous_peaks)/previous_peaks
    return drawdowns.min(axis=0)

def returns_heatmap(data, max_drawdowns, title, tickers, sortby='1-Day', reit='No', currencies='No', alok_secs='No', fg_data='No', india='No', style='Yes'):
    """

    """
    if reit=='Yes':
        df = pd.DataFrame(data = (data.pct_change(1).iloc[-1,:], data.pct_change(5).iloc[-1,:], data.pct_change(21).iloc[-1,:], data.pct_change(63).iloc[-1,:], data['2020':].iloc[-1,:]/data['2020':].iloc[0,:]-1, data['2020':].iloc[-1,:]/data['2020-03-23':].iloc[0,:]-1, data.pct_change(126).iloc[-1,:], data.pct_change(252).iloc[-1,:], max_drawdowns))
        df.index = ['1-Day', '1-Week', '1-Month', '3-Month', 'YTD', 'March-23 TD', '6-Month', '1-Year', 'Drawdowns']

    elif currencies=='Yes':
        df = pd.DataFrame(data = (data.pct_change(1).iloc[-1,:],  data.pct_change(5).iloc[-1,:], data.pct_change(21).iloc[-1,:], data.pct_change(42).iloc[-1,:], data.pct_change(63).iloc[-1,:], data['2020':].iloc[-1,:]/data['2020':].iloc[0,:]-1, data['2020':].iloc[-1,:]/data['2020-03-23':].iloc[0,:]-1, data.pct_change(126).iloc[-1,:], data.pct_change(252).iloc[-1,:], max_drawdowns))
        df.index = ['1-Day', '1-Week', '1-Month', '2-Month', '3-Month', 'YTD', 'March-23 TD', '6-Month', '1-Year', 'Drawdowns']

    elif alok_secs=='Yes':
        data = data.ffill()
        df = pd.DataFrame(data = (data.pct_change(1).iloc[-1,:],  data.pct_change(5).iloc[-1,:], data.pct_change(21).iloc[-1,:], data.pct_change(42).iloc[-1,:], data.pct_change(63).iloc[-1,:], data['2020':].iloc[-1,:]/data['2020-01-06':].iloc[0,:]-1, data['2020':].iloc[-1,:]/data['2020-03-23':].iloc[0,:]-1, max_drawdowns))
        df.index = ['1-Day', '1-Week', '1-Month', '2-Month', '3-Month', 'YTD', 'March-23 TD', 'Drawdowns']

    elif fg_data=='Yes':
        now = time.localtime()
        last = datetime.date(now.tm_year, now.tm_mon, 1) - datetime.timedelta(1)
        df = pd.DataFrame(data = (data.pct_change(1).iloc[-1,:],  data.pct_change(7).iloc[-1,:], data.pct_change(30).iloc[-1,:], data.iloc[-1,:]/data[last:].iloc[0,:]-1, data.pct_change(90).iloc[-1,:]))
        df.index = ['1-Day', '1-Week', '1-Month', 'MTD', '3-Month']


    else:
        df = pd.DataFrame(data = (data.pct_change(1).iloc[-1,:], data.pct_change(5).iloc[-1,:], data.pct_change(21).iloc[-1,:], data.pct_change(63).iloc[-1,:], data['2020':].iloc[-1,:]/data['2020':].iloc[0,:]-1, data['2020':].iloc[-1,:]/data['2020-03-23':].iloc[0,:]-1, data.pct_change(126).iloc[-1,:], data.pct_change(252).iloc[-1,:], data.pct_change(252*3).iloc[-1,:], max_drawdowns))
        df.index = ['1-Day', '1-Week', '1-Month', '3-Month', 'YTD', 'March-23 TD', '6-Month', '1-Year', '3-Year', 'Drawdowns']


    df_perf = (df.T*100)
    if india=='No':
        df_perf.insert(loc=0, column='Tickers', value=list(tickers))
        df_perf = df_perf.sort_values(by=sortby, ascending=False)

    #df_perf.insert(loc=1, column='Yields', value=list(yields.iloc[:,0].round(2).values))
    else:
        df_perf = df_perf.sort_values(by=sortby, ascending=False)

    df_perf.index.name = title

    if style=='Yes' and india=='No':
        df_perf = df_perf.round(2).style.format('{0:,.2f}%', subset=list(df_perf.drop(['Tickers'], axis=1).columns))\
                 .background_gradient(cmap='RdYlGn', subset=(df_perf.drop(['Tickers'], axis=1).columns))\
                 .set_properties(**{'font-size': '10pt',})

    elif style=='Yes' and india=='Yes':
        df_perf = df_perf.round(2).style.format('{0:,.2f}%')\
                 .background_gradient(cmap='RdYlGn')\
                 .set_properties(**{'font-size': '10pt',})

    else:
        df_perf = df_perf.round(2)

    return df_perf


def returns_heatmap_alok(data, max_drawdowns, title, sortby='1-Day', style='Yes'):
    """

    """
    data = data.ffill()
    df = pd.DataFrame(data = (data.pct_change(1).iloc[-1,:],  data.pct_change(5).iloc[-1,:], data.pct_change(21).iloc[-1,:], data.pct_change(63).iloc[-1,:], data.pct_change(126).iloc[-1,:], data['2020':].iloc[-1,:]/data['2020-01-06':].iloc[0,:]-1, data['2020':].iloc[-1,:]/data['2020-03-23':].iloc[0,:]-1, max_drawdowns))
    df.index = ['1-Day', '1-Week', '1-Month', '3-Month', '6-Month', 'YTD', 'March-23 TD', 'Drawdowns']

    df_perf = (df.T*100)
    df_perf = df_perf.sort_values(by=sortby, ascending=False)
    df_perf.index.name = title

    if style=='Yes':
        df_perf = df_perf.round(2).style.format('{0:,.2f}%')\
                     .background_gradient(cmap='RdYlGn')\
                     .set_properties(**{'font-size': '10pt',})

    else:
        df_perf = df_perf.round(2)
    return df_perf



def data_sov():
    #Soveriegn Fixed Income ETFs
    data_sov = yf.download('AGZ BWX EDV EMB EMLC GOVT HYD IEF IEI IGOV MUB PCY SHY SUB TFI TIP TLT TMF VWOB ZROZ', progress=False)['Adj Close']
    tickers = data_sov.columns
    data_sov.dropna(inplace=True)
    data_sov.columns = ["iShares Agency Bond ETF","SPDR  BBG Barclays International Treasury Bond ETF","Vanguard Extended Duration Treasury ETF","iShares JPM USD Emerging Markets Bond ETF","VanEck Vectors J.P. Morgan EM Local Currency Bond ETF","iShares U.S. Treasury Bond ETF","VanEck Vectors High-Yield Municipal Index ETF","iShares 7-10 Year Treasury Bond ETF","iShares 3-7 Year Treasury Bond ETF","iShares International Treasury Bond ETF","iShares National Muni Bond ETF","Invesco Emerging Markets Sovereign Debt ETF","iShares 1-3 Year Treasury Bond ETF","iShares Short-Term National Muni Bond ETF","SPDR Nuveen  BBG Barclays Municipal Bond ETF","iShares TIPS Bond ETF","iShares 20+ Year Treasury Bond ETF","Direxion Daily 20+ Year Treasury Bull 3X Shares","Vanguard Emerging Markets Government Bond ETF","PIMCO 25+ Year Zero Coupon US Treasury Index ETF"]
    return (data_sov,tickers)

def data_corp():
    #Corporate Fixed Income ETFs -  IG & HY in Developed & EM
    data_corp = yf.download('AGG ANGL BKLN BND BNDX CWB EMHY FALN FLOT FMB FPE HYEM HYG HYXE HYXU JNK LQD SHYG SRLN USIG VCIT VCLT VCSH', progress=False)['Adj Close']
    tickers = data_corp.columns
    data_corp.dropna(inplace=True)
    data_corp.columns = ["iShares Core U.S. Aggregate Bond ETF","VanEck Vectors Fallen Angel High Yield Bond ETF","Invesco Senior Loan ETF","Vanguard Total Bond Market ETF","Vanguard Total International Bond ETF","SPDR BBG Barclays Convertible Securities ETF","iShares EM High Yield Bond ETF","iShares Fallen Angels USD Bond ETF","iShares Floating Rate Bond ETF","First Trust Managed Municipal ETF","First Trust Preferred Securities & Income ETF","VanEck Emerging Markets High Yield Bond ETF","iShares USD High Yield Corporate Bond ETF","iShares USD High Yield ex Oil & Gas Corporate Bond ETF","iShares International High Yield Bond ETF","SPDR BBG Barclays High Yield Bond ETF","iShares USD IG Corporate Bond ETF","iShares 0-5 Year High Yield Corporate Bond ETF","SPDR Blackstone / GSO Senior Loan ETF","iShares Broad USD Investment Grade Bond ETF","Vanguard Intermediate-Term Corporate Bond ETF","Vanguard Long-Term Corporate Bond ETF","Vanguard Short-Term Corporate Bond ETF"]
    return (data_corp, tickers)

def data_reit(ticker='No'):
    #Real Estate Investment Trust (REIT) ETFs
    data_reit = yf.download('VNQ VNQI SRVR INDS HOMZ REZ PPTY IFEU REM MORT SRET RFI FFR GQRE CHIR FFR WPS IFGL KBWY BBRE ROOF NETL STOR', progress=False)['Adj Close']['2015':]
    tickers = data_reit.columns
    data_reit.dropna(inplace=True)
    if ticker == 'Yes':
        data_reit.columns = data_reit.columns
    else:
        data_reit.columns = ['Beta Builders', 'China RE', 'NAREIT Developed Market RE', 'Quality RE', 'Hoya Housing RE', 'Europe RE', ' International Developed RE', 'Industrial RE',
                     'Premium Yield Equity RE', 'VanEck Mortgage','NetLease RE', 'Divserified RE', 'iShares Mortgage', 'iShares Residential', 'Cohen & Steers RE',
                     'Small-Cap RE', 'Super Dividend', 'Data & Infrastructure RE',
                     'Store Retail', 'Vanguard US', 'Vanguard International', 'iShares Developed Market RE']

    return (data_reit,tickers)

def data_cur():
    #Currencies
    data_cur = yf.download('KRWUSD=X BRLUSD=X IDRUSD=X MXNUSD=X RUBUSD=X CADUSD=X JPYUSD=X EURUSD=X INRUSD=X TRYUSD=X NZDUSD=X GBPUSD=X DX-Y.NYB AUDUSD=X AUDJPY=X EURCHF=X TWDUSD=X THBUSD=X COPUSD=X CNYUSD=X CLPUSD=X ZARUSD=X HKDUSD=X CHFUSD=X SGDUSD=X',progress=False)['Adj Close']['2015':]
    tickers = data_cur.columns
    data_cur.dropna(inplace=True)
    data_cur.columns = ['AUD/JPY', 'Australian Dollar', 'Brazilian Real', 'Canadian Dollar', 'Swiss Francs', 'Chilean Peso', 'Chinese Yuan',
                    'Colombian Peso', 'Dollar Index', 'EUR/CHF', 'Euro', 'British Pound', 'Hong Kong Dollar', 'Indonesian Rupiah',
                    'Indian Rupee', 'Japanese Yen', 'Korean Won', 'Mexican Peso', 'New Zealand Dollar', 'Russian Ruble',
                    'Singapore Dollar', 'Thai Baht', 'Turkish Lira', 'Taiwanese Dollar', 'South African Rand']
    return (data_cur,tickers)


def data_comd():
    #Soveriegn Fixed Income ETFs
    data_comd = yf.download('COMT GSG DBC USO CL=F HG=F COPX GC=F GLD GDX PA=F PALL PPLT SI=F SIL ICLN TAN W=F ZC=F NG=F', progress=False)['Adj Close']
    tickers = data_comd.columns
    data_comd.dropna(inplace=True)
    data_comd.columns = ['Crude Oil WTI','COMT', 'Copper Miners', 'DB CMTY Fund', 'Gold Futures', 'Gold Miners',
                     'Gold ETF', 'GSCI ETF', 'Copper Futures', 'Clean Energy', 'NatGas Futures',
                     'Palladium Futures', 'Physical Palladium ETF', 'Physical Platinum ETF', 'Silver Futures', 'Silver ETF',
                     'Solar ETF', 'USO Oil ETF', 'Wheat Futures', 'Corn Futures']
    return (data_comd,tickers)

def data_country():
    #Country Equity ETFs
    data_count = yf.download('ECH EDEN EEM EEMA EEMS EEMV EFA EFAV EFG EFNL EFV EIDO EIRL EIS ENOR ENZL EPHE EPOL EPU ERUS EWA EWC EWD EWG EWGS EWH EWI EWJ EWK EWL EWM EWN EWO EWP EWQ EWS EWT EWU EWUS EWW EWY EWZ EZA EZU FXI INDY IWM SMIN SPY THD', progress=False)['Adj Close']['2015':]
    tickers = data_count.columns
    data_count.dropna(inplace=True)
    data_count.columns = ["Chile","Denmark","Emerging Markets","Asian EM","EM SmallCap","EM Low Volatility","EAFE","EAFE Low Volatility","EAFE Growth","Finland","EAFE Value","Indonesia","Ireland","Israel","Norway","New Zealand","Philippines","Poland","Peru","Russia","Australia","Canada","Sweden","Germany","Germany Small Cap","Hong Kong","Italy","Japan","Belgian","Switzerland","Malaysia","Netherlands","Austria","Spain","France","Singapore","Taiwan","UK","UK Small Cap","Mexico","South Korea","Brazil","South Africa","EuzoZone", "China","India", "US Russell 2000", "India Small Cap", "US S&P500", "Thailand"]
    return (data_count,tickers)


def data_equities():
    data_eq = yf.download("ARKK BLOK BOTZ CHIE CHIH CHII CHIK CHIM CHIQ CHIR CHIS CHIU CHIX FXI INTF JETS LRGF MTUM QFN.AX QQQ QRE.AX QUAL SIZE SPY STK.PA STN.PA STP.PA STQ.PA STR.PA STS.PA STU.PA STW.PA STZ.PA USMV VLUE VYM XAR XBI XEG.TO XES XFN.TO XHB XHE XHS XIT.TO XLB XLC XLE XLF XLI XLK XLP XLRE XLU XLV XLY XMA.TO XME XOP XPH XRT XSD XST.TO XSW XTN XUT.TO", progress=False)['Adj Close']['2018':]
    tickers = data_eq.columns
    data_eq.dropna(inplace=True)
    data_eq.columns = ["U.S. Innovation ETF","U.S. Blockchain ETF","U.S. Robotics&AI ETF","China Energy ETF","China Health Care ETF","China Industrial ETF","China Technology ETF","China Materials ETF","China Consumer Cyclical ETF","China Real Estate ETF","China Consumer Non-cyclical ETF","China Utilities ETF","China Finance ETF","China  ETF","DM Multi Factor ETF","U.S. Airlines ETF","U.S. Multi Factor ETF","U.S. Momentum ETF","Australia Finance ETF","US NASDAQ","Australia Resources ETF","U.S. Quality ETF","U.S. Size ETF","US S&P500 ETF","Europe Technology ETF","Europe Energy ETF","Europe Materials ETF","Europe Industrial ETF","Europe Consumer Cyclical ETF","Europe Consumer Non-cyclical ETF","Europe Utilities ETF","Europe Healthcare ETF","Europe Finance ETF","U.S. Low Volitality ETF","U.S. Value ETF","U.S. High Dividend Yield ETF","U.S. Aerospace&Defence ETF","U.S. Biotech ETF","Canada Energy ETF","U.S. Oil & Gas Equipment & Services ETF","Canada Finance ETF","U.S. Homebuilding ETF","U.S. Health Care Equipment & Supplies ETF","U.S. Health Care Services ETF","Canada Technology ETF","U.S. Basic Materials ETF","U.S. Telecom ETF","U.S. Energy ETF","U.S. Finance ETF","U.S. Industrial ETF","U.S. Technology ETF","U.S. Consumer Non-cyclical ETF","U.S. Real Estate ETF","U.S. Utilities ETF","U.S. Healthcare ETF","U.S. Consumer Cyclical ETF","Canada Materials ETF","U.S. Metals & Mining ETF","U.S. Oil&Gas Exploration ETF","U.S. Pharma ETF","U.S. Retail ETF","U.S. Semi Conductors ETF","Canada Consumer Non-cyclical ETF","U.S. Software ETF","U.S. Transportation ETF","Canada Utilities ETF"]
    return (data_eq, tickers)



def heatmap(rets, title='Cross Asset ETFs Heatmap', figsize=(15,10), annot_size=12, n_rows=10, n_cols=8, pct='Yes', mon=None):

    rets.columns = ['Return']
    rets = rets.sort_values('Return', ascending=False)
    symbols = (np.asarray(list(rets.index))).reshape(n_rows,n_cols)
    pct_rets = (np.asarray(rets['Return'].values)).reshape(n_rows,n_cols)
    rows =[]
    for i in range(1,n_rows+1):
        rows += list(np.repeat(i,n_cols))

    cols = list(list(np.arange(1,n_cols+1))*n_rows)
    rets['Rows'] = rows
    rets['Cols'] = cols

    result = rets.pivot(index = 'Rows', columns = 'Cols', values = 'Return')
    if pct=='Yes':
        labels = (np.asarray(["{0} \n {1:.2%} ".format(symb,value)
                     for symb, value in zip(symbols.flatten(), pct_rets.flatten())])).reshape(n_rows,n_cols)

    else:
        labels = (np.asarray(["{0} \n {1:.2f} \n {2}".format(symb,value, mon)
                     for symb, value, mon in zip(symbols.flatten(), pct_rets.flatten(), mon.flatten())])).reshape(n_rows,n_cols)
    fig, ax = plt.subplots(figsize=figsize)
    plt.title(title, fontsize=15)
    ttl = ax.title
    ttl.set_position([0.5,1.05])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')
    return sns.heatmap(result, annot=labels, fmt="", cmap = 'RdYlGn', linewidth=0.30, ax=ax, annot_kws={"size": annot_size})


def alok_heatmap():
    alok_data = yf.download("CADUSD=X EURUSD=X GBPUSD=X USDINR=X USDSGD=X ACWI URTH ^NDX ^NSEI ^GSPC BZ=F DBC GLD GC=F PALL USO O9P.SI ANGL EDV EMB EMHY EMLC FPE GOVT HYEM HYG JNK LQD SHYG TLT TMF VWOB ZROZ ABR A17U.SI AV.L J85.SI J91U.SI MERY.PA ME8U.SI RGL.L VNQ VNQI 8697.T AAPL ABT ADBE AHCO AIEQ AIIQ AMD AMZN B3SA3.SA BABA BB BBJP BOX CLX CMG CRWD CTXS DBX DOCU DPZ ERUS EWT EWZ FXI GILD GOOG HD JD LSE.L MA MFA MOEX.ME MSFT NDAQ NEM NFLX NLOK NTES NVDA PTON PYPL PZZA QQQ S68.SI SHOP SMG SNY SPY SQ TDOC TMO TWLO TWOU TWTR V WORK X.TO ZM PD MKC NKE LQDA.L UST.PA PSHZF VTWO IWM", start='2019-12-31',end=date.today())['Adj Close']

    alok_data = alok_data.rename(columns = {'^NSEI':'NIFTY50', 'BZ=F':'Crude', 'A17U.SI':'AREIT', 'AV.L':'AV', 'J85.SI':'CDREIT',
                                            'J91U.SI':'EREIT','S68.SI':'SGX', 'X.TO':'TMX Group', 'CADUSD=X':'CADUSD','ME8U.SI':'MINT',
                                            'EURUSD=X':'EURUSD', 'GBPUSD=X':'GBPUSD', 'USDINR=X':'USDINR', 'USDSGD=X':'USDSGD',
                                            '^NDX':'NASDAQ', 'URTH':'MSCI Wrld', '^GSPC':'S&P500', 'GC=F': 'GOLD', 'O9P.SI':'AHYG'})
    alok_data = alok_data.ffill().asfreq('B').dropna()
    alok_data1 = pd.DataFrame(data = (alok_data.iloc[-1,:],alok_data.pct_change().iloc[-1,:])).T
    alok_data1.columns = ['Price', 'Chg (%)']
    alok_data2 = alok_data1.style.format({'Chg (%)': "{:.2%}", 'Price': "{:.2f}"})
    alok_map = returns_heatmap_alok(alok_data, drawdowns2020(alok_data), title='Securities', style='No')
    prices = pd.DataFrame(alok_data1['Price'])
    prices.index.name = 'Securities'
    final = prices.merge(alok_map, on='Securities').sort_values(by='1-Day', ascending=False).style.format('{0:,.2f}%', subset=list(alok_map.columns))\
                     .format('{0:,.2f}', subset=['Price'])\
                     .background_gradient(cmap='RdYlGn', subset=list(alok_map.columns))\
                     .set_properties(**{'text-align':'left','font-family': 'Segoe UI','font-size': '10.5px'})
    return (final, alok_data1, alok_data)


def cross_asset_data():
    """
    """
    data_sov = yf.download('SHY IEF TLT IEI EMB EMLC AGZ BWX TIP', progress=False)['Adj Close']['2019':]
    data_sov.dropna(inplace=True)
    data_corp = yf.download('AGG BND BNDX LQD HYG SHYG JNK FALN ANGL FPE HYXE HYXU HYEM EMHY', progress=False)['Adj Close']['2019':]
    data_corp.dropna(inplace=True)
    data_reit = yf.download('VNQ VNQI SRVR INDS HOMZ REZ IFEU REM MORT SRET RFI FFR GQRE CHIR FFR WPS PPTY IFGL KBWY ROOF NETL SPG SKT STOR', progress=False)['Adj Close']['2019':]
    data_reit.dropna(inplace=True)
    data_cur = yf.download('KRWUSD=X  BRLUSD=X  IDRUSD=X  MXNUSD=X  RUBUSD=X  CADUSD=X  JPYUSD=X  EURUSD=X  INRUSD=X  TRYUSD=X  NZDUSD=X  GBPUSD=X  DX-Y.NYB  AUDUSD=X  AUDJPY=X  EURCHF=X', progress=False)['Adj Close']['2019':].iloc[:-1,:]
    data_cur.dropna(inplace=True)
    data_comd = yf.download('COMT USO CL=F HG=F COPX GC=F GLD GDX PA=F PALL PPLT SI=F SIL ICLN TAN W=F ZC=F NG=F', progress=False)['Adj Close']['2019':]
    data_comd.dropna(inplace=True)

    all_data = data_sov.merge(data_corp, on='Date').merge(data_reit, on='Date').merge(data_cur, on='Date').merge(data_comd, on='Date')
    return all_data


def cross_asset_heatmap(data, n_rows, n_cols, days=1, figsize=(15,10), annot_size=12, title='Cross Asset ETFs Heatmap'):
    """
    """
    rets = pd.DataFrame(data.pct_change(days).iloc[-1,:])
    return heatmap(rets, n_rows=n_rows, n_cols=n_cols, figsize=figsize, annot_size=annot_size, title=title)


#nbi:hide_in
def import_data(asset_class, sortby='1-Day'):
    """

    """
    tdy = str(date.today().day)+'/'+str(date.today().month)+'/'+str(date.today().year)
    oneyr = str(date.today().day)+'/'+str(date.today().month)+'/'+str(date.today().year-1)
    threeyr = str(date.today().day)+'/'+str(date.today().month)+'/'+str(date.today().year-3)
    #Import list of ETFs and Ticker Names
    etf_list = pd.read_excel('etf_names.xlsx', header=0, sheet_name=asset_class, engine='openpyxl')

    #Define function to fetch historical data from Investing.com
    def hist_data(name, country):
        df = investpy.get_etf_historical_data(etf=name, country=country, from_date=oneyr, to_date=tdy)['Close']
        df = pd.DataFrame(df)
        df.columns = [name]
        return df

    #Build an empty df to store historical 1 year data
    df = pd.DataFrame(index=pd.bdate_range(start=oneyr, end=date.today()))
    df.index.name='Date'

    #download and merge all data
    for i in range(len(etf_list)):
            df = df.join(hist_data(etf_list[asset_class][i], etf_list['Country'][i]), on='Date')

    #Forward fill for any missing days i.e. holidays
    df1 = df.iloc[:-1,:].ffill().dropna()
    df1.index.name = 'Date'


    #Generate multi timeframe returns table
    df0 = pd.DataFrame(data = (df1.pct_change(1).iloc[-1,:], df1.pct_change(5).iloc[-1,:], df1.pct_change(21).iloc[-1,:], df1.pct_change(63).iloc[-1,:], df1['2020':].iloc[-1,:]/df1['2020':].iloc[0,:]-1, df1['2020':].iloc[-1,:]/df1['2020-03-23':].iloc[0,:]-1, df1.pct_change(126).iloc[-1,:], df1.pct_change(252).iloc[-1,:], drawdowns2020(df1)))
    df0.index = ['1-Day', '1-Week', '1-Month', '3-Month', 'YTD', 'March-23 TD', '6-Month', '1-Year', 'Drawdowns']

    df_perf = (df0.T*100)
    df_perf.index.name = asset_class


    #Add Ticker Names and sort the dataframe
    tickers = pd.DataFrame(etf_list['Ticker'])
    tickers.index = etf_list[asset_class]
    df2 = tickers.merge(df_perf, on=asset_class)
    df2  = df2.sort_values(by=sortby, ascending=False)
    df2 = df2.round(2).style.format('{0:,.2f}%', subset=list(df2.drop(['Ticker'], axis=1).columns))\
                     .background_gradient(cmap='RdYlGn', subset=(df2.drop(['Ticker'], axis=1).columns))\
                     .set_properties(**{'font-size': '10pt',})

    return df2, df1




def import_heatmap(data, sortby):
    """
    data refers to return data for multiple timeframes
    sortby requires a str format input which includes 1-Day, 1-Week, 1-Month, YTD, 1-Year
    """
    #Sort values
    data = data.sort_values(by=sortby, ascending=False)

    #Conditionally format the dataframe
    data = data.round(2).style.format('{0:,.2f}%', subset=list(data.drop(['Ticker'], axis=1).columns))\
                     .background_gradient(cmap='RdYlGn', subset=(data.drop(['Ticker'], axis=1).columns))\
                     .set_properties(**{'font-size': '10pt',})
    return data


def analytics(price_data, returns_data):
    """

    """
    def heatmap_all(sortby):
        return import_heatmap(returns_data, sortby=sortby)

    interact(heatmap_all,
             sortby = widgets.Dropdown(options=['1-Day', '1-Week', '1-Month','3-Month', 'YTD', 'March-23 TD', '6-Month', '1-Year', 'Drawdowns'], value='1-Day', description = 'Sort By: '));

    def perf_chart_all(start_date):
        df = ((((1+price_data.dropna()[start_date:date.today()].pct_change().fillna(0.00))).cumprod()-1)).round(4)
        fig = px.line(df, x=df.index, y=df.columns)
        fig.update_layout(title = price_data.columns.name + ' ETFs',
                       xaxis_title='Date',
                       yaxis_title='Return (%)', font=dict(family="Segoe UI, monospace", size=13, color="#7f7f7f"),
                       legend_title_text='ETFs', plot_bgcolor = 'White', yaxis_tickformat = '%')
        fig.update_traces(hovertemplate='Date: %{x} <br>Return: %{y:.2%}')
        fig.update_yaxes(automargin=True)
        fig.show()


    interact(perf_chart_all,
             start_date = widgets.Dropdown(options=[('1 Month', one_m), ('3 Months', three_m), ('6 Month', six_m),
                                                ('YTD', ytd), ('1 Year', one_yr),
                                                ('From 23rd March', '2020-03-23')], value=ytd, description = 'Period'));


    #nbi:hide_in
    def trend_analysis(start_date,inv, ma):
        d = (price_data.pct_change(ma).dropna()[start_date:date.today()].resample(inv).agg(lambda x: (x + 1).prod() - 1).round(4)*100)
        fig = go.Figure(data=go.Heatmap(
                z=((d - d.mean())/d.std()).round(2).T.values,
                x=((d - d.mean())/d.std()).index,
                y=list(price_data.columns), zmax=3, zmin=-3,
                colorscale='rdylgn', hovertemplate='Date: %{x}<br>ETF: %{y}<br>Return Z-Score: %{z}<extra></extra>', colorbar = dict(title='Return Z-Score')))

        fig.update_layout(
            title= price_data.columns.name + ' ETFs Return Trend Analysis',
            xaxis_nticks=20, font=dict(family="Segoe UI, monospace", size=13, color="#7f7f7f"))

        fig.show()

    style = {'description_width': 'initial'}
    interact(trend_analysis,
             start_date = widgets.Dropdown(options=[('1 Month', one_m), ('3 Months', three_m), ('6 Months', six_m),
                                                    ('YTD', ytd), ('1 Year', one_yr),
                                                    ('From 23rd March', '2020-03-23')], value=ytd, description = 'Period'),
             inv = widgets.Dropdown(options=[('Daily', 'B'), ('Weekly', 'W'), ('Monthly', 'BM')], value='B', description='Return Interval:', style=style),
             ma = widgets.BoundedIntText(value=15, min=1, max=30, step=1, description='Rolling Return Period:', disabled=False, style=style));

#####    
def updated_world_indices(category='Major'):
    """
    
    """
    tdy = str(date.today().day)+'/'+str(date.today().month)+'/'+str(date.today().year)
    idxs = pd.read_excel('World_Indices_List.xlsx', index_col=0, header=0, sheet_name=category, engine='openpyxl')
    index_names = list(idxs['Indices'])
    country_names = list(idxs['Country'])
    
    def idx_data(index, country):
        df = investpy.get_index_historical_data(index=index, country=country, from_date='01/01/2020', to_date=tdy)['Close']
        df = pd.DataFrame(df)
        df.columns = [index]
        return df
    
    df = pd.DataFrame(index=pd.bdate_range(start='2020-01-01', end=date.today()))
    df.index.name='Date'
    
    #Stitch Local Currency Indices Data
    for i in range(len(idxs)):
        df = df.join(idx_data(index_names[i], country_names[i]), on='Date')
        
    if date.today().weekday() < 5:
        df1 = df.iloc[:-1,:].ffill().dropna()
    else:
        df1 = df.ffill().dropna()
    df1.index.name = 'Date'
    
    #Local Currency Returns Table
    oned_lcl = pd.concat([df1.iloc[-1,:],
                         df1.iloc[-1,:]-df1.iloc[-2,:],
                        (df1.iloc[-1,:]/df1.iloc[-2,:]-1),
                        (df1.iloc[-1,:]/df1.iloc[-6,:]-1),
                        (df1.iloc[-1,:]/df1.iloc[-22,:]-1),
                        (df1.iloc[-1,:]/df1.iloc[0,:]-1)], axis=1)
    oned_lcl.columns = ['Price (EOD)','1D Chg', '1D Chg (%)', '1W Chg (%)', '1M Chg (%)', 'Chg YTD (%)']
    
    #Add Country and Currency Names
    cntry = pd.DataFrame(idxs['Country'])
    cntry.index = idxs['Indices']

    currency = pd.DataFrame(idxs['Currency'])
    currency.index = idxs['Indices']
    
    oned_lcl = oned_lcl.sort_values('1D Chg (%)', ascending=False)
    oned_lcl.index.name = 'Indices'
    oned_lcl = oned_lcl.merge(cntry['Country'], on='Indices')
    oned_lcl = oned_lcl.merge(currency['Currency'], on='Indices')
    oned_lcl=oned_lcl[['Country', 'Price (EOD)','1D Chg', '1D Chg (%)', '1W Chg (%)', '1M Chg (%)', 'Chg YTD (%)', 'Currency']]
    
    #Import Currency Data

    ccyidx = yf.download("CADUSD=X BRLUSD=X MXNUSD=X EURUSD=X GBPUSD=X EURUSD=X EURUSD=X EURUSD=X EURUSD=X EURUSD=X CHFUSD=X EURUSD=X EURUSD=X EURUSD=X SEKUSD=X DKKUSD=X RUBUSD=X PLNUSD=X HUFUSD=X TRYUSD=X SARUSD=X JPYUSD=X AUDUSD=X NZDUSD=X CNYUSD=X CNYUSD=X CNYUSD=X CNYUSD=X HKDUSD=X TWDUSD=X THBUSD=X KRWUSD=X IDRUSD=X INRUSD=X INRUSD=X PHPUSD=X PKRUSD=X VNDUSD=X BHDUSD=X BGNUSD=X CLPUSD=X COPUSD=X EURUSD=X CZKUSD=X EGPUSD=X EURUSD=X EURUSD=X EURUSD=X MYRUSD=X OMRUSD=X PENUSD=X QARUSD=X SGDUSD=X ZARUSD=X KRWUSD=X TNDUSD=X", start='2020-01-01', progress=False)['Close'].ffill()
    dfmix = df1.merge(ccyidx, on='Date')
    ccys = dfmix.iloc[:,len(idxs):]
    
    #Calculate Currency Returns
    oned_ccy = pd.concat([ccys.iloc[-1,:],
                             ccys.iloc[-1,:]-ccys.iloc[-2,:],
                            (ccys.iloc[-1,:]/ccys.iloc[-2,:]-1),
                            (ccys.iloc[-1,:]/ccys.iloc[-6,:]-1),
                            (ccys.iloc[-1,:]/ccys.iloc[-22,:]-1),
                            (ccys.iloc[-1,:]/ccys.iloc[0,:]-1)], axis=1)
    oned_ccy.columns = ['Price (EOD)','1D Chg', '1D CChg (%)', '1W CChg (%)', '1M CChg (%)', 'CChg YTD (%)']
    
    abc = oned_ccy.copy()
    abc = abc.append({'Price (EOD)':0, '1D Chg':0, '1D CChg (%)':0, '1W CChg (%)':0, '1M CChg (%)':0, 'CChg YTD (%)':0}, ignore_index=True)
    abc.index = list(oned_ccy.index) + ['USD']
    abc.index.name='Currency'
    
    #Convert Local Currency to USD Returns
    oned_lcl_copy = oned_lcl.copy()
    oned_lcl_copy['Indices'] = oned_lcl_copy.index
    testa = oned_lcl_copy.merge(abc[['1D CChg (%)', '1W CChg (%)', '1M CChg (%)', 'CChg YTD (%)']], on='Currency')
    testa.index = testa['Indices']

    testa['$ 1D Chg (%)'] = (1+testa['1D Chg (%)'])*(1+testa['1D CChg (%)'])-1
    testa['$ 1W Chg (%)'] = (1+testa['1W Chg (%)'])*(1+testa['1W CChg (%)'])-1
    testa['$ 1M Chg (%)'] = (1+testa['1M Chg (%)'])*(1+testa['1M CChg (%)'])-1
    testa['$ Chg YTD (%)'] = (1+testa['Chg YTD (%)'])*(1+testa['CChg YTD (%)'])-1

    return [testa, df1.index[-1]]

def format_world_data(testa, usd='USD', country='No'):
    if country == 'Yes':
        if usd=='USD':
            testa = testa[['Country', 'Price (EOD)', '$ 1D Chg (%)', '$ 1W Chg (%)', '$ 1M Chg (%)', '$ Chg YTD (%)']]
        else:
            testa = testa[['Country', 'Price (EOD)', '1D Chg (%)', '1W Chg (%)', '1M Chg (%)', 'Chg YTD (%)']]
        testa.columns = ['Country', 'Price', '1-Day', '1-Week', '1-Month', 'YTD']
        testa = testa.sort_values('1-Day', ascending=False)
        formatted = testa.style.format({'Price': "{:.2f}", '1-Day': "{:.2%}", '1-Week': "{:.2%}", '1-Month': "{:.2%}", 'YTD': "{:.2%}"})\
                         .background_gradient(cmap='RdYlGn', subset=list(testa.drop(['Price','Country'], axis=1).columns))
    
    else:
        if usd=='USD':
            testa = testa[['Price (EOD)', '$ 1D Chg (%)', '$ 1W Chg (%)', '$ 1M Chg (%)', '$ Chg YTD (%)']]
        else:
            testa = testa[['Price (EOD)', '1D Chg (%)', '1W Chg (%)', '1M Chg (%)', 'Chg YTD (%)']]
        testa.columns = ['Price', '1-Day', '1-Week', '1-Month', 'YTD']
        testa = testa.sort_values('1-Day', ascending=False)
        formatted = testa.style.format({'Price': "{:.2f}", '1-Day': "{:.2%}", '1-Week': "{:.2%}", '1-Month': "{:.2%}", 'YTD': "{:.2%}"})\
                         .background_gradient(cmap='RdYlGn', subset=list(testa.drop(['Price'], axis=1).columns))
    return testa, formatted



def etf_details(ticker, option, asset):
    """
    """

    url = 'https://etfdb.com/etf/{}/#etf-holdings&sort_name=weight&sort_order=desc&page=1'.format(ticker)
    html = requests.get(url).content
    df_list = pd.read_html(html)

    url2 = 'https://finance.yahoo.com/quote/{}'.format(ticker)
    html2 = requests.get(url2).content
    df_list2 = pd.read_html(html2)
    
    if asset=='Equity/REIT ETF':
        if option == 'Top 15 Holdings':
            df_list[2] = df_list[2][:-1]
            df_list[2]['% Assets'] = df_list[2]['% Assets'].str.rstrip('%').astype('float')
            fig = go.Figure(go.Bar(
                                    x=list(df_list[2]['% Assets']),
                                    y=list(df_list[2]['Holding']), text=list(df_list[2]['Symbol']),
                                    orientation='h', hovertemplate='Ticker: %{text}<br>Holding: %{x}<extra></extra>'))
            fig.update_traces(texttemplate='%{x:.2f}%', textposition='outside')
            fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
            fig.update_layout(title="Top 15 Holdings", font=dict(family="Segoe UI",size=13,color="#7f7f7f"), yaxis=dict(autorange="reversed"), plot_bgcolor='rgb(255,255,255)')
            return fig

        elif option == 'Sector Exposure':
            fig = px.pie(list(df_list[4]['Percentage'].str.rstrip('%').astype('float64')), values=list(df_list[4]['Percentage'].str.rstrip('%').astype('float64')),
                                     names=list(df_list[4]['Sector']), color_discrete_sequence=px.colors.sequential.RdBu)
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(title="ETF Sector Exposure", font=dict(family="Segoe UI",size=15,color="#7f7f7f"), xaxis={'categoryorder':'category descending'})
            return fig


        elif option == 'Market Cap Exposure':
            fig = px.pie(list(df_list[5]['Percentage'].str.rstrip('%').astype('float')), values=list(df_list[5]['Percentage'].str.rstrip('%').astype('float')),
                                     names=list(df_list[5]['Market Cap']), color_discrete_sequence=px.colors.sequential.RdBu)
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(title="ETF Market Cap Exposure", font=dict(family="Segoe UI",size=15,color="#7f7f7f"))
            return fig


        elif option == 'Country Exposure':
            fig = px.pie(list(df_list[8]['Percentage'].str.rstrip('%').astype('float')), values=list(df_list[8]['Percentage'].str.rstrip('%').astype('float')),
                                     names=list(df_list[8]['Country']), color_discrete_sequence=px.colors.sequential.RdBu)
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(title="ETF Country Exposure", font=dict(family="Segoe UI",size=15,color="#7f7f7f"))
            return fig  


        elif option == 'Asset Allocation':
            fig = px.pie(list(df_list[9]['Percentage'].str.rstrip('%').astype('float')), values=list(df_list[9]['Percentage'].str.rstrip('%').astype('float')),
                                     names=list(df_list[9]['Asset']), color_discrete_sequence=px.colors.sequential.RdBu)
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(title="ETF Asset Class Exposure", font=dict(family="Segoe UI",size=15,color="#7f7f7f"))
            return fig


        elif option == 'General Overview':
            summary1 = df_list2[1]
            summary1.columns = ['Summary', 'Data']
            return summary1.set_index('Summary')
    
    elif asset=='Fixed Income ETF':
        if option == 'Top 15 Holdings':
            df_list[2] = df_list[2][:-1]
            df_list[2]['% Assets'] = df_list[2]['% Assets'].str.rstrip('%').astype('float')
            fig = go.Figure(go.Bar(
                            x=list(df_list[2]['% Assets']),
                            y=list(df_list[2]['Holding']), text=list(df_list[2]['Symbol']),
                            orientation='h', hovertemplate='Ticker: %{text}<br>Holding: %{x}<extra></extra>'))
            fig.update_traces(texttemplate='%{x:.2f}%', textposition='outside')
            fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
            fig.update_layout(title="Top 15 Holdings for {}".format(ticker), font=dict(family="Segoe UI",size=13,color="#7f7f7f"), yaxis=dict(autorange="reversed"), plot_bgcolor='rgb(255,255,255)')
            return fig       

        if option == 'Bond Sector Exposure':
            fig = px.pie(list(df_list[5]['Percentage'].str.rstrip('%').astype('float64')), values=list(df_list[5]['Percentage'].str.rstrip('%').astype('float64')),
                             names=list(df_list[5]['Bond Sector']), color_discrete_sequence=px.colors.sequential.RdBu)
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(title="Bond Sector Exposure for {}".format(ticker), font=dict(family="Segoe UI",size=15,color="#7f7f7f"), xaxis={'categoryorder':'category descending'})
            return fig

        if option == 'Coupon Breakdown':
            fig = px.pie(list(df_list[6]['Percentage'].str.rstrip('%').astype('float64')), values=list(df_list[6]['Percentage'].str.rstrip('%').astype('float64')),
                             names=list(df_list[6]['Coupon Range']), color_discrete_sequence=px.colors.sequential.RdBu)
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(title="Bonds Coupon Breakdown for {}".format(ticker), font=dict(family="Segoe UI",size=15,color="#7f7f7f"), xaxis={'categoryorder':'category descending'})
            return fig

        if option == 'Credit Quality Exposure':
            fig = px.pie(list(df_list[7]['Percentage'].str.rstrip('%').astype('float64')), values=list(df_list[7]['Percentage'].str.rstrip('%').astype('float64')),
                             names=list(df_list[7]['Credit']), color_discrete_sequence=px.colors.sequential.RdBu)
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(title="Bonds Credit Quality Exposure for {}".format(ticker), font=dict(family="Segoe UI",size=15,color="#7f7f7f"), xaxis={'categoryorder':'category descending'})
            return fig

        if option == 'Maturity Profile':
            fig = px.pie(list(df_list[8]['Percentage'].str.rstrip('%').astype('float64')), values=list(df_list[8]['Percentage'].str.rstrip('%').astype('float64')),
                             names=list(df_list[8]['Maturity']), color_discrete_sequence=px.colors.sequential.RdBu)
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(title="Bonds Maturity Profile for {}".format(ticker), font=dict(family="Segoe UI",size=15,color="#7f7f7f"), xaxis={'categoryorder':'category descending'})
            return fig
        
        if option == 'General Overview':
            summary1 = df_list2[1]
            summary1.columns = ['Summary', 'Data']
            return summary1.set_index('Summary')
        
        
### MACRO

def world_pmis(continent, sortby='Last'):
    if continent=='World':
        url1 = 'https://tradingeconomics.com/country-list/manufacturing-pmi'
    else:
        url1 = 'https://tradingeconomics.com/country-list/manufacturing-pmi?continent={}'.format(continent)

    html1 = requests.get(url1).content
    df_list1 = pd.read_html(html1)

    pmis = df_list1[0].iloc[:,:-1].set_index('Country')
    pmis['Change'] = pmis['Last'] - pmis['Previous']
    pmis = pmis[['Last', 'Previous', 'Change', 'Reference']]
    pmis = pmis.sort_values(by=sortby, ascending=False)
    return pmis.style.format({'Last': "{:.2f}", 'Previous': "{:.2f}", 'Change': "{:+.2f}"})\
        .background_gradient(cmap='RdYlGn', subset=list(pmis.drop(['Reference'], axis=1).columns))


def eco_calendar(importances=['Medium', 'High']):
    """ 
    
    """
    if importances=='All':
        calendar1 = investpy.get_calendar(time_zone = 'GMT +5:30')
    else:
        calendar1 = investpy.get_calendar(time_zone = 'GMT +5:30', importances=importances)
    cols = list(calendar1.columns)
    calendar1.columns = [cols.capitalize() for cols in cols]
    calendar1['Zone'] = calendar1['Zone'].str.capitalize()
    calendar1['Importance'] = calendar1['Importance'].str.capitalize()
    calendar1  = calendar1.drop('Id', axis=1).set_index('Date')
    return calendar1

#nbi:hide_in
def country_macros(country, data_type):
    url1 = 'https://tradingeconomics.com/{}/indicators'.format(country)
    html1 = requests.get(url1).content
    df_list1 = pd.read_html(html1)
    pd.set_option('display.float_format', lambda x: '%.2f' % x)
    
    if data_type == 'Overview':
        return df_list1[1].iloc[:,:-1].set_index(df_list1[1].columns[0])
    elif data_type == 'GDP':
        return df_list1[2].iloc[:,:-1].set_index(df_list1[2].columns[0])
    elif data_type == 'Labour':
        return df_list1[3].iloc[:,:-1].set_index(df_list1[3].columns[0])
    elif data_type == 'Inflation':
        return df_list1[5].iloc[:,:-1].set_index(df_list1[5].columns[0])
    elif data_type == 'Money':
        return df_list1[6].iloc[:,:-1].set_index(df_list1[6].columns[0])
    elif data_type == 'Trade':
        return df_list1[7].iloc[:,:-1].set_index(df_list1[7].columns[0])
    elif data_type == 'Government':
        return df_list1[8].iloc[:,:-1].set_index(df_list1[8].columns[0])
    elif data_type == 'Taxes':
        return df_list1[9].iloc[:,:-1].set_index(df_list1[9].columns[0])
    elif data_type == 'Business':
        return df_list1[10].iloc[:,:-1].set_index(df_list1[10].columns[0])
    elif data_type == 'Consumer':
        return df_list1[11].iloc[:,:-1].set_index(df_list1[11].columns[0])
    elif data_type == 'Housing':
        return df_list1[12].iloc[:,:-1].set_index(df_list1[12].columns[0])
    else:
        return 
    
def live_charts(opt='Yes'):
    if opt=='Yes':
        javascript = """ 
        <!-- TradingView Widget BEGIN -->
        <div class="tradingview-widget-container">
          <div id="tradingview_42617"></div>
          <div class="tradingview-widget-copyright"><a href="https://www.tradingview.com/symbols/AMEX-SPY/" rel="noopener" target="_blank"><span class="blue-text">SPY Chart</span></a> by TradingView</div>
          <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
          <script type="text/javascript">
          new TradingView.widget(
          {
          "width": 980,
          "height": 610,
          "symbol": "AMEX:SPY",
          "interval": "D",
          "timezone": "Asia/Kolkata",
          "theme": "light",
          "style": "1",
          "locale": "en",
          "toolbar_bg": "#f1f3f6",
          "enable_publishing": false,
          "hide_side_toolbar": false,
          "allow_symbol_change": true,
          "container_id": "tradingview_42617"
        }
          );
          </script>
        </div>
        <!-- TradingView Widget END -->
        """
        return HTML(javascript)
    elif opt=='No':
        return print('To view live charts of World Stocks, ETFs, FX & Commodities, select Yes')
    
def gdp(continent, sortby='Last'):   
    url1 = 'https://tradingeconomics.com/country-list/gdp-annual-growth-rate?continent={}'.format(continent)
    html1 = requests.get(url1).content
    df_list1 = pd.read_html(html1)
    df = df_list1[0].set_index('Country')
    df['Change'] = df['Last'] - df['Previous']
    df = df[['Last', 'Previous', 'Change', 'Reference', 'Unit']]
    df.drop('Unit', axis=1, inplace=True)

    df = df.sort_values(by=sortby, ascending=False)
    
    return df.style.format({'Last': "{:.2f}%", 'Previous': "{:.2f}%", 'Change': "{:+.2f}%"})\
            .background_gradient(cmap='RdYlGn', subset=list(df.drop(['Reference'], axis=1).columns))


def retail(continent, time='MoM', sortby='Last',):   
    url1 = 'https://tradingeconomics.com/country-list/retail-sales-{}?continent={}'.format(time, continent)
    html1 = requests.get(url1).content
    df_list1 = pd.read_html(html1)
    df = df_list1[0].set_index('Country')
    df['Change'] = df['Last'] - df['Previous']
    df = df[['Last', 'Previous', 'Change', 'Reference', 'Unit']]
    df.drop('Unit', axis=1, inplace=True)

    df = df.sort_values(by=sortby, ascending=False)
    
    return df.style.format({'Last': "{:.2f}%", 'Previous': "{:.2f}%", 'Change': "{:+.2f}%"})\
            .background_gradient(cmap='RdYlGn', subset=list(df.drop(['Reference'], axis=1).columns))

def inflation(continent, cat='', sortby='Last'):   
    url1 = 'https://tradingeconomics.com/country-list/{}inflation-rate?continent={}'.format(cat, continent)
    html1 = requests.get(url1).content
    df_list1 = pd.read_html(html1)
    df = df_list1[0].set_index('Country')
    df['Change'] = df['Last'] - df['Previous']
    df = df[['Last', 'Previous', 'Change', 'Reference', 'Unit']]
    df.drop('Unit', axis=1, inplace=True)

    df = df.sort_values(by=sortby, ascending=False)
    
    return df.style.format({'Last': "{:.2f}%", 'Previous': "{:.2f}%", 'Change': "{:+.2f}%"})\
            .background_gradient(cmap='RdYlGn', vmax=7, subset=list(df.drop(['Reference'], axis=1).columns))

def unemp(continent, sortby='Last'):   
    url1 = 'https://tradingeconomics.com/country-list/unemployment-rate?continent={}'.format(continent)
    html1 = requests.get(url1).content
    df_list1 = pd.read_html(html1)
    unemp = df_list1[0].set_index('Country')
    unemp['Change'] = unemp['Last'] - unemp['Previous']
    unemp = unemp[['Last', 'Previous', 'Change', 'Reference', 'Unit']]
    unemp.drop('Unit', axis=1, inplace=True)

    unemp = unemp.sort_values(by=sortby, ascending=True)
    return unemp.style.format({'Last': "{:.2f}%", 'Previous': "{:.2f}%", 'Change': "{:+.2f}%"})\
            .background_gradient(cmap='RdYlGn', subset=list(unemp.drop(['Reference'], axis=1).columns))


def live_indices():
    prices=[]
    names=[]
    changes=[]
    percentChanges=[]
    marketCaps=[]
    totalVolumes=[]
    circulatingSupplys=[]
     
    CryptoCurrenciesUrl = "https://in.finance.yahoo.com/world-indices"
    r= requests.get(CryptoCurrenciesUrl)
    data=r.text
    soup=BeautifulSoup(data)
     
    counter = 40
    for i in range(40, 404, 14):
          for row in soup.find_all('tbody'):
            for srow in row.find_all('tr'):
                for name in srow.find_all('td', attrs={'class':'data-col1'}):
                    names.append(name.text)
                for price in srow.find_all('td', attrs={'class':'data-col2'}):
                    prices.append(price.text)
                for change in srow.find_all('td', attrs={'class':'data-col3'}):
                    changes.append(change.text)
                for percentChange in srow.find_all('td', attrs={'class':'data-col4'}):
                    percentChanges.append(percentChange.text)
     
    df = pd.DataFrame({"Names": names, "Prices": prices, "Change": changes, "% Change": percentChanges}).drop_duplicates().set_index('Names')
    df['% Change'] = pd.to_numeric(df['% Change'].str.strip('%'))
    df.drop('CBOE Volatility Index', axis=0, inplace=True)
    df = df.style.format({'% Change': "{:.2f}%"}).background_gradient(cmap='RdYlGn', subset=['% Change'])
    return df

def live_comds():
    prices=[]
    names=[]
    changes=[]
    percentChanges=[]
    marketCaps=[]
    marketTimes=[]
    totalVolumes=[]
    openInterests=[]

    CryptoCurrenciesUrl = "https://in.finance.yahoo.com/commodities"
    r= requests.get(CryptoCurrenciesUrl)
    data=r.text
    soup=BeautifulSoup(data)

    counter = 40
    for i in range(40, 404, 14):
        for row in soup.find_all('tbody'):
            for srow in row.find_all('tr'):
                for name in srow.find_all('td', attrs={'class':'data-col1'}):
                    names.append(name.text)
                for price in srow.find_all('td', attrs={'class':'data-col2'}):
                    prices.append(price.text)
                for time in srow.find_all('td', attrs={'class':'data-col3'}):
                    marketTimes.append(time.text)
                for change in srow.find_all('td', attrs={'class':'data-col4'}):
                    changes.append(change.text)
                for percentChange in srow.find_all('td', attrs={'class':'data-col5'}):
                    percentChanges.append(percentChange.text)
                for volume in srow.find_all('td', attrs={'class':'data-col6'}):
                    totalVolumes.append(volume.text)
                for openInterest in srow.find_all('td', attrs={'class':'data-col7'}):
                    openInterests.append(openInterest.text)

    df = pd.DataFrame({"Names": names, "Prices": prices, "Change": changes, "% Change": percentChanges, "Market Time": marketTimes,'Open Interest': openInterests ,"Volume": totalVolumes}).drop_duplicates().set_index('Names')
    df['% Change'] = pd.to_numeric(df['% Change'].str.strip('%'))
    df = df.sort_values(by='% Change', ascending=False).style.format({'% Change': "{:.2f}%"}).background_gradient(cmap='RdYlGn', subset=['% Change'])
    return df

def live_curr():
    names=[]
    prices=[]
    changes=[]
    percentChanges=[]
    marketCaps=[]
    totalVolumes=[]
    circulatingSupplys=[]

    CryptoCurrenciesUrl = "https://in.finance.yahoo.com/currencies"
    r= requests.get(CryptoCurrenciesUrl)
    data=r.text
    soup=BeautifulSoup(data)

    counter = 40
    for i in range(40, 404, 14):
        for listing in soup.find_all('tr', attrs={'data-reactid':i}):
            for name in listing.find_all('td', attrs={'data-reactid':i+3}):
                 names.append(name.text)
            for price in listing.find_all('td', attrs={'data-reactid':i+4}):
                 prices.append(price.text)
            for change in listing.find_all('td', attrs={'data-reactid':i+5}):
                 changes.append(change.text)
            for percentChange in listing.find_all('td', attrs={'data-reactid':i+7}):
                 percentChanges.append(percentChange.text)
    df = pd.DataFrame({"Names": names, "Prices": prices, "Change": changes, "% Change": percentChanges}).drop_duplicates().set_index('Names')
    df['% Change'] = pd.to_numeric(df['% Change'].str.strip('%'))
    df = df.sort_values(by='% Change', ascending=False).style.format({'% Change': "{:.2f}%"}).background_gradient(cmap='RdYlGn', subset=['% Change'])
    return df


def india_inds(industry='auto'):
    prices=[]
    names=[]
    changes=[]
    percentChanges=[]
    marketCaps=[]
    marketTimes=[]
    totalVolumes=[]
    openInterests=[]

    CryptoCurrenciesUrl = "https://in.finance.yahoo.com/industries/" + str(industry)
    r= requests.get(CryptoCurrenciesUrl)
    data=r.text
    soup=BeautifulSoup(data)

    counter = 40
    for i in range(40, 404, 14):
        for row in soup.find_all('tbody'):
            for srow in row.find_all('tr'):
                for name in srow.find_all('td', attrs={'class':'data-col1'}):
                    names.append(name.text)
                for price in srow.find_all('td', attrs={'class':'data-col2'}):
                    prices.append(price.text)
                for time in srow.find_all('td', attrs={'class':'data-col3'}):
                    marketTimes.append(time.text)
                for change in srow.find_all('td', attrs={'class':'data-col4'}):
                    changes.append(change.text)
                for percentChange in srow.find_all('td', attrs={'class':'data-col5'}):
                    percentChanges.append(percentChange.text)
                for volume in srow.find_all('td', attrs={'class':'data-col6'}):
                    totalVolumes.append(volume.text)
                for openInterest in srow.find_all('td', attrs={'class':'data-col7'}):
                    openInterests.append(openInterest.text)

    df = pd.DataFrame({"Names": names, "Prices": prices, "Change": changes, "% Change": percentChanges, "Market Time": marketTimes,'Volume': openInterests ,"Avg Volume(3M)": totalVolumes}).drop_duplicates().set_index('Names')
    df = df[df['Prices'] !='-']
    df['% Change'] = pd.to_numeric(df['% Change'].str.strip('%'))
    df = df.sort_values(by='% Change', ascending=False).style.format({'% Change': "{:.2f}%"}).background_gradient(cmap='RdYlGn', subset=['% Change'])
    return df


##FIXED INCOME - YIELD CURVES


def yield_curve(country='United States'):    
    df = investpy.bonds.get_bonds_overview(country=country)
    df.set_index('name', inplace=True)
    if country=='United States':
        df.index = df.index.str.strip('U.S.')
    elif country =='United Kingdom':
        df.index = df.index.str.strip('U.K.')
    else:
        df.index = df.index.str.strip(country)
    return df['last']
    
def show_yc():
    us = yield_curve('United States')
    uk = yield_curve('United Kingdom')
    china = yield_curve('China')
    aus = yield_curve('Australia')
    germany = yield_curve('Germany')
    japan = yield_curve('Japan')
    can = yield_curve('Canada')
    ind = yield_curve('India')
    
    fig = make_subplots(
        rows=4, cols=2,
        subplot_titles=("United States", "United Kingdom", "China", "Australia", "Germany", "Japan", "Canada", "India"))

    fig.add_trace(go.Scatter(x=us.index, y=us, mode='lines+markers', name='US', line_shape='spline'),
                  row=1, col=1)

    fig.add_trace(go.Scatter(x=uk.index, y=uk, mode='lines+markers', name='UK', line_shape='spline'),
                  row=1, col=2)

    fig.add_trace(go.Scatter(x=china.index, y=china, mode='lines+markers', name='China', line_shape='spline'),
                  row=2, col=1)

    fig.add_trace(go.Scatter(x=aus.index, y=aus, mode='lines+markers', name='Australia', line_shape='spline'),
                  row=2, col=2)

    fig.add_trace(go.Scatter(x=germany.index, y=germany, mode='lines+markers', name='Germany', line_shape='spline'),
                  row=3, col=1)

    fig.add_trace(go.Scatter(x=japan.index, y=japan, mode='lines+markers', name='Japan', line_shape='spline'),
                  row=3, col=2)

    fig.add_trace(go.Scatter(x=can.index, y=can, mode='lines+markers', name='Canada', line_shape='spline'),
                  row=4, col=1)

    fig.add_trace(go.Scatter(x=ind.index, y=ind, mode='lines+markers', name='India', line_shape='spline'),
                  row=4, col=2)

    fig.update_layout(height=1600, width=1100,
                      title_text="Global Sovereign Yield Curves")
    fig.update_yaxes(title_text="Yield (%)", showgrid=True, zeroline=True, zerolinecolor='red', tickformat = '.3f')
    fig.update_xaxes(title_text="Maturity (Yrs)")
    fig.update_layout(font=dict(family="Segoe UI, monospace", size=13, color="#7f7f7f")
                  ,plot_bgcolor = 'White', hovermode='x')
    fig.update_traces(hovertemplate='Maturity: %{x} <br>Yield: %{y:.3f}%')
    fig.update_yaxes(automargin=True)

    return fig


def global_yields(countries=['U.S.', 'Germany', 'U.K.', 'Canada', 'Australia', 'Japan', 'India']):
    """
    
    """
    tdy = str(date.today().day)+'/'+str(date.today().month)+'/'+str(date.today().year)
    oneyr = str(date.today().day)+'/'+str(date.today().month)+'/'+str(date.today().year-1)
    
    tens = pd.DataFrame(index=pd.bdate_range(start=oneyr, end=date.today()))
    tens.index.name='Date'

    fives = pd.DataFrame(index=pd.bdate_range(start=oneyr, end=date.today()))
    fives.index.name='Date'

    twos = pd.DataFrame(index=pd.bdate_range(start=oneyr, end=date.today()))
    twos.index.name='Date'

    def ytm(country, maturity):
        df = pd.DataFrame(investpy.get_bond_historical_data(bond= str(country)+' '+str(maturity), from_date=oneyr, to_date=tdy)['Close'])
        df.columns = [str(country)]
        df.index = pd.to_datetime(df.index)
        return pd.DataFrame(df)

    cntry = countries
    
    for i in range(len(cntry)):
                tens = tens.merge(ytm(cntry[i], '10Y'), on='Date')
    for i in range(len(cntry)):
                fives = fives.merge(ytm(cntry[i], '5Y'), on='Date')
    for i in range(len(cntry)):
                twos = twos.merge(ytm(cntry[i], '2Y'), on='Date')
      
    ytd = date.today() - offsets.YearBegin()
    #10 Year
    teny = pd.DataFrame(data= (tens.iloc[-1,:], tens.diff(1).iloc[-1,:]*100, tens.diff(1).iloc[-5,:]*100, (tens.iloc[-1,:] - tens[ytd:].iloc[0,:])*100, (tens.iloc[-1,:]-tens.iloc[0,:])*100))
    teny = teny.T
    cols = [('10Y', 'Yield'),('10Y', '1 Day'), ('10Y', '1 Week'), ('10Y', 'YTD'), ('10Y', '1 Year')]
    teny.columns = pd.MultiIndex.from_tuples(cols)
    teny.index.name='Countries'
    
    #5 Year
    fivey = pd.DataFrame(data= (fives.iloc[-1,:], fives.diff(1).iloc[-1,:]*100, fives.diff(1).iloc[-6,:]*100,(fives.iloc[-1,:] - fives[ytd:].iloc[0,:])*100, (fives.iloc[-1,:]-fives.iloc[0,:])*100))
    fivey = fivey.T
    cols = [('5Y', 'Yield'),('5Y', '1 Day'), ('5Y', '1 Week'), ('5Y', 'YTD'), ('5Y', '1 Year')]
    fivey.columns = pd.MultiIndex.from_tuples(cols)
    fivey.index.name='Countries'
    
    #2 Year
    twoy = pd.DataFrame(data= (twos.iloc[-1,:], twos.diff(1).iloc[-1,:]*100, twos.diff(1).iloc[-6,:]*100, (twos.iloc[-1,:] - twos[ytd:].iloc[0,:])*100, (twos.iloc[-1,:]-twos.iloc[0,:])*100))
    twoy = twoy.T
    cols = [('2Y', 'Yield'),('2Y', '1 Day'), ('2Y', '1 Week'), ('2Y', 'YTD'), ('2Y', '1 Year')]
    twoy.columns = pd.MultiIndex.from_tuples(cols)
    twoy.index.name='Countries'
    
    yields = twoy.merge(fivey, on='Countries').merge(teny, on='Countries')
    
    data = yields.style.format('{0:,.3f}%', subset=[('2Y', 'Yield'), ('5Y', 'Yield'), ('10Y', 'Yield')])\
            .background_gradient(cmap='RdYlGn', subset=list(yields.columns.drop(('2Y', 'Yield')).drop(('5Y', 'Yield')).drop(('10Y', 'Yield')))).set_precision(2)
    return data
