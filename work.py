import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import edhec_risk_kit as erk
import edhec_risk_kit_206 as erk1
from datetime import date, timedelta
import matplotlib.pyplot as plt
from matplotlib import colors
import openpyxl
from nsepython import *



def index_cons_2005(securities, weights, freq='Quarterly'):
    """
    Securities = Dataframe of 2 Securities Time Series Data (Price Index)
    Weights =  List including 2 weights (float or int), adding up to 1
    Frequency = 'Quarterly' or 'Monthly'
    """
    if freq=='Quarterly':
        month1 = pd.Series(securities.index.quarter)
        month2 = pd.Series(securities.index.quarter).shift(-1)
        mask = (month1 != month2)
        bmk2q = securities[mask.values]
    elif freq =='Monthly':
        month1 = pd.Series(securities.index.month)
        month2 = pd.Series(securities.index.month).shift(-1)
        mask = (month1 != month2)
        bmk2q = securities[mask.values]

    bmk2q = bmk2q*0
    bmk2q = bmk2q + weights

    bmk2q.columns = bmk2q.columns + 'W' 
    bmk2f = securities.join(bmk2q, on='Date')
    bmk2f.iloc[0,2:] = weights
    bmk2f['Index'] = bmk2f.iloc[:,0].copy()*0
    bmk2f['Index'][0] = 10000

    bmk2f['Sec-1'] = bmk2f.iloc[:,0].copy()*0
    bmk2f['Sec-2'] = bmk2f.iloc[:,0].copy()*0

    for i in range(len(bmk2f)-1):
        if bmk2f.iloc[:,2][i] > 0:
            bmk2f.iloc[:,5][i] = (bmk2f.iloc[:,4][i] * bmk2f.iloc[:,2][i])/bmk2f.iloc[:,0][i]
            bmk2f.iloc[:,6][i] = (bmk2f.iloc[:,4][i] * bmk2f.iloc[:,3][i])/bmk2f.iloc[:,1][i]

        else:
            bmk2f.iloc[:,5][i] = bmk2f.iloc[:,5][i-1]
            bmk2f.iloc[:,6][i] = bmk2f.iloc[:,6][i-1]

        bmk2f.iloc[:,4][i+1] = (bmk2f.iloc[:,5][i]*bmk2f.iloc[:,0][i+1]) + (bmk2f.iloc[:,6][i]*bmk2f.iloc[:,1][i+1])

    return pd.DataFrame(bmk2f[['Index']])




def index_cons(securities, weights, freq='Quarterly'):
    """
    Securities = Dataframe of 3 Securities Time Series Data (Price Index)
    Weights =  List including 3 weights (float or int), adding up to 1
    Frequency = 'Quarterly' or 'Monthly'
    """
    if freq=='Quarterly':
        month1 = pd.Series(securities.index.quarter)
        month2 = pd.Series(securities.index.quarter).shift(-1)
        mask = (month1 != month2)
        bmk2q = securities[mask.values]
    elif freq =='Monthly':
        month1 = pd.Series(securities.index.month)
        month2 = pd.Series(securities.index.month).shift(-1)
        mask = (month1 != month2)
        bmk2q = securities[mask.values]

    bmk2q = bmk2q*0
    bmk2q = bmk2q + weights

    bmk2q.columns = bmk2q.columns + 'W' 
    bmk2f = securities.join(bmk2q, on='Date')
    bmk2f.iloc[0,3:] = weights
    bmk2f['Index'] = bmk2f.iloc[:,0].copy()*0
    bmk2f['Index'][0] = 10000

    bmk2f['Sec-1'] = bmk2f.iloc[:,0].copy()*0
    bmk2f['Sec-2'] = bmk2f.iloc[:,0].copy()*0
    bmk2f['Sec-3'] = bmk2f.iloc[:,0].copy()*0

    for i in range(len(bmk2f)-1):
        if bmk2f.iloc[:,3][i] > 0:
            bmk2f.iloc[:,7][i] = (bmk2f.iloc[:,6][i] * bmk2f.iloc[:,3][i])/bmk2f.iloc[:,0][i]
            bmk2f.iloc[:,8][i] = (bmk2f.iloc[:,6][i] * bmk2f.iloc[:,4][i])/bmk2f.iloc[:,1][i]
            bmk2f.iloc[:,9][i] = (bmk2f.iloc[:,6][i] * bmk2f.iloc[:,5][i])/bmk2f.iloc[:,2][i] 
        else:
            bmk2f.iloc[:,7][i] = bmk2f.iloc[:,7][i-1]
            bmk2f.iloc[:,8][i] = bmk2f.iloc[:,8][i-1]
            bmk2f.iloc[:,9][i] = bmk2f.iloc[:,9][i-1]

        bmk2f.iloc[:,6][i+1] = (bmk2f.iloc[:,7][i]*bmk2f.iloc[:,0][i+1]) + (bmk2f.iloc[:,8][i]*bmk2f.iloc[:,1][i+1]) + (bmk2f.iloc[:,9][i]*bmk2f.iloc[:,2][i+1])

    return pd.DataFrame(bmk2f[['Index']])



def drawdowns(data):
    """
    Max Drawdown in the current calendar year
    """
    data = data.ffill()
    return_series = pd.DataFrame(data.pct_change().dropna()[str(date.today().year):])
    wealth_index = 1000*(1+return_series).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index - previous_peaks)/previous_peaks
    return drawdowns.min(axis=0)


def get_nse_index(symbol, series_type, start_date, end_date):
    """
    """
    if series_type=="Price":
        index = index_history(symbol, start_date, end_date)
        index = index[['HistoricalDate','CLOSE']].set_index('HistoricalDate')
    elif series_type=="TRI":
        index = index_total_returns(symbol, start_date, end_date)
        index = index[['Date','TotalReturnsIndex']].set_index('Date')
        
    index.columns = [symbol+" "+series_type]
    index.index.name = 'Date'
    index.index = pd.to_datetime(index.index, format="%d %b %Y")
    index[index.columns[0]] = pd.to_numeric(index[index.columns[0]])
    return index.sort_values(by='Date')


def rebase_timeframe(securities, freq):
    """
    freq can be either "Monthly", "Quarterly" or "Annual"
    """
    if freq=='Annual':
        period1 = pd.Series(securities.index.year)
        period2 = pd.Series(securities.index.year).shift(-1)
        mask = (period1 != period2)
        updated_freq = securities[mask.values]
    if freq=='Quarterly':
        period1 = pd.Series(securities.index.quarter)
        period2 = pd.Series(securities.index.quarter).shift(-1)
        mask = (period1 != period2)
        updated_freq = securities[mask.values]
    if freq=='Monthly':
        period1 = pd.Series(securities.index.month)
        period2 = pd.Series(securities.index.month).shift(-1)
        mask = (period1 != period2)
        updated_freq = securities[mask.values]

    return updated_freq
