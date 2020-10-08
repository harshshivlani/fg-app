import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import yfinance as yf
import yahooquery
import investpy
import streamlit as st
import streamlit.components.v1 as components
import warnings
import work
import etf as etf
from datetime import date
import datetime
from yahooquery import Ticker
warnings.filterwarnings("ignore")

import plotly.express as px
import plotly.graph_objects as go
import plotly
from plotly.subplots import make_subplots

import ipywidgets as widgets
from ipywidgets import interact, interact_manual
from IPython.core.display import display, HTML

st.write("""
# Cross Asset Market Analytics
""")
components.iframe("https://harshshivlani.github.io/x-asset/liveticker")

st.sidebar.header('Cross Asset Monitor: Contents')
side_options = st.sidebar.radio('Please Select One:', ('Cross Asset Summary', 'Equities', 'Fixed Income', 'Global Sovereign Yields', 'REITs', 'Commodities', 'FX', 'Macroeconomic Data', 'Country Macroeconomic Profile','Economic Calendar', 'ETF Details'))


#Import Master Data
data = pd.read_excel('GSTOCKS.xlsx')
data['Country'].replace("United Arab Emirates", "UAE", inplace=True)
data['Country'].replace("Trindad & Tobago", "Trindad", inplace=True)
all_countries = ["All"] + list(data['Country'].unique())
data.iloc[:,5:] = data.iloc[:,5:].replace('--', np.nan).astype(float)


@st.cache(allow_output_mutation=True)
def indices_func():
	return work.updated_world_indices('All', 'Daily')

@st.cache()
def world_map(timeperiod):
    """
    """
    iso = pd.read_excel('World_Indices_List.xlsx', sheet_name='iso')
    iso.set_index('Country', inplace=True)
    rawdata = indices_func()[1]
    rawdata['Index'] = rawdata.index
    rawdata = rawdata.drop(['China A50', 'SmallCap 2000', 'BSE Sensex', 'Euro Stoxx 50', 'Nasdaq', 'KOSDAQ', 'RTSI', 'DJ Shanghai', 'SZSE Component'], axis=0)
    data2 = rawdata.merge(iso['iso_alpha'], on='Country')

    data2[['Chg (%)', 'Chg YTD (%)', '$ Chg (%)','$ Chg YTD (%)']] = data2[['Chg (%)', 'Chg YTD (%)', '$ Chg (%)','$ Chg YTD (%)']].round(4)*100

    df = data2
    for col in df.columns:
        df[col] = df[col].astype(str)

    df['text'] = 'Return: '+df[timeperiod]+'%' + '<br>' \
                  'Country: '+ df['Country'] + '<br>' \
                  'Index: '+ df['Index'] + '<br>' \

    fig1 = go.Figure(data=go.Choropleth(locations=df['iso_alpha'], z=df[timeperiod].astype(float).round(2), colorscale='RdYlGn', autocolorscale=False,
        text=df['text'], colorbar_ticksuffix = '%', colorbar_title = "Return"))

    return fig1.update_layout(width=950, height=500, margin=dict(l=0,r=0,b=0,t=0,pad=1),
                        xaxis=dict(scaleanchor='x', constrain='domain'), coloraxis_colorbar_x=1)





#EQUITIES - SCREENER
def filter_table(country, ind, subind, mcap):
    df = data.copy()
    if country != "All":
        df = df[df['Country']==country]

    if ind != "All":
        df = df[df["Industry"]==ind]

    if subind != "All":
        df = df[df["Sub-Industry"]==subind]
    

    if mcap == 'Mega Cap':
        df = df[df["Market Cap"]>200]
    elif mcap =='Large Cap':
        df = df[(df["Market Cap"]<=200) & (df["Market Cap"]>10)]
    elif mcap == 'Mid Cap':
        df = df[(df["Market Cap"]<=10) & (df["Market Cap"]>2)]
    elif mcap == 'Small Cap':
        df = df[(df["Market Cap"]<=2) & (df["Market Cap"]>0.3)]
    elif mcap == 'Micro Cap':
        df = df[(df["Market Cap"]<=0.3)]
    elif mcap == "All":
        df = df[:]

    df_style = df.set_index('Ticker').sort_values(by='Market Cap', ascending=False).round(2).style.format('{0:,.2f}%', subset=data.columns[6:-1])\
                                    .format('{0:,.2f}', subset=data.columns[5])\
                                    .format('{0:,.2f}', subset=data.columns[-1])\
                                    .background_gradient(cmap='RdYlGn', subset=data.columns[6:12])   
    return df_style

#EQUITIES - PIVOT TABLE



def pivot_table(country, ind, mcap):
	df = data.copy()

	if mcap == 'Mega Cap':
		df = df[df["Market Cap"]>200]
	elif mcap =='Large Cap':
		df = df[(df["Market Cap"]<=200) & (df["Market Cap"]>10)]
	elif mcap == 'Mid Cap':
		df = df[(df["Market Cap"]<=10) & (df["Market Cap"]>2)]
	elif mcap == 'Small Cap':
		df = df[(df["Market Cap"]<=2) & (df["Market Cap"]>0.3)]
	elif mcap == 'Micro Cap':
		df = df[(df["Market Cap"]<=0.3)]
	elif mcap == "All":
		df = df[:]

	if country != "All" and ind=="All":
		df = df[df['Country']==country].groupby(by="Industry").median()
	elif country == "All" and ind=="All":
		df = df.groupby(by="Industry").median()
	elif country == "All" and ind!="All":
		df = df[df['Industry']==ind].groupby(by='Country').median()
	else:
		df = df[(df['Country']==country) & (df['Industry']==ind)].set_index('Ticker')

	df = df.style.format('{0:,.2f}%', subset=data.columns[6:-1])\
					.format('{0:,.2f}B', subset=data.columns[5])\
					.format('{0:,.2f}', subset=data.columns[-1])\
					.background_gradient(cmap='RdYlGn', subset=data.columns[6:12])
	return df




# ----------------------------- EQUITIES SIDEPANEL ---------------------------------------------------------------------
if side_options == 'Equities':
	st.title('Global Equities - Pivot Table')
	st.subheader('Compare Median Return across Countries & GICS Industries')
	if st.checkbox('Show Global Equities Pivot Table', value=True):
		country = st.selectbox('Country: ', all_countries, key='pivot')
		if country == "All":
			all_ind = ["All"] + list(data['Industry'].unique())
			all_subind = ["All"] + list(data['Sub-Industry'].unique())
		else:
			all_ind = ["All"] + list(data[data['Country']==country]['Industry'].unique())
			all_subind = ["All"] + list(data[data['Country']==country]['Sub-Industry'].unique())

		ind = st.selectbox('GICS Industry Name: ', all_ind, key='pivot')
		mcap = st.selectbox('Market Cap: ', ['All', 'Mega Cap', 'Large Cap', 'Mid Cap', 'Small Cap', 'Micro Cap'], index=0, key='pivot')
		print(st.dataframe(pivot_table(country=country, ind=ind, mcap=mcap), height=700))


	st.title('Global Equities Screener')
	st.subheader('Filter Global Stocks by Countries, GICS Industries, GICS Sub-Industries & Market Capitalization (USD)')
	if st.checkbox('Show Global Equities Filter'):
		country = st.selectbox('Country: ', all_countries, index=2)
		if country == "All":
			all_ind = ["All"] + list(data['Industry'].unique())
			all_subind = ["All"] + list(data['Sub-Industry'].unique())
		else:
			all_ind = ["All"] + list(data[data['Country']==country]['Industry'].unique())
			all_subind = ["All"] + list(data[data['Country']==country]['Sub-Industry'].unique())

		ind = st.selectbox('GICS Industry Name: ', all_ind)
		if ind!="All" and country!="All":
			subind = st.selectbox('GICS Sub Industry Name: ', ["All"] + list(data[data['Country']==country][data['Industry']==ind]['Sub-Industry'].unique()))
		elif ind!="All" and country=="All":
			subind = st.selectbox('GICS Sub Industry Name: ', ["All"] + list(data[data['Industry']==ind]['Sub-Industry'].unique()))
		else:
			subind = st.selectbox('GICS Sub Industry Name: ', all_subind)

		mcap = st.selectbox('Market Cap: ', ['All', 'Mega Cap', 'Large Cap', 'Mid Cap', 'Small Cap', 'Micro Cap'], index=0, key='table')
		print(st.dataframe(filter_table(country=country, ind=ind, subind=subind, mcap=mcap), height=600))

# ----------------------------- REITS SIDEPANEL ---------------------------------------------------------------------
reits = pd.read_excel('GREITS.xlsx')
reits['Country'].replace('Virgin Islands (United States)','Virgin Islands (US)', inplace=True)
reit_countries = ['All'] + list(reits['Country'].unique())

def filter_reit(country, subind, mcap):
    df = reits.copy()
    df.replace('--', np.nan, inplace=True)
    df.iloc[:, 4:13] = df.iloc[:, 4:13].astype(float)
    if country != "All":
        df = df[df['Country']==country]

    if subind != "All":
        df = df[df["Sub-Industry"]==subind]
    

    if mcap == 'Mega Cap':
        df = df[df["Market Cap"]>200]
    elif mcap =='Large Cap':
        df = df[(df["Market Cap"]<=200) & (df["Market Cap"]>10)]
    elif mcap == 'Mid Cap':
        df = df[(df["Market Cap"]<=10) & (df["Market Cap"]>2)]
    elif mcap == 'Small Cap':
        df = df[(df["Market Cap"]<=2) & (df["Market Cap"]>0.3)]
    elif mcap == 'Micro Cap':
        df = df[(df["Market Cap"]<=0.3)]
    elif mcap == "All":
        df = df[:]

    df_style = df.set_index('Ticker').sort_values(by='Market Cap', ascending=False).round(2).style.format('{0:,.2f}%', subset=df.columns[5:13])\
                   .format('{0:,.2f}B', subset=df.columns[4])\
                   .background_gradient(cmap='RdYlGn', subset=df.columns[6:12])
    return df_style


def reit_pivot_table(country, ind, mcap):
    df = reits.copy()
    df.replace('--', np.nan, inplace=True)
    df.iloc[:, 4:13] = df.iloc[:, 4:13].astype(float)

    if mcap == 'Mega Cap':
    	df = df[df["Market Cap"]>200]
    elif mcap =='Large Cap':
    	df = df[(df["Market Cap"]<=200) & (df["Market Cap"]>10)]
    elif mcap == 'Mid Cap':
    	df = df[(df["Market Cap"]<=10) & (df["Market Cap"]>2)]
    elif mcap == 'Small Cap':
    	df = df[(df["Market Cap"]<=2) & (df["Market Cap"]>0.3)]
    elif mcap == 'Micro Cap':
    	df = df[(df["Market Cap"]<=0.3)]
    elif mcap == "All":
    	df = df[:]

    if country != "All" and ind=="All":
    	df = df[df['Country']==country].groupby(by="Sub-Industry").median()
    	df = df.sort_values(by='Market Cap', ascending=False).round(2).style.format('{0:,.2f}%', subset=df.columns[1:9])\
                   .format('{0:,.2f}B', subset=df.columns[0])\
                   .background_gradient(cmap='RdYlGn', subset=df.columns[2:9])
    elif country == "All" and ind=="All":
    	df = df.groupby(by="Sub-Industry").median()
    	df = df.sort_values(by='Market Cap', ascending=False).round(2).style.format('{0:,.2f}%', subset=df.columns[1:9])\
                   .format('{0:,.2f}B', subset=df.columns[0])\
                   .background_gradient(cmap='RdYlGn', subset=df.columns[2:9])
    elif country == "All" and ind!="All":
    	df = df[df['Sub-Industry']==ind].groupby(by='Country').median()
    	df = df.sort_values(by='Market Cap', ascending=False).round(2).style.format('{0:,.2f}%', subset=df.columns[1:9])\
                   .format('{0:,.2f}B', subset=df.columns[0])\
                   .background_gradient(cmap='RdYlGn', subset=df.columns[2:9])
    else:
    	df = df[(df['Country']==country) & (df['Sub-Industry']==ind)].set_index('Ticker')
    	df = df.sort_values(by='Market Cap', ascending=False).style.format('{0:,.2f}%', subset=df.columns[4:12])\
                   .format('{0:,.2f}B', subset=df.columns[3])\
                   .background_gradient(cmap='RdYlGn', subset=df.columns[4:12])
    return df

if side_options == 'REITs':
	st.title('Global REITs Pivot Table')
	st.subheader('Compare Median Return across Countries & GICS Sub-Industries')
	if st.checkbox('Show Global REITs Pivot Table', value=True):
		data = reits.copy()
		country = st.selectbox('Country: ', reit_countries, key='pivot')
		if country == "All":
			all_subind = ["All"] + list(data['Sub-Industry'].unique())
		else:
			all_subind = ["All"] + list(data[data['Country']==country]['Sub-Industry'].unique())

		ind = st.selectbox('GICS Industry Name: ', all_subind, key='pivot')
		mcap = st.selectbox('Market Cap: ', ['All', 'Mega Cap', 'Large Cap', 'Mid Cap', 'Small Cap', 'Micro Cap'], index=0, key='reit_pivot')
		print(st.dataframe(reit_pivot_table(country=country, ind=ind, mcap=mcap), height=700))



	st.title('Global REITs Screener')
	st.subheader('Filter Global REITs by Countries, GICS Sub-Industries & Market Capitalization (USD)')
	if st.checkbox('Show Global REITs Filter'):
		data = reits.copy()
		country = st.selectbox('Country: ', reit_countries, index=1)
		if country == "All":
			all_subind = ["All"] + list(data['Sub-Industry'].unique())
		else:
			all_subind = ["All"] + list(data[data['Country']==country]['Sub-Industry'].unique())

		subind = st.selectbox('GICS Industry Name: ', all_subind)

		mcap = st.selectbox('Market Cap: ', ['All', 'Mega Cap', 'Large Cap', 'Mid Cap', 'Small Cap', 'Micro Cap'], index=0, key='table')
		print(st.dataframe(filter_reit(country=country, subind=subind, mcap=mcap), height=600))




























# ----------------------------- COMMODITIES SIDEPANEL ---------------------------------------------------------------------
@st.cache
def import_data(asset_class):
    """
    Imports Historical Data for Mentioned ETFs
    asset_class = mention the asset class for ETFs data import (str)
    options available = 'Fixed Income', 'REIT', 'Currencies', 'Commodities', 'World Equities', 'Sector Equities'
    """
    #Import list of ETFs and Ticker Names
    etf_list = pd.read_excel('etf_names.xlsx', header=0, sheet_name='Commodities')

    #Build an empty df to store historical 1 year data
    df = pd.DataFrame(index=pd.bdate_range(start=oneyr, end=date.today()))
    df.index.name='Date'


    def hist_data_comd(name):
    	oneyr = str(date.today().day)+'/'+str(date.today().month)+'/'+str(date.today().year-1)
    	tdy = str(date.today().day)+'/'+str(date.today().month)+'/'+str(date.today().year)
    	df = investpy.get_commodity_historical_data(commodity=name, from_date=oneyr, to_date=tdy)['Close']
    	df = pd.DataFrame(df)
    	df.columns = [name]
    	return df

    #download and merge all data
    for i in range(len(etf_list)):
    	df = df.join(hist_data_comd(etf_list['Commodities'][i]), on='Date')
    df = df[:yest].ffill().dropna()
    df.index.name = 'Date'
    return df

from pandas.tseries import offsets
one_m = date.today() - datetime.timedelta(30)
three_m = date.today() - datetime.timedelta(90)
six_m = date.today() - datetime.timedelta(120)
one_yr = date.today() - datetime.timedelta(370)
ytd = date.today() - offsets.YearBegin()
year = date.today().year
yest = date.today() - datetime.timedelta(1)
now = datetime.datetime.now()
now = now.strftime("%b %d, %Y %H:%M")

tdy = str(date.today().day)+'/'+str(date.today().month)+'/'+str(date.today().year)
oneyr = str(date.today().day)+'/'+str(date.today().month)+'/'+str(date.today().year-1)

def drawdowns(data):
    """
    Max Drawdown in the current calendar year
    """
    return_series = pd.DataFrame(data.pct_change().dropna()[str(date.today().year):])
    wealth_index = 1000*(1+return_series).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index - previous_peaks)/previous_peaks
    return drawdowns.min(axis=0)

#SORTED AND CONDITIONALLY FORMATTED RETURNS DATAFRAME
def returns_hmap(data, cat, asset_class, start=date(2020,3,23), end=date.today(), sortby='1-Day'):
    """
    data = Price Data for the ETFs (dataframe)
    asset_class = asset class of the ETF (str)
    cat = subset or category (list), default is all ETFs mentioned
    """
    st.subheader("Multi Timeframe Returns of " + str(asset_class))
    st.markdown("Data as of :  " + str(data.index[-1].strftime("%b %d, %Y")))
    df = pd.DataFrame(data = (data.iloc[-1,:], data.pct_change(1).iloc[-1,:], data.pct_change(5).iloc[-1,:], data.pct_change(21).iloc[-1,:],
                              data.pct_change(63).iloc[-1,:], data[str(year):].iloc[-1,:]/data[str(year):].iloc[0,:]-1, data[start:end].iloc[-1,:]/data[start:end].iloc[0,:]-1,
                              data.pct_change(126).iloc[-1,:], data.pct_change(252).iloc[-1,:], drawdowns(data)))
    df.index = ['Price','1-Day', '1-Week', '1-Month', '3-Month', 'YTD', 'Custom', '6-Month', '1-Year', 'Max DD']
    df_perf = df.T
    df_perf.iloc[:,1:] = (df_perf.iloc[:,1:]*100)
    df_perf.index.name = asset_class

    #Add Ticker Names and sort the dataframe
    etf_list = pd.read_excel('etf_names.xlsx', header=0, sheet_name=asset_class)
    if asset_class=='Indian Equities':
        df2 = df_perf.copy()
        df2  = df2.sort_values(by=sortby, ascending=False)
        df2 = df2.round(2).style.format('{0:,.2f}%')\
                     .background_gradient(cmap='RdYlGn')\
                     .set_properties(**{'font-size': '10pt',})
    else:
        df2 = df_perf
        df2  = df2.sort_values(by=sortby, ascending=False)
        df2 = df2.round(2).style.format('{0:,.2f}%', subset=(df2.drop(['Price'], axis=1).columns))\
        		 .format('{0:,.2f}', subset='Price')\
                 .background_gradient(cmap='RdYlGn', subset=(df2.drop(['Price'], axis=1).columns))\
                 .set_properties(**{'font-size': '10pt',})
    
    return df2

#PLOT RETURN CHART BY PLOTLY
def plot_chart(data, cat, start_date=one_yr):
    """
    Returns a Plotly Interactive Chart for the given timeseries data (price)
    data = price data for the ETFs (dataframe)
    """
    df = ((((1+data[cat].dropna()[start_date:date.today()].pct_change().fillna(0.00))).cumprod()-1)).round(4)
    fig = px.line(df, x=df.index, y=df.columns)
    fig.update_layout(xaxis_title='Date',
                      yaxis_title='Return (%)', font=dict(family="Segoe UI, monospace", size=14, color="#7f7f7f"),
                      legend_title_text='Securities', plot_bgcolor = 'White', yaxis_tickformat = '%', width=950, height=600)
    fig.update_traces(hovertemplate='Date: %{x} <br>Return: %{y:.2%}')
    fig.update_yaxes(automargin=True)
    return fig

#TREND ANALYSIS
def trend_analysis(data, cat, start_date=one_yr, inv='B', ma=15):
    """
    data = price data (dataframe)
    inv = daily to be resampled to weekly ('W') or monthly ('BM') return data
    ma = rolling return lookback period, chose 1 for just daily/monthly etc based incase you resample via inv variable
    cat =  any subset of data (list - column names of data)
    """
    d = (data[cat].pct_change(ma).dropna()[start_date:date.today()].resample(inv).agg(lambda x: (x + 1).prod() - 1).round(4)*100)
    fig = go.Figure(data=go.Heatmap(
            z=((d - d.mean())/d.std()).round(2).T.values,
            x=((d - d.mean())/d.std()).index,
            y=list(data[cat].columns), zmax=3, zmin=-3,
            colorscale='rdylgn', hovertemplate='Date: %{x}<br>Security: %{y}<br>Return Z-Score: %{z}<extra></extra>', colorbar = dict(title='Return Z-Score')))

    fig.update_layout(xaxis_nticks=20, font=dict(family="Segoe UI, monospace", size=14, color="#7f7f7f"),width=950, height=600)
    return fig

# Additional Settings for Interactive Widget Buttons for Charts & Plots
#Select Time Frame Options
disp_opts = {one_m: '1 Month', three_m: '3 Months', six_m:'6 Months', ytd: 'Year-to-Date', one_yr:'1-Year'} #To show text in options but have numbers in the backgroud
def format_func(option):
    return disp_opts[option]

#Select Daily/Weekly/Monthly data for Trend Analysis
inv = {'B': 'Daily', 'W': 'Weekly', 'BM': 'Monthly'}
def format_inv(option):
    return inv[option]

def display_items(data, asset_class, cat):
	start= st.date_input("Custom Start Date: ", date(2020,3,23))
	end = st.date_input("Custom End Date: ", date.today())
	st.dataframe(returns_hmap(data=data[cat], asset_class=asset_class, cat=cat, start=start, end=end), height=1500)
	st.subheader("Price Return Performance")
	start_date = st.selectbox('Select Period', list(disp_opts.keys()), index=3, format_func = format_func, key='chart')
	print(st.plotly_chart(plot_chart(data=data[cat], start_date=start_date, cat=cat)))
	st.subheader("Rolling Return Trend Heatmap")
	start_date = st.selectbox('Select Period: ', list(disp_opts.keys()), index=3, format_func = format_func, key='trend')
	inv_opt = st.selectbox('Select Timescale: ', list(inv.keys()), index=0, format_func = format_inv)
	ma = st.number_input('Select Rolling Return Period: ', value=15, min_value=1)
	print(st.plotly_chart(trend_analysis(data=data[cat], cat=cat, start_date=start_date, inv=inv_opt, ma=ma)))

#@st.cache
def import_data_yahoo(asset_class):
    """
    Imports Historical Data for Mentioned ETFs
    asset_class = mention the asset class for ETFs data import (str)
    options available = 'Fixed Income', 'REIT', 'Currencies', 'Commodities', 'World Equities', 'Sectoral'
    """
    #Import list of ETFs and Ticker Names
    etf_list = pd.read_excel('etf_names.xlsx', header=0, sheet_name=asset_class)
    etf_list = etf_list.sort_values(by='Ticker')

    #Build an empty df to store historical 1 year data
    df = pd.DataFrame(index=pd.bdate_range(start=one_yr, end=date.today()))
    df.index.name='Date'

    #download and merge all data
    df1 = Ticker(list(etf_list['Ticker']), asynchronous=True).history(start=date(date.today().year -1 , date.today().month-1, date.today().day))['adjclose']
    df1 = pd.DataFrame(df1).unstack().T.reset_index(0).drop('level_0', axis=1)
    df1.index.name = 'Date'
    df1.index = pd.to_datetime(df1.index)
    df = df.merge(df1, on='Date')
    #Forward fill for any missing days i.e. holidays
    df = df.ffill().dropna()
    df.index.name = 'Date'
    df.columns = list(etf_list[asset_class])
    return df

comd = import_data('Commodities')
fx = import_data_yahoo('Currencies')

if side_options == 'Commodities':
	if st.checkbox('Show Live Data', value=True):
		components.iframe("https://harshshivlani.github.io/x-asset/comd", width=670, height=500)
	if st.checkbox('Show Live Chart'):
		components.iframe("https://harshshivlani.github.io/x-asset/equity-chart", width=1000, height=550)
	st.write('**Note:** All returns are in USD')
	print(display_items(comd, 'Commodities', cat=list(comd.columns)))

if side_options == 'FX':
	if st.checkbox('Show Live Data', value=True):
		components.iframe("https://harshshivlani.github.io/x-asset/cur", width=670, height=500)
	if st.checkbox('Show Live Chart'):
		components.iframe("https://harshshivlani.github.io/x-asset/equity-chart", height=500)
	print(display_items(fx, 'Currencies', cat=list(fx.columns)))


#################-------------------------Fixed Income-------------------------------#######################
fi_etfs = pd.read_excel('Fixed Income-Daily.xlsx')
fi_cats = ['None', 'All'] + list(fi_etfs['Category'].unique())


def fi_filter(category):
	df = fi_etfs.copy()
	df.replace('--', np.nan, inplace=True)
	df.replace(' ', np.nan, inplace=True)
	df.iloc[:, 4:15] = df.iloc[:, 4:15].astype(float)

	if category == 'All':
	    return df.set_index('Ticker').sort_values(by='1D', ascending=False).round(2).style.format('{0:,.2f}%', subset=df.columns[4:13])\
	                   .format('{0:,.2f}B', subset=df.columns[14])\
	                   .format('{0:,.2f}M', subset=df.columns[13])\
	                   .background_gradient(cmap='RdYlGn', subset=df.columns[5:13])

	elif category == 'None':
	    df = df.groupby(by="Category").median()
	    return df.sort_values(by='1D', ascending=False).round(2).style.format('{0:,.2f}%', subset=df.columns[:9])\
	                   .format('{0:,.2f}B', subset=df.columns[10])\
	                   .format('{0:,.2f}M', subset=df.columns[9])\
	                   .background_gradient(cmap='RdYlGn', subset=df.columns[1:9])
	else:
	    df = df[df['Category'] == category]
	    return df.set_index('Ticker').sort_values(by='1D', ascending=False).round(2).style.format('{0:,.2f}%', subset=df.columns[4:13])\
	                   .format('{0:,.2f}B', subset=df.columns[14])\
	                   .format('{0:,.2f}M', subset=df.columns[13])\
	                   .background_gradient(cmap='RdYlGn', subset=df.columns[5:13])

if side_options =='Fixed Income':
	fi_category = st.selectbox('Category: ', fi_cats, key='fi_pivot')
	print(st.dataframe(fi_filter(category=fi_category), height=700))

##################-------------------Global Yields----------------#############################

@st.cache(allow_output_mutation=True)
def global_yields(countries=['U.S.', 'Germany', 'U.K.', 'Italy', 'France', 'Canada', 'China', 'Australia', 'Japan', 'India', 'Russia', 'Brazil', 'Philippines', 'Thailand']):
    """
    
    """
    def ytm(country, maturity):
    	df = pd.DataFrame(investpy.get_bond_historical_data(bond= str(country)+' '+str(maturity), from_date=oneyr, to_date=tdy)['Close'])
    	df.columns = [str(country)]
    	df.index = pd.to_datetime(df.index)
    	return pd.DataFrame(df)
    
    tdy = str(date.today().day)+'/'+str(date.today().month)+'/'+str(date.today().year)
    oneyr = str(date.today().day)+'/'+str(date.today().month)+'/'+str(date.today().year-1)
    
    tens = pd.DataFrame(index=pd.bdate_range(start=oneyr, end=date.today()))
    tens.index.name='Date'

    fives = pd.DataFrame(index=pd.bdate_range(start=oneyr, end=date.today()))
    fives.index.name='Date'

    twos = pd.DataFrame(index=pd.bdate_range(start=oneyr, end=date.today()))
    twos.index.name='Date'

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
            .background_gradient(cmap='RdYlGn_r', subset=list(yields.columns.drop(('2Y', 'Yield')).drop(('5Y', 'Yield')).drop(('10Y', 'Yield')))).set_precision(2)
    return data

@st.cache
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

us = yield_curve('United States')
uk = yield_curve('United Kingdom')
china = yield_curve('China')
aus = yield_curve('Australia')
germany = yield_curve('Germany')
japan = yield_curve('Japan')
can = yield_curve('Canada')
ind = yield_curve('India')
italy = yield_curve('Italy')
france = yield_curve('France')
rus = yield_curve('Russia')
phil = yield_curve('Philippines')
thai = yield_curve('Thailand')
brazil = yield_curve('Brazil')

def show_yc():
    fig = make_subplots(
        rows=7, cols=2,
        subplot_titles=("United States", "United Kingdom", "China", "Australia", "Germany", "Japan", "Canada", "India", "Italy", "France", "Brazil", "Thailand", "Philippines", "Russia"))

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

    fig.add_trace(go.Scatter(x=italy.index, y=italy, mode='lines+markers', name='Italy', line_shape='spline'),
                  row=5, col=1)

    fig.add_trace(go.Scatter(x=france.index, y=france, mode='lines+markers', name='France', line_shape='spline'),
                  row=5, col=2)

    fig.add_trace(go.Scatter(x=brazil.index, y=brazil, mode='lines+markers', name='Brazil', line_shape='spline'),
                  row=6, col=1)

    fig.add_trace(go.Scatter(x=thai.index, y=thai, mode='lines+markers', name='Thailand', line_shape='spline'),
                  row=6, col=2)

    fig.add_trace(go.Scatter(x=phil.index, y=phil, mode='lines+markers', name='Philippines', line_shape='spline'),
                  row=7, col=1)

    fig.add_trace(go.Scatter(x=rus.index, y=rus, mode='lines+markers', name='Russia', line_shape='spline'),
                  row=7, col=2)

    fig.update_layout(height=3000, width=900, showlegend=False)
    fig.update_yaxes(title_text="Yield (%)", showgrid=True, zeroline=True, zerolinecolor='red', tickformat = '.3f')
    fig.update_xaxes(title_text="Maturity (Yrs)")
    fig.update_layout(font=dict(family="Segoe UI, monospace", size=13, color="#7f7f7f")
                  ,plot_bgcolor = 'White', hovermode='x')
    fig.update_traces(hovertemplate='Maturity: %{x} <br>Yield: %{y:.3f}%')
    fig.update_yaxes(automargin=True)

    return fig


if side_options == 'Global Sovereign Yields':
	st.write(global_yields())
	st.header('Global Sovereign Yield Curves')
	st.write(show_yc())
	st.markdown('Data Source: Investing.com')





###############------------------Extras---------------------------##############################
if side_options=='Macroeconomic Data':
     st.subheader('Macroeconomic Data')
     cat = st.selectbox('Select Data Category: ', ('World Manufacturing PMIs', 'GDP', 'Retail Sales', 'Inflation', 'Unemployment'))
     if cat == 'World Manufacturing PMIs':
         st.subheader('World Manufacturing PMIs')
         continent = st.selectbox('Select Continent', ('World', 'G20', 'America', 'Europe', 'Asia', 'Africa'))
         st.dataframe(etf.world_pmis(continent=continent), width=1000, height=1500)
     elif cat == 'GDP':
         st.subheader('World GDP Data')
         continent = st.selectbox('Select Continent', ('G20', 'World', 'America', 'Europe', 'Asia', 'Africa'))
         st.dataframe(etf.gdp(continent=continent), width=1200, height=2000)
     elif cat=='Retail Sales':
         st.subheader('Retail Sales')
         continent = st.selectbox('Select Continent', ('G20', 'World', 'America', 'Europe', 'Asia', 'Africa'))
         time = st.selectbox('Select Period: ', ('YoY', 'MoM'))
         st.dataframe(etf.retail(continent=continent, time=time), width=1200, height=2000)
     elif cat == 'Inflation':
         st.subheader('World Inflation Data')
         continent = st.selectbox('Select Continent', ('G20', 'World', 'America', 'Europe', 'Asia', 'Africa'))
         st.dataframe(etf.inflation(continent=continent), width=1200, height=2000)
     elif cat == 'Unemployment':
         st.subheader('World Unemployment Data')
         continent = st.selectbox('Select Continent', ('G20', 'World', 'America', 'Europe', 'Asia', 'Africa'))
         st.dataframe(etf.unemp(continent=continent), width=1200, height=2000)

if side_options == 'Economic Calendar':
     st.subheader('Economic Calendar')
     components.iframe("https://harshshivlani.github.io/x-asset/ecocalendar", height=800)
     #importances = st.multiselect('Importance: ', ['Low', 'Medium', 'High'], ['Medium', 'High'])
     #st.dataframe(etf.eco_calendar(importances=importances), width=2000, height=1200)

if side_options == 'Country Macroeconomic Profile':
     st.subheader('Country Macroeconomic Profile')
     countries_list = st.selectbox('Select Country: ', ["United-States", "Afghanistan","Albania","Algeria","Andorra","Angola","Antigua-and-Barbuda","Argentina","Armenia","Aruba","Australia","Austria","Azerbaijan","Bahamas","Bahrain","Bangladesh","Barbados","Belarus","Belgium","Belize","Benin","Bermuda","Bhutan","Bolivia","Bosnia-and-Herzegovina","Botswana","Brazil","Brunei","Bulgaria","Burkina-Faso","Burundi","Cambodia","Cameroon","Canada","Cape-Verde","Cayman-Islands","Central-African-Republic","Chad","Chile","China","Colombia","Comoros","Congo","Costa-Rica","Croatia","Cuba","Cyprus","Czech-Republic","Denmark","Djibouti","Dominica","Dominican-Republic","East-Timor","Ecuador","Egypt","El-Salvador","Equatorial-Guinea","Eritrea","Estonia","Ethiopia","Euro-Area","Faroe-Islands","Finland","France","Gabon","Gambia","Georgia","Germany","Ghana","Greece","Grenada","Guatemala","Guinea","Guinea-Bissau","Guyana","Haiti","Honduras","Hong-Kong","Hungary","Iceland","India","Indonesia","Iran","Iraq","Ireland","Isle-of-Man","Israel","Italy","Ivory-Coast","Jamaica","Japan","Jordan","Kazakhstan","Kenya","Kosovo","Kuwait","Kyrgyzstan","Laos","Latvia","Lebanon","Lesotho","Liberia","Libya","Liechtenstein","Lithuania","Luxembourg","Macao","Macedonia","Madagascar","Malawi","Malaysia","Maldives","Mali","Malta","Mauritania","Mauritius","Mexico","Moldova","Monaco","Mongolia","Montenegro","Morocco","Mozambique","Myanmar","Namibia","Nepal","Netherlands","New-Zealand","Nicaragua","Niger","Nigeria","North-Korea","Norway","Oman","Pakistan","Palestine","Panama","Paraguay","Peru","Philippines","Poland","Portugal","Puerto-Rico","Qatar","Republic-of-the-Congo","Romania","Russia","Rwanda","Sao-Tome-and-Principe","Saudi-Arabia","Senegal","Serbia","Seychelles","Sierra-Leone","Singapore","Slovakia","Slovenia","Somalia","South-Africa","South-Korea","South-Sudan","Spain","Sri-Lanka","Sudan","Suriname","Swaziland","Sweden","Switzerland","Syria","Taiwan","Tajikistan","Tanzania","Thailand","Togo","Trinidad-and-Tobago","Tunisia","Turkey","Turkmenistan","Uganda","Ukraine","United-Arab-Emirates","United-Kingdom","United-States","Uruguay","Uzbekistan","Venezuela","Vietnam","Yemen","Zambia","Zimbabwe"])
     data_type = st.selectbox('Data Category: ', ['Overview', 'GDP', 'Labour', 'Inflation', 'Money', 'Trade', 'Government', 'Taxes', 'Business', 'Consumer'])
     st.dataframe(etf.country_macros(country=countries_list, data_type=data_type), height=1200)


if side_options == 'ETF Details':
    def etf_details():
        ticker_name = st.text_input('Enter Ticker Name', value='URTH')
        asset =  st.selectbox('ETF Asset Class:', ('Equity/REIT ETF', 'Fixed Income ETF'))
        if asset=='Equity/REIT ETF':
            details = st.selectbox('Select Data Type:', ('General Overview', 'Top 15 Holdings', 'Sector Exposure',
                                    'Market Cap Exposure', 'Country Exposure', 'Asset Allocation'))
        elif asset=='Fixed Income ETF':
            details = st.selectbox('Select Data Type:', ('General Overview', 'Top 15 Holdings', 'Bond Sector Exposure',
                                    'Coupon Breakdown', 'Credit Quality Exposure', 'Maturity Profile'))
        return [ticker_name, details, asset]

    etf_details = etf_details()
    st.write(etf.etf_details(etf_details[0].upper(), etf_details[1], etf_details[2]))
    st.write('Data Source: ETFdb.com')




#################-------------------------Summary-------------------------------#######################
if side_options == 'Cross Asset Summary':
	st.header('Global Equities')
	#components.iframe("http://www.teletrader.com/yadea/stocks/details/tts-161915467", height=500)
	st.subheader('World Equity Heatmap')
	timeframe = st.selectbox('Timeframe: ', ['Chg (%)', 'Chg YTD (%)', '$ Chg (%)','$ Chg YTD (%)'], index=2)
	print(st.plotly_chart(world_map(timeperiod=timeframe)))

	st.subheader('Global Equity Indices')
	indices = indices_func()
	st.dataframe(indices[0])

	st.subheader('Industry Summary: ')
	if st.checkbox('Show Global Equities Industry Return Summary', value=True):
		st.write('*Median industry wise USD returns across the World')
		ret_inds_summ = st.selectbox('Median Return TimeFrame: ', ['1D', '1W', '1M', '3M', '6M', 'YTD'], key='sort_inds_summ')
		country_inds = st.selectbox('Country: ', all_countries, key='sum inds')
		if country_inds !='All':
			inds_eq_summ = data[data['Country']==country_inds].groupby(by="Industry").median()
		elif country_inds =='All':
			inds_eq_summ = data.groupby(by="Industry").median()  														

		top_inds = pd.DataFrame(inds_eq_summ.sort_values(by=ret_inds_summ, ascending=False).head()).style.format('{0:,.2f}%', subset=data.columns[6:-1])\
															.format('{0:,.2f}B', subset=data.columns[5])\
															.format('{0:,.2f}', subset=data.columns[-1])\
	    													.background_gradient(cmap='YlGn', subset=data.columns[6:-9])
		bot_inds = pd.DataFrame(inds_eq_summ.sort_values(by=ret_inds_summ, ascending=False).tail()).style.format('{0:,.2f}%', subset=data.columns[6:-1])\
															.format('{0:,.2f}B', subset=data.columns[5])\
															.format('{0:,.2f}', subset=data.columns[-1])\
	    													.background_gradient(cmap='YlOrRd_r', subset=data.columns[6:-9])												

		st.markdown('**Global Industries  -  Top Gainers**')
		st.write(top_inds)
		st.markdown('**Global Industries  -  Top Losers**')
		st.write(bot_inds)

	st.subheader('Global Individual Stocks Summary: ')
	if st.checkbox('Show Global Individual Stocks Return Summary', value=True):
		st.write('*Top USD returns across the World')
		sort_summ = st.selectbox('Return TimeFrame: ', ['1D', '1W', '1M', '3M', '6M', 'YTD'], key='stock_summary')
		country_summ = st.selectbox('Country: ', all_countries)

		if country_summ !='All':
			inds_summ = ["All"] + list(data[data['Country']==country_summ]['Industry'].unique())
			eq_inds_summ = st.selectbox('GICS Industry Name: ', inds_summ)
			if eq_inds_summ != "All":
				comp = data[(data['Country']==country_summ) & (data['Industry']==eq_inds_summ)]
			elif eq_inds_summ == "All":
				comp = data[data['Country']==country_summ]
		elif country_summ =='All':
			inds_summ = ["All"] + list(data['Industry'].unique())
			eq_inds_summ = st.selectbox('GICS Industry Name: ', inds_summ)
			if eq_inds_summ != "All":
				comp = data[data['Industry']==eq_inds_summ]
			elif eq_inds_summ == "All":
				comp = data
	    
		mcap_slider = st.number_input('Minimum MCap (USD): ', min_value=0.5, max_value=1000.0, value=2.0, step=0.1)			
		top_comp = comp[comp['Market Cap']>=mcap_slider].set_index('Ticker').sort_values(by=sort_summ, ascending=False).round(2).head(10).style.format('{0:,.2f}%', subset=data.columns[6:-1])\
															.format('{0:,.2f}B', subset=data.columns[5])\
															.format('{0:,.2f}', subset=data.columns[-1])\
	    													.background_gradient(cmap='YlGn', subset=data.columns[6:-9])
		bot_comp = comp[comp['Market Cap']>=mcap_slider].set_index('Ticker')
		bot_comp = bot_comp.T[bot_comp[sort_summ].dropna().index].T.sort_values(by=sort_summ, ascending=False).round(2).tail(10).style.format('{0:,.2f}%', subset=data.columns[6:-1])\
															.format('{0:,.2f}B', subset=data.columns[5])\
															.format('{0:,.2f}', subset=data.columns[-1])\
	    													.background_gradient(cmap='YlOrRd_r', subset=data.columns[6:-9])
		    	 											
		st.markdown('**Top Gainers - Equities**')	 												
		st.write(top_comp)
		st.markdown('**Top Losers - Equities**')	 												
		st.write(bot_comp)




	st.header('Fixed Income ETFs')
	if st.checkbox('Show Fixed Income ETFs Category Return Summary', value=True):
		df = fi_etfs.copy()
		df.replace('--', np.nan, inplace=True)
		df.replace(' ', np.nan, inplace=True)
		df.iloc[:, 4:15] = df.iloc[:, 4:15].astype(float)
		df = df.groupby(by="Category").median()
		print(st.dataframe(df.sort_values(by='1D', ascending=False).round(2).style.format('{0:,.2f}%', subset=df.columns[:9])\
                   .format('{0:,.2f}B', subset=df.columns[10])\
                   .format('{0:,.2f}M', subset=df.columns[9])\
                   .background_gradient(cmap='RdYlGn', subset=df.columns[1:9])))

	if st.checkbox('Show Fixed Income ETFs Top Movers', value=True):
		df = fi_etfs.copy()
		df.replace('--', np.nan, inplace=True)
		df.replace(' ', np.nan, inplace=True)
		df.iloc[:, 4:15] = df.iloc[:, 4:15].astype(float)
		sort_summ = st.selectbox('Return TimeFrame: ', ['1D', '1W', '1M', '3M', '6M', 'YTD'], key='fi_in_summ')
		

		top_fi = df.set_index('Ticker').sort_values(by=sort_summ, ascending=False).round(2).head(10).style.format('{0:,.2f}%', subset=df.columns[4:13])\
	                   .format('{0:,.2f}B', subset=df.columns[14])\
	                   .format('{0:,.2f}M', subset=df.columns[13])\
	                   .background_gradient(cmap='YlGn', subset=df.columns[5:13])
		bot_fi = df.set_index('Ticker').sort_values(by=sort_summ, ascending=False).round(2).tail(10).style.format('{0:,.2f}%', subset=df.columns[4:13])\
	                   .format('{0:,.2f}B', subset=df.columns[14])\
	                   .format('{0:,.2f}M', subset=df.columns[13])\
	                   .background_gradient(cmap='YlOrRd_r', subset=df.columns[5:13])
		st.markdown('**Top Gainers - Fixed Income ETFs**')     
		print(st.dataframe(top_fi))

		st.markdown('**Top Losers - Fixed Income ETFs**')     
		print(st.dataframe(bot_fi))




	st.header('Global REITs')
	st.subheader('Global REITs Summary: ')
	if st.checkbox('Show Global REITs Industry Return Summary', value=True):
		df = reits.copy()
		df.replace('--', np.nan, inplace=True)
		df.iloc[:, 4:13] = df.iloc[:, 4:13].astype(float)
		country = st.selectbox('Country: ', reit_countries, key='reit_pivot_summ')
		if country == 'All':
			df = df.groupby(by="Sub-Industry").median()
			df = df.drop('Market Cap', axis=1)
			df = df.sort_values(by='1D', ascending=False).round(2).style.format('{0:,.2f}%', subset=df.columns[1:9])\
                   .format('{0:,.2f}B', subset=df.columns[0])\
                   .background_gradient(cmap='RdYlGn', subset=df.columns[1:9])
			print(st.dataframe(df))
		else:
			df = df[df['Country']==country].groupby(by="Sub-Industry").median()
			df = df.drop('Market Cap', axis=1)
			df = df.sort_values(by='1D', ascending=False).round(2).style.format('{0:,.2f}%', subset=df.columns[1:9])\
                   .format('{0:,.2f}B', subset=df.columns[0])\
                   .background_gradient(cmap='RdYlGn', subset=df.columns[1:9])
			print(st.dataframe(df))

	if st.checkbox('Show Top Global REITs Movers', value=True):
		df = reits.copy()
		df.replace('--', np.nan, inplace=True)
		df.iloc[:, 4:13] = df.iloc[:, 4:13].astype(float)
		country = st.selectbox('Country: ', reit_countries, key='reit_pivot_summ1')
		sort_summ = st.selectbox('Return TimeFrame: ', ['1D', '1W', '1M', '3M', '6M', 'YTD'], key='reit_in_summ')
		if country == 'All':
			top_reit = df.set_index('Ticker').sort_values(by=sort_summ, ascending=False).round(2).head(10).style.format('{0:,.2f}%', subset=df.columns[5:13])\
                   .format('{0:,.2f}B', subset=df.columns[4])\
                   .background_gradient(cmap='YlGn', subset=df.columns[6:13])
			st.markdown('**Top Gainers - REIT**')     
			print(st.dataframe(top_reit))

			bot_reit = df.set_index('Ticker').sort_values(by=sort_summ, ascending=False).round(2).tail(10).style.format('{0:,.2f}%', subset=df.columns[5:13])\
                   .format('{0:,.2f}B', subset=df.columns[4])\
                   .background_gradient(cmap='YlOrRd_r', subset=df.columns[6:13])
			st.markdown('**Top Losers - REIT**')
			print(st.dataframe(bot_reit))

		else:
			top_reit = df[df['Country']==country].set_index('Ticker').sort_values(by=sort_summ, ascending=False).round(2).head(5).style.format('{0:,.2f}%', subset=df.columns[5:13])\
                   .format('{0:,.2f}B', subset=df.columns[4])\
                   .background_gradient(cmap='YlGn', subset=df.columns[6:13])
			st.markdown('**Top Gainers - REIT**')
			print(st.dataframe(top_reit))

			bot_reit = df[df['Country']==country].set_index('Ticker').sort_values(by=sort_summ, ascending=False).round(2).tail(5).style.format('{0:,.2f}%', subset=df.columns[5:13])\
                   .format('{0:,.2f}B', subset=df.columns[4])\
                   .background_gradient(cmap='YlOrRd_r', subset=df.columns[6:13])
			st.markdown('**Top Losers - REIT**')
			print(st.dataframe(bot_reit))			
		
	st.header('Commodities')
	if st.checkbox('Show Commodities Performance', value=True):
		start= st.date_input("Custom Start Date: ", date(2020,3,23))
		end = st.date_input("Custom End Date: ", date.today())
		print(st.dataframe(returns_hmap(data=comd[list(comd.columns)], asset_class='Commodities', cat=list(comd.columns), start=start, end=end), height=1500))

	st.header('FX/Currencies')
	if st.checkbox('Show FX Performance', value=True):
		start= st.date_input("Custom Start Date: ", date(2020,3,23), key='fx_summ')
		end = st.date_input("Custom End Date: ", date.today(), key='fx_summ1')
		print(st.dataframe(returns_hmap(data=fx[list(fx.columns)], asset_class='Currencies', cat=list(fx.columns), start=start, end=end), height=1500))


st.sidebar.markdown('Developed by Harsh Shivlani')