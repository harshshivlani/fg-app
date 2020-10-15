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
@st.cache(allow_output_mutation=True)
def load_eq_data():
	data = pd.read_excel('GSTOCKS_N.xlsx')
	data.columns = ["Ticker","Name","Market Cap","Country","Industry","Sub-Industry","1D","1M","3M","6M","YTD","ROE","ROCE","EBITDA (%)","Profit (%)","P/E","Rev YoY","EBITDA YoY","Profit YoY","Rev T12M","FCF T12M"]
	data['Market Cap'] = data['Market Cap']/10**9
	data['Rev T12M'] = data['Rev T12M']/10**9
	data['FCF T12M'] = data['FCF T12M']/10**9
	data['FCF Yield'] = (data['FCF T12M']/data['Market Cap'])*100
	data['MCap-OG'] = data['Market Cap']/(1+data['YTD']/100)
	data = data[['Ticker', 'Name', 'Country', 'Industry', 'Sub-Industry', 'Market Cap', 'MCap-OG',
	'1D', '1M', '3M', '6M', 'YTD', 'ROE', 'ROCE', 'EBITDA (%)',
	'Profit (%)', 'Rev YoY', 'EBITDA YoY', 'Profit YoY', 'FCF Yield', 'Rev T12M',
	'FCF T12M', 'P/E']]
	data['Country'].replace("United Arab Emirates", "UAE", inplace=True)
	data['Country'].replace("Trinidad & Tobago", "Trinidad", inplace=True)
	return data

data = load_eq_data()

nums = ['1D', '1M', '3M', '6M', 'YTD', 'ROE', 'ROCE', 'EBITDA (%)',
       'Profit (%)', 'Rev YoY', 'EBITDA YoY', 'Profit YoY', 'FCF Yield', 'Rev T12M',
       'FCF T12M', 'P/E']
gradient = ['1D', '1M', '3M', '6M', 'YTD']
percs = ['1D', '1M', '3M', '6M', 'YTD', 'ROE', 'ROCE', 'EBITDA (%)',
       'Profit (%)', 'Rev YoY', 'EBITDA YoY', 'Profit YoY', 'FCF Yield']
bns = ['Market Cap', 'MCap-OG', 'Rev T12M','FCF T12M']
all_countries = ["All"] + list(data['Country'].unique())


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
@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def filter_table(country, ind, subind, maxmcap, minmcap):
    df = data.copy()
    df = df[(df["Market Cap"].values<=maxmcap) & (df["Market Cap"].values>minmcap)]
    #df = df[(df['Country'].values==country) & (df["Industry"].values.isin(ind)) & (df["Sub-Industry"].values == subind)]
    if ind != ["All"]:
        df = df[df["Industry"].isin(ind)]
    
    if subind != "All":
    	df = df[df["Sub-Industry"].values == subind]

    if country != "All":
    	df = df[df['Country'].values==country]

    df[percs] = df[percs].fillna(0.00)	
    st.write('There are {} securities in this screen'.format(len(df)))
    st.write('Maximum Market Cap is {}B USD'.format(df['Market Cap'].max().round(0)))
    st.write('Minimum Market Cap is {}B USD'.format(df['Market Cap'].min().round(2)))
    df_style = df.set_index('Ticker').sort_values(by='1D', ascending=False)
    df_style = df_style.round(2)   
    return df_style

#EQUITIES - PIVOT TABLE
@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def mcap_weighted(df, rets_cols, groupby):
	df[rets_cols] = df[rets_cols]/100
	old_mcap = (1/(1+df[rets_cols])).multiply(df['Market Cap'], axis='rows')
	old = old_mcap.join(df['Market Cap'])
	old.iloc[:,:-1] = -old.iloc[:,:-1].subtract(old.iloc[:,-1], axis='rows')
	change_sum = old.join(df[groupby]).groupby(groupby).sum().iloc[:,:-1]
	old_sum = old_mcap.join(df[groupby]).groupby(groupby).sum()
	mcap_weight = pd.DataFrame(df.groupby(groupby).sum()['Market Cap']).merge(change_sum.divide(old_sum, axis='rows'), on=groupby)
	df = mcap_weight
	df[rets_cols] = df[rets_cols]*100
	df = df.sort_values(by='1D', ascending=False).style.format('{0:,.2f}%', subset=["1D","1M","3M","6M","YTD"])\
	                   .format('{0:,.2f}B', subset=["Market Cap"])\
	                   .background_gradient(cmap='RdYlGn', subset=["1D","1M","3M","6M","YTD"])
	return df

#@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def pivot_table(country, ind, maxmcap, minmcap):
	df = data.copy()
	df = df[(df["Market Cap"]<=maxmcap) & (df["Market Cap"]>minmcap)]
	rets_cols = ['1D', '1M', '3M', '6M', 'YTD']
	if country != "All" and ind==["All"]:
		df = df[df['Country'].values==country]
		df = mcap_weighted(df, rets_cols, 'Industry')
	elif country == "All" and ind==["All"]:
		df = mcap_weighted(df, rets_cols, 'Industry')
	elif country == "All" and ind==["None"]:
		df = mcap_weighted(df, rets_cols, 'Country')                   		
	elif country == "All" and ind!=["All"]:
		df = df[df['Industry'].isin(ind)]
		df = mcap_weighted(df, rets_cols, 'Country')		
	else:
		df = df[(df['Country'].values==country) & (df['Industry'].isin(ind))]
		df = df.set_index('Ticker').drop(['MCap-OG'], axis=1)
		df = df.sort_values(by='1D', ascending=False).style.format('{0:,.2f}%', subset=percs)\
					.format('{0:,.2f}B', subset=['Market Cap','Rev T12M','FCF T12M'])\
					.format('{0:,.2f}', subset=nums[-1])\
					.background_gradient(cmap='RdYlGn', subset=gradient)
	return df



#########################################--------------------EQUITY-ETFS--------------------##########################
@st.cache(suppress_st_warning=True)
def load_eqetf_data():
	eq_etfs = pd.read_excel('GEQETF_N.xlsx')
	eq_etfs.columns = ["Ticker","Name","Country","Category","Dividend Yield","Currency","Market Cap","1D","1W","1M","3M","6M","YTD","1Y","3Y","Dividend Type","Exchange","20D T/O","52W High","Price"]
	eq_etfs["% 52W High"] = (eq_etfs["Price"]/eq_etfs["52W High"])*100
	eq_etfs["Market Cap"] = eq_etfs["Market Cap"]/(10**9)
	eq_etfs["20D T/O"] = eq_etfs["20D T/O"]/10**6 
	eq_etfs = eq_etfs.drop(["52W High", "Price", "Dividend Type"], axis=1)
	eq_etfs = eq_etfs[["Ticker","Name","Country","Category","Market Cap","1D","1W","1M","3M","6M","YTD","1Y","3Y","% 52W High","Dividend Yield","Currency","Exchange","20D T/O"]]
	eq_etfs['Country'] = eq_etfs['Country'].replace('Ireland', 'London') 
	npercs = gradient + ['1Y', '3Y', '1W']
	eq_etfs[npercs] = eq_etfs[npercs].fillna(0.00)
	return eq_etfs

retsetf = ["1D","1W","1M","3M","6M","YTD","1Y","3Y","% 52W High"]
eq_etfs = load_eqetf_data()

@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def eqetf_filter(category, country, currency):
	df = eq_etfs.copy()
	if country != 'All':
		df = df[df['Country']==country]

	if currency != 'All':
		df = df[df['Currency']==currency]

	if category == 'All':
		df=df[:].set_index('Ticker')
	elif category == 'None':
		st.write('Median Returns')
		df = df.groupby(by="Category").median()
	else:
	    df = df[df['Category'].values == category].set_index('Ticker')

	return df.sort_values(by='1D', ascending=False).style.format('{0:,.2f}%', subset=["Dividend Yield"]+retsetf)\
                   .format('{0:,.2f}B', subset=["Market Cap"])\
                   .format('{0:,.2f}M', subset=["20D T/O"])\
                   .background_gradient(cmap='RdYlGn', subset=retsetf)



# ----------------------------- EQUITIES SIDEPANEL ---------------------------------------------------------------------
@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def num_func(df, sortby, num):
    return df.sort_values(by=sortby, ascending=False)[:num] if num>0 else df.sort_values(by=sortby, ascending=False)[num:].sort_values(by=sortby)

if side_options == 'Equities':
	#PIVOT - EQUITY
	st.title('Global Equities - Pivot Table')
	st.subheader('Compare Market Cap Weighted Returns across Countries & GICS Industries')
	country = st.selectbox('Country: ', all_countries, key='pivot')
	if country == "All":
		all_ind = ["All"] + list(data['Industry'].unique())
		ind = st.multiselect(label='GICS Industry Name: ', options =  ['None'] + all_ind, default=['All'])
	else:
		all_ind = ["All"] + list(data[data['Country'].values==country]['Industry'].unique())
		ind = st.multiselect(label='GICS Industry Name: ', options = all_ind, default=['All'])

	st.write('Select Market Cap Range:')
	maxmcap = st.number_input('Maximum MCap (Bn USD): ', min_value=0.5, max_value=data['Market Cap'].max(), value=data['Market Cap'].max(), step=0.1, key='eqpivot-max')
	minmcap = st.number_input('Minimum MCap (Bn USD): ', min_value=0.5, max_value=data['Market Cap'].max(), value=1.0, step=0.1, key ='eqpivot-min')
	print(st.dataframe(pivot_table(country=country, ind=ind, maxmcap=maxmcap, minmcap=minmcap), height=500))

	#SCREENER - EQUITY
	st.title('Global Equities Screener')
	st.subheader('Filter Global Stocks by Countries, GICS Industries, GICS Sub-Industries & Market Capitalization (USD)')
	maxmcap = st.number_input('Maximum MCap (Bn USD): ', min_value=0.50, max_value=data['Market Cap'].max(), value=data['Market Cap'].max(), step=0.1, key='eqscreener-max')
	minmcap = st.number_input('Minimum MCap (Bn USD): ', min_value=0.50, max_value=data['Market Cap'].max(), value=1.0, step=0.1, key='eqscreener-min')
	country = st.selectbox('Country: ', all_countries, index=0)
	data = data[(data["Market Cap"]<=maxmcap) & (data["Market Cap"]>minmcap)]
	if country == "All":
		all_ind = ["All"] + list(data['Industry'].unique())
		all_subind = ["All"] + list(data['Sub-Industry'].unique())
	else:
		all_ind = ["All"] + list(data[data['Country'].values==country]['Industry'].unique())
		all_subind = ["All"] + list(data[data['Country'].values==country]['Sub-Industry'].unique())

	ind = st.multiselect(label='GICS Industry Name: ', options = all_ind, default=['All'])
	if ind!=["All"] and country!="All":
		subind = st.selectbox('GICS Sub Industry Name: ', ["All"] + list(data[data['Country'].values==country][data['Industry'].isin(ind)]['Sub-Industry'].unique()))
	elif ind!=["All"] and country=="All":
		subind = st.selectbox('GICS Sub Industry Name: ', ["All"] + list(data[data['Industry'].isin(ind)]['Sub-Industry'].unique()))
	elif ind==["All"] and country!="All":
		subind = st.selectbox('GICS Sub Industry Name: ', ["All"] + list(data[data['Country'].values==country]['Sub-Industry'].unique()))
	else:
		subind = st.selectbox('GICS Sub Industry Name: ', all_subind)

	if country == 'All':
		st.write('*Since the table consists of 10000+ securities, please select sorting and number of securities to be displayed below (- for bottom)')
		sortby = st.selectbox('Sort By: ', ['1D','1W', '1M', '3M', '6M', 'YTD'])
		num = st.number_input('Show Top/Bottom ', min_value=-len(data), max_value=len(data), value=100, step=1, key='eqscreener_num')
		st.dataframe(num_func(filter_table(country=country, ind=ind, subind=subind, maxmcap=maxmcap, minmcap=minmcap), sortby, num).style.format('{0:,.2f}%', subset=percs)\
	    								.format('{0:,.2f}', subset=nums)\
	                                    .format('{0:,.2f}B', subset=bns)\
	                                    .background_gradient(cmap='RdYlGn', subset=gradient), height=600)
	else:
		st.dataframe(filter_table(country=country, ind=ind, subind=subind, maxmcap=maxmcap, minmcap=minmcap).sort_values(by='1D', ascending=False).style.format('{0:,.2f}%', subset=percs)\
	    								.format('{0:,.2f}', subset=nums)\
	                                    .format('{0:,.2f}B', subset=bns)\
	                                    .background_gradient(cmap='RdYlGn', subset=gradient), height=600)

	#ETFS - EQUITY
	st.title('Global Equity ETFs')
	eq_cntry = ['All'] + list(eq_etfs['Country'].unique())
	eq_country = st.selectbox('Country: ', eq_cntry, key='fi_cnt_pivot', index=0)
	if eq_country != "All":
		eq_cur = ['All'] + list(eq_etfs[eq_etfs['Country']==eq_country]['Currency'].unique())
	else:
		eq_cur = ['All'] + list(eq_etfs['Currency'].unique())

	eq_currency = st.selectbox('Currency: ', eq_cur, key='fi_cur_pivot')
	if eq_country=="All" and eq_currency !="All":
		eq_cats = ['None', 'All'] + list(eq_etfs[eq_etfs['Currency']==eq_currency]['Category'].unique())
	elif eq_country !="All" and eq_currency!="All":
		eq_cats = ['None', 'All'] + list(eq_etfs[(eq_etfs['Country']==eq_country) & (eq_etfs['Currency']==eq_currency)]['Category'].unique())
	elif eq_country !="All" and eq_currency=="All":
		eq_cats = ['None', 'All'] + list(eq_etfs[eq_etfs['Country']==eq_country]['Category'].unique())
	else:
		eq_cats = ['None', 'All'] + list(eq_etfs['Category'].unique())

	eq_category = st.selectbox('Category: ', eq_cats, key='fi_pivot')
	print(st.dataframe(eqetf_filter(category=eq_category, country=eq_country, currency=eq_currency), height=700))


# ----------------------------- REITS SIDEPANEL ---------------------------------------------------------------------
@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def load_reits():
	reits = pd.read_excel('GREITS_N.xlsx')
	reits.columns = ["Ticker","Name","Market Cap","Country","Sub-Industry","20D T/O","Dividend Yield","1D","1W","1M","3M","6M","YTD","1Y","3Y","Dividend Type","Div Payout Date","Currency","52W High","Price"]
	reits["% 52W High"] = (reits["Price"]/reits["52W High"])*100
	reits["Market Cap"] = reits["Market Cap"]/(10**9)
	reits["20D T/O"] = reits["20D T/O"]/10**6 
	reits = reits.drop(["52W High", "Price"], axis=1)
	reits = reits[["Ticker","Name","Country","Sub-Industry","Market Cap","Dividend Yield","1D","1W","1M","3M","6M","YTD","1Y","3Y","% 52W High","Dividend Type","Div Payout Date","Currency", "20D T/O"]]

	reits['Country'].replace('Virgin Islands (United States)','Virgin Islands (US)', inplace=True)
	return reits

reits = load_reits()
reit_countries = ['All'] + list(reits['Country'].unique())

@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def filter_reit(country, subind, maxmcap, minmcap):
    df = reits.copy()
    df[df.columns[4:14]] = df[df.columns[4:14]].fillna(0.00) 
    df = df[(df["Market Cap"]<=maxmcap) & (df["Market Cap"]>minmcap)]
    if country != "All":
        df = df[df['Country']==country]

    if subind != "All":
        df = df[df["Sub-Industry"]==subind]

    df_style = df.set_index('Ticker').sort_values(by='Market Cap', ascending=False).style.format('{0:,.2f}%', subset=df.columns[5:15])\
                   .format('{0:,.2f}B', subset=df.columns[4])\
                   .format('{0:,.2f}M', subset=df.columns[-1])\
                   .background_gradient(cmap='RdYlGn', subset=df.columns[6:15])

    return df_style

@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def reit_pivot_table(country, ind, maxmcap, minmcap):
    df = reits.copy()
    df = df[(df["Market Cap"]<=maxmcap) & (df["Market Cap"]>minmcap)]
    rets_cols = ['1D', '1M', '3M', '6M', 'YTD']
    if country != "All" and ind=="All":
    	df = df[df['Country'].values==country]
    	df = mcap_weighted(df, rets_cols, groupby='Sub-Industry')
    	#df = df.sort_values(by='Market Cap', ascending=False)
    elif country == "All" and ind=="All":
    	df = mcap_weighted(df, rets_cols, groupby='Sub-Industry')
    	#df = df.sort_values(by='Market Cap', ascending=False)
    elif country == "All" and ind!="All":
    	df = df[df['Sub-Industry'].values==ind]
    	df = mcap_weighted(df, rets_cols, groupby='Country')
    	#df = df.sort_values(by='Market Cap', ascending=False)
    else:
    	df = df[(df['Country'].values==country) & (df['Sub-Industry'].values==ind)].set_index('Ticker')
    	df[df.columns[5:13]] = df[df.columns[5:13]].fillna(0.00) 
    	df = df.sort_values(by='Market Cap', ascending=False)

    if country!="All" and ind!="All":
    	return df.style.format('{0:,.2f}%', subset=df.columns[4:14])\
    			   .format('{0:,.2f}%', subset=df.columns[4])\
                   .format('{0:,.2f}B', subset=df.columns[3])\
                   .format('{0:,.2f}M', subset=df.columns[-1])\
                   .background_gradient(cmap='RdYlGn', subset=df.columns[5:14])
    else:
    	return df

if side_options == 'REITs':
	st.title('Global REITs Pivot Table')
	st.subheader('Compare Median Return across Countries & GICS Sub-Industries')
	data = reits.copy()
	country = st.selectbox('Country: ', reit_countries, key='pivot')
	if country == "All":
		all_subind = ["All"] + list(data['Sub-Industry'].unique())
	else:
		all_subind = ["All"] + list(data[data['Country']==country]['Sub-Industry'].unique())

	ind = st.selectbox('GICS Industry Name: ', all_subind, key='pivot')
	maxmcap = st.number_input('Maximum MCap (Bn USD): ', min_value=0.5, max_value=data['Market Cap'].max().astype(float), value=data['Market Cap'].max().astype(float), step=0.1, key='reitspivot-max')
	minmcap = st.number_input('Minimum MCap (Bn USD): ', min_value=0.5, max_value=data['Market Cap'].max().astype(float), value=1.0, step=0.1, key='reitspivot-min')
	print(st.dataframe(reit_pivot_table(country=country, ind=ind, maxmcap=maxmcap, minmcap=minmcap), height=700))

	st.title('Global REITs Screener')
	st.subheader('Filter Global REITs by Countries, GICS Sub-Industries & Market Capitalization (USD)')
	data = reits.copy()
	country = st.selectbox('Country: ', reit_countries, index=1)
	if country == "All":
		all_subind = ["All"] + list(data['Sub-Industry'].unique())
	else:
		all_subind = ["All"] + list(data[data['Country']==country]['Sub-Industry'].unique())

	subind = st.selectbox('GICS Industry Name: ', all_subind)
	maxmcap = st.number_input('Maximum MCap (Bn USD): ', min_value=0.5, max_value=data['Market Cap'].max(), value=data['Market Cap'].max(), step=0.1, key='reitscreener-max')
	minmcap = st.number_input('Minimum MCap (Bn USD): ', min_value=0.5, max_value=data['Market Cap'].max(), value=1.0, step=0.1, key='reitscreener-min')

	print(st.dataframe(filter_reit(country=country, subind=subind, maxmcap=maxmcap, minmcap=minmcap), height=600))




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

@st.cache(suppress_st_warning=True, allow_output_mutation=True)
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
@st.cache(suppress_st_warning=True, allow_output_mutation=True)
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
@st.cache(suppress_st_warning=True, allow_output_mutation=True)
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
@st.cache(suppress_st_warning=True, allow_output_mutation=True)
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
@st.cache(suppress_st_warning=True, allow_output_mutation=True)
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
	if st.checkbox('Show Live Data', value=False):
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
@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def load_fi_etfs():
	fi_etfs = pd.read_excel('GFI_N.xlsx')
	fi_etfs.columns = ["Ticker","Name","20D T/O","Market Cap","Category","Dividend Yield","Currency","1D","1W","1M","3M","6M","YTD","1Y","Dividend Type","Exchange","Country","52W High","Price"]
	fi_etfs["% 52W High"] = (fi_etfs["Price"]/fi_etfs["52W High"])*100
	fi_etfs["Market Cap"] = fi_etfs["Market Cap"]/(10**9)
	fi_etfs["20D T/O"] = fi_etfs["20D T/O"]/10**6 
	fi_etfs = fi_etfs.drop(["52W High", "Price"], axis=1)
	fi_etfs = fi_etfs[["Ticker","Name","Country","Category","Market Cap","Dividend Yield","Currency","1D","1W","1M","3M","6M","YTD","1Y","% 52W High","Dividend Type","Exchange","20D T/O"]]
	fi_etfs['Country'] = fi_etfs['Country'].replace('Ireland', 'London')
	return fi_etfs

fi_etfs = load_fi_etfs()
fi_cats = ['None', 'All'] + list(fi_etfs['Category'].unique())
fi_cats1 = ['All'] + list(fi_etfs['Category'].unique())
fi_cntry = ['All'] + list(fi_etfs['Country'].unique())
fi_cur = ['All'] + list(fi_etfs['Currency'].unique())

@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def fi_filter(category, country, currency):
	df = fi_etfs.copy()
	if country != 'All':
		df = df[df['Country']==country]

	if currency != 'All':
		df = df[df['Currency']==currency]

	if category == 'All':
	    return df.set_index('Ticker').sort_values(by='1D', ascending=False).style.format('{0:,.2f}%', subset=["Dividend Yield","1D","1W","1M","3M","6M","YTD","1Y","% 52W High"])\
	                   .format('{0:,.2f}B', subset=["Market Cap"])\
	                   .format('{0:,.2f}M', subset=["20D T/O"])\
	                   .background_gradient(cmap='RdYlGn', subset=["1D","1W","1M","3M","6M","YTD","1Y","% 52W High"])

	elif category == 'None':
	    df = df.groupby(by="Category").median()
	    return df.sort_values(by='1D', ascending=False).style.format('{0:,.2f}%', subset=["Dividend Yield","1D","1W","1M","3M","6M","YTD","1Y","% 52W High"])\
	                   .format('{0:,.2f}B', subset=["Market Cap"])\
	                   .format('{0:,.2f}M', subset=["20D T/O"])\
	                   .background_gradient(cmap='RdYlGn', subset=["1D","1W","1M","3M","6M","YTD","1Y","% 52W High"])
	else:
	    df = df[df['Category'] == category]
	    return df.set_index('Ticker').sort_values(by='1D', ascending=False).style.format('{0:,.2f}%', subset=["Dividend Yield","1D","1W","1M","3M","6M","YTD","1Y","% 52W High"])\
	                   .format('{0:,.2f}B', subset=["Market Cap"])\
	                   .format('{0:,.2f}M', subset=["20D T/O"])\
	                   .background_gradient(cmap='RdYlGn', subset=["1D","1W","1M","3M","6M","YTD","1Y","% 52W High"])

if side_options =='Fixed Income':
	fi_country = st.selectbox('Country: ', fi_cntry, key='fi_cnt_pivot')
	fi_currency = st.selectbox('Currency: ', fi_cur, key='fi_cur_pivot')
	if fi_currency !="All":
		fi_category = st.selectbox('Category: ', fi_cats1, key='fi_pivot')
	else:
		fi_category = st.selectbox('Category: ', fi_cats, key='fi_pivot')

	print(st.dataframe(fi_filter(category=fi_category, country=fi_country, currency=fi_currency), height=700))


##################-------------------Global Yields----------------#############################

@st.cache(allow_output_mutation=True)
def global_yields(countries=['U.S.', 'Germany', 'U.K.', 'Italy', 'France', 'Canada', 'China', 'Australia', 'Japan', 'India', 'Russia', 'Brazil', 'Philippines', 'Thailand']):
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

@st.cache(suppress_st_warning=True, allow_output_mutation=True)
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
	@st.cache(suppress_st_warning=True, allow_output_mutation=True)
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
	components.iframe("https://harshshivlani.github.io/x-asset/livesummary", width=670, height=500)

	st.header('Global Equities')
	#components.iframe("http://www.teletrader.com/yadea/stocks/details/tts-161915467", height=500)
	st.subheader('World Equity Heatmap')
	timeframe = st.selectbox('Timeframe: ', ['Chg (%)', 'Chg YTD (%)', '$ Chg (%)','$ Chg YTD (%)'], index=2)
	print(st.plotly_chart(world_map(timeperiod=timeframe)))

	st.subheader('Global Equity Indices')
	indices = indices_func()
	st.dataframe(indices[0], height=500)

	st.subheader('Industry Summary: ')
	st.write('*Market Cap Weighted Industry Wise USD Returns - Global')
	ret_inds_summ = st.selectbox('Median Return TimeFrame: ', ['1D', '1M', '3M', '6M', 'YTD'], key='sort_inds_summ')
	country_inds = st.selectbox('Country: ', all_countries, key='sum inds')
	df = data.copy()
	rets_cols = ['1D', '1M', '3M', '6M', 'YTD']
	df[rets_cols] = df[rets_cols]/100
	if country_inds !='All':
		df = df[df['Country']==country_inds]
		#inds_eq_summ = data[data['Country']==country_inds].groupby(by="Industry").median()
	elif country_inds =='All':
		df = df[:]
		#inds_eq_summ = data.groupby(by="Industry").median()

	old_mcap = (1/(1+df[rets_cols])).multiply(df['Market Cap'], axis='rows')
	old = old_mcap.join(df['Market Cap'])
	old.iloc[:,:-1] = -old.iloc[:,:-1].subtract(old.iloc[:,-1], axis='rows')
	change_sum = old.join(df['Industry']).groupby('Industry').sum().iloc[:,:-1]
	old_sum = old_mcap.join(df['Industry']).groupby('Industry').sum()
	mcap_weighted = pd.DataFrame(df.groupby('Industry').sum()['Market Cap']).merge(change_sum.divide(old_sum, axis='rows'), on='Industry')
	df = mcap_weighted
	df[rets_cols] = df[rets_cols]*100
	inds_eq_summ = df[:]

	top_inds = inds_eq_summ.sort_values(by=ret_inds_summ, ascending=False).head().style.format('{0:,.2f}%', subset=["1D","1M","3M","6M","YTD"])\
               							.format('{0:,.2f}B', subset=["Market Cap"])\
               							.background_gradient(cmap='YlGn', subset=["1D","1M","3M","6M","YTD"])

	bot_inds = inds_eq_summ.sort_values(by=ret_inds_summ).head().style.format('{0:,.2f}%', subset=["1D","1M","3M","6M","YTD"])\
               							.format('{0:,.2f}B', subset=["Market Cap"])\
               							.background_gradient(cmap='YlOrRd_r', subset=["1D","1M","3M","6M","YTD"])											

	st.markdown('**Global Industries  -  Top Gainers**')
	st.write(top_inds)
	st.markdown('**Global Industries  -  Top Losers**')
	st.write(bot_inds)

	st.subheader('Global Individual Stocks Summary: ')
	st.write('*Top USD returns across the World')
	sort_summ = st.selectbox('Return TimeFrame: ', ['1D', '1M', '3M', '6M', 'YTD'], key='stock_summary')
	country_summ = st.selectbox('Country: ', all_countries)
	data[percs] = data[percs].fillna(0.00)

	if country_summ !='All':
		inds_summ = ["All"] + list(data[data['Country'].values==country_summ]['Industry'].unique())
		eq_inds_summ = st.selectbox('GICS Industry Name: ', inds_summ)
		if eq_inds_summ != "All":
			comp = data[(data['Country'].values==country_summ) & (data['Industry'].values==eq_inds_summ)]
		elif eq_inds_summ == "All":
			comp = data[data['Country'].values==country_summ]
	elif country_summ =='All':
		inds_summ = ["All"] + list(data['Industry'].unique())
		eq_inds_summ = st.selectbox('GICS Industry Name: ', inds_summ)
		if eq_inds_summ != "All":
			comp = data[data['Industry'].values==eq_inds_summ]
		elif eq_inds_summ == "All":
			comp = data
    
	mcap_slider = st.number_input('Minimum MCap (USD): ', min_value=0.5, max_value=1000.0, value=2.0, step=0.1)			
	top_comp = comp[comp['Market Cap']>=mcap_slider].set_index('Ticker').sort_values(by=sort_summ, ascending=False).round(2).head(10).style.format('{0:,.2f}%', subset=percs)\
														.format('{0:,.2f}B', subset=bns)\
														.format('{0:,.2f}', subset=nums[-1])\
    													.background_gradient(cmap='YlGn', subset=gradient)
	bot_comp = comp[comp['Market Cap']>=mcap_slider].set_index('Ticker')
	bot_comp = bot_comp.T[bot_comp[sort_summ].dropna().index].T.sort_values(by=sort_summ).round(2).head(10).style.format('{0:,.2f}%', subset=percs)\
														.format('{0:,.2f}B', subset=bns)\
														.format('{0:,.2f}', subset=nums[-1])\
    													.background_gradient(cmap='YlOrRd_r', subset=gradient)
	    	 											
	st.markdown('**Top Gainers - Equities**')	 												
	st.write(top_comp)
	st.markdown('**Top Losers - Equities**')	 												
	st.write(bot_comp)




	st.header('Fixed Income ETFs')
	st.write('Median Category Returns:')
	df = fi_etfs.copy()
	df = df.groupby(by="Category").median()
	print(st.dataframe(df.sort_values(by='1D', ascending=False).round(2).style.format('{0:,.2f}%', subset=df.columns[1:10])\
               .format('{0:,.2f}B', subset=df.columns[0])\
               .format('{0:,.2f}M', subset=df.columns[-1])\
               .background_gradient(cmap='RdYlGn', subset=df.columns[2:10])))


	df = fi_etfs.copy()
	cut = gradient + ['1Y', '1W']
	df[cut] = df[cut].fillna(0.00)

	sort_summ = st.selectbox('Return TimeFrame: ', ['1D', '1W', '1M', '3M', '6M', 'YTD', '1Y'], key='fi_in_summ')

	top_fi = df.set_index('Ticker').sort_values(by=sort_summ, ascending=False).round(2).head(10).style.format('{0:,.2f}%', subset=df.columns[7:15])\
                   .format('{0:,.2f}%', subset=df.columns[5])\
                   .format('{0:,.2f}B', subset=df.columns[4])\
                   .format('{0:,.2f}M', subset=df.columns[-1])\
                   .background_gradient(cmap='YlGn', subset=df.columns[7:15])
	bot_fi = df.set_index('Ticker').sort_values(by=sort_summ).round(2).head(10).style.format('{0:,.2f}%', subset=df.columns[7:15])\
                   .format('{0:,.2f}%', subset=df.columns[5])\
                   .format('{0:,.2f}B', subset=df.columns[4])\
                   .format('{0:,.2f}M', subset=df.columns[-1])\
                   .background_gradient(cmap='YlOrRd_r', subset=df.columns[7:15])
	st.markdown('**Top Gainers - Fixed Income ETFs**')     
	print(st.dataframe(top_fi))

	st.markdown('**Top Losers - Fixed Income ETFs**')     
	print(st.dataframe(bot_fi))




	st.header('Global REITs')
	st.subheader('Global REITs Summary: ')
	if st.checkbox('Show Global REITs Industry Median Return Summary', value=True):
		df = reits.copy()

		country = st.selectbox('Country: ', reit_countries, key='reit_pivot_summ')
		if country == 'All':
			df = df.groupby(by="Sub-Industry").median()
			#df = df.drop('Market Cap', axis=1)
			df = df.sort_values(by='1D', ascending=False).round(2).style.format('{0:,.2f}%', subset=df.columns[1:11])\
                   .format('{0:,.2f}B', subset=df.columns[0])\
                   .format('{0:,.2f}M', subset=df.columns[-1])\
                   .background_gradient(cmap='RdYlGn', subset=df.columns[2:11])
			print(st.dataframe(df))
		else:
			df = df[df['Country']==country].groupby(by="Sub-Industry").median()
			#df = df.drop('Market Cap', axis=1)
			df = df.sort_values(by='1D', ascending=False).round(2).style.format('{0:,.2f}%', subset=df.columns[1:11])\
                   .format('{0:,.2f}B', subset=df.columns[0])\
                   .format('{0:,.2f}M', subset=df.columns[-1])\
                   .background_gradient(cmap='RdYlGn', subset=df.columns[2:11])
			print(st.dataframe(df))

	if st.checkbox('Show Top Global REITs Movers', value=True):
		df = reits.copy()
		df.replace('--', np.nan, inplace=True)
		cut = gradient + ['1Y', '1W', '3Y']
		df[cut] = df[cut].fillna(0.00)

		country = st.selectbox('Country: ', reit_countries, key='reit_pivot_summ1')
		sort_summ = st.selectbox('Return TimeFrame: ', ['1D', '1W', '1M', '3M', '6M', 'YTD', '1Y', '3Y'], key='reit_in_summ')
		if country == 'All':
			df = df[:]
		else:
			df = df[df['Country']==country]

		top_reit = df.set_index('Ticker').sort_values(by=sort_summ, ascending=False).round(2).head(10).style.format('{0:,.2f}%', subset=df.columns[5:15])\
               .format('{0:,.2f}B', subset=df.columns[4])\
               .format('{0:,.2f}M', subset=df.columns[-1])\
               .background_gradient(cmap='YlGn', subset=df.columns[6:15])
		st.markdown('**Top Gainers - REIT**')     
		print(st.dataframe(top_reit))

		bot_reit = df.set_index('Ticker').sort_values(by=sort_summ).round(2).head(10).style.format('{0:,.2f}%', subset=df.columns[5:15])\
               .format('{0:,.2f}B', subset=df.columns[4])\
               .format('{0:,.2f}M', subset=df.columns[-1])\
               .background_gradient(cmap='YlOrRd_r', subset=df.columns[6:15])
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