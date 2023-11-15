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
import datetime
from datetime import date, timedelta
import yahooquery
from yahooquery import Ticker
import base64
from io import BytesIO
import openpyxl
import quandl
quandl.ApiConfig.api_key = "KZ69tzkHfXscfQ1qcJ5K"

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
st.write('Data as of 14th November, 2023 (EOD)')
components.iframe("https://harshshivlani.github.io/x-asset/liveticker")
st.sidebar.header('Cross Asset Monitor: Contents')
side_options = st.sidebar.radio('Please Select One:', ('Equities', 'Fixed Income', 'REITs', 'Commodities', 'FX'))
st.sidebar.write('Developed by Harsh Shivlani')

def color_positive_green(val):
    """
    Takes a scalar and returns a string with
    the css property `'color: green'` for positive
    strings, black otherwise.
    """
    if val > 0:
        color = 'green'
    else:
        color = 'red'
    return 'color: %s' % color

#Import Master Data
@st.cache(allow_output_mutation=True)
def load_eq_data():
	data = pd.read_excel("GSTOCKS_N.xlsx", engine='openpyxl')
	#sep=","
	data.columns = ["Ticker","Name","Market Cap","Country","Industry","Sub-Industry","1D","1M","3M","YTD","ROE","ROCE","EBITDA (%)",
	"Profit (%)","P/E","Rev YoY","EBITDA YoY","Profit YoY","Rev T12M","FCF T12M", "1W"]
	data['Market Cap'] = data['Market Cap']/10**9
	data['Rev T12M'] = data['Rev T12M']/10**9
	data['FCF T12M'] = data['FCF T12M']/10**9
	data['FCF Yield'] = (data['FCF T12M']/data['Market Cap'])*100
	data['MCap-OG'] = data['Market Cap']/(1+data['YTD']/100)
	data = data[['Ticker', 'Name', 'Country', 'Industry', 'Sub-Industry', 'Market Cap', 'MCap-OG',
	'1D', '1W', '1M', '3M', 'YTD', 'ROE', 'ROCE', 'EBITDA (%)',
	'Profit (%)', 'Rev YoY', 'EBITDA YoY', 'Profit YoY', 'FCF Yield', 'Rev T12M',
	'FCF T12M', 'P/E']]
	data['Country'].replace("United Arab Emirates", "UAE", inplace=True)
	data['Country'].replace("Trinidad & Tobago", "Trinidad", inplace=True)
	return data

data = load_eq_data()

nums = ['1D', '1W', '1M', '3M', 'YTD', 'ROE', 'ROCE', 'EBITDA (%)',
       'Profit (%)', 'Rev YoY', 'EBITDA YoY', 'Profit YoY', 'FCF Yield', 'Rev T12M',
       'FCF T12M', 'P/E']
gradient = ['1D', '1W', '1M', '3M', 'YTD']
percs = ['1D', '1W', '1M', '3M', 'YTD', 'ROE', 'ROCE', 'EBITDA (%)',
       'Profit (%)', 'Rev YoY', 'EBITDA YoY', 'Profit YoY', 'FCF Yield']
bns = ['Market Cap', 'MCap-OG', 'Rev T12M','FCF T12M']
all_countries = ["All"] + list(data['Country'].unique())



#EQUITIES - SCREENER
@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def filter_table(country, maxmcap, minmcap, ind=['All'], subind='All'):
    df = data.copy()
    df = df[(df["Market Cap"].values<=maxmcap) & (df["Market Cap"].values>minmcap)]
    #df = df[(df['Country'].values==country) & (df["Industry"].values.isin(ind)) & (df["Sub-Industry"].values == subind)]
    if country != "All":
    	if country =='Emerging Markets':
    		df = df[df['Country'].isin(ems)]
    	elif country=='Asian Markets':
    		df = df[df['Country'].isin(asia)]
    	elif country =='European Markets':
    		df = df[df['Country'].isin(europe)]
    	else:
    		df = df[df['Country'].values==country]

    if ind != ["All"]:
        df = df[df["Industry"].isin(ind)]
    
    if subind != "All":
    	df = df[df["Sub-Industry"].values == subind]

    df[percs] = df[percs].fillna(0.00)	
    st.write('There are {} securities in this screen'.format(len(df)))
    st.write('Maximum Market Cap is {}B USD'.format(df['Market Cap'].max().round(2)))
    st.write('Minimum Market Cap is {}B USD'.format(df['Market Cap'].min().round(2)))
    df_style = df.set_index('Ticker').sort_values(by='1D', ascending=False)
    df_style = df_style.round(2)   
    return df_style

#EQUITIES - PIVOT TABLE
@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def mcap_weighted(df, rets_cols, groupby, reit=False):
	df[rets_cols] = df[rets_cols]/100
	old_mcap = (1/(1+df[rets_cols])).multiply(df['Market Cap'], axis='rows')
	old = old_mcap.join(df['Market Cap'])
	old.iloc[:,:-1] = -old.iloc[:,:-1].subtract(old.iloc[:,-1], axis='rows')
	change_sum = old.join(df[groupby]).groupby(groupby).sum().iloc[:,:-1]
	old_sum = old_mcap.join(df[groupby]).groupby(groupby).sum()
	mcap_weight = pd.DataFrame(df.groupby(groupby).sum()['Market Cap']).merge(change_sum.divide(old_sum, axis='rows'), on=groupby)
	df = mcap_weight
	df[rets_cols] = df[rets_cols]*100
	if reit == True:
		subs = ["1D","1W","1M","3M","6M","YTD"]
	else:
		subs = ["1D","1W","1M","3M","YTD"]
	df = df.sort_values(by='1D', ascending=False).style.format('{0:,.2f}%', subset=subs)\
	                   .format('{0:,.2f}B', subset=["Market Cap"])\
	                   .applymap(color_positive_green, subset=subs)
	return df

#@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def pivot_table(country, ind, maxmcap, minmcap):
	df = data.copy()
	df = df[(df["Market Cap"]<=maxmcap) & (df["Market Cap"]>minmcap)]
	rets_cols = ['1D', '1W', '1M', '3M', 'YTD']
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
					.applymap(color_positive_green, subset=gradient)
	return df


def regional_sect_perf(period='1M', level = 'Industry'):
    """
    """
    def mcap_weighted11(df, rets_cols, groupby, reit=False):
    	df[rets_cols] = df[rets_cols]/100
    	old_mcap = (1/(1+df[rets_cols])).multiply(df['Market Cap'], axis='rows')
    	old = old_mcap.join(df['Market Cap'])
    	old.iloc[:,:-1] = -old.iloc[:,:-1].subtract(old.iloc[:,-1], axis='rows')
    	change_sum = old.join(df[groupby]).groupby(groupby).sum().iloc[:,:-1]
    	old_sum = old_mcap.join(df[groupby]).groupby(groupby).sum()
    	mcap_weight = pd.DataFrame(df.groupby(groupby).sum()['Market Cap']).merge(change_sum.divide(old_sum, axis='rows'), on=groupby)
    	df = mcap_weight
    	df[rets_cols] = df[rets_cols]*100
    	if reit == True:
    		subs = ["1D","1W","1M","3M","6M","YTD"]
    	else:
    		subs = ["1D","1W","1M","3M","YTD"]
    	df = df.sort_values(by='1D', ascending=False)
    	return df

    region_names = ['All', 'DM', 'US', 'Europe', 'EM', 'China', 'EM ex-China', 'EM-Asia','EMEA','LatAm','Middle East']
    region_dfs = [full, dm, us, europe, ems, china, emexchina, emasia, emea, latam, me]
    matrix = pd.DataFrame(index=data[level].unique())
    matrix.index.name = level
    rets_cols = ['1D', '1W', '1M', '3M', 'YTD']
    for i in range(11):
        new_df = data[data['Country'].isin(region_dfs[i])]
        new_df = new_df[(new_df["Market Cap"]<=10000) & (new_df["Market Cap"]>0.5)]    
        reg  = pd.DataFrame(mcap_weighted11(new_df, rets_cols, level)[period])
        reg.columns = [region_names[i]]
        matrix = matrix.join(reg, on=level)

    matrix = matrix.sort_values(by='All', ascending = False).drop(np.nan, axis=0).fillna(0).style.format('{0:,.2f}%').applymap(color_positive_green)
    #.drop(['Mortgage Real Estate Investment Trusts (REITs)', 'Equity Real Estate Investment Trusts (REITs)'], axis=0)
    
    return matrix


#########################################--------------------EQUITY-ETFS--------------------##########################
@st.cache(suppress_st_warning=True)
def load_eqetf_data():
	eq_etfs = pd.read_excel('GEQETF_N.xlsx', engine='openpyxl')
	eq_etfs.columns = ["Ticker","Name","Country","Category","Dividend Yield","Currency","Market Cap","1D","1W","1M","3M","6M","YTD","1Y","3Y","Dividend Type","Exchange","52W High","Price","20D T/O"]
	eq_etfs["% 52W High"] = (eq_etfs["Price"]/eq_etfs["52W High"])*100
	eq_etfs["Market Cap"] = eq_etfs["Market Cap"]/(10**9)
	eq_etfs["20D T/O"] = eq_etfs["20D T/O"]/10**6 
	eq_etfs = eq_etfs.drop(["52W High", "Price", "Dividend Type"], axis=1)
	eq_etfs = eq_etfs[["Ticker","Name","Country","Category","Market Cap","1D","1W","1M","3M","6M","YTD","1Y","3Y","% 52W High","Dividend Yield","Currency","Exchange","20D T/O"]]
	eq_etfs['Country'] = eq_etfs['Country'].replace('Ireland', 'London') 
	npercs = gradient + ['1Y', '3Y', '1W']
	#eq_etfs[npercs] = eq_etfs[npercs].fillna(0)
	return eq_etfs

retsetf = ["1D","1W","1M","3M","6M","YTD","1Y","3Y","% 52W High"]
eq_etfs = load_eqetf_data()

#DEFINE REGION LISTS
full = list(data['Country'].unique())

allclean = data.groupby('Country').sum()
allclean = allclean[allclean['Market Cap']>=50].index.to_list()

ems = ['Argentina', 'Brazil', 'Chile', 'China', 'Colombia', 'Czech Republic', 'Egypt', 'Greece', 'Hungary', 'India', 'Indonesia',
  'South Korea', 'Kuwait', 'Malaysia', 'Mexico', 'Peru', 'Philippines', 'Poland', 'Qatar', 'South Africa',
  'Taiwan', 'Thailand', 'Turkey', 'United Arab Emirates', 'UAE', 'Vietnam', 'Saudi Arabia', 'Pakistan']

dm = list(set(full).difference(ems))

dmexus = list(set(full).difference(ems).difference(['United States']))

asia_pacific = ['China', 'India', 'Indonesia', 'South Korea', 'Malaysia', 'Pakistan', 'Philippines',
           'Taiwan', 'Thailand', 'Hong Kong', 'Japan', 'New Zealand', 'Australia', 'Singapore']

asia = ['China', 'India', 'Indonesia', 'South Korea', 'Malaysia', 'Pakistan', 'Philippines',
           'Taiwan', 'Thailand', 'Hong Kong', 'Japan', 'Singapore']

asiaexindia = ['China', 'Indonesia', 'South Korea', 'Malaysia', 'Pakistan', 'Philippines',
           'Taiwan', 'Thailand', 'Hong Kong', 'Japan', 'Singapore']

asiaexindiachina = ['Indonesia', 'South Korea', 'Malaysia', 'Pakistan', 'Philippines',
           'Taiwan', 'Thailand', 'Hong Kong', 'Japan', 'Singapore']

emexchina = ['Argentina', 'Brazil', 'Chile', 'Colombia', 'Czech Republic', 'Egypt', 'Greece', 'Hungary', 'India', 'Indonesia',
  'South Korea', 'Kuwait', 'Malaysia', 'Mexico', 'Peru', 'Philippines', 'Poland', 'Qatar', 'Russia', 'South Africa',
  'Taiwan', 'Thailand', 'Turkey', 'United Arab Emirates', 'Vietnam', 'Saudi Arabia', 'Pakistan']

emasia = ['China', 'India', 'Indonesia', 'South Korea', 'Malaysia', 'Pakistan', 'Philippines', 'Taiwan', 'Thailand']

europe = ['Austria', 'Belgium', 'Denmark', 'Finland', 'France', 'Germany', 'Italy', 'Netherlands', 'Norway',
          'Spain', 'Sweden', 'Switzerland', 'United Kingdom', 'Portugal']

latam = ['Argentina', 'Brazil', 'Chile', 'Colombia', 'Mexico', 'Peru']

emea = ['Czech Republic', 'Egypt', 'Greece', 'Hungary', 'Kuwait', 'Poland', 'Qatar', 'Russia', 'Saudi Arabia',
        'South Africa', 'Turkey', 'United Arab Emirates']

me=['Bahrain', 'Cyprus', 'Egypt', 'Iran', 'Iraq','Israel','Jordan','Kuwait','Lebanon','Oman','Qatar','Saudi Arabia','Turkey','UAE']

india = ['India']

us= ['United States']

china = ['China']

@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def eqetf_filter(category, country, currency):
	df = eq_etfs.copy()
	if country == 'All':
		df = df[:]
	elif country =='Emerging Markets':
		df = df[df['Country'].isin(ems)]
	elif country=='Asian Markets':
		df = df[df['Country'].isin(asia)]
	elif country =='European Markets':
		df = df[df['Country'].isin(europe)]
	else:
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
                   .applymap(color_positive_green, subset=retsetf)



# ----------------------------- EQUITIES SIDEPANEL ---------------------------------------------------------------------
@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def num_func(df, sortby, num):
    return df.sort_values(by=sortby, ascending=False)[:num] if num>0 else df.sort_values(by=sortby, ascending=False)[num:].sort_values(by=sortby)

def to_excel(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Sheet1')
    writer.close()
    processed_data = output.getvalue()
    return processed_data

def get_table_download_link(df):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    val = to_excel(df)
    b64 = base64.b64encode(val)  # val looks like b'...'
    return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="extract.xlsx">Export to Excel</a>' # decode b'abc' => abc


if side_options == 'Equities':
	#INDEX DATA
	st.title('Global Equity Indices')
	eqidx = pd.read_excel("EQ-FX.xlsx", engine='openpyxl', sheet_name='EQ', header=0, index_col=0)
	eqidx = eqidx.sort_values(by='Chg USD(%)', ascending=False).style.format('{0:,.2%}', subset=['Chg L Cy(%)', 'Chg USD(%)', '$1W(%)',
																					  '1M (%)', '$YTD(%)'])\
					.format('{0:,.2f}', subset=['Price (EOD)'])\
					.applymap(color_positive_green, subset=['Chg L Cy(%)', 'Chg USD(%)', '$1W(%)','1M (%)', '$YTD(%)'])\
					#.background_gradient(cmap='RdYlGn', subset=['Chg L Cy(%)', 'Chg USD(%)', '$1W(%)','1M (%)', '$YTD(%)'])
	print(st.dataframe(eqidx, height=500))
	st.markdown(get_table_download_link(eqidx), unsafe_allow_html=True)


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
	eq_pivot_styled = pivot_table(country=country, ind=ind, maxmcap=maxmcap, minmcap=minmcap)
	print(st.dataframe(eq_pivot_styled, height=500))
	st.markdown(get_table_download_link(eq_pivot_styled), unsafe_allow_html=True)

	#REGIONAL - EQUITY
	st.title('Global Equities - Regional Performance Table')
	st.subheader('Compare Market Cap Weighted Industry Returns across Major Regions')
	period = st.selectbox('Period: ', ['1D', '1W', '1M', '3M', 'YTD'], key='rot')
	level=st.selectbox('Level: ', ['Industry', 'Sub-Industry'], key='rot1')
	reg_equity = regional_sect_perf(period, level)
	print(st.dataframe(reg_equity, height=500))
	st.markdown(get_table_download_link(reg_equity), unsafe_allow_html=True)

	#SCREENER - EQUITY
	st.title('Global Equities Screener')
	st.subheader('Filter Global Stocks by Countries, GICS Industries, GICS Sub-Industries & Market Capitalization (USD)')
	maxmcap = st.number_input('Maximum MCap (Bn USD): ', min_value=0.50, max_value=data['Market Cap'].max(), value=data['Market Cap'].max(), step=0.1, key='eqscreener-max')
	minmcap = st.number_input('Minimum MCap (Bn USD): ', min_value=0.50, max_value=data['Market Cap'].max(), value=1.0, step=0.1, key='eqscreener-min')
	country = st.selectbox('Country: ',all_countries + ['Emerging Markets', 'Asian Markets', 'European Markets'], index=0)
	data = data[(data["Market Cap"]<=maxmcap) & (data["Market Cap"]>minmcap)]
	if country == "All":
		all_ind = ["All"] + list(data['Industry'].unique())
		all_subind = ["All"] + list(data['Sub-Industry'].unique())
	elif country=='Emerging Markets':
		all_ind = ["All"] + list(data[data['Country'].isin(ems)]['Industry'].unique())
		all_subind = ["All"] + list(data[data['Country'].isin(ems)]['Sub-Industry'].unique())
	elif country=='Asian Markets':
		all_ind = ["All"] + list(data[data['Country'].isin(asia)]['Industry'].unique())
		all_subind = ["All"] + list(data[data['Country'].isin(asia)]['Sub-Industry'].unique())
	elif country=='European Markets':
		all_ind = ["All"] + list(data[data['Country'].isin(europe)]['Industry'].unique())
		all_subind = ["All"] + list(data[data['Country'].isin(europe)]['Sub-Industry'].unique())
	else:
		all_ind = ["All"] + list(data[data['Country'].values==country]['Industry'].unique())
		all_subind = ["All"] + list(data[data['Country'].values==country]['Sub-Industry'].unique())

	ind = st.multiselect(label='GICS Industry Name: ', options = all_ind, default=['All'])
	if ind!=["All"] and country!="All":
		if country=='Emerging Markets':
			subind = st.selectbox('GICS Sub Industry Name: ', ["All"] + list(data[data['Country'].isin(ems)][data['Industry'].isin(ind)]['Sub-Industry'].unique()))
		elif country=='Asian Markets':
			subind = st.selectbox('GICS Sub Industry Name: ', ["All"] + list(data[data['Country'].isin(asia)][data['Industry'].isin(ind)]['Sub-Industry'].unique()))
		elif country=='European Markets':
			subind = st.selectbox('GICS Sub Industry Name: ', ["All"] + list(data[data['Country'].isin(europe)][data['Industry'].isin(ind)]['Sub-Industry'].unique()))
		else:
			subind = st.selectbox('GICS Sub Industry Name: ', ["All"] + list(data[data['Country'].values==country][data['Industry'].isin(ind)]['Sub-Industry'].unique()))
	elif ind!=["All"] and country=="All":
		subind = st.selectbox('GICS Sub Industry Name: ', ["All"] + list(data[data['Industry'].isin(ind)]['Sub-Industry'].unique()))
	else:
		subind='All'

	if country == 'All':
		st.write('*Since the table consists of 10000+ securities, please select sorting and number of securities to be displayed below (- for bottom)')
		sortby = st.selectbox('Sort By: ', ['1D','1W', '1M', '3M', 'YTD'])
		num = st.number_input('Show Top/Bottom ', min_value=-len(data), max_value=len(data), value=100, step=1, key='eqscreener_num')
		if st.button('Show Results'):
			all_eq1 = num_func(filter_table(country=country, ind=ind, subind=subind, maxmcap=maxmcap, minmcap=minmcap), sortby, num).style.format('{0:,.2f}%', subset=percs)\
	    								.format('{0:,.2f}', subset=nums)\
	                                    .format('{0:,.2f}B', subset=bns)\
	                                    .applymap(color_positive_green, subset=gradient)
			st.dataframe(all_eq1, height=600)
			st.markdown(get_table_download_link(all_eq1), unsafe_allow_html=True)

	else:
		eqs_f = filter_table(country=country, ind=ind, subind=subind, maxmcap=maxmcap, minmcap=minmcap).sort_values(by='1D', ascending=False)
		eqs_f_styled = eqs_f.style.format('{0:,.2f}%', subset=percs)\
	    								.format('{0:,.2f}', subset=nums)\
	                                    .format('{0:,.2f}B', subset=bns)\
	                                    .applymap(color_positive_green, subset=gradient)
		if st.button('Show Results'):
			st.dataframe(eqs_f_styled, height=600)
			st.markdown(get_table_download_link(eqs_f_styled), unsafe_allow_html=True)


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
	eq_etfs_styled = eqetf_filter(category=eq_category, country=eq_country, currency=eq_currency)
	print(st.dataframe(eq_etfs_styled, height=700))
	st.markdown(get_table_download_link(eq_etfs_styled), unsafe_allow_html=True)


# ----------------------------- REITS SIDEPANEL ---------------------------------------------------------------------
@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def load_reits():
	reits = pd.read_excel('GREITS_N.xlsx', engine='openpyxl')
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


def reit_boxplot(country, industry, period):
    data = reits.copy().round(2)
    if country=='All':
    	data = data[:]
    else:
    	data = data[data['Country']==country]


    if industry=="None":
        fig = px.box(data, x="Country", y=period, color="Country", 
                     hover_data=["Sub-Industry", "Ticker", "Name","Market Cap",
                                "Dividend Yield", "Dividend Type", "Currency", '1D', '1W', '1M','3M',
                                 '6M', 'YTD', '1Y', '% 52W High'], points="all")
    elif industry=="All":
    	fig = px.box(data, x="Sub-Industry", y=period, color="Sub-Industry", 
                     hover_data=["Country", "Ticker", "Name","Market Cap",
                                "Dividend Yield", "Dividend Type", "Currency", '1D', '1W', '1M','3M',
                                 '6M', 'YTD', '1Y', '% 52W High'], points="all")
    else:
        fig = px.box(data[data["Sub-Industry"]==industry], x="Country", y=period, color="Country", 
                     hover_data=["Sub-Industry", "Ticker", "Name","Market Cap",
                                "Dividend Yield", "Dividend Type", "Currency", '1D', '1W', '1M','3M',
                                 '6M', 'YTD', '1Y', '% 52W High'], points="all")       

    fig.update_layout(title = 'Global REITs Country Wise USD Returns - '+str(period), 
                      font=dict(family="Segoe UI, monospace", size=13, color="#7f7f7f"),
                      legend_title_text='Countries', plot_bgcolor = 'White')
    fig.update_yaxes(ticksuffix="%")
    #fig.update_traces(hovertemplate='Date: %{x} <br>Return: %{y:.2%}') 
    return fig

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
                   .applymap(color_positive_green, subset=df.columns[6:15])

    return df_style

@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def reit_pivot_table(country, ind, maxmcap, minmcap):
    df = reits.copy()
    df = df[(df["Market Cap"]<=maxmcap) & (df["Market Cap"]>minmcap)]
    rets_cols = ['1D', '1W', '1M', '3M', '6M', 'YTD']
    if country != "All" and ind=="All":
    	df = df[df['Country'].values==country]
    	df = mcap_weighted(df, rets_cols, groupby='Sub-Industry', reit=True)
    	#df = df.sort_values(by='Market Cap', ascending=False)
    elif country == "All" and ind=="All":
    	df = mcap_weighted(df, rets_cols, groupby='Sub-Industry', reit=True)
    	#df = df.sort_values(by='Market Cap', ascending=False)
    elif country == "All" and ind!="All":
    	df = df[df['Sub-Industry'].values==ind]
    	df = mcap_weighted(df, rets_cols, groupby='Country', reit=True)
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
                   .applymap(color_positive_green, subset=df.columns[5:14])
    else:
    	return df

if side_options == 'REITs':
	st.title('Global REITs Pivot Table')
	st.subheader('Compare Market Cap Weighted Returns across Countries & GICS Sub-Industries')
	data = reits.copy()
	country = st.selectbox('Country: ', reit_countries, key='pivot')
	if country == "All":
		all_subind = ["All"] + list(data['Sub-Industry'].unique())
	else:
		all_subind = ["All"] + list(data[data['Country']==country]['Sub-Industry'].unique())

	ind = st.selectbox('GICS Industry Name: ', all_subind, key='pivot1o')
	maxmcap = st.number_input('Maximum MCap (Bn USD): ', min_value=0.5, max_value=data['Market Cap'].max().astype(float), value=data['Market Cap'].max().astype(float), step=0.1, key='reitspivot-max')
	minmcap = st.number_input('Minimum MCap (Bn USD): ', min_value=0.5, max_value=data['Market Cap'].max().astype(float), value=1.0, step=0.1, key='reitspivot-min')
	print(st.dataframe(reit_pivot_table(country=country, ind=ind, maxmcap=maxmcap, minmcap=minmcap), height=300))

	st.title('Global REITs Box Plot')
	countries_reit = st.selectbox('Country: ', reit_countries, index=0)
	period_reit = st.selectbox('Period: ', reits.columns[6:13], index=0)
	if countries_reit!='All':
		industry_reit = st.selectbox('Industry: ', ["All"] + list(reits[reits['Country']==countries_reit]["Sub-Industry"].unique()), index=0)
	else:
		industry_reit = st.selectbox('Industry: ', ["All", "None"] + list(reits["Sub-Industry"].unique()), index=0)
	st.plotly_chart(reit_boxplot(country=countries_reit,period=period_reit, industry=industry_reit), use_container_width=True)


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
	reits_filter1 = filter_reit(country=country, subind=subind, maxmcap=maxmcap, minmcap=minmcap)
	print(st.dataframe(reits_filter1, height=600))
	st.markdown(get_table_download_link(reits_filter1), unsafe_allow_html=True)





# ----------------------------- COMMODITIES SIDEPANEL ---------------------------------------------------------------------

if side_options == 'Commodities':
	st.title('Commodity Futures Performance')
	comdidx = pd.read_excel("EQ-FX.xlsx", engine='openpyxl', sheet_name='Comd', header=0, index_col=0)
	comdidx = comdidx.sort_values(by='1D (%)', ascending=False).style.format('{0:,.2%}')\
					.applymap(color_positive_green)\
					#.background_gradient(cmap='RdYlGn', subset=['Chg L Cy(%)', 'Chg USD(%)', '$1W(%)','1M (%)', '$YTD(%)'])
	print(st.dataframe(comdidx, height=500))
	st.markdown(get_table_download_link(comdidx), unsafe_allow_html=True)


	if st.checkbox('Show Live Data', value=True):
		components.iframe("https://harshshivlani.github.io/x-asset/comd", width=670, height=500)
	if st.checkbox('Show Live Chart',value=True):
		components.iframe("https://harshshivlani.github.io/x-asset/equity-chart", width=1000, height=550)
	

#################-------------------------Fixed Income-------------------------------#######################

if side_options == 'FX':
	st.title('Spot FX Performance')
	fxidx = pd.read_excel("EQ-FX.xlsx", engine='openpyxl', sheet_name='FX', header=0, index_col=0)
	fxidx = fxidx.sort_values(by='1D', ascending=False).style.format('{0:,.2%}', subset=['Interest Rates', '1D', '1W','1M','3M','6M','YTD'])\
					.applymap(color_positive_green, subset=['1D', '1W','1M','3M','6M','YTD'])\
					#.background_gradient(cmap='RdYlGn', subset=['Chg L Cy(%)', 'Chg USD(%)', '$1W(%)','1M (%)', '$YTD(%)'])
	print(st.dataframe(fxidx, height=500))
	st.markdown(get_table_download_link(fxidx), unsafe_allow_html=True)



	if st.checkbox('Show Live Data', value=True):
		components.iframe("https://harshshivlani.github.io/x-asset/cur", width=670, height=500)
	if st.checkbox('Show Live Chart'):
		components.iframe("https://harshshivlani.github.io/x-asset/equity-chart", height=500)


#################-------------------------Fixed Income-------------------------------#######################
@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def load_fi_etfs():
	fi_etfs = pd.read_excel('GFI_N.xlsx', engine='openpyxl')
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
	                   .applymap(color_positive_green,  subset=["1D","1W","1M","3M","6M","YTD","1Y","% 52W High"])

	elif category == 'None':
	    df = df.groupby(by="Category").median()
	    return df.sort_values(by='1D', ascending=False).style.format('{0:,.2f}%', subset=["Dividend Yield","1D","1W","1M","3M","6M","YTD","1Y","% 52W High"])\
	                   .format('{0:,.2f}B', subset=["Market Cap"])\
	                   .format('{0:,.2f}M', subset=["20D T/O"])\
	                   .applymap(color_positive_green, subset=["1D","1W","1M","3M","6M","YTD","1Y","% 52W High"])
	else:
	    df = df[df['Category'] == category]
	    return df.set_index('Ticker').sort_values(by='1D', ascending=False).style.format('{0:,.2f}%', subset=["Dividend Yield","1D","1W","1M","3M","6M","YTD","1Y","% 52W High"])\
	                   .format('{0:,.2f}B', subset=["Market Cap"])\
	                   .format('{0:,.2f}M', subset=["20D T/O"])\
	                   .applymap(color_positive_green, subset=["1D","1W","1M","3M","6M","YTD","1Y","% 52W High"])

if side_options =='Fixed Income':
	st.subheader('Sovereign 10-Year Bond Yield Movements')
	st.write('Note: Yield change is in local currency and is denoted in basis points (bps). Source: Bloomberg.')
	bondidx = pd.read_excel("EQ-FX.xlsx", engine='openpyxl', sheet_name='Bonds', header=0, index_col=0)
	bondidx = bondidx.sort_values(by='1D', ascending=False).style.format('{0:,.2f}%', subset=['Yield'])\
					 .format('{0:,.1f}', subset=['1D', '1W','1M','3M','YTD', '6M', '1Y'])\
					.applymap(color_positive_green, subset=['1D', '1W','1M','3M','YTD', '6M', '1Y'])\
					#.background_gradient(cmap='RdYlGn', subset=['Chg L Cy(%)', 'Chg USD(%)', '$1W(%)','1M (%)', '$YTD(%)'])
	print(st.dataframe(bondidx, height=500))
	st.markdown(get_table_download_link(bondidx), unsafe_allow_html=True)


	st.subheader('Bond ETF Performance')
	fi_country = st.selectbox('Country: ', fi_cntry, key='fi_cnt_pivot')
	fi_currency = st.selectbox('Currency: ', fi_cur, key='fi_cur_pivot')
	if fi_currency !="All":
		fi_category = st.selectbox('Category: ', fi_cats1, key='fi_pivot')
	else:
		fi_category = st.selectbox('Category: ', fi_cats, key='fi_pivot')

	print(st.dataframe(fi_filter(category=fi_category, country=fi_country, currency=fi_currency), height=700))

