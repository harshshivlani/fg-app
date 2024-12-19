import numpy as np
import pandas as pd
import streamlit as st
import work as work
import plotly.express as px
import plotly.graph_objects as go
import plotly
from plotly.subplots import make_subplots
import ipywidgets as widgets
from ipywidgets import interact, interact_manual
from IPython.core.display import display, HTML
import streamlit.components.v1 as components
import datetime as dt
import edhec_risk_kit as erk
from mftool import Mftool
from operator import itemgetter
mf = Mftool()


st.write("""
### KAAML Discretionary Solutions: Analytics Toolkit
""")
st.sidebar.header('Contents')
side_options = st.sidebar.radio('Please Select One:', ('Equity Factors', 'Mutual Funds'))
st.sidebar.markdown(''' :blue[Developed by Harsh Shivlani]''')


#Import Master Data
@st.cache_data()
def load_indices_data():
       """
       Downloads historical TRI data for all major benchmark, sector and factor indices in India via NIFTY Indices website

       """
       indices_list = ['NIFTY NEXT 50', 'NIFTY 100', 'NIFTY 200', 'NIFTY 500', 'NIFTY MIDCAP 150', 'NIFTY SMALLCAP 250',
                       'NIFTY200 MOMENTUM 30', 'NIFTY500 MOMENTUM 50', 'NIFTY100 ALPHA 30', 'NIFTY200 ALPHA 30', 'NIFTY ALPHA 50', 
                       'NIFTY MIDCAP150 MOMENTUM 50',
                       'NIFTY HIGH BETA 50', 'NIFTY200 VALUE 30', 'NIFTY500 VALUE 50', 
                       'NIFTY100 LOW VOLATILITY 30', 'NIFTY LOW VOLATILITY 50', 'NIFTY200 QUALITY 30', 'NIFTY MIDCAP150 QUALITY 50', 
                       'NIFTY QUALITY LOW-VOLATILITY 30']
       
       sectors_list = ['NIFTY AUTO', 'NIFTY BANK', 'NIFTY FINANCIAL SERVICES', 'NIFTY FINANCIAL SERVICES EX BANK', 'NIFTY FMCG', 'NIFTY HEALTHCARE',
                       'NIFTY IT', 'NIFTY MEDIA', 'NIFTY METAL', 'NIFTY PHARMA', 'NIFTY PRIVATE BANK', 'NIFTY PSU BANK', 'NIFTY REALTY', 'NIFTY CONSUMER DURABLES',
                       'NIFTY INDIA DEFENCE', 'NIFTY PSE', 'NIFTY COMMODITIES']
       
       #data = pd.DataFrame(index=pd.date_range("1/1/2005", str(dt.date.today())))
       #data.index.name = 'Date'
       data = work.get_nse_index("NIFTY 50", "TRI", "01-Jan-1990", dt.date.today().strftime("%d-%b-%Y"))

       for i in indices_list:
              indices = work.get_nse_index(i, "TRI", "01-Jan-1990", dt.date.today().strftime("%d-%b-%Y")) 
              data = data.join(indices, on='Date')

       return data


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


indices_data = load_indices_data()


def performance_table(df, factors="Yes"):
       """
       """
       st.write(f"#### Periodic Return of Factor & Benchmark Indices")
       st.caption("Total Return Index (TRI) performance snapshot of multiple factors and benchmarks")

       perf_as_of = st.date_input("Performance as of:", df.index[-1])
       df = df.copy()[:perf_as_of]
       
       def px_offset(months):
              return df.loc[df.index <= (df.index[-1] - pd.DateOffset(months=months))].iloc[-1]
       
       last_px = df.iloc[-1]

       cytd = df.loc[df[:str(df.index[-1].year-1)].index[-1]]
       weeks1 = df.loc[df.index <= (df.index[-1] - pd.DateOffset(weeks=1))].iloc[-1]
       weeks2 = df.loc[df.index <= (df.index[-1] - pd.DateOffset(weeks=2))].iloc[-1]

       perf = pd.concat([df.pct_change(1).iloc[-1,:], last_px/weeks1-1, last_px/weeks2-1, 
                         last_px/px_offset(1)-1, last_px/px_offset(2)-1, last_px/px_offset(3)-1, 
                         last_px/px_offset(6)-1, last_px/cytd-1, last_px/px_offset(12)-1,
                        (last_px/px_offset(24))**(1/2)-1, (last_px/px_offset(36))**(1/3)-1,
                         (last_px/px_offset(60))**(1/5)-1, (last_px/px_offset(120))**(1/10)-1], axis=1)

       perf.columns=['1D', '1W', '2W', '1M', '2M', '3M', '6M', 'YTD', '1Y', '2Y', '3Y', '5Y', '10Y']
       if factors=="Yes":
              perf.insert(0, "Category", ["Benchmark"] * 6 + ["Factors"] * (len(df.columns)-6))
              perf = perf.style.format('{0:,.2%}', subset=perf.columns[1:]).applymap(color_positive_green, subset=perf.columns[1:])
       else:
              perf = perf.style.format('{0:,.2%}').applymap(color_positive_green)

       return st.dataframe(perf)


def equity_curve(df, category='Factors'):
       """
       """
       st.write(f"#### Normalized Performance Graph")
       st.caption("Compare performance of selected securities over a specific period")
       indices = list(df.columns)

       if category == 'Factors':
              selected_indices = st.multiselect("Select securities to display", indices, default=indices[:1], key='equity_curve')
       else:
              selected_indices = list(df.columns)
       col1, col2 = st.columns(2)
       with col1:
              start_date = st.date_input('Start:', dt.date(2023,12,31))
       with col2:
              end_date = st.date_input('End:', dt.date.today())
       
       eq_curve = (1+df[selected_indices][start_date:end_date].dropna().pct_change().fillna(0)).cumprod()*100

       
       fig = px.line(
        eq_curve,
        x=eq_curve.index,
        y=eq_curve.columns,
        labels={"value": "Value", "index": "Date", "variable": "Indices"})
       
       fig.add_hline(y=100, line_dash="dot", line_color="red", line_width=2)
       fig.update_layout(hovermode="x unified")
       # Display Plot
       st.plotly_chart(fig, use_container_width=True)


def rolling_cagr_plots(df):
       """
       """
       st.write(f"#### Rolling Returns of Factor Indices")
       st.caption("Analyse annualized rolling returns of multiple factors and benchmarks")
       
       col1, col2 = st.columns(2)
       with col1:
              yrs = st.number_input('Select Period (yrs):', 1, 10 , value=3)
       indices = list(df.columns)
       
       with col2:
              selected_indices = st.multiselect("Select indices to display", indices, default=indices[:1])
       roll_cagr = ((1+df[selected_indices].pct_change(252*yrs))**(1/yrs)-1).dropna()

       
       fig = px.line(
        roll_cagr,
        x=roll_cagr.index,
        y=roll_cagr.columns,
        title= str(yrs)+"-Year Rolling CAGR for Selected Indices",
        labels={"value": "Rolling CAGR", "index": "Date", "variable": "Indices"})
       
       fig.add_hline(y=0, line_dash="dot", line_color="red", line_width=2)
       fig.update_yaxes(tickformat=".0%")

       fig.update_layout(hovermode="x unified")
       # Display Plot
       st.plotly_chart(fig, use_container_width=True)


def factor_cagr_spread(df):
       """
       """
       st.write(f"#### Return Spreads: Factor Indices")
       st.caption("Analyse rolling return spreads for mean reversion between factors. Choose from actual return spreads and Z-Score of spreads")

       indices = list(df.columns)

       col1, col2, col3 = st.columns(3)
       with col1:
              yrs = st.number_input('Select Period (yrs):', 1,10)
       with col2:
              index1 = st.selectbox("1st index", indices, index=None)
       with col3:
              index2 = st.selectbox("2nd index", indices, index=None)

       col4, col5 = st.columns(2)
       with col4:
              data_type = st.pills("Select Analysis Type:", ['Spread', 'Z-Score'], selection_mode="single")
       
       if index1:
              if index2:
                     roll_cagr = ((1+df[[index1, index2]].pct_change(252*yrs))**(1/yrs)-1).dropna()
                     spread = pd.DataFrame((roll_cagr.iloc[:,0]-roll_cagr.iloc[:,1]).dropna())
                     spread.columns = [index1+ '-' +index2]

                     if data_type == 'Z-Score':
                            with col5:
                                   zscore_lb = st.number_input("Z-Score Lookback (yrs)", 1, 10)
                            zscore = pd.DataFrame((spread-spread.rolling(252*zscore_lb).mean())/spread.rolling(252*zscore_lb).std(ddof=1)).dropna()
                            zscore.columns = [index1+ '-' +index2]
                            fig = px.area(zscore, x=zscore.index, y=zscore.columns,
                                   title= str(yrs)+"-Year Rolling CAGR Spread: Z-Score (" + str(zscore_lb)+" Yr)",
                                   labels={"value": "Z-Score", "index": "Date", "variable": "Indices"})
                            fig.update_layout(hovermode="x unified")
                            return st.plotly_chart(fig, use_container_width=True)
                     
                     elif data_type == 'Spread':
                            fig = px.area(spread, x=spread.index, y=spread.columns,
                                   title= str(yrs)+"-Year Rolling CAGR Spread of Selected Indices",
                                   labels={"value": "Rolling CAGR", "index": "Date", "variable": "Indices"})
                            fig.update_yaxes(tickformat=".0%")
                            fig.update_layout(hovermode="x unified")
                            return st.plotly_chart(fig, use_container_width=True)       
       else:
              st.write("Please select an index to display the graph.")


def factor_correlations(df):
       """
       """
       st.write(f"#### Time-Varying Factor Correlations")
       st.caption("Analyse time-varying excess return correlations with five major factors: Momentum, Value, Low Volatility, Quality and High Beta. Upload Fund NAVs (with dates and NAVs in 2 columns) for analysis or select an any index.")

       col1, col2, col3 = st.columns(3)
       with col1:
              period = st.number_input("Lookback (months)", 1, 60, value=12)
       
       four_factors = ['NIFTY200 MOMENTUM 30 TRI', 'NIFTY100 LOW VOLATILITY 30 TRI', 'NIFTY200 QUALITY 30 TRI',
                       'NIFTY200 VALUE 30 TRI', 'NIFTY HIGH BETA 50 TRI']
       with col2:
              benchmark = st.selectbox("Benchmark", ['NIFTY 50 TRI', 'NIFTY 200 TRI', 'NIFTY 500 TRI'], index=1)
       with col3:
              upload_question = st.selectbox("Upload File?:", ['No', 'Yes'], index=0)
       
       if upload_question =='No':
              y_variable = st.selectbox("Fund/Index:", df.columns, index=0)
       else:
              uploaded_file = st.file_uploader("Upload Fund NAV Excel File:")
              fund_navs = pd.read_excel(uploaded_file, header=0, index_col=0)
              fund_navs.index.name = 'Date'
              df = df.merge(fund_navs, on='Date')
              y_variable = fund_navs.columns[0]


       rets = df.pct_change(period*21)
       excess_rets = (rets.drop(benchmark, axis=1).T - rets[benchmark]).T       

       correl = (excess_rets[four_factors].rolling(period*21).corr(excess_rets[y_variable])).dropna()

       fig = px.line(correl, x=correl.index, y=correl.columns,
                                   title= str(period)+"-Month Rolling Excess Return Correlation of " +str(y_variable),
                                   labels={"value": "Correlation", "index": "Date", "variable": "Indices"})
       fig.update_yaxes(tickformat=".0%")
       fig.update_layout(hovermode="x unified")
       fig.add_hline(y=0, line_dash="dot", line_color="red", line_width=2)
       return st.plotly_chart(fig, use_container_width=True)


#Get MF Scheme names and codes:
scheme_codes = mf.get_scheme_codes()
#Swap keys with values to filter scheme code from scheme name
scheme_codes = {v: k for k, v in scheme_codes.items()}

#Get a list of all mutual fund schemes names
scheme_names = mf.get_scheme_codes()
scheme_names = list(scheme_names.values())[1:]


@st.cache_data()
def get_mf_hist_navs(fund_names):
    """
    Gets historical NAVs for a list of selected mutual funds by scraping data from AMFI
    """

    #fetch scheme codes for selected scheme names
    fund_codes = itemgetter(*fund_names)(scheme_codes)
    fund_codes = list(fund_codes) if isinstance(fund_codes, tuple) else [fund_codes]

    #Get historical NAVs for first scheme
    hist_navs = mf.get_scheme_historical_nav(fund_codes[0],as_Dataframe=True)[['nav']]
    hist_navs.columns = [fund_names[0]]

    if len(fund_names)>1:
        for i in range(1, len(fund_codes)):
            fund_nav = mf.get_scheme_historical_nav(fund_codes[i],as_Dataframe=True)[['nav']]
            fund_nav.columns = [fund_names[i]]
            hist_navs = hist_navs.merge(fund_nav, on='date')
    
    hist_navs.index = pd.to_datetime(hist_navs.index, format='%d-%m-%Y')
    hist_navs[hist_navs.select_dtypes(include='object').columns] = hist_navs.select_dtypes(include='object').apply(pd.to_numeric, errors='coerce')
    hist_navs.index.name = 'Date'
    hist_navs.sort_values(by='Date', inplace=True)
    
    return hist_navs



if side_options == 'Equity Factors':
       st.caption('Data as of '+ str(indices_data.index[-1].strftime('%d %b %Y')) + ' (EOD)')

       performance_table(indices_data)

       equity_curve(indices_data)

       rolling_cagr_plots(indices_data)

       factor_cagr_spread(indices_data)

       factor_correlations(indices_data)


if side_options == 'Mutual Funds':

       selected_mfs = st.multiselect("Select list of Mutual Funds", scheme_names, default=None, key='select_mfs')
       mf_benchmark = st.selectbox("Select Benchmark", list(indices_data.columns), index=None, key='mf_benchmark')

       def mf_performance_table(df):
              """
              """
              st.write(f"#### Performance of Selected Mutual Funds")

              perf_as_of = st.date_input("Performance as of:", df.index[-1])
              df = df.copy()[:perf_as_of]

              def px_offset(months):
                     #Check if data exists for the required offset period
                     target_date = df.index[-1] - pd.DateOffset(months=months)
                     valid_data = df.loc[df.index <= target_date]
                     return valid_data.iloc[-1] if not valid_data.empty else None

              #Offsets for specific periods (with handling for missing data)
              def safe_offset(offset_func):
                     result = offset_func()
                     return result if result is not None else pd.Series([pd.NA] * len(df.columns), index=df.columns)

              last_px = df.iloc[-1]

              #Precompute necessary offsets
              cytd = safe_offset(lambda: df.loc[df[:str(df.index[-1].year-1)].index[-1]])
              weeks1 = safe_offset(lambda: df.loc[df.index <= (df.index[-1] - pd.DateOffset(weeks=1))].iloc[-1])
              weeks2 = safe_offset(lambda: df.loc[df.index <= (df.index[-1] - pd.DateOffset(weeks=2))].iloc[-1])

              #Offsets for months and years
              offsets = {
                      "1M": safe_offset(lambda: px_offset(1)),
                      "2M": safe_offset(lambda: px_offset(2)),
                      "3M": safe_offset(lambda: px_offset(3)),
                      "6M": safe_offset(lambda: px_offset(6)),
                      "1Y": safe_offset(lambda: px_offset(12)),
                      "2Y": safe_offset(lambda: px_offset(24)),
                      "3Y": safe_offset(lambda: px_offset(36)),
                      "5Y": safe_offset(lambda: px_offset(60)),
                      "10Y": safe_offset(lambda: px_offset(120))
                  }


               # Compute returns with checks for None values
              def calculate_return(offset_value, periods=1):
                     if offset_value is not None:
                            return (last_px / offset_value) ** (1 / periods) - 1
                     else:
                            return pd.Series([pd.NA] * len(df.columns), index=df.columns)

              perf = pd.concat([
                      df.pct_change(1).iloc[-1, :],                      # 1D
                      last_px / weeks1 - 1,                              # 1W
                      last_px / weeks2 - 1,                              # 2W
                      calculate_return(offsets["1M"]),                   # 1M
                      calculate_return(offsets["2M"]),                   # 2M
                      calculate_return(offsets["3M"]),                   # 3M
                      calculate_return(offsets["6M"]),                   # 6M
                      last_px / cytd - 1,                                # YTD
                      calculate_return(offsets["1Y"]),                   # 1Y
                      calculate_return(offsets["2Y"], 2),                # 2Y
                      calculate_return(offsets["3Y"], 3),                # 3Y
                      calculate_return(offsets["5Y"], 5),                # 5Y
                      calculate_return(offsets["10Y"], 10)               # 10Y
                  ], axis=1)


              perf.columns=['1D', '1W', '2W', '1M', '2M', '3M', '6M', 'YTD', '1Y', '2Y', '3Y', '5Y', '10Y']
              perf = perf.dropna(axis=1).style.format('{0:,.2%}').applymap(color_positive_green)

              return st.dataframe(perf)

       def annual_mf_performance(df):
              """
              """
              st.caption("Calendar-Year Performance Data:")
              cy_data = work.rebase_timeframe(df, 'Annual')
              perf = cy_data.pct_change().dropna().T
              perf.columns = perf.columns.year
              perf.rename(columns={perf.columns[-1]: str(perf.columns[-1])+str(' TD')}, inplace=True)
              perf = perf.style.format('{0:,.2%}').applymap(color_positive_green)
              return st.dataframe(perf)

       def mf_factor_correlations(factor_df):
              """
              Requires inputs of factor index data (factor_df) and a single mutual fund NAV data (mf_df) 
              """
              st.write(f"#### Time-Varying Factor Correlations")
              st.caption("Analyse time-varying excess return correlations with five major factors: Momentum, Value, Low Volatility, Quality and High Beta")

              col1, col2 = st.columns(2)
              with col1:
                     period = st.number_input("Lookback (months)", 1, 60, value=12, key='mf_factor_correlations')
              
              four_factors = ['NIFTY200 MOMENTUM 30 TRI', 'NIFTY100 LOW VOLATILITY 30 TRI', 'NIFTY200 QUALITY 30 TRI',
                              'NIFTY200 VALUE 30 TRI', 'NIFTY HIGH BETA 50 TRI']
              with col2:
                     benchmark = st.selectbox("Benchmark", ['NIFTY 50 TRI', 'NIFTY 200 TRI', 'NIFTY 500 TRI'], index=1)

              selected_mf = st.selectbox("Select a Mutual Fund:", scheme_names, index=None, key='select_mf_correl') 
              
              #Check if a mutual fund was selected
              if selected_mf is None:
                     st.write('Please select a mutual fund to display results.')
                     return              

              fund_nav = get_mf_hist_navs([selected_mf])
              fund_nav.index.name = 'Date'
              factor_df = factor_df.merge(fund_nav, on='Date')
              y_variable = fund_nav.columns[0]


              rets = factor_df.pct_change(period*21)
              excess_rets = (rets.drop(benchmark, axis=1).T - rets[benchmark]).T       

              correl = (excess_rets[four_factors].rolling(period*21).corr(excess_rets[y_variable])).dropna()

              fig = px.line(correl, x=correl.index, y=correl.columns,
                                          title= str(period)+"-Month Rolling Excess Return Correlation of " +str(y_variable),
                                          labels={"value": "Correlation", "index": "Date", "variable": "Indices"})
              fig.update_yaxes(tickformat=".0%")
              fig.update_layout(hovermode="x unified")
              fig.add_hline(y=0, line_dash="dot", line_color="red", line_width=2)
              return st.plotly_chart(fig, use_container_width=True)

       if st.checkbox('Submit'):
              my_navs = get_mf_hist_navs(selected_mfs)
              my_navs = my_navs.copy().merge(indices_data[mf_benchmark], on='Date')

              mf_performance_table(my_navs)
              annual_mf_performance(my_navs)
              equity_curve(my_navs, 'Mutual Funds')
              st.write(f'#### Risk/Reward Summary Statistics')
              st.caption('Analyse risk/reward metrics like Sharpe, Sortino, IR, Max Drawdowns, etc. Risk-free rate is assumed to 6.5%.')
              st.dataframe(erk.summary_stats(my_navs.pct_change().dropna(), my_navs.iloc[:,-1].pct_change().dropna(), 0.065, 252).drop([str(my_navs.columns[-1])+'_x',
                                                                                                                                     str(my_navs.columns[-1])+'_y'],
                                                                                                                                     axis=0).fillna("").replace("nan%", ""))
              mf_factor_correlations(indices_data)







