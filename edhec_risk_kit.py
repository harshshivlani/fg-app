import pandas as pd

def drawdown(return_series: pd.Series):
    """
    What does this function do?
    It takes as input a TimeSeries data of Asset Returns
    What does it output?
    It outputs a pandas DataFrame which includes,
    
    1. A Wealth or Normalized Index with base of $1000
    2. Previous Peaks of Wealth Index
    3. Percentage Drawdowns
    """
    wealth_index = 1000*(1+return_series).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index - previous_peaks)/previous_peaks
    return pd.DataFrame({ "Wealth Index": wealth_index,
                          "Previous Peaks": previous_peaks,
                          "Drawdowns": drawdowns
                         })  

#Adding a convenience code i.e. last time we had to convert the series, select only hi & low and so on, let's build function that does just that
def get_ffme_returns():
    """
    Returns the Fama French US Stock Market Returns Dataset for the Top & Bottom Deciles i.e. "Lo 10" & "Hi 10"
    """
    me_m = pd.read_csv("C:/Users/Harsh/Documents/Portfolio Construction Python EDHEC Coursera/Files/data/Portfolios_Formed_on_ME_monthly_EW.csv",
                           header=0, index_col=0, parse_dates=True, na_values=-99.99
                         )  #Importing the previous file on US Stock Return Data based on Market Cap in Deciles & Quintiles
    rets = me_m[["Lo 10","Hi 10"]]  #Selecting just the lowest decile and the highest decline (i.e. SmallCap & LargeCap)
    rets.columns = ["SmallCap","LargeCap"] #Renaming the columns
    rets = rets/100 #Dividing by 100 for returns
    rets.index = pd.to_datetime(rets.index, format = "%Y%m").to_period("M")
    return rets

def get_hfi_returns():
    """
    Returns the EDHEC Hedge Fund Indices Dataset
    """
    hfi = pd.read_csv("C:/Users/Harsh/Documents/Portfolio Construction Python EDHEC Coursera/Files/data/edhec-hedgefundindices.csv",
                           header=0, index_col=0, parse_dates=True, na_values=-99.99
                         )  #Importing the Hedge fund dataset
    
    hfi = hfi/100 #Dividing by 100 for returns
    hfi.index = hfi.index.to_period("M")
    return hfi

def get_fff_returns():
    """
    Imports the Fama-French Model 'monthly' data for various Factor Returns from 1926 onwards 
    """
    rets = pd.read_csv("C:/Users/Harsh/Documents/Portfolio Construction Python EDHEC Coursera/Files/data/F-F_Research_Data_Factors_m.csv", index_col=0, header=0, na_values=-99.99)/100
    rets.index = pd.to_datetime(rets.index, format="%Y%m").to_period('M')
    return rets

def get_ind_file(filetype, weighting="vw", n_inds=30):
    """
    Load and format the Ken French Industry Portfolios files
    Variant is a tuple of (weighting, size) where:
        weighting is one of "ew", "vw"
        number of inds is 30 or 49
    """    
    if filetype is "returns":
        name = f"{weighting}_rets" 
        divisor = 100
    elif filetype is "nfirms":
        name = "nfirms"
        divisor = 1
    elif filetype is "size":
        name = "size"
        divisor = 1
    else:
        raise ValueError(f"filetype must be one of: returns, nfirms, size")
    
    ind = pd.read_csv(f"data/ind{n_inds}_m_{name}.csv", header=0, index_col=0, na_values=-99.99)/divisor
    ind.index = pd.to_datetime(ind.index, format="%Y%m").to_period('M')
    ind.columns = ind.columns.str.strip()
    return ind

def get_ind_returns(weighting="vw", n_inds=30):
    """
    Load and format the Ken French Industry Portfolios Monthly Returns
    """
    return get_ind_file("returns", weighting=weighting, n_inds=n_inds)

def get_ind_nfirms(n_inds=30):
    """
    Load and format the Ken French 30 Industry Portfolios Average number of Firms
    """
    return get_ind_file("nfirms", n_inds=n_inds)

def get_ind_size(n_inds=30):
    """
    Load and format the Ken French 30 Industry Portfolios Average size (market cap)
    """
    return get_ind_file("size", n_inds=n_inds)


def get_ind_market_caps(n_inds=30, weights=False):
    """
    Load the industry portfolio data and derive the market caps
    """
    ind_nfirms = get_ind_nfirms(n_inds=n_inds)
    ind_size = get_ind_size(n_inds=n_inds)
    ind_mktcap = ind_nfirms * ind_size
    if weights:
        total_mktcap = ind_mktcap.sum(axis=1)
        ind_capweight = ind_mktcap.divide(total_mktcap, axis="rows")
        return ind_capweight
    #else
    return ind_mktcap

def get_total_market_index_returns(n_inds=30):
    """
    Load the 30 industry portfolio data and derive the returns of a capweighted total market index
    """
    ind_capweight = get_ind_market_caps(n_inds=n_inds)
    ind_return = get_ind_returns(weighting="vw", n_inds=n_inds)
    total_market_return = (ind_capweight * ind_return).sum(axis="columns")
    return total_market_return


def compound(r):
    """
    Returns the result of compounding the set of returns in r
    """
    return np.expm1(np.log1p(r).sum())

def semideviation(r, periods_per_year):
    """
    Returns Downside Deviation i.e. deviation of returns that are negative
    """
    return r[r<0].std(ddof=0)*(periods_per_year**0.5)


def skewness(r):
    """
    Returns Skewness for a given a Series or DataFrame
    Shortcut Alternative is to directly use scipy.stats.skew()
    Returns a float or a series
    """
    demeaned_r = r - r.mean()
    sigma_r = r.std(ddof=0) #Degree of freedom is 0 so returns population standard deviation
    exp = (demeaned_r**3).mean()
    return exp/sigma_r**3
    
def kurtosis(r):
    """
    Returns Kurtosis for a given a Series or DataFrame
    Shortcut Alternative is to directly use scipy.stats.kurtosis()
    Returns a float or a series
    """
    demeaned_r = r - r.mean()
    sigma_r = r.std(ddof=0)       #Degree of freedom is 0 so returns population standard deviation
    exp = (demeaned_r**4).mean()
    return exp/sigma_r**4
    

def is_normal(r, level=0.01): #Says that it takes r as input and a level of confidence ( if not given, the default assumed is 1%)
    
    """
    Applies Jarque Bera test from Scipy to check the normality of the given returns dataset 
    at 1% level of confidence by default.
    Returns True if Hypothesis of Normality is as Accepted and False if it is not.
    """
    import scipy.stats
    statistic, p_value = scipy.stats.jarque_bera(r)  #You are assigning the names to the tuple output that you will get by running the scipy code
    return p_value > level   #Returns True if p_value > 1% or given level and False if not


def var_historic(r, level=5):
    """
    Returns the historic Value at Risk(VaR) for a specified level (alpha)
    i.e. returns such that if level =5 then 5% of the time returns may be at or lower than the resulted return
    """
    import numpy as np
    
    if isinstance(r, pd.DataFrame):                     #Checks if the return series given is a Dataframe
        return r.aggregate(var_historic, level=level)   #If it's a Dataframe it applies the check on each column then
    elif isinstance(r, pd.Series):                      #If not a dataframe, checks if it is a series
        return -np.percentile(r, level) #If a series, then applies the VaR formula via numpy, negative sign since we always report VaR as a positive number
    else:
        raise TypeError("Expected r to be a Series or a DataFrame")  #If none of the above then data is not in correct format so raise an error
        
from scipy.stats import norm
def gaussian_var(r, level=5, modified=False):
    """
    Returns the Guassian VaR for a given returns Series or DataFrame at a specified level
    If Modified = True, then returns the Modified Semi-Parametric VaR, as given
    by Cornish-Fisher Modification
    """
    #Compute Z-Score assuming it was Gaussian
    
    Z = norm.ppf(level/100)             #PPF is Percentage Point Function, specify the probabiltiy in point terms, returns the z-score
    if modified:
        #modify the Z-score based on Skewness(S) & Kurtosis(K)
        s = skewness(r)
        k = kurtosis(r)
        Z = (Z +
                 (Z**2-1)*s/6 +
                 (Z**3-3*Z)*(k-3)/24 -
                 (2*Z**3 - 5*Z)*(s**2)/36
            )
    
    return -(r.mean() + Z*r.std(ddof=0))   #Mean + Z*Vol = VaR for normal distribution, if S=0,K=3, only z will be left or it will be modified if S,K are different from what a normal distributio has, Negative Sign for positive reporting of VaR

def cvar_historic(r, level=5):
    """
    Returns the historic Conditional Value at Risk(VaR) for a specified level (alpha)
    i.e. returns such that if level =5 then 5% of the time  average returns will be the result
    """ 
    if isinstance(r, pd.Series):                          #Checks if the return series given is a series
        is_beyond = r <= -var_historic(r, level=level) #checks if return is less than VaR
        return -r[is_beyond].mean()                     #Reports the mean of all returns less than (is_beyond) historic_var
    elif isinstance(r, pd.DataFrame):                     #If not a series, checks if it is a dataframe
        return r.aggregate(cvar_historic, level=level)
    else:
        raise TypeError("Expected r to be a Series or a DataFrame")  #If none of the above then data is not in correct format so raise an error

def annualized_ret(r, periods_per_year):
    """
    Annualizes the return (CAGR) for any given Series or DataFrame 
    """
    compounded_growth = (1+r).prod() #ending value of portfolio by compounding returns
    no_of_periods= r.shape[0] #no of rows in the dataset 'r'
    return compounded_growth**(periods_per_year/no_of_periods)-1

def annualized_vol(r, periods_per_year):
    """
    Annualizes the volatility of a set of returns
    Periods per year is 12 for monthly data, 252 for daily and so on
    """
    return r.std()*(periods_per_year**0.5)

def sharpe_ratio(r, periods_per_year, rf):
    """
    Calucates annualized sharpe ratio for a return series
    rf is your risk free rate
    """
    rf_per_period = (1+rf)**(1/periods_per_year)-1
    excess_return = r - rf_per_period
    ann_excess_ret = annualized_ret(excess_return, periods_per_year)
    ann_vol = annualized_vol(r, periods_per_year)
    return ann_excess_ret/ann_vol

def sortino_ratio(r, periods_per_year, rf):
    rf_per_period = (1+rf)**(1/periods_per_year)-1
    excess_return = r - rf_per_period
    ann_excess_ret = annualized_ret(excess_return, periods_per_year)
    ann_vol = semideviation(r, periods_per_year)
    return ann_excess_ret/ann_vol
    
#Lets Define Portfolio Returns & Volatility for a given weight matrix
def portfolio_returns(weights, returns):
    """
    Returns portfolio return for a given weight matrix
    """
    return weights.T @ returns #Weights transpose multiplied by a returns matrix

def portfolio_vol(weights, covmat):
    """
    Returns portfolio volatility for a given set of weights and covariances (matrix) of the assets in the portfolio
    """
    vol= (weights.T @ covmat @ weights)**0.5
    return vol                                 #Square root because this results the variance so to get standard deviation

def plot_ef2(n_points, er, cov, style=".-",color="goldenrod"):
    """
    Plots a 2-Asset Efficient Frontier
    """
    import numpy as np
    import edhec_risk_kit as erk
    if er.shape[0] != 2 or er.shape[0] !=2: #If your columns i.e. assets are not equal to 2 then raise an error as min and max requirement is 2 assets
        raise ValueError("plot_ef2 function can only plot 2 Asset Efficient Frontiers")
    weights = [np.array([w, 1-w]) for w in np.linspace(0, 1, n_points)]  #Creates a list of weights, each set of weights is an array hence the np.array
#but the weights are between 0 and 1 that are linearly spaced that equal distance from each other, so such "X" points specified by n_points here
    rets = [erk.portfolio_returns(w, er) for w in weights]
    vols = [erk.portfolio_vol(w, cov) for w in weights]
    eff_frontier = pd.DataFrame({"Returns": rets,
                                "Volatility": vols
                                })
    return eff_frontier.plot.line(x="Volatility", y="Returns", style=style, color=color)




def minimize_vol(target_return, er, cov):
    """
    Gives minimum volatility portfolio weights for a given level of expected return for n-asset portfolio
    """
    import numpy as np
    import edhec_risk_kit as erk
    from scipy.optimize import minimize
    
    n = er.shape[0] #no of assets, since er row headers will be the number of assets
    init_guess = np.repeat(1/n, n) #Initial guess of what the weights should  be for a target return on portfolio, Equally weighted portfolio.
    #Repeat function inputs, 1. The number to repeated 
    #                        2. No. of times you want to repeat the number
    
    #A tuple of min and max weights that is 0% and 100% for each asset, hence multiplied by n. Try running this in console for better understanding.
    bounds = ((0.0, 1.0),)*n     
    
    #Adding Constraints
    return_met = {
        'type':'eq',      #constraint type is equality
        'args': (er,),    #extra argument required is er
        'fun': lambda weights, er: target_return - erk.portfolio_returns(weights, er)
    }
    
    weights_sum_to_1 = {
        'type':'eq',
        'fun': lambda weights: np.sum(weights)-1
    }
    
    #Calling the scipy.optimize function of "Minimize", now that we have all the inputs i.e. Function, Constraints, Bounds
    results = minimize(erk.portfolio_vol, init_guess,
                       args= (cov,), method = "SLSQP",   #Args means additional arguments required i.e. covariance matrix here for the portfolio vol function above #"SLSQP" is the method name for "Quadratic Optimization"
                       bounds=bounds,                 #Bounds define minimum and maximum weights as we defined above
                       constraints=(return_met, weights_sum_to_1),
                       options={'disp':False}
                      )
    return results.x
                       
    
def optimal_weights(n_points, er, cov):
    """
    List of weights to run the optimizer on, to minimize the volatility
    """
    import numpy as np
    import pandas as pd
    import edhec_risk_kit as erk
    target_rets = np.linspace(er.min(), er.max(), n_points) #Our minimize vol function from scipy optimize will give optimal weights if we provide target returns,
    #so this func gives the target returns ranging from lowest to highest individual asset returns and linearly spaces it
    weights = [erk.minimize_vol(target_return, er, cov) for target_return in target_rets]
    return weights
     

#Maximum Sharpe Ratio Portfolio Weights Generation Code    
def msr(riskfree_rate, er, cov):
    """
    Gives the sharpe ratio portfolio returns & weights for a multi-asset portfolio
    """
    import numpy as np
    import edhec_risk_kit as erk
    from scipy.optimize import minimize
    
    n = er.shape[0] #no of assets, since er row headers will be the number of assets
    init_guess = np.repeat(1/n, n) #Initial guess of what the weights should  be for a target return on portfolio, Equally weighted portfolio.
    #Repeat function inputs, 1. The number to repeated 
    #                        2. No. of times you want to repeat the number
    
    #A tuple of min and max weights that is 0% and 100% for each asset, hence multiplied by n. Try running this in console for better understanding.
    bounds = ((0.0, 1.0),)*n     
    
    #Adding Constraints
    
    weights_sum_to_1 = {
        'type':'eq',
        'fun': lambda weights: np.sum(weights)-1
    }
    
    #Defining the negative sharpe ratio that we want to minimize i.e. maximize the sharpe ratio
    def neg_sharpe_ratio(weights, riskfree_rate, er, cov):
        """
        Returns the negative of the sharpe ratio given the weights
        """
        r = portfolio_returns(weights, er)
        vol = portfolio_vol(weights, cov)
        return -(r - riskfree_rate)/vol
    
    #Calling the scipy.optimize function of "Minimize", now that we have all the inputs i.e. Function, Constraints, Bounds
    results = minimize(neg_sharpe_ratio, init_guess,
                       args= (riskfree_rate, er, cov), method = "SLSQP",   #Args means additional arguments required i.e. covariance matrix here for the portfolio vol function above #"SLSQP" is the method name for "Quadratic Optimization"
                       bounds=bounds,                 #Bounds define minimum and maximum weights as we defined above
                       constraints=(weights_sum_to_1,),
                       options={'disp':False}
                      )
    return results.x


def gmv(cov):
    """
    Global Minimum Volatility Portfolio Weights (GMV): Returns the portfolio weights that minimizes the portfolio volatility for a given covariance matrix
    """
    import numpy as np
    n = cov.shape[0]
    return msr(0, np.repeat(1, n), cov) #You assume mean returns to be 1 and the same for all assets in the matrix and maximize sharpe so the only way to improve is by reducing volatility, that's what we want, global minimum volatility portfolio weights


#Efficient Frontier Plotting Code given weights, expected returns & covariance matrix

def plot_ef(n_points, er, cov, show_cml=False, riskfree_rate=0, show_ew=False, show_gmv=False, style=".-",color="goldenrod"):
    """
    Plots a Multi-Asset Efficient Frontier with the Capital Market Line if needed
    """
    import numpy as np
    import pandas as pd
    import edhec_risk_kit as erk
    weights = optimal_weights(n_points, er, cov)
    rets = [portfolio_returns(w, er) for w in weights]
    vols = [portfolio_vol(w, cov) for w in weights]
    eff_frontier = pd.DataFrame({"Returns": rets,
                                "Volatility": vols
                                })
    ax = eff_frontier.plot.line(x="Volatility", y="Returns", style=style, color=color)
    if show_ew:
        n = er.shape[0]
        w_ew = np.repeat(1/n, n)
        r_ew = portfolio_returns(w_ew, er)
        vol_ew = portfolio_vol(w_ew, cov)
        #Display EW - Equally Weighted Portfolio's Return & Volatility on the chart
        ax.plot([vol_ew], [r_ew], color="purple", marker="o", markersize=12)      

    if show_gmv:                                               #Global Minimum Volatility Portfolio Weights & Plot on Chart: Only a function of volatility
        n = er.shape[0]
        w_gmv = gmv(cov)
        r_gmv = portfolio_returns(w_gmv, er)
        vol_gmv = portfolio_vol(w_gmv, cov)
        #Display Global Minimum Volatility Portfolio (GMV)
        ax.plot([vol_gmv], [r_gmv], color="midnightblue", marker="o", markersize=10)      
    
    if show_cml:
        ax.set_xlim(left = 0)
        w_msr = msr(riskfree_rate, er, cov)                 #Generates the weights of the Max Sharpe Ratio Portfolio
        r_msr = portfolio_returns(w_msr, er) #Gives the Y-axis (Return) point of the Max Sharpe Ratio Portfolio
        vol_msr = portfolio_vol(w_msr, cov)  #Gives the X-axis (Volatility) point of the Max Sharpe Ratio Portfolio
        #Add CML
        cml_x = [0, vol_msr]  #X-Axis is Volatility so it's a combination of 0% Vol for Rf and the MSR Portfolio Vol depending on their weights
        cml_y = [riskfree_rate, r_msr]   #Y-Axis is Returns so it's a combination of Rf Rate for Rf and the MSR Portfolio Return depending on their weights
        ax.plot(cml_x, cml_y, color="green", marker ="o", linestyle="dashed", markersize=12, linewidth=2)
        
    return ax  

def run_cppi(risky_r, safe_r=None, m=3, start=1000, floor=0.8, riskfree_rate=0.03, drawdown=None):
    """
    
    """
    import numpy as np
    #Set up the CPPI Parameters
    dates = risky_r.index              #Creating and index of dates
    n_steps = len(dates)               #Length of date gives you the total number i.e. 24 for 2 years worth of monthly data and so on
    account_value = start              #Initial start value of $1000 assumed
    floor_value  = start*floor         #Hence, intital floor value will be $800 assuming floor of 0.8
    peak = start
    
    if isinstance(risky_r, pd.Series):
        risky_r = pd.DataFrame(risky_r, columns=["R"])
        
    if safe_r is None:
        safe_r = pd.DataFrame().reindex_like(risky_r)
        safe_r.values[:] = riskfree_rate/12           #Fast way to set all the values to a number


    #Some data points we might want to look at constantly across time after the backtest 
    account_history  = pd.DataFrame().reindex_like(risky_r)     #Portfolio value over time
    cushion_history  = pd.DataFrame().reindex_like(risky_r)     #History of cushion changing over time 
    risky_w_history  = pd.DataFrame().reindex_like(risky_r)     #History of weight allocated to risky asset over time
    floor_history    = pd.DataFrame().reindex_like(risky_r)
    for step in range(n_steps):
        if drawdown is not None:
            peak = np.maximum(peak, account_value)
            floor_value=peak*(1-drawdown)
        cushion = (account_value - floor_value)/account_value
        risky_w = m*cushion
        risky_w = np.minimum(risky_w, 1)                        #Max Allocation of 100% to Risky Asset
        risky_w = np.maximum(risky_w, 0)                        #Minimum Allocation of 0% and not negative to Risky Asset
        safe_w = 1-risky_w
        risky_alloc = account_value*risky_w                     #Intial Amount Allocated to Risky Asset
        safe_alloc = account_value*safe_w                       #Intial Amount Allocated to Safe Asset
    
        #Update the Account Value for this time step
        account_value = risky_alloc*(1+risky_r.iloc[step]) + safe_alloc*(1+safe_r.iloc[step])
    
        #Save the values so we can look at the history and plot it
        cushion_history.iloc[step] = cushion
        risky_w_history.iloc[step] = risky_w
        account_history.iloc[step] = account_value
        floor_history.iloc[step]   = floor_value
    
    risky_wealth = start*(1+risky_r).cumprod()
    backtest_result={
        "Wealth": account_history,
        "Risky Wealth": risky_wealth,
        "Risk Budget": cushion_history,
        "Risky Allocation": risky_w_history,
        "Multiplier": m,
        "Start": start,
        "Floor": floor_history,
        "Risky Return": risky_r,
        "Safe Return": safe_r
                    }
    return backtest_result


def summary_stats(r, riskfree_rate=0.03, periods_per_year=12):
    """
    Return a DataFrame that contains aggregated summary stats for the returns in the columns of r
    """
    wealth_index = (1+r).cumprod()
    total_rets = ((wealth_index.iloc[-1,:]/wealth_index.iloc[0,:]-1)*100).round(2).astype(str) + '%'
    ann_r = (r.aggregate(annualized_ret, periods_per_year=periods_per_year)*100).round(2).astype(str) + '%'
    ann_vol = (r.aggregate(annualized_vol, periods_per_year=periods_per_year)*100).round(2).astype(str) + '%'
    ann_sr = r.aggregate(sharpe_ratio, rf=riskfree_rate, periods_per_year=periods_per_year).round(2)
    sortino = r.aggregate(sortino_ratio, rf=riskfree_rate, periods_per_year=periods_per_year).round(2)
    dd = (r.aggregate(lambda r: drawdown(r).Drawdowns.min())*100).round(2).astype(str) + '%'
    skew = r.aggregate(skewness).round(2)
    kurt = r.aggregate(kurtosis).round(2)
    cf_var5 = (r.aggregate(gaussian_var, modified=True)*100).round(2).astype(str) + '%'
    hist_cvar5 = (r.aggregate(cvar_historic)*100).round(2).astype(str) + '%'
    return pd.DataFrame({
        "Total Return": total_rets,
        "Annualized Return": ann_r,
        "Annualized Vol": ann_vol,
        "Skewness": skew,
        "Kurtosis": kurt,
        "Cornish-Fisher VaR (5%)": cf_var5,
        "Historic CVaR (5%)": hist_cvar5,
        "Sharpe Ratio": ann_sr,
        "Sortino Ratio": sortino,
        "Max Drawdown": dd
    })

def summary_stats1(r, riskfree_rate=0.03, periods_per_year=12):
    """
    Return a DataFrame that contains aggregated summary stats for the returns in the columns of r
    """
    wealth_index = (1+r).cumprod()
    total_rets = ((wealth_index.iloc[-1,:]/wealth_index.iloc[0,:]-1)*100).round(2)
    ann_r = (r.aggregate(annualized_ret, periods_per_year=periods_per_year)*100).round(2)
    ann_vol = (r.aggregate(annualized_vol, periods_per_year=periods_per_year)*100).round(2)
    ann_sr = r.aggregate(sharpe_ratio, rf=riskfree_rate, periods_per_year=periods_per_year).round(2)
    dd = (r.aggregate(lambda r: drawdown(r).Drawdowns.min())*100).round(2)
    sortino = r.aggregate(sortino_ratio, rf=riskfree_rate, periods_per_year=periods_per_year).round(2)
    skew = r.aggregate(skewness).round(2)
    kurt = r.aggregate(kurtosis).round(2)
    cf_var5 = (r.aggregate(gaussian_var, modified=True)*100).round(2)
    hist_cvar5 = (r.aggregate(cvar_historic)*100).round(2)
    return pd.DataFrame({
        "Total Return": total_rets,
        "Annualized Return": ann_r,
        "Annualized Vol": ann_vol,
        "Skewness": skew,
        "Kurtosis": kurt,
        "Cornish-Fisher VaR (5%)": cf_var5,
        "Historic CVaR (5%)": hist_cvar5,
        "Sharpe Ratio": ann_sr,
        "Sortino Ratio": sortino,
        "Max Drawdown": dd
    })

#Defining a Geometric Brownian Motion(GBM) function:
import numpy as np
import pandas as pd

def gbm(n_years=10, n_scenarios=3, steps_per_year=12, mu=0.07, sigma=0.15, s_0=100.0, prices=True):
    """
    Generates the Evolution of Asset Prices using a Geometric Brownian Motion Model
    Input Paramters: No. of years
                     No. of Scenarios
                     Mean
                     Volatility
                     Steps per year (eg. 12 for monthly)
                     Initial Asset Price(S0)
    """
    dt=1/steps_per_year
    n_steps= int(n_years*steps_per_year) + 1
    # the standard way ...
    # rets_plus_1 = np.random.normal(loc=mu*dt+1, scale=sigma*np.sqrt(dt), size=(n_steps, n_scenarios))
    # without discretization error ...
    rets_plus_1 = np.random.normal(loc=(1 + mu)**dt, scale= (sigma*np.sqrt(dt)), size=(n_steps, n_scenarios))
    rets_plus_1[0] = 1                      #Making the First Random Return 0, so that first row of each scenario is the saw i.e. 100
    rets = pd.DataFrame(rets_plus_1)
    ret_val= s_0*(rets).cumprod() if prices else rets_plus_1-1
    return ret_val
  
    
#Week 4: Asset-Liability Management

def discount(t, r):
    """
    Gives the price of a pure discount bond that pays a dollar at time "t", given "r" i.e. per period interest rate
    Returns a |t| x |r| Series or DataFrame
    "r" can be a float, Series or DataFrame
    Returns a DataFrame indexed by "t"
    """
    discounts = pd.DataFrame([(1+ r)**-i for i in t])
    discounts.index = t
    return discounts

def pv(flows, r):
    """
   Computes the present value of a series of cashflows given by the time (as an index) and amounts "r" can be scalar
   or a Series or a DataFrame with the number of rows matching the number of rows in "flows"
    """
    dates = flows.index
    discounts = discount(dates, r)
    return discounts.multiply(flows, axis="rows").sum()

def funding_ratio(assets, liabilities, r):
    """
    Returns the funding ratio as assets/present value of liabilities,
    by discounting the liabilities at the annual interest of "r"
    Note: "liabilities" here is a DataFrame of all liabities indexed by time (in terms of years, such as 10yrs : 200Cr and so on),
          "assets" is just a single number i.e. current value of all your assets i.e investments
    Interpretation: A Ratio below 1 means your underfunded and above means your overfunded or you have a surplus
    """
    return pv(assets, r)/pv(liabilities, r)


#Convert short rate to annualized rate by continous compounding
def inst_to_ann(r):
    """
    Convert short rate to annualized rate by continous compounding
    """
    return np.exp(r)-1  #Same as np.exp(r)-1

def ann_to_inst(r):
    """
    Convert annualized rate to short rate
    """
    return np.log1p(r)



#Updated CIR Model that generates the zero coupon bond prices at the generated implied rates:
import math
def cir(n_years = 10, n_scenarios=1, a=0.05, b=0.03, sigma=0.05, steps_per_year=12, r_0=None):
    """
    Generate random interest rate evolution over time using the CIR model
    b and r_0 are assumed to be the annualized rates, not the short rate
    and the returned values are the annualized rates as well
    """
    if r_0 is None: r_0 = b 
    r_0 = ann_to_inst(r_0)
    dt = 1/steps_per_year
    num_steps = int(n_years*steps_per_year) + 1 # because n_years might be a float
    
    shock = np.random.normal(0, scale=np.sqrt(dt), size=(num_steps, n_scenarios))
    rates = np.empty_like(shock)
    rates[0] = r_0

    ## For Price Generation
    h = math.sqrt(a**2 + 2*sigma**2)
    prices = np.empty_like(shock)
    ####

    def price(ttm, r):
        _A = ((2*h*math.exp((h+a)*ttm/2))/(2*h+(h+a)*(math.exp(h*ttm)-1)))**(2*a*b/sigma**2)
        _B = (2*(math.exp(h*ttm)-1))/(2*h + (h+a)*(math.exp(h*ttm)-1))
        _P = _A*np.exp(-_B*r)
        return _P
    prices[0] = price(n_years, r_0)
    ####
    
    for step in range(1, num_steps):
        r_t = rates[step-1]
        d_r_t = a*(b-r_t)*dt + sigma*np.sqrt(r_t)*shock[step]
        rates[step] = abs(r_t + d_r_t)
        # generate prices at time t as well ...
        prices[step] = price(n_years-step*dt, rates[step])

    rates = pd.DataFrame(data=inst_to_ann(rates), index=range(num_steps))
    ### for prices
    prices = pd.DataFrame(data=prices, index=range(num_steps))
    ###
    return rates, prices
           
#Bond Price Calculation using the First Principles i.e. Cash Flow Discounting Method
#First we create a function to calculate bond's cashflows:

def bond_cash_flows(maturity, principal=100, coupon_rate=0.03, coupons_per_year=12):
    """
    Returns a Series of Bond Cash Flows given the maturity(in years), coupon rate, principal and frequency.
    """
    n_coupons=round(maturity*coupons_per_year)
    coupon_times = np.arange(1, n_coupons+1)                   #Creates an array ranging from 1 to to the no.of coupons + 1, since its not inclusive
    coupon_amt = principal*coupon_rate/coupons_per_year
    cash_flows = pd.Series(data=coupon_amt, index=coupon_times)  
    cash_flows.iloc[-1] += principal                      #Since you will recieve back the principal at maturity
    return cash_flows

#Since we have the cashflows we can calculate the bond price by summing all the PV of Cashflows using the PV function we just developed that takes a series of cash flows as an input with a discount rate i.e. YTM
def bond_price(maturity, principal=100, coupon_rate=0.03, coupons_per_year=12, discount_rate=0.03):
    """
    Computes the price of a bond that pays regular coupons until maturity
    at which time the principal and the final coupon is returned
    This is not designed to be efficient, rather,
    it is to illustrate the underlying principle behind bond pricing!
    If discount_rate is a DataFrame, then this is assumed to be the rate on each coupon date
    and the bond value is computed over time.
    i.e. The index of the discount_rate DataFrame is assumed to be the coupon number
    """
    if isinstance(discount_rate, pd.DataFrame):
        pricing_dates = discount_rate.index
        prices = pd.DataFrame(index=pricing_dates, columns=discount_rate.columns)
        for t in pricing_dates:
            prices.loc[t] = bond_price(maturity-t/coupons_per_year, principal, coupon_rate, coupons_per_year,
                                      discount_rate.loc[t])
        return prices
    else: # base case ... single time period
        if maturity <= 0: return principal+principal*coupon_rate/coupons_per_year
        cash_flows = bond_cash_flows(maturity, principal, coupon_rate, coupons_per_year)
        return pv(cash_flows, discount_rate/coupons_per_year)
                                     
#NOT THROUGH THE COURSE, MY OWN
def bond_duration(maturity, principal=100, coupon_rate=0.03, discount_rate=0.03, coupons_per_year=12):
    """
    Calculates the Modified Duration of a bond given the maturity, principal, coupon rate, YTM and frequency. 
    """
    up_yield = discount_rate-0.0001
    down_yield = discount_rate+0.0001
    bond_price_up = bond_price(maturity=maturity, principal=principal, coupon_rate=coupon_rate,  coupons_per_year=coupons_per_year, discount_rate=up_yield)
    bond_price_down = bond_price(maturity=maturity, principal=principal, coupon_rate=coupon_rate, coupons_per_year=coupons_per_year, discount_rate=down_yield)
    bond_price_c = bond_price(maturity=maturity, principal=principal, coupon_rate=coupon_rate, coupons_per_year=coupons_per_year, discount_rate=discount_rate)
    duration = (bond_price_up - bond_price_down)/(2*0.0001*bond_price_c)
    
    return duration


#THROUGH THE COURSE, AGAIN
def macaulay_duration(flows, discount_rate):
    """
    Computes the Macaulay Duration of a sequence of cash flows, given a per-period discount rate
    """
    discounted_flows = discount(flows.index, discount_rate)*pd.DataFrame(flows)
    weights = discounted_flows/discounted_flows.sum()
    return np.average(flows.index, weights=weights.iloc[:,0])      #Weighted Average of Discounted Cash Flows

def match_durations(cf_t, cf_s, cf_l, discount_rate):
    
    d_t = macaulay_duration(cf_t, discount_rate)
    d_s = macaulay_duration(cf_s, discount_rate)
    d_l = macaulay_duration(cf_l, discount_rate)
    return (d_t - d_l)/(d_l - d_s)

def bond_total_return(monthly_prices, principal, coupon_rate, coupons_per_year):
    """
    Computes the total return of a Bond based on monthly bond prices and coupon payments
    Assumes that dividends (coupons) are paid out at the end of the period (e.g. end of 3 months for quarterly div)
    and that dividends are reinvested in the bond
    """
    coupons = pd.DataFrame(data = 0, index=monthly_prices.index, columns=monthly_prices.columns)
    t_max = monthly_prices.index.max()
    pay_date = np.linspace(12/coupons_per_year, t_max, int(coupons_per_year*t_max/12), dtype=int)
    coupons.iloc[pay_date] = principal*coupon_rate/coupons_per_year
    total_returns = (monthly_prices + coupons)/monthly_prices.shift()-1
    return total_returns.dropna()


def bt_mix(r1, r2, allocator, **kwargs):
    """
    Runs a back test (simulation) of allocating between a two sets of returns
    r1 and r2 are T x N DataFrames or returns where T is the time step index and N is the number of scenarios.
    allocator is a function that takes two sets of returns and allocator specific parameters, and produces
    an allocation to the first portfolio (the rest of the money is invested in the GHP) as a T x 1 DataFrame
    Returns a T x N DataFrame of the resulting N portfolio scenarios
    """
    if not r1.shape == r2.shape:
        raise ValueError("r1 and r2 need to have the same shape")
    weights = allocator(r1, r2, **kwargs)
    if not weights.shape == r1.shape:
        raise ValueError("Allocator returned weights with a different shape than the returns")
    r_mix = weights*r1 + (1-weights)*r2
    return r_mix

def fixedmix_allocator(r1, r2, w1, **kwargs):
    """
    Produces a time series over T steps of allocations between the PSP and GHP across N scenarios
    PSP and GHP are T x N DataFrames that represent the returns of the PSP and GHP such that:
    each column is a scenario
    each row is the price for a timestep
    Returns an T x N DataFrame of PSP Weights
    """
    return pd.DataFrame(data=w1, index=r1.index, columns=r1.columns)

def terminal_values(rets):
    """
    Computes the terminal values from a set of returns supplied as a T x N DataFrame
    Return a Series of length N indexed by the columns of rets
    """
    return (rets+1).prod()

def terminal_stats(rets, floor=0.8, cap=np.inf, name="Stats"):
    """
    Produce Summary Statistics on the terminal values per invested dollar
    across a range of N scenarios
    rets is a T x N DataFrame of returns, where T is the time-step (we assume rets is sorted by time)
    Returns a 1 column DataFrame of Summary Stats indexed by the stat name 
    """
    terminal_wealth = (rets+1).prod()
    breach = terminal_wealth < floor
    reach = terminal_wealth >= floor
    p_breach = breach.mean() if breach.sum() > 0 else np.nan
    p_reach = breach.mean() if reach.sum() > 0 else np.nan
    e_short = (floor-terminal_wealth[breach]).mean() if breach.sum() > 0 else np.nan
    e_surplus = (cap-terminal_wealth[reach]).mean() if reach.sum() > 0 else np.nan
    sum_stats = pd.DataFrame.from_dict({
        "Mean": terminal_wealth.mean(),
        "Volatility" : terminal_wealth.std(),
        "Probability of Breach": p_breach,
        "Expected Shortfall":e_short,
        "Probability of Reach": p_reach,
        "Expected Surplus": e_surplus
    }, orient="index", columns=[name])
    return sum_stats

def glidepath_allocator(r1, r2, start_glide=1, end_glide=1):
    """
    Allocates weights to r1 starting at start_glide and ends at end_glide
    by gradually moving from start_glide to end_glide over time
    """
    n_points = r1.shape[0]
    n_cols = r1.shape[1]
    path = pd.Series(np.linspace(start_glide, end_glide, num=n_points))
    #We need N number of paths depending the number of columns of returns i.e. No of Scenarios
    paths = pd.concat([path]*n_cols, axis=1)    #Replicate path in list form [path], by multiplying it with the number of scenarios and place them side by side in columns hence axis=1
    paths.index=r1.index
    paths.columns=r1.columns
    return paths 


def floor_allocator(psp_r, ghp_r, floor, zc_prices, m=3):
    """
    Allocate between PSP and GHP with the goal to provide exposure to the upside
    of the PSP without going violating the floor.
    Uses a CPPI-style dynamic risk budgeting algorithm by investing a multiple
    of the cushion in the PSP
    Returns a DataFrame with the same shape as the psp/ghp representing the weights in the PSP
    """
    if zc_prices.shape != psp_r.shape:
        raise ValueError("PSP and ZC Prices must have the same shape")
    n_steps, n_scenarios = psp_r.shape
    account_value = np.repeat(1, n_scenarios)
    floor_value = np.repeat(1, n_scenarios)
    w_history = pd.DataFrame(index=psp_r.index, columns=psp_r.columns)
    for step in range(n_steps):
        floor_value = floor*zc_prices.iloc[step] ## PV of Floor assuming today's rates and flat YC
        cushion = (account_value - floor_value)/account_value
        psp_w = (m*cushion).clip(0, 1) # same as applying min and max
        ghp_w = 1-psp_w
        psp_alloc = account_value*psp_w
        ghp_alloc = account_value*ghp_w
        # recompute the new account value at the end of this step
        account_value = psp_alloc*(1+psp_r.iloc[step]) + ghp_alloc*(1+ghp_r.iloc[step])
        w_history.iloc[step] = psp_w
    return w_history

def drawdown_allocator(psp_r, ghp_r, maxdd, m=3):
    """
    Allocate between PSP and GHP with the goal to provide exposure to the upside
    of the PSP without going violating the floor.
    Uses a CPPI-style dynamic risk budgeting algorithm by investing a multiple
    of the cushion in the PSP
    Returns a DataFrame with the same shape as the psp/ghp representing the weights in the PSP
    """
    n_steps, n_scenarios = psp_r.shape
    account_value = np.repeat(1, n_scenarios)
    floor_value = np.repeat(1, n_scenarios)
    peak_value = np.repeat(1, n_scenarios)
    w_history = pd.DataFrame(index=psp_r.index, columns=psp_r.columns)
    for step in range(n_steps):
        floor_value = (1-maxdd)*peak_value ### Floor is based on Prev Peak
        cushion = (account_value - floor_value)/account_value
        psp_w = (m*cushion).clip(0, 1) # same as applying min and max
        ghp_w = 1-psp_w
        psp_alloc = account_value*psp_w
        ghp_alloc = account_value*ghp_w
        # recompute the new account value and prev peak at the end of this step
        account_value = psp_alloc*(1+psp_r.iloc[step]) + ghp_alloc*(1+ghp_r.iloc[step])
        peak_value = np.maximum(peak_value, account_value)
        w_history.iloc[step] = psp_w
    return w_history

##### ##### ##### ##### #####  ##### ##### ##### ##### ##### ##### COURSE 2 ###### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### 
def regress(dependent_variable, explanatory_variables, alpha=True):
    """
    Runs a linear regression to decompose the dependent variable into the explanatory variables
    returns an object of type statsmodel's RegressionResults on which you can call
       .summary() to print a full summary
       .params for the coefficients
       .tvalues and .pvalues for the significance levels
       .rsquared_adj and .rsquared for quality of fit
    """
    import statsmodels.api as sm
    if alpha:
        explanatory_variables = explanatory_variables.copy()
        explanatory_variables["Alpha"] = 1
    lm = sm.OLS(dependent_variable, explanatory_variables).fit()
    return lm

def tracking_error(r_a, r_b):
    """
    Returns the Tracking Error between the two return series
    """
    return np.sqrt(((r_a - r_b)**2).sum())

def portfolio_tracking_error(weights, ref_r, bb_r):
    """
    Returns the tracking error between the reference returns
    and a portfolio of building block returns held with given weights
    """
    return tracking_error(ref_r, (weights*bb_r).sum(axis=1))

from scipy.optimize import minimize
def style_analysis(dependant_variable, explanatory_variables):
    """
    Returns the optimal weights that minimizes the Tracking error between
    a portfolio of the explanatory variables and the dependent variable
    """
    n =  explanatory_variables.shape[1]                  #Number of Explanatory variables i.e. factors that you have
    init_guess = np.repeat(1/n, n)                       #Inital guess of weightage is equal weight to each factor 
    bounds = ((0.00, 1.00),)*n                           #Min and max weights of 0% and 100% to each factor (n) hence multipled by n to get bounds for factors
    constraint_1 = {'type': 'eq',
                    'fun': lambda weights: np.sum(weights) - 1}           #Constraint: Weights should sum to 1, 'eq' means equality that the equation = 0
    solution = minimize(portfolio_tracking_error, init_guess,
                       args=(dependant_variable, explanatory_variables,),
                       method='SLSQP',
                       bounds = bounds,
                       constraints= (constraint_1,),
                       options={'disp':False})
    weights = pd.Series(solution.x, index=explanatory_variables.columns)
    return weights

def weight_ew(r, cap_weights=None, max_cw_mult=None, microcap_threshold=None, **kwargs):
    """
    Returns the weights of the EW portfolio based on the asset returns "r" as a DataFrame
    If supplied a set of capweights and a capweight tether, it is applied and reweighted 
    """
    n = len(r.columns)
    ew = pd.Series(1/n, index=r.columns)
    if cap_weights is not None:
        cw = cap_weights.loc[r.index[0]] # starting cap weight
        ## exclude microcaps
        if microcap_threshold is not None and microcap_threshold > 0:
            microcap = cw < microcap_threshold
            ew[microcap] = 0
            ew = ew/ew.sum()
        #limit weight to a multiple of capweight
        if max_cw_mult is not None and max_cw_mult > 0:
            ew = np.minimum(ew, cw*max_cw_mult)
            ew = ew/ew.sum() #reweight
    return ew

def weight_cw(r, cap_weights, **kwargs):
    """
    Returns the weights of the CW portfolio based on the time series of capweights
    """
    w = cap_weights.loc[r.index[0]]
    return w/w.sum()

def backtest_ws(r, estimation_window=60, weighting=weight_ew, verbose=False, prices=None, **kwargs):
    """
    Backtests a given weighting scheme, given some parameters:
    r : asset returns to use to build the portfolio
    estimation_window: the window to use to estimate parameters
    weighting: the weighting scheme to use, must be a function that takes "r", and a variable number of keyword-value arguments
    """
    n_periods = r.shape[0]
    # return windows
    windows = [(start, start+estimation_window) for start in range(n_periods-estimation_window)]
    if prices is not None:
        weights = [weighting(prices.iloc[win[0]:win[1]], **kwargs) for win in windows]
    else:
        weights = [weighting(r.iloc[win[0]:win[1]], **kwargs) for win in windows]
    
    # convert List of weights to DataFrame
    weights = pd.DataFrame(weights, index=r.iloc[estimation_window:].index, columns=r.columns)
    returns = (weights * r).sum(axis="columns",  min_count=1) #mincount is to generate NAs if all inputs are NAs
    return returns

def sample_cov(r, **kwargs):
    """
    Returns the sample covariance of the supplied returns
    """
    return r.cov()

def weight_gmv(r, cov_estimator=sample_cov, **kwargs):
    """
    Produces the weights of the GMV portfolio given a covariance matrix of the returns 
    """
    est_cov = cov_estimator(r, **kwargs)
    return gmv(est_cov)

def cc_cov(r, **kwargs):
    """
    Estimates a covariance matrix by using the Elton/Gruber Constant Correlation model
    """
    rhos = r.corr()
    n = rhos.shape[0]
    # this is a symmetric matrix with diagonals all 1 - so the mean correlation is ...
    rho_bar = (rhos.values.sum()-n)/(n*(n-1))
    ccor = np.full_like(rhos, rho_bar)
    np.fill_diagonal(ccor, 1.)
    sd = r.std()
    return pd.DataFrame(ccor * np.outer(sd, sd), index=r.columns, columns=r.columns)

def shrinkage_cov(r, delta=0.5, **kwargs):
    """
    Covariance estimator that shrinks between the Sample Covariance and the Constant Correlation Estimators
    """
    prior = cc_cov(r, **kwargs)
    sample = sample_cov(r, **kwargs)
    return delta*prior + (1-delta)*sample
