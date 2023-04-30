# Import numpy

import numpy as np
import pandas as pd
from datetime import datetime
from scipy import stats

#%% Current water year (1 July to 30 June)

def current_water_year():
    '''    
    Obtains the current water year. The 2020 water year is 1 July 2019 through 30 June 2020.
    
    Returns
    -------
    current_water_year : Integer
        The current water year       
    '''
    
    current_month = datetime.now().month
    
    if current_month < 7:
        current_water_year = datetime.now().year
    else:
        current_water_year = datetime.now().year + 1
        
    return current_water_year

#%% Seasonality Test

def kruskal_wallis_test(df,
                        results_col,
                        season_col=None,
                        alpha=0.05):
    '''
    A function that runs the Kruskal-Wallis seasonality test on a dataset.

    Parameters
    ----------
    df : DataFrame
        A Dataframe that contains a column with results and a column indicating
        the season for each result.
    results_col : string
        The column name for the column containing the results with a numeric
        data type.
    season_col : string
        The column name for the column indicating the season for each result
        The default is None.
    alpha : float
        The threshold p-value for seasonal vs. non-seasonal.
        The default is 0.05.

    Returns
    -------
    Series
        A Series that contains:
            -the p-value result from the Kruskal-Wallis function
            -the resulting seasonality (using a 0.05 cutoff)
            -a comment that explains why particular datasets may not have a seasonality result

    '''
    
    # Copy the DataFrame
    df = df.copy()
    
    # Set default outputs
    KW_p = np.nan
    KW_seasonality = 'N/A'
    
    # Use default outputs if no seasonal column
    if season_col:
        # Convert results column to float
        df[results_col] = df[results_col].astype(float)
        
        # Determine the unique seasons
        seasons = df[season_col].unique()
        
        # Use default values if there are not multiple seasons
        if len(seasons) < 2:
            pass
        # If all values are identical, consider the result non-seasonal
        elif len(df[results_col].unique()) < 2:
            KW_p = 1.0
            KW_seasonality = 'Non-seasonal'
        else:
            # Perform the Kruskal-Wallis test
            KW = stats.kruskal(*[df[df[season_col]==i][results_col] for i in seasons])
            # Record the p-value
            KW_p = KW.pvalue
            # Using 0.05 as the cutoff, determine seasonality
            if KW_p <= alpha:
                KW_seasonality = 'Seasonal'
            elif KW_p > alpha:
                KW_seasonality = 'Non-seasonal'
    
    return pd.Series([KW_p,KW_seasonality],
                     index=['KW-pValue','Seasonality'])

#%% Trend Direction

def mann_kendall(df,
                 results_col):
    '''
    A function that calculates the Mann-Kendall S-statistic and variance
    for a dataset that has already been sorted by time

    Parameters
    ----------
    df : DataFrame
        A DataFrame containing the results to be analysed in a Mann-Kendall analysis
    results_col : string
        The column name for the column containing the results with a numeric
        data type.

    Returns
    -------
    Series
        A Series that contains:
            - the Mann-Kendall S-statistic result
            - the variance of the S-statistic

    '''
    
    # Copy the DataFrame
    df = df.copy()
    
    # Convert results column to float
    df[results_col] = df[results_col].astype(float)
    
    # Convert Series to numpy array
    data = df[results_col].to_numpy()
    
    # Determine length of dataset
    n = data.size
    
    # Create array/matrix of increases (+1), decreases (-1), and ties (0)
    # to determine the Mann-Kendall S-statistic
    s = np.sign(np.subtract.outer(data, data))
    # Sum the upper triangle of the matrix
    s = np.triu(s).sum()
    
    # Determine the unique values and number of multiple occurences(ties)
    counts = np.unique(data, return_counts=True)[1]
    
    # Use the number of repeated values to adjust the variance calculation
    if len(counts) == n:
        tt = 0
    else:
        counts = counts[counts>1]
        tt = np.sum(counts * (counts-1) * (2 * counts + 5))
    var = (n*(n-1)*(2*n+5)-tt)/18.
    
    return pd.Series([s,var],
                     index=['MK-S','MK-Var'])

def mann_kendall_seasonal(df,
                          results_col,
                          season_col):
    '''
    A function that analyses data within each season using a Mann-Kendall test
    and combines the results into an overall S-statistic and variance result

    Parameters
    ----------
    df : DataFrame
        A DataFrame with columns indicating the season and a column indicating
        the result
    results_col : string
        The column name for the column containing the results with a numeric
        data type.
    season_col : string
        The column name for the column indicating the season

    Returns
    -------
    output : Series
        A Series that contains:
            - the overall Mann-Kendall S-statistic result
            - the overall variance of the S-statistic

    '''
    
    # Group by season and sum each seasons S and variance to get
    # an overall S and variance
    output = df.groupby(season_col).apply(mann_kendall, results_col).sum()
    
    return output

def trend_direction(df,
                    results_col,
                    seasonal_test = False,
                    season_col = None,
                    confidence_categories = {0.90:'Very likely', 0.67:'Likely'},
                    neutral_category = 'Indeterminate'):
    '''
    A function that applies a Mann-Kendall trend test, determines a continuous
    scale confidence that the trend is increasing and categorises the result

    Parameters
    ----------
    df : DataFrame
        The DataFrame that contains the results to be analysed.
    results_col : string
        The column name for the column containing results with a numeric
        data type.
    seasonal_test : boolean, optional
        Set as True to perform a seasonal Mann-Kendall test
        The default is False.
    season_col : string, optional
        The optional column name for a column indicating the season.
        The default is None.
    confidence_categories : dictionary with decimal keys and string values, optional
        A dictionary that creates the symmetrical boundaries for confidence
        categories and the names that should be used for those categories.
        Only the upper thresholds need defined so values should be between 0.5
        and 1.0. Lower thresholds will be made symmetric, where 'increasing' 
        and 'decreasing' will be added to the categories that are not the
        neutral category. Trends between the smallest cutoff and 1 minus that
        cutoff will be assigned to the neutral category.
        The default is {0.90:'Very likely', 0.67:'Likely'}.
    neutral_category : string, optional
        The name to be used for the category where the trend is as likely
        increasing as it is decreasing.
        The default is 'Indeterminate'.

    Raises
    ------
    ValueError
        An error raised if the confidence category dictionary includes a cutoff
        outside the range of 0.5 to 1.0.

    Returns
    -------
    Series
        A Series that contains:
            - the trend method (seasonal or non-seasonal)
            - the Mann-Kendall S-statistic result
            - the variance of the S-statistic
            - the Mann-Kendall p-value
            - the confidence of an increasing scale ranging from 0% - 100%
            - the trend category

    '''
    
    # Determine whether to use seasonal or non-seasonal test based on
    # whether a season_col is provided.
    if seasonal_test:
        method = 'Seasonal'
        s, var = mann_kendall_seasonal(df,results_col,season_col)
    else:
        method = 'Non-seasonal'
        s, var = mann_kendall(df,results_col)
    
    # Calculate the Z-score using the S-statistic and the variance
    z = s - np.sign(s)
    if z != 0:
        z = z / np.sqrt(var)
    
    # Calculate the p-value from the Z-score
    # where the p-value is determined for a two-tailed test
    p = stats.norm.cdf(z)
    p = 2*min(p, 1-p)
    
    # Convert p-value to confidence in the trend direction
    C = 1 - 0.5*p
    
    # Convert confidence to a continuous scale of confidence that the trend is increasing
    if s > 0:
        C = 1 - C
    
    # Set default trend category
    trend = neutral_category
    
    # Sort confidence categories from largest to smallest
    confidence_categories = dict(sorted(confidence_categories.items(), reverse=True))
    
    # Convert confidence to a trend category
    for cutoff, category in confidence_categories.items():
        # Check cutoff values
        if cutoff >= 1.0 or cutoff <= 0.5:
            raise ValueError(f'A cutoff value of {cutoff} was included. Cutoffs'
                             'must be between 0.5 and 1.0.')
        if max(C, 1-C) >= cutoff:
            trend = category
            if C > 0.5:
                trend += ' increasing'
            else:
                trend += ' decreasing'
            break
    
    return pd.Series([method,s,var,p,round(C*100,2),trend],
                     index=['DirectionMethod','MK-S','MK-Var','MK-pValue','IncreasingConfidence%','TrendCategory'])

#%% Trend Magnitude

def slopes(df,
           results_col,
           year_col,
           season_col=None,
           seasons_per_year=1):
    '''
    A function that determines all combinations of slopes that will be used in
    a Sen slope analysis. The data must be of either annual, quarterly, or
    monthly frequency.

    Parameters
    ----------
    df : DataFrame
        A DataFrame containing a numeric results and time information
    results_col : string
        The column name for the column containing numeric results
    year_col : string
        The column name for the column containing the year
    season_col : string
        The column name for a column with an integer representation of the season.
    seasons_per_year : integer
        The frequency of results to be analysed for a Sen slope.
        The default is 1 season or annual data.

    Raises
    ------
    Exception
        Raised if the frequency is not one of the accepted strings or if there
        are multiple results taken within a season

    Returns
    -------
    slopes : list of floats
        A list of slopes that can be used to determine the Sen slope.

    '''
    
    # Copy the DataFrame
    df = df.copy()
    
    # Convert year column to integer
    df[year_col] = df[year_col].astype(int)
    
    # Convert results column to float
    df[results_col] = df[results_col].astype(float)
    
    # Check that seasons_per_year is a factor of 12
    if seasons_per_year not in [1,2,3,4,6,12]:
        raise ValueError('seasons_per_year must be a factor of 12')
    
    # Determine the decimal value of the year for use in slopes
    df['__DecimalYear__'] = df[year_col]
    if season_col:
        # Convert season column to integer
        df[season_col] = df[season_col].astype(int)
        # Add partial year based on season
        df['__DecimalYear__'] += df[season_col] / (seasons_per_year)
    
    # Check that only a single result occurs for each year
    if not df['__DecimalYear__'].is_unique:
        raise Exception('Only one result can be supplied within each season.')
    
    # Set y values and x values
    y = np.array(df[results_col])
    x = np.array(df['__DecimalYear__'])

    # Compute sorted slopes only when deltax > 0
    deltax = x[:, np.newaxis] - x
    deltay = y[:, np.newaxis] - y
    slopes = deltay[deltax > 0] / deltax[deltax > 0]
    
    return slopes

def slopes_seasonal(df,
                    results_col,
                    year_col,
                    season_col):
    '''
    A function that determines all combinations of slopes within each season
    that will be used in a Sen slope analysis. The data must be of either
    annual, quarterly, or monthly frequency.

    Parameters
    ----------
    df : DataFrame
        A DataFrame containing a numeric results and time information
    results_col : string
        The column name for the column containing numeric results
    year_col : string
        The column name for the column containing the year
    season_col : string
        The column name for the column indicating the season

    Returns
    -------
    slopes : list of floats
        A list of slopes that can be used to determine the Sen slope.

    '''
    # Determine the unique seasons
    seasons = df[season_col].unique()
    
    slopes_seasonal = np.concatenate([slopes(df[df[season_col]==season],
                                    results_col,
                                    year_col,
                                    seasons_per_year=1) for season in seasons])
    
    return slopes_seasonal


def trend_magnitude(df,
                    results_col,
                    year_col,
                    seasonal_test = False,
                    season_col=None,
                    seasons_per_year=1,
                    percentile_method='hazen',
                    confidence_interval=90):
    '''
    A function that determines the Sen slope for a dataset along with upper and
    lower bound estimates for within a specific confidence interval.

    Parameters
    ----------
    df : DataFrame
        A DataFrame containing a numeric results and time information
    results_col : string
        The column name for the column containing numeric results
    year_col : string
        The column name for the column containing the year
    seasonal_test : boolean, optional
        Set as True to perform a seasonal Sen-slope analysis
        The default is False.
    season_col : string, optional
        The column name for the column indicating the season.
        The default is None.
    seasons_per_year : integer
        The frequency of results to be analysed for a Sen slope.
        The default is 1 season or annual data.
    percentile_method : string, optional
        The percentile method. The options and definitions come from:
            https://environment.govt.nz/assets/Publications/Files/hazen-percentile-calculator-2.xls
            Options include the following, ordered from largest to smallest result.
            - weiball
            - tukey
            - blom
            - hazen
            - excel
        The default is hazen.
    confidence_interval : integer or float, optional
        The size of the confidence interval used to determine the lower and
        upper bounds of the Sen slope. A 90 percent confidence interval
        uses the 5th percentile for the lower bound and 95th percentile for
        the upper bound.
        The default is 90.

    Returns
    -------
    Series
        A Series that contains:
            - the Sen slope method (seasonal or non-seasonal)
            - the resulting Sen slope (median of all slopes)
            - the lower bound of the confidence interval
            - the upper bound of the confidence interval
    '''
    
    # Copy the DataFrame
    df = df.copy()
    
    # Determine whether to use seasonal or non-seasonal set of slopes based on
    # whether a season_col is provided.
    if seasonal_test:
        method = 'Seasonal'
        slopes = slopes_seasonal(df,results_col,year_col,season_col)
    else:
        method = 'Non-seasonal'
        slopes = slopes(df,results_col,year_col,season_col,seasons_per_year)
    
    # Calculate median slope
    median_slope = np.nan
    if len(slopes) != 0:
        median_slope = np.median(slopes)
    
    # Calculate confidence interval
    
    # Initialise bounds of confidence interval
    lower_slope = np.nan
    upper_slope = np.nan
    
    # Calculate the percentiles of the confidence interval
    percentile_lower = (100 - confidence_interval)/2
    percentile_upper = 100 - percentile_lower
    
    # Set values for percentile methods
    # https://environment.govt.nz/assets/Publications/Files/hazen-percentile-calculator-2.xls
    method_dict = {'weiball':0.0, 'tukey':1/3, 'blom':3/8, 'hazen':1/2, 'excel':1.0}
    # https://en.wikipedia.org/wiki/Percentile
    C = method_dict[percentile_method]
    
    # Calculate minimum data size for percentile method to 
    # ensure rank is at least 1 and no more than len(data)
    # Note the same size is required for both the upper and lower percentile
    minimum_size = round(C + (1-C)*(1-percentile_lower)/percentile_lower, 10)
    
    # Confidence interval can only be determined if there are enough slopes
    # for the percentile method used to generate the interval
    if len(slopes) >= minimum_size:
        
        # Map MfE method names to numpy method names
        mfe_to_numpy_percentile = {
            'weiball':'weibull',
            'tukey':'median_unbiased',
            'blom':'normal_unbiased',
            'hazen':'hazen',
            'excel':'linear'
            }
        
        # Calculate percentiles for lower and upper bound of confidence interval
        lower_slope = np.percentile(slopes,
                                    percentile_lower,
                                    method=mfe_to_numpy_percentile[percentile_method])
        
        upper_slope = np.percentile(slopes,
                                    percentile_upper,
                                    method=mfe_to_numpy_percentile[percentile_method])        
    
    return pd.Series([method,median_slope,lower_slope,upper_slope],
                     index=['SlopeMethod','SenSlope','LowerBound','UpperBound'])

