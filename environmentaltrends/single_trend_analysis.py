'''
This module includes functions for analysing a single trend within a DataFrame

'''

import pandas as pd
import numpy as np
import censoredsummarystats as css
from scipy import stats

#%% Trend analysis for an individual dataset

def trend_analysis(df, tdf, seasons_per_year, seasonal_test=None):
    
    # Copy the DataFrame
    df = df.copy()
    
    # Initialise an empty output series
    output = []
    
    # Describe the dataset depending on whether results are censored or numeric
    if tdf.censored_values:
        # Create censored dataframe object
        cdf = css.CensoredData(df, tdf.value_col, **tdf.censored_kwargs)
        # Describe data
        output.append(describe_censored_data(tdf, cdf))
        # Convert censored data to have common censorship thresholds
        df = prep_censored_data(tdf, cdf)
    else:
        output.append(describe_data(tdf, df))
        
    # Describe the season counts of the dataset
    output.append(season_counts(tdf, df, seasons_per_year))
    
    # Replace value col with numeric col, if censored values
    if tdf.censored_values:
        df[tdf.value_col] = df[cdf.numeric_col]
    
    # Sort by date
    df = df.sort_values(by=[tdf.trend_year_col, tdf.trend_month_col])
    
    # Perform a Seasonality test
    kw_output = kruskal_wallis_test(df, tdf)
    seasonality = kw_output[tdf.kw_seasonality_col]
    output.append(kw_output)
    
    # Determine seasonal test from seasonality test if seasonal_test is None
    if seasonal_test == None:
        if seasonality == 'Seasonal':
            seasonal_test = True
        else:
            seasonal_test = False
    
    # Perform a trend direction analysis
    output.append(trend_direction(df, tdf, seasonal_test))
    
    # Perform a trend magnitude analysis
    output.append(trend_magnitude(df, tdf, seasons_per_year, seasonal_test))
    
    return pd.concat(output)

def describe_data(tdf, df):
    
    # Determine summary stats of the numeric data
    count = len(df)
    min_value = df[tdf.value_col].min()
    median_value = df[tdf.value_col].median()
    average_value = df[tdf.value_col].mean()
    max_value = df[tdf.value_col].max()
    
    return pd.Series([count,min_value,median_value,average_value,max_value],
                     index=[tdf.count_col,
                            tdf.minimum_col,
                            tdf.median_col,
                            tdf.average_col,
                            tdf.maximum_col])

def describe_censored_data(tdf, cdf):
    
    # Isolate dataframe
    df = cdf.data
    
    # Determine summary stats of the numeric data
    count = len(df)
    # Uncensored data stats
    non_censored = df[df[cdf.censor_col]==''][cdf.numeric_col]
    min_value = non_censored.min()
    max_value = non_censored.max()
    # Left censored data stats
    left_censored = df[df[cdf.censor_col].isin(['<','≤'])][cdf.numeric_col]
    count_left_censored = left_censored.count()
    min_detect_limit = left_censored.min()
    max_detect_limit = left_censored.max()
    # Right censored data stats
    right_censored = df[df[cdf.censor_col].isin(['≥','>'])][cdf.numeric_col]
    count_right_censored = right_censored.count()
    min_quant_limit = right_censored.min()
    max_quant_limit = right_censored.max()
    
    return pd.Series([count,min_value,max_value,
                      count_left_censored,min_detect_limit,max_detect_limit,
                      count_right_censored,min_quant_limit,max_quant_limit],
                     index=[tdf.count_col,
                            tdf.minimum_col,
                            tdf.maximum_col,
                            tdf.left_censored_count_col,
                            tdf.left_censored_min_col,
                            tdf.left_censored_max_col,
                            tdf.right_censored_count_col,
                            tdf.right_censored_min_col,
                            tdf.right_censored_max_col])

def prep_censored_data(tdf, cdf):
    
    # Isolate dataframe
    df = cdf.data
    # Create True/False indicator for uncensored data
    non_censored_check = (df[cdf.censor_col] == '')
    # Create True/False indicator for left censored data
    left_censored_check = df[cdf.censor_col].isin(['<','≤'])
    # Determine the maximum detection limit
    max_detect_limit = df[left_censored_check][cdf.numeric_col].max()
    # Create True/False indicator for right censored data
    right_censored_check = df[cdf.censor_col].isin(['≥','>'])
    # Determine the minimum quantification limit
    min_quant_limit = df[right_censored_check][cdf.numeric_col].min()
    
    # Convert all right censored data and all uncensored data that is greater
    # than the minimum quanitification limit to be equal to one
    # another and slightly larger than the limit
    if sum(right_censored_check) > 0:
        df[cdf.numeric_col] = np.where((right_censored_check) |
            (non_censored_check & (df[cdf.numeric_col] > min_quant_limit)),
            tdf.upper_conversion_factor*min_quant_limit,
            df[cdf.numeric_col])
    
    # Convert all left censored data and all uncensored data that is less
    # than the largest detection limit to be equal to one
    # another and less than the detection limit
    if sum(left_censored_check) > 0:
        df[cdf.numeric_col] = np.where((left_censored_check) | \
            (non_censored_check & (df[cdf.numeric_col] < max_detect_limit)),
            tdf.lower_conversion_factor*max_detect_limit,
            df[cdf.numeric_col])
    
    return df

def season_counts(tdf, df, seasons_per_year):
    
    # Determine trend_length and seasons per year
    trend_length = df[tdf.trend_len_col].iloc[0]
    
    years = len(df.groupby([tdf.trend_year_col]))
    seasons = len(df.groupby([tdf.trend_year_col,tdf.trend_season_col]))
    percent_years = 100 * years / trend_length
    percent_seasons = 100 * seasons / (seasons_per_year * trend_length)
    
    return pd.Series([years,seasons,percent_years,percent_seasons],
                     index=[tdf.years_in_trend_col,
                            tdf.seasons_in_trend_col,
                            tdf.percent_of_years_col,
                            tdf.percent_of_seasons_col])

#%% Seasonality Test

def kruskal_wallis_test(df, tdf):
    
    # Set default outputs
    KW_p = np.nan
    KW_seasonality = 'N/A'
        
    # Determine the unique seasons
    seasons = df[tdf.trend_season_col].unique()
    
    # Use default values if there are not multiple seasons
    if len(seasons) < 2:
        pass
    # If all values are identical, consider the result non-seasonal
    elif len(df[tdf.value_col].unique()) < 2:
        KW_p = 1.0
        KW_seasonality = 'Non-seasonal'
    else:
        # Perform the Kruskal-Wallis test
        KW = stats.kruskal(*[df[df[tdf.trend_season_col]==i][tdf.value_col]
                                                         for i in seasons])
        # Record the p-value
        KW_p = KW.pvalue
        # Using 0.05 as the cutoff, determine seasonality
        if KW_p <= tdf.seasonality_alpha:
            KW_seasonality = 'Seasonal'
        elif KW_p > tdf.seasonality_alpha:
            KW_seasonality = 'Non-seasonal'
    
    return pd.Series([KW_p,KW_seasonality],
                     index=[tdf.kw_pvalue_col,tdf.kw_seasonality_col])

#%% Trend Direction

def mann_kendall(df, tdf):
    
    # Convert Series to numpy array
    data = df[tdf.value_col].to_numpy()
    
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
                     index=[tdf.mk_svalue_col,tdf.mk_variance_col])

def mann_kendall_seasonal(df, tdf):
    
    # Group by season and sum each seasons S and variance to get
    # an overall S and variance
    output = (df.groupby(tdf.trend_season_col)
                  .apply(mann_kendall, tdf)
                  .sum())
    
    return output

def trend_direction(df, tdf, seasonal_test):
    
    # Determine whether to use seasonal or non-seasonal test based on
    # whether a season_col is provided.
    if seasonal_test:
        method = 'Seasonal'
        s, var = mann_kendall_seasonal(df, tdf)
    else:
        method = 'Non-seasonal'
        s, var = mann_kendall(df, tdf)
    
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
    trend = tdf.neutral_category
    
    # Sort confidence categories from largest to smallest
    confidence_categories = dict(sorted(tdf.confidence_categories.items(),
                                        reverse=True))
    
    # Convert confidence to a trend category
    for cutoff, category in confidence_categories.items():
        # Check cutoff values
        if cutoff >= 1.0 or cutoff <= 0.5:
            raise ValueError(f'A cutoff value of {cutoff} was included. '
                             'Cutoffs must be between 0.5 and 1.0.')
        if max(C, 1-C) >= cutoff:
            trend = category
            if C > 0.5:
                trend += ' increasing'
            else:
                trend += ' decreasing'
            break
    
    return pd.Series([method, s, var, p, 100*C, trend],
                     index=[tdf.applied_seasonality_col,
                            tdf.mk_svalue_col,
                            tdf.mk_variance_col,
                            tdf.mk_pvalue_col,
                            tdf.confidence_col,
                            tdf.trend_category_col])

#%% Trend Magnitude

def _slopes(df, tdf, seasons_per_year):
    
    # Copy the DataFrame
    df = df.copy()
    
    # Determine the decimal value of the year for use in slopes
    df['__DecimalYear__'] = df[tdf.trend_year_col]
    # Add partial year based on season
    df['__DecimalYear__'] += df[tdf.trend_season_col] / (seasons_per_year)
    
    # Set y values and x values
    y = np.array(df[tdf.value_col])
    x = np.array(df['__DecimalYear__'])

    # Compute sorted slopes only when deltax > 0
    deltax = x[:, np.newaxis] - x
    deltay = y[:, np.newaxis] - y
    slopes = deltay[deltax > 0] / deltax[deltax > 0]
    
    return slopes

def slopes_seasonal(df, tdf):
    
    # Determine the unique seasons
    seasons = df[tdf.trend_season_col].unique()
    
    slopes_seasonal = np.concatenate([
        _slopes(df[df[tdf.trend_season_col]==season], tdf, seasons_per_year=1)
                                                        for season in seasons])
    
    return slopes_seasonal


def trend_magnitude(df,
                    tdf,
                    seasons_per_year,
                    seasonal_test):
    
    # Determine whether to use seasonal or non-seasonal set of slopes based on
    # whether a season_col is provided.
    if seasonal_test:
        slopes = slopes_seasonal(df, tdf)
    else:
        slopes = _slopes(df, tdf, seasons_per_year)
    
    # Calculate median slope
    median_slope = np.nan
    if len(slopes) != 0:
        median_slope = np.median(slopes)
    
    # Calculate confidence interval
    
    # Initialise bounds of confidence interval
    lower_slope = np.nan
    upper_slope = np.nan
    
    # Calculate the percentiles of the confidence interval
    percentile_lower = (100 - tdf.confidence_interval)/2
    percentile_upper = 100 - percentile_lower
        
    # Map MfE method names to numpy method names
    mfe_to_numpy_percentile = {
        'weiball':'weibull',
        'tukey':'median_unbiased',
        'blom':'normal_unbiased',
        'hazen':'hazen',
        'excel':'linear'
        }
    
    if len(slopes) != 0:
        # Calculate percentiles for lower/upper bound of confidence interval
        lower_slope = np.percentile(slopes, percentile_lower,
                        method=mfe_to_numpy_percentile[tdf.percentile_method])
        
        upper_slope = np.percentile(slopes, percentile_upper,
                        method=mfe_to_numpy_percentile[tdf.percentile_method])        
    
    return pd.Series([median_slope,lower_slope,upper_slope],
                     index=[tdf.median_slope_col,
                            tdf.lower_slope_col,
                            tdf.upper_slope_col])