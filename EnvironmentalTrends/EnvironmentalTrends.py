# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 10:39:17 2023

@author: KurtV
"""

import numpy as np
import pandas as pd
from scipy import stats

df = pd.DataFrame({'Season':[1,2,3,1,2,3], 'Results':[2,2,1,2,5,10]})

def kruskal_wallis_test(df,
                        results_column,
                        season_column):
    '''
    A function that runs the Kruskal-Wallis seasonality test on a dataset.

    Parameters
    ----------
    df : DataFrame
        A Dataframe that contains a column with results and a column indicating
        the season for each result.
    results_column : string
        The column name for the column containing the results with a numeric
        data type.
    season_column : string
        The column name for the column indicating the season for each result

    Returns
    -------
    Series
        A Series that contains:
            -the p-value result from the Kruskal-Wallis function
            -the resulting seasonality (using a 0.05 cutoff)
            -a comment that explains why particular datasets may not have a seasonality result

    '''
    
    # Set default outputs
    KW_p = np.nan
    KW_seasonality = 'N/A'
    KW_comment = ''
    
    # Determine the unique seasons
    seasons = df[season_column].unique()
    
    # Check that there are multiple seasons
    if len(seasons) < 2:
        KW_comment = f'At least 2 seasons are required. Seasons: {seasons}'
    # Check that there are at least 2 distinct results
    elif len(df[results_column].unique()) < 2:
        KW_comment = f'All results are identical. Result values: {df[results_column].unique()}'
    
    else:
        # Perform the Kruskal-Wallis test
        KW = stats.kruskal(*[df[df[season_column]==i][results_column] for i in seasons])
        # Record the p-value
        KW_p = KW.pvalue
        # Using 0.05 as the cutoff, determine seasonality
        if KW_p <= 0.05:
            KW_seasonality = 'Seasonal'
        elif KW_p > 0.05:
            KW_seasonality = 'Non-seasonal'
        else:
            KW_seasonality = 'N/A'
            KW_comment = f'Unexpected KW p-value: {KW_p}'
    
    return pd.Series([KW_p,KW_seasonality,KW_comment],
                     index=['KW-pValue','Seasonality','SeasonalityComment'])

def mann_kendall(df,
                 results_column):
    '''
    A function that calculates the Mann-Kendall S-statistic and variance
    for a dataset

    Parameters
    ----------
    df : DataFrame
        A DataFrame containing the results to be analysed in a Mann-Kendall analysis
    results_column : string
        The column name for the column containing the results

    Returns
    -------
    Series
        A Series that contains:
            - the Mann-Kendall S-statistic result
            - the variance of the S-statistic

    '''
    
    # Convert Series to numpy array
    data = df[results_column].to_numpy()
    
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
                          results_column,
                          season_column):
    '''
    A function that analyses data within each season using a Mann-Kendall test
    and combines the results into an overall S-statistic and variance result

    Parameters
    ----------
    df : DataFrame
        A DataFrame with columns indicating the season and a column indicating
        the result
    results_column : string
        The column name for the column containing the results
    season_column : string
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
    output = df.groupby(season_column).apply(mann_kendall, results_column).sum()
    
    return output

def trend_direction(df,
                    results_column,
                    season_column = None,
                    confidence_categories = {0.67:'Likely', 0.9:'Very likely'},
                    neutral_category = 'Indeterminate'):
    
    # Determine whether to use seasonal or non-seasonal test based on
    # whether a season_column is provided.
    if season_column:
        method = 'Seasonal'
        s, var = mann_kendall_seasonal(df,results_column,season_column)
    else:
        method = 'Non-seasonal'
        s, var = mann_kendall(df,results_column)
    
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
        if cutoff >= 1.0 or cutoff <= 0.5:
            raise ValueError(f'A cutoff value of {cutoff} was included. Cutoffs'
                             'must be between 0.5 and 1.0.')
        if max(C, 1-C) >= cutoff:
            trend = category
            break
    if C > 0.5:
        trend += ' increasing'
    else:
        trend += ' decreasing'
    
    return method, p, C, trend