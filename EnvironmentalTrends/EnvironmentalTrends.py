# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 10:39:17 2023

@author: KurtV
"""

import numpy as np
import pandas as pd
from scipy import stats
from datetime import datetime
from censoredsummarystats import result_to_components, median, average, maximum, minimum

#%% Seasonality Test

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
    
    # Copy the DataFrame
    df = df.copy()
    
    # Set default outputs
    KW_p = np.nan
    KW_seasonality = 'N/A'
    KW_comment = ''
    
    # Convert results column to float
    df[results_column] = df[results_column].astype(float)
    
    # Determine the unique seasons
    seasons = df[season_column].unique()
    
    # Check that there are multiple seasons
    if len(seasons) < 2:
        KW_comment = 'Only {} seasons provided in the data.'.format(len(seasons))
    # Check that there are at least 2 distinct results
    elif len(df[results_column].unique()) < 2:
        KW_comment = 'All results are identical. Result values: {}'.format(df[results_column].unique())
    
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

#%% Trend Direction

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
    
    # Copy the DataFrame
    df = df.copy()
    
    # Convert results column to float
    df[results_column] = df[results_column].astype(float)
    
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
                    confidence_categories = {0.90:'Very likely', 0.67:'Likely'},
                    neutral_category = 'Indeterminate'):
    '''
    A function that applies a Mann-Kendall trend test, determines a continuous
    scale confidence that the trend is increasing and categorises the result

    Parameters
    ----------
    df : DataFrame
        The DataFrame that contains the results to be analysed.
    results_column : string
        The column name for the column containing results
    season_column : string, optional
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
           year_column,
           results_column,
           seasons_per_year=1,
           season_column=None):
    '''
    A function that determines all combinations of slopes that will be used in
    a Sen slope analysis. The data must be of either annual, quarterly, or
    monthly frequency.

    Parameters
    ----------
    df : DataFrame
        A DataFrame containing a numeric results and time information
    year_column : string
        The column name for the column containing the year
    results_column : string
        The column name for the column containing numeric results
    seasons_per_year : integer
        The frequency of results to be analysed for a Sen slope.
        The default is 1 season or annual data.
    season_column : string
        The column name that has an integer representation of the season.

    Raises
    ------
    Exception
        Raised if the frequency is not one of the accepted strings or if there
        are multiple results taken within one interval of the defined frequency

    Returns
    -------
    slopes : list of floats
        A list of slopes that can be used to determine the Sen slope.

    '''
    
    # Copy the DataFrame
    df = df.copy()
    
    # Convert year column to integer
    df[year_column] = df[year_column].astype(int)
    
    # Convert results column to float
    df[results_column] = df[results_column].astype(float)
    
    # Check that intervals_per_year is a factor of 12
    if seasons_per_year not in [1,2,3,4,6,12]:
        raise ValueError('seasons_per_year must be a factor of 12')
    
    # Convert season column to integer
    if season_column:
        df[season_column] = df[season_column].astype(int)
    
    # Determine the decimal value of the year for use in slopes
    df['DecimalYear'] = df[year_column]
    if season_column:
        df['DecimalYear'] += df[season_column] / (seasons_per_year)
    
    # Check that only a single result occurs for each year
    if not df['DecimalYear'].is_unique:
        raise Exception('Only one result can be supplied within each interval ' \
                        'of the defined frequency.')
    
    # Set y values and x values
    y = np.array(df[results_column])
    x = np.array(df['DecimalYear'])

    # Compute sorted slopes only when deltax > 0
    deltax = x[:, np.newaxis] - x
    deltay = y[:, np.newaxis] - y
    slopes = deltay[deltax > 0] / deltax[deltax > 0]
    
    return slopes

def slopes_seasonal(df,
                    year_column,
                    results_column,
                    season_column):
    '''
    A function that determines all combinations of slopes within each season
    that will be used in a Sen slope analysis. The data must be of either
    annual, quarterly, or monthly frequency.

    Parameters
    ----------
    df : DataFrame
        A DataFrame containing a numeric results and time information
    year_column : string
        The column name for the column containing the year
    results_column : string
        The column name for the column containing numeric results
    season_column : string
        The column name for the column indicating the season

    Returns
    -------
    slopes : list of floats
        A list of slopes that can be used to determine the Sen slope.

    '''
    # Determine the unique seasons
    seasons = df[season_column].unique()
    
    slopes_seasonal = np.concatenate([slopes(df[df[season_column]==season],
                                    year_column, 
                                    results_column,
                                    seasons_per_year=1) for season in seasons])
    
    return slopes_seasonal


def trend_magnitude(df,
                    year_column,
                    results_column,
                    seasons_per_year=1,
                    season_column=None,
                    percentile_method='hazen',
                    confidence_interval=90):
    '''
    A function that determines the Sen slope for a dataset along with upper and
    lower bound estimates for within a specific confidence interval.

    Parameters
    ----------
    df : DataFrame
        A DataFrame containing a numeric results and time information
    year_column : string
        The column name for the column containing the year
    results_column : string
        The column name for the column containing numeric results
    seasons_per_year : integer
        The frequency of results to be analysed for a Sen slope.
        The default is 1 season or annual data.
    season_column : string, optional
        The column name for the column indicating the season.
        The default is None.
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
    # whether a season_column is provided.
    if season_column:
        method = 'Seasonal'
        slopes = slopes_seasonal(df,year_column,results_column,season_column)
    else:
        method = 'Non-seasonal'
        slopes = slopes(df,year_column,results_column,seasons_per_year,season_column)
    
    # Calculate median slope
    median_slope = np.nan
    if len(slopes) != 0:
        median_slope = np.median(slopes)
    
    # Calculate confidence interval
    
    # Initialise bounds of confidence interval
    lower_slope = np.nan
    upper_slope = np.nan
    
    # Calculate the percentiles of the confidence interval
    percentile_lower = (100-confidence_interval)/2
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

#%% Preprocessing datasets for trends

def define_intervals(df,
                     intervals_per_year,
                     date_columns=['DateTime'],
                     date_format=None,
                     end_month=6,
                     groupby_columns=[],
                     reduction_method=None,
                     reduction_inputs=None):
    '''
    A function that adds columns for trend frequency, trend year, and
    trend interval (for the trend year) i.e. quarter, month, etc. If multiple
    results are within an interval, a reduction method is needed.

    Parameters
    ----------
    df : DataFrame
        A DataFrame containing data for trend information including column(s)
        that describe the time of results to be analysed in a trend analysis
    intervals_per_year : integer
        The number of results per year that should be analysed in a trend analysis
    date_columns : list of strings, optional
        The column name(s) for the column(s) that contains year and month information.
        A single value indicates a combined year month column (could also be
        date or datetime) with the format defined by date_format, two values
        can be provided if there are columns for the year and month (1-12)
        separately, where the year column should be listed first.
        The default is [DateTime].
    date_format : string or None, optional
        The date or datetime format used in date_columns when a single value is
        provided. This parameter should be left as 'None' if the date column is 
        already in a pandas datetime format or if the year and month are split
        into two columns.
        The default is None.
        https://strftime.org/
    end_month : integer, optional
        The number for the last month that should be included in the trend period.
        The default is 6 (June).
    groupby_columns : list of strings, optional
        List of column names that should be used to create groups of datasets for trends.
        The default is [].
    reduction_method : string, optional
        The method used to reduce multiple values within an interval to a
        single result. If None specified, then only a single result should be
        included per interval. Options include:
            - midpoint: the result closest to the middle of the interval
            - first: the first result in the interval
            - last: the last result in the interval
            - median: the median result in the interval
            - average: the average result in the interval
            - maximum: the maximum result in the interval
            - minimum: the minimum result in the interval
            - nearest: the result closest to a particular date (only applies
                        for an annual frequency (i.e., intervals_per_year = 1))
    reduction_inputs : list of strings
        A list of extra input needed to apply a specified reduction method.
        The extra input required for each method includes:
            - midpoint, first, last: None. No information needed.
            - median, average, maximum, minimum:
                - if the results are not censored, then provide a single string
                    for the name of the numeric column (e.g., 'Result'); else,
                - a list of the column name(s) for the column(s) that contain
                    the results. If a single column name is given, it is
                    assumed that the column contains combined censor and numeric
                    components. If two column names are provided, then the first
                    should only contain one of five censors (<,≤,,≥,>) and the
                    second should contain only numeric data.
                - A True/False value for whether to focus on the highest/lowest
                    potential, respectively (only for average/median reduction)
                - A True/False value for whether to include the negative interval
                    as potential values for left censored values.
                - A float value for the precision percentage tolerance at which
                    to drop a censor (<0.3 vs 0.25 for a result between 0.2 and 0.3)
                - A True/False value as to whether to apply a precision rounding
                    function to the statistical result
            - nearest: a month and day of month expressed as a string.
                October 16 should be written as '10-16'.
                The time will be set as midnight (start) of the given day.
                Or is it [10,16]?

    Raises
    ------
    ValueError or Exception
        Errors raised if inputs do not meet the requirements

    Returns
    -------
    df : DataFrame
        The input dataframe with additional columns that include the trend
        frequency, the trend year, the trend interval that the result
        is in (quarter, month, etc.). If multiple results are in an interval,
        then a reduction method must be chosen so there is one result per interval.

    '''
    
    # Copy the DataFrame
    df = df.copy()
    
    # Check that intervals_per_year is a factor of 12
    if intervals_per_year not in [1,2,3,4,6,12]:
        raise ValueError('intervals_per_year must be a factor of 12')
    # Add columns for the data frequency and number of intervals per year
    else:
        freq_dict = {
            1:'Annually',
            2:'Semiannually',
            3:'Triannually',
            4:'Quarterly',
            6:'Bimonthly',
            12:'Monthly'
            }
        df['Frequency'] = freq_dict[intervals_per_year]
        df['Intervals/year'] = intervals_per_year
    
    # Check that the end month provided is between 1 and 12
    if end_month - 1 not in range(12):
        raise ValueError('end_month must be an integer from 1 to 12')
    
    # If year month columns provided then use them to generate Trend year and month
    if len(date_columns) == 2:
        
        # Assign date_column_format
        date_column_format = 'Year/Month'
        
        # Determine year and month column names
        year_column = date_columns[0]
        month_column = date_columns[1]
        
        # Sort by year then month
        df = df.sort_values(by=[year_column,month_column])
        
        # Generate artifical dates for the dateframe
        dates = pd.to_datetime(df[[year_column, month_column]].assign(DAY=15)).copy()
        
    # If single column name, year month are within a single column
    elif len(date_columns) == 1:
        
        # Assign date_column_format
        date_column_format = 'DateTime'
        
        # Determine date column name
        date_column = date_columns[0]
        
        # Convert the date column using the provided format
        if date_format:
            df[date_column] = pd.to_datetime(df[date_column], format=date_format)
        
        # Sort by datetime
        df = df.sort_values(by=[date_column])
        
        # Copy the dates
        dates = df[date_column].copy()
    
    # If there are not 1 or 2 date column names provided, raise an error
    else:
        raise ValueError('date_columns requires a list with one or two column names.')
    
    # Shift months forward by 12 months minus the end month so that the interval
    # and trend year align with the calendar year. For example,
    # the first month in the trend period will be month 1 and the first three months
    # will be quarter 1 and the year will be the year that the trend period ends
    dates = dates + pd.DateOffset(months=12-end_month)
    
    # Add a column for the trend year
    df['TrendYear'] = dates.dt.year
    
    # Add a column to specify which interval the result is in
    # Convert month to a 0-based index and floor divide by months per interval
    # and then adjust back to 1-based index
    df['TrendInterval'] = (dates.dt.month-1)//int(12/intervals_per_year) + 1
    
    # Add created columns to groupby_columns
    groupby_columns = groupby_columns.copy()
    groupby_columns += ['Frequency','TrendYear','TrendInterval']
    
    # Apply reduction method
    
    # If statistical reduction
    if reduction_method in ['median','average','maximum','minimum']:
        # Check for valid inputs
        if not isinstance(reduction_inputs,str):
            expected_inputs = {'median':5, 'average':5, 'maximum':4, 'minimum':4}
            if len(reduction_inputs) != expected_inputs[reduction_method]:
                raise ValueError('The "{}" reduction method requires {} inputs.' \
                                 .format(reduction_method,expected_inputs[reduction_method()]))                    
        # Apply censored stat calculations
        if reduction_method == 'median':
            if isinstance(reduction_inputs,str):
                df = df.groupby(groupby_columns).median()
            else:
                df = median(df,
                            groupby_columns = groupby_columns,
                            *reduction_inputs)
        elif reduction_method == 'average':
            if isinstance(reduction_inputs,str):
                df = df.groupby(groupby_columns).mean()
            else:
                df = average(df,
                             groupby_columns = groupby_columns,
                             *reduction_inputs)
        elif reduction_method == 'maximum':
            if isinstance(reduction_inputs,str):
                df = df.groupby(groupby_columns).max()
            else:
                df = maximum(df,
                             groupby_columns = groupby_columns,
                             *reduction_inputs)
        elif reduction_method == 'minimum':
            if isinstance(reduction_inputs,str):
                df = df.groupby(groupby_columns).min()
            else:
                df = minimum(df,
                             groupby_columns = groupby_columns,
                             *reduction_inputs)
        # Reset index
        df = df.reset_index()
        
    # If date based reduction
    elif reduction_method in ['first','last','nearest','midpoint']:
        # Check that there is a datetime column
        if date_column_format != 'DateTime':
            raise Exception('A single date column is required to use the ' \
                            '"{}" reduction method.'.format(reduction_method))
    
        # If reduction method is to keep first or last in interval
        if reduction_method in ['first','last']:
            # Sort by date column
            df.sort_values(by=date_column, inplace=True)
            # Keep first/last result in each interval
            df = df.drop_duplicates(subset=groupby_columns,
                                    keep=reduction_method)
        # If reduction method is midpoint or nearest, a date needs defined for each interval
        elif reduction_method in ['nearest','midpoint']:
            # If nearest
            if reduction_method == 'nearest':
                # Check for single interval per year
                if intervals_per_year != 1:
                    raise Exception('The nearest reduction method can only be applied ' \
                                'if there is one interval per year.')
                # If the provided month is not after the last month in the
                # trend period, then the relevant date will have the same
                # year as the trend period even when the trend period
                if reduction_inputs[0] <= end_month:
                    annual_date = pd.to_datetime(df['TrendYear'].astype(str)+ \
                                                    '-{}-{}'.format(*reduction_inputs))
                # Otherwise, subtract a year
                else:
                    annual_date = pd.to_datetime((df['TrendYear']-1).astype(str)+ \
                                                    '-{}-{}'.format(*reduction_inputs))
                # Determine proximity of each date to the relevant interval date
                df['DateProximity'] = abs(df[date_column]-annual_date)
            # If midpoint
            else:
                # Check that there are no reduction method inputs
                if reduction_inputs:
                    raise ValueError('No inputs are required for the midpoint reduction method')
                else:
                    # Create dictionary for the date of the midpoint for each interval
                    # when the trend year has been shifted to align with calendar year
                    if intervals_per_year == 1:
                        midpoint_date = {
                            1:'-7-1'
                            }
                    elif intervals_per_year == 2:
                        midpoint_date = {
                            1:'-4-1',
                            2:'-10-1'
                            }
                    elif intervals_per_year == 3:
                        midpoint_date = {
                            1:'-3-1',
                            2:'-7-1',
                            3:'-11-1'
                            }
                    elif intervals_per_year == 4:
                        midpoint_date = {
                            1:'-2-16',
                            2:'-5-16',
                            3:'-8-16',
                            4:'-11-16'
                            }
                    elif intervals_per_year == 6:
                        midpoint_date = {
                            1:'-2-1',
                            2:'-4-1',
                            3:'-6-1',
                            4:'-8-1',
                            5:'-10-1',
                            6:'-12-1'
                            }
                    else:
                        midpoint_date = {
                            1:'-1-16',
                            2:'-2-16',
                            3:'-3-16',
                            4:'-4-16',
                            5:'-5-16',
                            6:'-6-16',
                            7:'-7-16',
                            8:'-8-16',
                            9:'-9-16',
                            10:'-10-16',
                            11:'-11-16',
                            12:'-12-16'
                            }
                # Use the trend year to determine the relevant midpoint date
                # when intervals have been aligned with the calendar year
                midpoint_dates = pd.to_datetime(df['TrendYear'].astype(str) + df['TrendInterval'].map(midpoint_date))
                
                # Determine the date proximity using the dates for the results
                # that have been shifted so the trend period aligns with the
                # calendar year
                df['DateProximity'] = abs(dates-midpoint_dates)
            
            # Sort by Proximity and take later sample time if ties
            df = df.sort_values(by=['DateProximity',date_column],ascending=[True,False])
            # Keep single sample closest to target date
            df = df.drop_duplicates(subset=groupby_columns)
            # Resort data
            df = df.sort_values(by=date_column)
            # Drop temporary column
            df = df.drop(columns=['DateProximity']).reset_index(drop=True)
    else:
        # If no reduction method, check that each interval has one result
        if df.groupby(groupby_columns).ngroups != len(df):
            raise Exception('At least one interval contains multiple results. ' \
                    'Specify a reduction method to reduce multiple results ' \
                    'within an interval to a single result.')
    
    return df

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

def trend_periods(df,
                  trend_lengths,
                  end_years=[current_water_year()-1],
                  end_month=6,
                  output_format_end='%Y',
                  output_format_period='%b %Y'):
    '''
    Function that filters data for one or more trend periods.

    Parameters
    ----------
    df : DateFrame
        A DataFrame that has a DateTime column and a results column for trend analysis
    trend_lengths : list of integers
        A list the trend lengths where each trend length is the number of years
        to include in a given trend period. For example, 10- and 20-year trend
        periods can be obtained by setting this parameter to [10,20].
    end_years : list of integers, optional
        A list of years for which trend periods will be generated with the
        trend period ending in the listed year. The default is a single year
        which is the year of the most recent June 30.
    end_month : integer, optional
        The number for the last month that should be included in the trend period.
        The default is 6 (June).
    output_format_end : string or None, optional
        The format that will be used in a TrendEnd column that describes the
        end date of the trend period. A value of None will prevent this column
        from being generated. The default is to show the last year of the trend period ('%Y').
        https://strftime.org/
    output_format_period : string or None, optional
        The format that will be used in a TrendPeriod column that describes
        the start and end of the trend period. A value of None will prevent
        this column from being generated. The default value is to include
        the first three letters of the month and the year for the start and end
        dates of the trend period ('%b %Y').
        https://strftime.org/

    Raises
    ------
    ValueError
        Errors raised if inputs do not meet requirements

    Returns
    -------
    output_df : DataFrame
        A DataFrame with each trend period containing its own set of data to
        be analysed in a trend analysis.

    '''
    
    # Copy the DataFrame
    df = df.copy()
    
    # Check that trend_lengths are postive integers
    for trend_length in trend_lengths:
        if (trend_length <= 0) | (not isinstance(trend_length, int)):
            raise ValueError('All trend lengths must be positive integers.')
    
    # Check that each end year is an integer
    for end_year in end_years:
        if not isinstance(end_year, int):
            raise ValueError('All end years must be integers.')
    
    # Initiate output DataFrame
    output_df = []
    
    # Sort trend lengths in descending order
    # This allows the use of the same data variable for each trend length
    trend_lengths.sort(reverse=True)
    
    # Loop through end dates
    for end_year in end_years:
        
        # Copy data
        data = df.copy()
        
        # Only consider data up through the end year
        data = data[data['TrendYear'] <= end_year]
        
        # Determine DateTime for the end of the trend period
        end = pd.to_datetime('{}-{}-1'.format(end_year,end_month)) + \
                pd.DateOffset(months=1) + \
                pd.Timedelta(seconds=-1)
        
        # Add a column for the end of the trend, if output format specified
        if output_format_end:
            data['TrendEnd'] = end.strftime(output_format_end)
        
        # Loop through trend lengths for each end date
        for trend_length in trend_lengths:
            
            # Determine the start year (the difference between the first year
            # and last year in a trend period is 1 less than the trend length)
            start_year = end_year - (trend_length - 1)
            
            # Only keep years after and including the start year
            data = data[data['TrendYear'] >= start_year].copy()
            
            # Add a column for the trend period, if output format specified
            if output_format_period:
                # Determine DateTime for the start of the trend period
                start = pd.to_datetime('{}-{}-1'.format(end_year,end_month)) + \
                            pd.DateOffset(months=1) + \
                            pd.DateOffset(years=-trend_length)
                
                data['TrendPeriod'] = \
                    start.strftime(output_format_period) + ' to ' + \
                    end.strftime(output_format_period)
            
            # Add additional information columns
            data['TrendLength'] = trend_length
            
            # Append trend data to output
            output_df.append(data)
        
    # Combine dataframes
    output_df = pd.concat(output_df).reset_index(drop=True)
    
    return output_df


#%% Dataset Summary

def describe_numeric_data(df,
                          results_column):
    '''
    A function that provides a summary of a numeric column of a DataFrame.

    Parameters
    ----------
    df : DataFrame
        A DataFrame with a numeric column to describe.
    results_column : string
        The name of the column containing numeric results.

    Returns
    -------
    Series
        A Series that contains a summary of the results which includes the
        total count followed by the min/median/average/max.
    '''
    
    # Determine summary stats of the numeric data
    count = len(df)
    min_value = df[results_column].min()
    median_value = df[results_column].max()
    average_value = df[results_column].mean()
    max_value = df[results_column].max()
    
    return pd.Series([count,min_value,median_value,average_value,max_value],
                     index=['Count','Minimum','Median','Average','Maximum'])


def describe_censored_data(df,
                           results_column = ['CensorComponent','NumericComponent']):
    '''
    A function that provides a summary of censored data in a DataFrame.

    Parameters
    ----------
    df : DataFrame
        A DataFrame with a column with censored results or
        two columns containing the censor symbol and numeric component, respectively.
    results_column : list of strings, optional
        The column name(s) for the column(s) that contain the results. If a 
        single column name is given, it is assumed that the column contains
        combined censor and numeric components. If two column names are
        provided, then the first should only contain one of five censors (<,≤,,≥,>)
        and the second should contain only numeric data.
        The default is ['CensorComponent','NumericComponent'].

    Returns
    -------
    Series
        A Series that contains a summary of the results which includes the
        total count followed by the min/max uncensored results, the count/min/max
        of the left censored results and the count/min/max of the right
        censored results.

    '''
    
    # If single result column provided, then split column
    if len(results_column) == 1:
        censor_column = 'CensorComponent'
        numeric_column = 'NumericComponent'
        df = result_to_components(df,results_column[0])
    # Else define the names to use for the censor and numeric columns
    else:
        censor_column = results_column[0]
        numeric_column = results_column[1]
    
    # Calculate summary stats on the censored data
    count = len(df)
    # Uncensored data stats
    non_censored = df[df[censor_column]==''][numeric_column]
    min_value = non_censored.min()
    max_value = non_censored.max()
    # Left censored data stats
    left_censored = df[df[censor_column].isin(['<','≤'])][numeric_column]
    count_left_censored = left_censored.count()
    min_detect_limit = left_censored.min()
    max_detect_limit = left_censored.max()
    # Right censored data stats
    right_censored = df[df[censor_column].isin(['≥','>'])][numeric_column]
    count_right_censored = right_censored.count()
    min_quant_limit = right_censored.min()
    max_quant_limit = right_censored.max()
    
    return pd.Series([count,min_value,max_value,
                      count_left_censored,min_detect_limit,max_detect_limit,
                      count_right_censored,min_quant_limit,max_quant_limit],
                     index=['Count','Minimum','Maximum',
                            'CountBelowDetect','MinDetectLimit','MaxDetectLimit',
                            'CountAboveQuant','MinQuantLimit','MaxQuantLimit'])

#%% Preprocessing dataset for individual trend result

def prep_censored_data_for_trends(df,
                                  results_column = ['CensorComponent','NumericComponent'],
                                  lower_conversion_factor = 0.5,
                                  upper_conversion_factor = 1.1):
    '''
    This function adjusts results to a common detection limit and quantification
    limit. This is necessary to ensure that trends are not detected as a
    result of changing detection and quantification limits.

    Parameters
    ----------
    df : DataFrame
        A DataFrame containing results that may be censored
    results_column : list of strings, optional
        The column name(s) for the column(s) that contain the results. If a 
        single column name is given, it is assumed that the column contains
        combined censor and numeric components. If two column names are
        provided, then the first should only contain one of five censors (<,≤,,≥,>)
        and the second should contain only numeric data.
        The default is ['CensorComponent','NumericComponent'].
        DESCRIPTION. The default is 1.1.
    lower_conversion_factor : float, optional
        The factor that will be used to scale all left censored results and all
        uncensored results that are less than the greatest detection limit.
        The default is 0.5.
    upper_conversion_factor : float, optional
        The factor that will be used to scale all right censored results and all
        uncensored results that are greater than the smallest quantification limit.
        The default is 1.1.

    Raises
    ------
    ValueError
        Raised if the upper and lower conversion factors are not above and below
        1.0, respectively.

    Returns
    -------
    df : DataFrame
        The input DataFrame where the numeric component has been adjusted for use
        in a trend analysis.

    '''
    
    # Copy the DataFrame
    df = df.copy()
    
    # Check that the upper conversion factor is greater than 1
    if upper_conversion_factor <= 1:
        raise ValueError('The upper conversion factor needs to be greater than 1.')
    # Check that the lower conversion factor is less than 1
    if lower_conversion_factor >= 1:
        raise ValueError('The lower conversion factor needs to be less than 1.')
    
    # If single result column provided, then split column
    if len(results_column) == 1:
        censor_column = 'CensorComponent'
        numeric_column = 'NumericComponent'
        df = result_to_components(df,results_column[0])
    # Else define the names to use for the censor and numeric columns
    else:
        censor_column = results_column[0]
        numeric_column = results_column[1]
    
    # # Create True/False indicator for uncensored data
    non_censored_check = df[censor_column]==''
    # Create True/False indicator for left censored data
    left_censored_check = df[censor_column].isin(['<','≤'])
    # Determine the maximum detection limit
    max_detect_limit = df[left_censored_check][numeric_column].max()
    # Create True/False indicator for right censored data
    right_censored_check = df[censor_column].isin(['≥','>'])
    # Determine the minimum quantification limit
    min_quant_limit = df[right_censored_check][numeric_column].min()
    
    # Convert all right censored data and all uncensored data that is greater
    # than the minimum quanitification limit to be equal to one
    # another and slightly larger than the limit
    if sum(right_censored_check) > 0:
        df[numeric_column] = np.where((right_censored_check) | \
                                      (non_censored_check & (df[numeric_column] > min_quant_limit)),
                                      upper_conversion_factor*min_quant_limit,
                                      df[numeric_column])
    
    # Convert all left censored data and all uncensored data that is less
    # than the largest detection limit to be equal to one
    # another and less than the detection limit
    if sum(left_censored_check) > 0:
        df[numeric_column] = np.where((left_censored_check) | \
                                      (non_censored_check & (df[numeric_column] < max_detect_limit)),
                                      lower_conversion_factor*max_detect_limit,
                                      df[numeric_column])
    
    return df


def interval_counts(df,
                    seasons_per_year):
    '''
    Function that describes how complete a dataset is within a trend period

    Parameters
    ----------
    df : DataFrame
        A DataFrame that contains data for a single trend analysis result

    Returns
    -------
    Series
        A Series which describes the intervals within a trend analysis dataset:
            - The number of years with results
            - The number of intervals with results
            - The percentage of years with results
            - The percentage of intervals with results

    '''
    
    # Copy the DataFrame
    df = df.copy()
    
    # Determine trend_length and intervals per year
    trend_length = df['TrendLength'].iloc[0]
    
    years = len(df.groupby(['TrendYear']))
    intervals = len(df.groupby(['TrendYear','TrendInterval']))
    percent_years = 100 * years / trend_length
    percent_intervals = 100 * intervals / (seasons_per_year * trend_length)
    
    return pd.Series([years,intervals,percent_years,percent_intervals],
                     index=['YearsInPeriod','IntervalsInPeriod',
                            'PercentOfYears','PercentOfIntervals'])
    

#%% Trend analysis for an individual dataset

def trend_analysis(df,
                   results_column,
                   seasons_per_year,
                   year_column_for_slopes='TrendYear',
                   season_column = 'TrendInterval',
                   censored_results=False,
                   censored_conversions=[0.5, 1.1],
                   direction_confidence_categories = {0.90:'Very likely', 0.67:'Likely'},
                   direction_neutral_category = 'Indeterminate',
                   slope_percentile_method='hazen',
                   slope_confidence_interval=90):
    
    # Initialise an empty output series
    output = []
    
    # Describe the dataset depending on whether results are censored or numeric
    if censored_results:
        output.append(describe_censored_data(df,results_column))
        # Convert censored data to have common censorship thresholds
        df = prep_censored_data_for_trends(df,
                                           results_column = results_column,
                                           lower_conversion_factor = censored_conversions[0],
                                           upper_conversion_factor = censored_conversions[1])
    else:
        output.append(describe_numeric_data(df,results_column))
    
    # Describe the interval counts of the dataset
    output.append(interval_counts(df,seasons_per_year))
    
    #results_column = 'NumericComponent' #!!!!!!!!!!!!!!!!!
    
    # Perform a Seasonality test
    output.append(kruskal_wallis_test(df,results_column,season_column))
    
    # Perform a trend direction analysis
    output.append(trend_direction(df,results_column,season_column,
                                  direction_confidence_categories,
                                  direction_neutral_category))
    
    # Perform a trend magnitude analysis
    output.append(trend_magnitude(df,year_column_for_slopes,results_column,seasons_per_year,
                                  season_column,slope_percentile_method,slope_confidence_interval))
    
    return pd.concat(output)

#%% Trend analysis for a dataset

def trends(df,
           results_column,
           seasons_per_year,
           trend_lengths,
           end_years=[current_water_year()-1],
           end_month=6,
           groupby_columns=[],
           date_columns=['DateTime'],
           date_format=None,
           reduction_method=None,
           reduction_inputs=None,
           output_format_end='%Y',
           output_format_period='%b %Y',
           year_column_for_slopes='TrendYear',
           season_column = 'TrendInterval',
           censored_results=False,
           censored_conversions=[0.5, 1.1],
           direction_confidence_categories = {0.90:'Very likely', 0.67:'Likely'},
           direction_neutral_category = 'Indeterminate',
           slope_percentile_method='hazen',
           slope_confidence_interval=90):
    
    # Copy the DataFrame
    df = df.copy()
    groupby_columns = groupby_columns.copy()
    
    # Determine trend intervals based on specified data frequency and trend year
    df = define_intervals(df,seasons_per_year,date_columns,date_format,end_month,
                          groupby_columns,reduction_method,reduction_inputs)
    
    # Create dataset for each trend period
    df = trend_periods(df,trend_lengths,end_years,end_month,
                        output_format_end,output_format_period)
    
    groupby_columns += ['TrendEnd','TrendLength','TrendPeriod','Frequency','Intervals/year']
    
    # Analyse each dataset
    df = df.groupby(groupby_columns).apply(trend_analysis,
                                            results_column,
                                            seasons_per_year,
                                            year_column_for_slopes,
                                            season_column,
                                            censored_results,
                                            censored_conversions,
                                            direction_confidence_categories,
                                            direction_neutral_category,
                                            slope_percentile_method,
                                            slope_confidence_interval)
    
    
    return df
    
    
    
    
    