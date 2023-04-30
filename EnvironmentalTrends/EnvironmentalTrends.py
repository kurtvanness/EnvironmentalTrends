
import numpy as np
import pandas as pd
from .utils import kruskal_wallis_test, trend_direction, trend_magnitude, current_water_year
import censoredsummarystats as censored

# Set default column names
trend_end_col = 'TrendEnd'
trend_len_col = 'TrendLength'
trend_period_col = 'TrendPeriod'
freq_col = 'Frequency'
freq_num_col = 'Seasons/year'
trend_date_col = '__TrendDate__'
trend_year_col = '__TrendYear__'
trend_season_col = '__TrendSeason__'
date_proximity_col = '__DateProximity__'
season_mid_col = '__SeasonMidpoint__'
censor_col = 'CensorComponent'
numeric_col = 'NumericComponent'


#%% Preprocessing datasets for trends

def define_seasons(df,
                   seasons_per_year,
                   date_col='DateTime',
                   date_format=None,
                   end_month=6):
    '''
    A function that adds columns for trend frequency, trend year, and
    trend season (for the trend year) i.e. quarter, month, etc.

    Parameters
    ----------
    df : DataFrame
        A DataFrame containing data for trend information including column(s)
        that describe the time of results to be analysed in a trend analysis
    seasons_per_year : integer
        The number of results per year that should be analysed in a trend analysis
    date_col : string or list of two integers, optional
        The column name(s) for the column(s) that contains year and month information.
        A string value indicates a combined year month column (could also be
        date or datetime) with the format defined by date_format, two values
        can be provided if there are columns for the year and month (1-12)
        separately, where the year column should be listed first.
        The default is 'DateTime'.
    date_format : string or None, optional
        The date or datetime format used in date_col when a single value is
        provided. This parameter should be left as 'None' if the date column is 
        already in a pandas datetime format or if the year and month are split
        into two columns.
        The default is None.
        https://strftime.org/
    end_month : integer, optional
        The number for the last month that should be included in the trend period.
        The default is 6 (June).

    Raises
    ------
    ValueError or Exception
        Errors raised if inputs do not meet the requirements

    Returns
    -------
    df : DataFrame
        The input dataframe with additional columns that include the trend
        frequency, the trend year, the trend season that the result
        is in (quarter, month, etc.). If multiple results are in a season,
        then a reduction method must be chosen so there is one result per season.

    '''
    
    # Copy the DataFrame
    df = df.copy()
    
    # Check that seasons_per_year is a factor of 12
    if seasons_per_year not in [1,2,3,4,6,12]:
        raise ValueError('seasons_per_year must be a factor of 12')
    # Add columns for the data frequency and number of seasons per year
    else:
        freq_dict = {
            1:'Annually',
            2:'Semiannually',
            3:'Triannually',
            4:'Quarterly',
            6:'Bimonthly',
            12:'Monthly'
            }
        df[freq_col] = freq_dict[seasons_per_year]
        df[freq_num_col] = seasons_per_year
    
    # Check that the end month provided is between 1 and 12
    if end_month - 1 not in range(12):
        raise ValueError('end_month must be an integer from 1 to 12')
    
    # If single column name, year month are within a single column
    if isinstance(date_col,str):
        
        # Convert the date column using the provided format
        if date_format:
            df[date_col] = pd.to_datetime(df[date_col], format=date_format)
        
        # Copy the dates
        df[trend_date_col] = df[date_col].copy()
    
    # If year month columns provided then use them to generate Trend year and month
    elif (isinstance(date_col,list)) & (len(date_col) == 2):
        
        # # Assign date_column_format
        # date_column_format = 'Year/Month'
        
        # Determine year and month column names
        year_col = date_col[0]
        month_col = date_col[1]
        
        # Generate artifical dates for the dateframe
        df[trend_date_col] = pd.to_datetime(df[[year_col, month_col]].assign(DAY=15)).copy()
    
    # If there are not 1 or 2 date column names provided, raise an error
    else:
        raise ValueError('date_col requires a string or a list of two strings.')
    
    # Shift months forward by 12 months minus the end month so that the seasons
    # and trend year align with the calendar year. For example,
    # the first month in the trend period will be month 1 and the first three months
    # will be quarter 1 and the year will be the year that the trend period ends
    df[trend_date_col] = df[trend_date_col] + pd.DateOffset(months=12-end_month)
    
    # Add a column for the trend year
    df[trend_year_col] = df[trend_date_col].dt.year
    
    # Add a column to specify which season the result is in
    # Convert month to a 0-based index and floor divide by months per season
    # and then adjust back to 1-based index
    df[trend_season_col] = (df[trend_date_col].dt.month-1)//int(12/seasons_per_year) + 1
    
    return df

def season_reduction(df,
                     results_col,
                     groupby_columns,
                     reduction_method=None,
                     end_month=None,
                     annual_date='-10-16'):
    '''
    A function that reduces multiple results within a season to a single result.

    Parameters
    ----------
    df : DataFrame
        A DataFrame containing data for trend information including column(s)
        that describe the time of results to be analysed in a trend analysis
    results_col : string
        The column name for the column that contains the numeric results. This is
        only used if the results are non-censored and a stat reduction method is used.
    groupby_columns : list of strings
        List of column names that should be used to create groups of datasets for trends.
    reduction_method : string, optional
        The method used to reduce multiple values within a season to a
        single result. If None specified, then only a single result should be
        included per season. Options include:
            - Stat-based reduction
                - median: the median result in the season
                - average: the average result in the season
                - maximum: the maximum result in the season
                - minimum: the minimum result in the season
            - Date-based reduction
                - midpoint: the result closest to the middle of the season
                - first: the first result in the season
                - last: the last result in the season
                - nearest: the result closest to a particular date (only applies
                        for an annual frequency)
    end_month : integer, optional
        This is only required for the 'nearest' reduction method.
    annual_date : string
        The string of the month and day that will be used for the 'nearest' method.
        Hyphens should be included such that reducing to the result in the year
        closest to 16 October should be written as '-10-16'.
    
    Raises
    ------
    ValueError or Exception
        Errors raised if inputs do not meet the requirements

    Returns
    -------
    df : DataFrame
        The input dataframe with additional columns that include the trend
        frequency, the trend year, the trend season that the result
        is in (quarter, month, etc.). If multiple results are in a season,
        then a reduction method must be chosen so there is one result per season.

    '''
    
    # Copy the DataFrame
    df = df.copy()
    
    # Add created columns to groupby_columns
    reduction_groupby_columns = groupby_columns + [freq_col,freq_num_col,trend_year_col,trend_season_col]
    
    # Apply reduction method
    
    # If statistical reduction
    if reduction_method in ['median','average','maximum','minimum']:
        # Check for valid inputs
        if not isinstance(results_col,str):
            raise ValueError('The "{}" reduction method requires a column \
                             name for the results.'.format(reduction_method))
        # Convert results column to numeric
        df[results_col] = df[results_col].astype(float)
        # Apply censored stat calculations
        if reduction_method == 'median':
            df = df.groupby(reduction_groupby_columns)[results_col].median()
        elif reduction_method == 'average':
            df = df.groupby(reduction_groupby_columns)[results_col].mean()
        elif reduction_method == 'maximum':
            df = df.groupby(reduction_groupby_columns)[results_col].max()
        elif reduction_method == 'minimum':
            df = df.groupby(reduction_groupby_columns)[results_col].min()
        # Reset index
        df = df.reset_index()
        
    # If date based reduction
    elif reduction_method in ['first','last','nearest','midpoint']:
        # If reduction method is to keep first or last in season
        if reduction_method in ['first','last']:
            # Sort by date column
            df.sort_values(by=trend_date_col, inplace=True)
            # Keep first/last result in each season
            df = df.drop_duplicates(subset=reduction_groupby_columns,
                                    keep=reduction_method)
        # If reduction method is midpoint or nearest, a date needs defined for each season
        elif reduction_method in ['nearest','midpoint']:
            # If nearest
            if reduction_method == 'nearest':
                # Check for single season per year
                if df[freq_num_col].max() != 1:
                    raise Exception('The nearest reduction method can only be applied ' \
                                'if there is one season per year.')
                # If the provided month is not after the last month in the
                # trend period, then the relevant date will have the same
                # year as the trend period even when the trend period
                nearest_month = int(annual_date.split('-')[1]) + (12-end_month)
                if nearest_month > 12:
                    nearest_month -= 12
                trend_annual_date = pd.to_datetime(df[trend_year_col].astype(str) + \
                                                       nearest_month.astype(str) + \
                                                       annual_date.split('-')[2])
                # Determine proximity of each date to the relevant season date
                df[date_proximity_col] = abs(df[trend_date_col] - trend_annual_date)
            # If midpoint
            else:
                # Specify midpoint for each season and each possible number of seasons per year
                conditions = [
                    (df[freq_num_col] == 1) & (df[trend_season_col] == 1),
                    
                    (df[freq_num_col] == 2) & (df[trend_season_col] == 1),
                    (df[freq_num_col] == 2) & (df[trend_season_col] == 2),
                    
                    (df[freq_num_col] == 3) & (df[trend_season_col] == 1),
                    (df[freq_num_col] == 3) & (df[trend_season_col] == 2),
                    (df[freq_num_col] == 3) & (df[trend_season_col] == 3),
                    
                    (df[freq_num_col] == 4) & (df[trend_season_col] == 1),
                    (df[freq_num_col] == 4) & (df[trend_season_col] == 2),
                    (df[freq_num_col] == 4) & (df[trend_season_col] == 3),
                    (df[freq_num_col] == 4) & (df[trend_season_col] == 4),
                    
                    (df[freq_num_col] == 6) & (df[trend_season_col] == 1),
                    (df[freq_num_col] == 6) & (df[trend_season_col] == 2),
                    (df[freq_num_col] == 6) & (df[trend_season_col] == 3),
                    (df[freq_num_col] == 6) & (df[trend_season_col] == 4),
                    (df[freq_num_col] == 6) & (df[trend_season_col] == 5),
                    (df[freq_num_col] == 6) & (df[trend_season_col] == 6),
                    
                    (df[freq_num_col] == 12) & (df[trend_season_col] == 1),
                    (df[freq_num_col] == 12) & (df[trend_season_col] == 2),
                    (df[freq_num_col] == 12) & (df[trend_season_col] == 3),
                    (df[freq_num_col] == 12) & (df[trend_season_col] == 4),
                    (df[freq_num_col] == 12) & (df[trend_season_col] == 5),
                    (df[freq_num_col] == 12) & (df[trend_season_col] == 6),
                    (df[freq_num_col] == 12) & (df[trend_season_col] == 7),
                    (df[freq_num_col] == 12) & (df[trend_season_col] == 8),
                    (df[freq_num_col] == 12) & (df[trend_season_col] == 9),
                    (df[freq_num_col] == 12) & (df[trend_season_col] == 10),
                    (df[freq_num_col] == 12) & (df[trend_season_col] == 11),
                    (df[freq_num_col] == 12) & (df[trend_season_col] == 12)
                    ]

                results = [
                    '-7-1',
                    
                    '-4-1',
                    '-10-1',
                    
                    '-3-1',
                    '-7-1',
                    '-11-1',
                    
                    '-2-16',
                    '-5-16',
                    '-8-16',
                    '-11-16',
                    
                    '-2-1',
                    '-4-1',
                    '-6-1',
                    '-8-1',
                    '-10-1',
                    '-12-1',
                    
                    '-1-16',
                    '-2-16',
                    '-3-16',
                    '-4-16',
                    '-5-16',
                    '-6-16',
                    '-7-16',
                    '-8-16',
                    '-9-16',
                    '-10-16',
                    '-11-16',
                    '-12-16'
                    ]
                # Use the trend year to determine the relevant midpoint date
                # when seasons have been aligned with the calendar year
                df[season_mid_col] = np.select(conditions,
                                                   pd.to_datetime(df[trend_year_col].astype(str) + results),
                                                   np.nan)
                
                # Determine the date proximity using the dates for the results
                # that have been shifted so the trend period aligns with the
                # calendar year
                df[date_proximity_col] = abs(df[trend_date_col]-df[season_mid_col])
            
            # Sort by Proximity and take later sample time if ties
            df = df.sort_values(by=[date_proximity_col,trend_date_col],ascending=[True,False])
            # Keep single sample closest to target date
            df = df.drop_duplicates(subset=reduction_groupby_columns)
            # Drop temporary column
            df = df.drop(columns=[date_proximity_col]).reset_index(drop=True)
    else:
        # If no reduction method, check that each season has one result
        if df.groupby(reduction_groupby_columns).ngroups != len(df):
            raise Exception('At least one season contains multiple results. ' \
                    'Specify a reduction method to reduce multiple results ' \
                    'within a season to a single result.')
    
    return df
    
def season_reduction_censored_stats(df,
                                    results_col,
                                    groupby_columns,
                                    reduction_method=None,
                                    focus_high_potential = True,
                                    include_negative_interval = False,
                                    precision_tolerance_to_drop_censor = 0.25,
                                    precision_rounding = True):
    '''
    A function that adds columns for trend frequency, trend year, and
    trend season (for the trend year) i.e. quarter, month, etc. If multiple
    results are within a season, a reduction method is needed.

    Parameters
    ----------
    df : DataFrame
        A DataFrame containing data for trend information including column(s)
        that describe the time of results to be analysed in a trend analysis
    result_col : string
        The column name for the column that contain the results as text.
        Only four possible censors should be used (<,≤,≥,>).
    groupby_columns : list of strings
        List of column names that should be used to create groups of datasets for trends.
    reduction_method : string, optional
        The method used to reduce multiple values within a season to a
        single result. If None specified, then only a single result should be
        included per season. Options include:
            - median: the median result in the season
            - average: the average result in the season
            - maximum: the maximum result in the season
            - minimum: the minimum result in the season
    focus_high_potential : boolean, optional
        If True, then information on the highest potential result will be
        focused over the lowest potential result. Only relevant for median and
        average reduction methods
    include_negative_interval : boolean, optional
        If True, then all positive and negative values are considered
        e.g., <0.5 would be converted to (-np.inf,5).
        If False, then only non-negative values are considered
        e.g., <0.5 would be converted to [0,5).
        This setting only affects results if focus_high_potential is False.
        The default is False.
    precision_tolerance_to_drop_censor : float, optional
        Threshold for reporting censored vs non-censored results.
        Using the default, a result that is known to be in the interval (0.3, 0.5)
        would be returned as 0.4, whereas a tolerance of 0 would yield a
        result of <0.5 or >0.3 depending on the value of focus_highest_potential.
        The default is 0.25.
    precision_rounding : boolean, optional
        If True, a rounding method is applied to round results to have no more
        decimals than what can be measured.
        The default is True.

    Raises
    ------
    ValueError or Exception
        Errors raised if inputs do not meet the requirements

    Returns
    -------
    df : DataFrame
        The input dataframe with additional columns that include the trend
        frequency, the trend year, the trend season that the result
        is in (quarter, month, etc.). If multiple results are in a season,
        then a reduction method must be chosen so there is one result per season.

    '''
    
    # Copy the DataFrame
    df = df.copy()
    
    # Add created columns to groupby_columns
    reduction_groupby_columns = groupby_columns + [freq_col,freq_num_col,trend_year_col,trend_season_col]
    
    # Apply reduction method
    
    # If statistical reduction
    if reduction_method in ['median','average','maximum','minimum']:
        # Apply censored stat calculations
        if reduction_method == 'median':
            df = censored.median(df,
                                 results_col,
                                 reduction_groupby_columns,
                                 focus_high_potential,
                                 include_negative_interval,
                                 precision_tolerance_to_drop_censor,
                                 precision_rounding)
        elif reduction_method == 'average':
            df = censored.average(df,
                                  results_col,
                                  reduction_groupby_columns,
                                  focus_high_potential,
                                  include_negative_interval,
                                  precision_tolerance_to_drop_censor,
                                  precision_rounding)
        elif reduction_method == 'maximum':
            df = censored.maximum(df,
                                  results_col,
                                  reduction_groupby_columns,
                                  include_negative_interval,
                                  precision_tolerance_to_drop_censor,
                                  precision_rounding)
        elif reduction_method == 'minimum':
            df = censored.minimum(df,
                                  results_col,
                                  reduction_groupby_columns,
                                  include_negative_interval,
                                  precision_tolerance_to_drop_censor,
                                  precision_rounding)
        # Reset index
        df = df.reset_index()
        
    else:
        # If no reduction method, check that each season has one result
        if df.groupby(reduction_groupby_columns).ngroups != len(df):
            raise Exception('At least one season contains multiple results. ' \
                    'Specify a reduction method to reduce multiple results ' \
                    'within a season to a single result.')
    
    return df

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
        data = data[data[trend_year_col] <= end_year]
        
        # Determine DateTime for the end of the trend period
        end = pd.to_datetime('{}-{}-1'.format(end_year,end_month)) + \
                pd.DateOffset(months=1) + \
                pd.Timedelta(seconds=-1)
        
        # Add a column for the end of the trend, if output format specified
        if output_format_end:
            data[trend_end_col] = end.strftime(output_format_end)
        
        # Loop through trend lengths for each end date
        for trend_length in trend_lengths:
            
            # Determine the start year (the difference between the first year
            # and last year in a trend period is 1 less than the trend length)
            start_year = end_year - (trend_length - 1)
            
            # Only keep years after and including the start year
            data = data[data[trend_year_col] >= start_year].copy()
            
            # Add a column for the trend period, if output format specified
            if output_format_period:
                # Determine DateTime for the start of the trend period
                start = pd.to_datetime('{}-{}-1'.format(end_year,end_month)) + \
                            pd.DateOffset(months=1) + \
                            pd.DateOffset(years=-trend_length)
                
                data[trend_period_col] = \
                    start.strftime(output_format_period) + ' to ' + \
                    end.strftime(output_format_period)
            
            # Add additional information columns
            data[trend_len_col] = trend_length
            
            # Append trend data to output
            output_df.append(data)
        
    # Combine dataframes
    output_df = pd.concat(output_df).reset_index(drop=True)
    
    return output_df


#%% Dataset Summary

def describe_numeric_data(df,
                          results_col):
    '''
    A function that provides a summary of a numeric column of a DataFrame.

    Parameters
    ----------
    df : DataFrame
        A DataFrame with a numeric column to describe.
    results_col : string
        The name of the column containing numeric results.

    Returns
    -------
    Series
        A Series that contains a summary of the results which includes the
        total count followed by the min/median/average/max.
    '''
    
    # Determine summary stats of the numeric data
    count = len(df)
    min_value = df[results_col].min()
    median_value = df[results_col].max()
    average_value = df[results_col].mean()
    max_value = df[results_col].max()
    
    return pd.Series([count,min_value,median_value,average_value,max_value],
                     index=['Count','Minimum','Median','Average','Maximum'])


def describe_censored_data(df,
                           results_col):
    '''
    A function that provides a summary of censored data in a DataFrame.

    Parameters
    ----------
    df : DataFrame
        A DataFrame with a column with censored results or
        two columns containing the censor symbol and numeric component, respectively.
    result_col : string
        The column name for the column that contain the results as text.
        Only four possible censors should be used (<,≤,≥,>).

    Returns
    -------
    Series
        A Series that contains a summary of the results which includes the
        total count followed by the min/max uncensored results, the count/min/max
        of the left censored results and the count/min/max of the right
        censored results.

    '''
    
    # Create a copy of the DataFrame
    df = df.copy()
    
    # Split the result into censor and numeric components
    df = censored.result_to_components(df,results_col,censor_col,numeric_col)
    
    # Calculate summary stats on the censored data
    count = len(df)
    # Uncensored data stats
    non_censored = df[df[censor_col]==''][numeric_col]
    min_value = non_censored.min()
    max_value = non_censored.max()
    # Left censored data stats
    left_censored = df[df[censor_col].isin(['<','≤'])][numeric_col]
    count_left_censored = left_censored.count()
    min_detect_limit = left_censored.min()
    max_detect_limit = left_censored.max()
    # Right censored data stats
    right_censored = df[df[censor_col].isin(['≥','>'])][numeric_col]
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
                                  results_col,
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
    result_col : string
        The column name for the column that contain the results as text.
        Only four possible censors should be used (<,≤,≥,>).
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
    
    
    df = censored.result_to_components(df,results_col,censor_col,numeric_col)
    
    # Create True/False indicator for uncensored data
    non_censored_check = (df[censor_col] == '')
    # Create True/False indicator for left censored data
    left_censored_check = df[censor_col].isin(['<','≤'])
    # Determine the maximum detection limit
    max_detect_limit = df[left_censored_check][numeric_col].max()
    # Create True/False indicator for right censored data
    right_censored_check = df[censor_col].isin(['≥','>'])
    # Determine the minimum quantification limit
    min_quant_limit = df[right_censored_check][numeric_col].min()
    
    # Convert all right censored data and all uncensored data that is greater
    # than the minimum quanitification limit to be equal to one
    # another and slightly larger than the limit
    if sum(right_censored_check) > 0:
        df[numeric_col] = np.where((right_censored_check) | \
                                      (non_censored_check & (df[numeric_col] > min_quant_limit)),
                                      upper_conversion_factor*min_quant_limit,
                                      df[numeric_col])
    
    # Convert all left censored data and all uncensored data that is less
    # than the largest detection limit to be equal to one
    # another and less than the detection limit
    if sum(left_censored_check) > 0:
        df[numeric_col] = np.where((left_censored_check) | \
                                      (non_censored_check & (df[numeric_col] < max_detect_limit)),
                                      lower_conversion_factor*max_detect_limit,
                                      df[numeric_col])
    
    return df


def season_counts(df,
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
        A Series which describes the seasons within a trend analysis dataset:
            - The number of years with results
            - The number of seasons with results
            - The percentage of years with results
            - The percentage of seasons with results

    '''
    
    # Copy the DataFrame
    df = df.copy()
    
    # Determine trend_length and seasons per year
    trend_length = df[trend_len_col].iloc[0]
    
    years = len(df.groupby([trend_year_col]))
    seasons = len(df.groupby([trend_year_col,trend_season_col]))
    percent_years = 100 * years / trend_length
    percent_seasons = 100 * seasons / (seasons_per_year * trend_length)
    
    return pd.Series([years,seasons,percent_years,percent_seasons],
                     index=['YearsInPeriod','SeasonsInPeriod',
                            'PercentOfYears','PercentOfSeasons'])
    

#%% Trend analysis for an individual dataset

def trend_analysis(df,
                   results_col,
                   seasons_per_year,
                   seasonal_test=False,
                   year_column_for_slopes=trend_year_col,
                   season_column=trend_season_col,
                   censored_results=False,
                   censored_conversions=[0.5, 1.1],
                   seasonality_alpha=0.05,
                   direction_confidence_categories = {0.90:'Very likely', 0.67:'Likely'},
                   direction_neutral_category='Indeterminate',
                   slope_percentile_method='hazen',
                   slope_confidence_interval=90):
    
    # Copy the DataFrame
    df = df.copy()
    
    # Initialise an empty output series
    output = []
    
    # Describe the dataset depending on whether results are censored or numeric
    if censored_results:
        output.append(describe_censored_data(df,results_col))
        # Convert censored data to have common censorship thresholds
        df = prep_censored_data_for_trends(df,
                                           results_col = results_col,
                                           lower_conversion_factor = censored_conversions[0],
                                           upper_conversion_factor = censored_conversions[1])
    else:
        output.append(describe_numeric_data(df,results_col))
    
    # Describe the season counts of the dataset
    output.append(season_counts(df,seasons_per_year))
    
    # Create variable for numeric column
    if censored_results:
        numeric_column = results_col[1]
    else:
        numeric_column = results_col
    
    # Sort by date
    df = df.sort_values(by=[trend_date_col])
    
    # Perform a Seasonality test
    output.append(kruskal_wallis_test(df,
                                      numeric_column,
                                      season_column,
                                      seasonality_alpha))
    
    # Perform a trend direction analysis
    output.append(trend_direction(df,
                                  numeric_column,
                                  seasonal_test,
                                  season_column,
                                  direction_confidence_categories,
                                  direction_neutral_category))
    
    # Perform a trend magnitude analysis
    output.append(trend_magnitude(df,
                                  numeric_column,
                                  year_column_for_slopes,
                                  seasonal_test,
                                  season_column,
                                  seasons_per_year,
                                  slope_percentile_method,
                                  slope_confidence_interval))
    
    return pd.concat(output)

#%% Trend analysis for a dataset

def trends(df,
           results_col,
           seasons_per_year,
           trend_lengths,
           end_years=[current_water_year()-1],
           end_month=6,
           seasonal_test=False,
           groupby_columns=None,
           date_col='DateTime',
           date_format=None,
           reduction_method=None,
           annual_date='-10-16',
           focus_high_potential=True,
           include_negative_interval=False,
           precision_tolerance_to_drop_censor=0.25,
           precision_rounding=True,
           output_format_end='%Y',
           output_format_period='%b %Y',
           year_column_for_slopes=trend_year_col,
           season_column = trend_season_col,
           censored_results=False,
           censored_conversions=[0.5, 1.1],
           seasonality_alpha=0.05,
           direction_confidence_categories = {0.90:'Very likely', 0.67:'Likely'},
           direction_neutral_category = 'Indeterminate',
           slope_percentile_method='hazen',
           slope_confidence_interval=90):
    
    # Copy the DataFrame
    df = df.copy()
    
    # If no groups create empty list
    if groupby_columns == None:
        groupby_columns = []
    
    # Classify results into a trend season based on season frequency and trend year
    df = define_seasons(df,seasons_per_year,date_col,date_format,end_month)
    
    # Reduce multiple results within a season to a single value
    if (censored_results) & (reduction_method in ['median','average','maximum','minimum']):
        df = season_reduction_censored_stats(df,
                                             results_col,  
                                             groupby_columns,
                                             reduction_method,
                                             focus_high_potential,
                                             include_negative_interval,
                                             precision_tolerance_to_drop_censor,
                                             precision_rounding)
    else:
        df = season_reduction(df,
                             results_col,
                             groupby_columns,
                             reduction_method,
                             end_month,
                             annual_date)
    
    # Create dataset for each trend period
    df = trend_periods(df,
                       trend_lengths,
                       end_years,
                       end_month,
                       output_format_end,
                       output_format_period)
    
    trend_groups = groupby_columns + [trend_end_col,trend_len_col,trend_period_col,freq_col,freq_num_col]
    
    # Analyse each dataset
    df = df.groupby(trend_groups).apply(trend_analysis,
                                        results_col,
                                        seasons_per_year,
                                        seasonal_test,
                                        year_column_for_slopes,
                                        season_column,
                                        censored_results,
                                        censored_conversions,
                                        seasonality_alpha,
                                        direction_confidence_categories,
                                        direction_neutral_category,
                                        slope_percentile_method,
                                        slope_confidence_interval)
    
    
    return df