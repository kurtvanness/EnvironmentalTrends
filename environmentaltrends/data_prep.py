'''
These functions prep the data for trend analyses

'''

import numpy as np
import pandas as pd

def _define_seasons(tdf,
                    seasons_per_year,
                    trend_end_month):
    
    # Copy the DataFrame
    df = tdf.data.copy()
    
    # Add columns for the data frequency and number of seasons per year
    freq_dict = {
        1:'Annually',
        2:'Semiannually',
        3:'Triannually',
        4:'Quarterly',
        6:'Bimonthly',
        12:'Monthly'
        }
    df[tdf.freq_col] = freq_dict[seasons_per_year]
    df[tdf.freq_num_col] = seasons_per_year
    
    # Determine trend year (where year is aligned with trend period)
    df[tdf.trend_year_col] = np.where(df[tdf.month_col] > trend_end_month,
                                      df[tdf.year_col] + 1,
                                      df[tdf.year_col])
    
    # Determine trend month (shifted from Calender months)
    # Adjustements by 1 are for converting between 0- and 1-based index
    df[tdf.trend_month_col] = ((df[tdf.month_col] + (12-trend_end_month) - 1)
                                           %12 + 1)
    
    # Determine trend season (shifted from Calendar seasons)
    df[tdf.trend_season_col] = ((df[tdf.trend_month_col] - 1)
                                            //int(12/seasons_per_year) + 1)
    
    return df


def _season_reduction(tdf,
                      df,
                      trend_end_month,
                      groupby_cols=None,
                      reduction_method=None,
                      annual_midpoint_date=None):
    
    # Add created columns to groupby_cols
    if groupby_cols == None:
        reduction_groupby_cols = []
    else:
        reduction_groupby_cols = groupby_cols.copy()
    
    reduction_groupby_cols += [tdf.freq_col,
                               tdf.freq_num_col,
                               tdf.trend_year_col,
                               tdf.trend_season_col]
    
    # Check if there are any seasons with multiple results
    if not df.duplicated(subset = reduction_groupby_cols).any():
        return df
    # If multiple results, then check that reduction method is not None
    elif reduction_method == None:
        raise ValueError('Multiple results exist within at least one season. '
            'Resolve any instances or choose a reduction method so that there '
            'is only ever a single result within each season for each group.')
    # Else check for valid reduction method
    elif reduction_method not in ['first','last','midpoint',
                            'median', 'mean','average','maximum','minimum']:
        raise ValueError('The value provided for reduction_method is not '
            f'a valid option. The provided value was {reduction_method}')
    
    # Apply reduction method
    
    # If statistical reduction
    if reduction_method in ['median','mean','average','maximum','minimum']:
        # Check for censored data
        if tdf.censored_values:
            raise ValueError(f'The {reduction_method} reduction method is not '
                'available for censored data at this time. Choose another '
                'reduction method or reduce the data without this package.')
        # Apply simple stat reduction
        if reduction_method == 'median':
            df = df.groupby(reduction_groupby_cols)[tdf.value_col].median()
        elif reduction_method in ['mean','average']:
            df = df.groupby(reduction_groupby_cols)[tdf.value_col].mean()
        elif reduction_method == 'maximum':
            df = df.groupby(reduction_groupby_cols)[tdf.value_col].max()
        elif reduction_method == 'minimum':
            df = df.groupby(reduction_groupby_cols)[tdf.value_col].min()
        # Reset index
        df = df.reset_index()
    # If date based reduction, check for multiple values for same date or
    # for same month depending on supplied time format
    else:
        if tdf.date_col != None:
            if df.duplicated(subset = reduction_groupby_cols +
                                                     [tdf.date_col]).any():
                raise Exception('There are multiple values supplied for the '
                    'same DateTime within a group. Date based reduction '
                    'cannot be used as these values are not able to be '
                    'sorted. Resolve these values or choose a statistical '
                    'reduction method (min, median, mean, max).')
        else:
            if df.duplicated(subset = reduction_groupby_cols +
                                                     [tdf.month_col]).any():
                raise Exception('There are multiple values supplied for the '
                    'same month within a group. Date based reduction '
                    'cannot be used as these values are not able to be '
                    'sorted. Resolve these values or choose a statistical '
                    'reduction method (min, median, mean, max).')
    
    if reduction_method in ['first','last']:
        if tdf.date_col != None:
            # Sort by date column
            df = df.sort_values(by=tdf.date_col)
        else:
            # Sort by date column
            df = df.sort_values(by=tdf.month_col)
            
        # Keep first/last result in each season
        df = df.drop_duplicates(subset=reduction_groupby_cols,
                                keep=reduction_method)
    elif reduction_method == 'midpoint':
        # If date column provided, determine the date of the midpoint
        if tdf.date_col != None:
            
            # Only quarterly and monthly seasons have midpoints mid-month
            # Use midnight of the 16th (end of 15th)
            df['__MidpointDay__'] = np.where(df[tdf.freq_num_col].isin([4,12]),
                                             16,
                                             1)
            
            # Determine the trend month that corresponds with the provided
            # annual midpoint month
            if annual_midpoint_date != None:
                # Correct midpoint day if there is an annual override date
                df['__MidpointDay__'] = np.where(df[tdf.freq_num_col] == 1,
                                                 annual_midpoint_date['day'],
                                                 df['__MidpointDay__'])
                # Determine the trend month that corresponds with the provided
                # annual midpoint month
                trend_annual_midpoint_month = (
                    (annual_midpoint_date['month'] - trend_end_month -1)%12 + 1
                    )
            else:
                trend_annual_midpoint_month = 7
            
            # Determine the different between 
            trend_annual_midpoint_month_diff = trend_annual_midpoint_month - 7
            
            
            # Determine the midpoint year and month for all seasons per year
            conditions = [
                # Use the same year and month of date for monthly data
                (df[tdf.freq_num_col] == 12),
                
                # For odd trend months (in first month of 2-month interval)
                # then use the year/month from the month that follows
                # Use the same year and month of date for bimonthly data
                # if the trend month is even (second month of 2-month period)
                # else add a month to the date and use that year/month
                (df[tdf.freq_num_col] == 6) & (df[tdf.trend_month_col]%2 == 1),
                (df[tdf.freq_num_col] == 6) & (df[tdf.trend_month_col]%2 == 0),
                
                # Use similar logic for quarterly data
                (df[tdf.freq_num_col] == 4) & (df[tdf.trend_month_col]%3 == 1),
                (df[tdf.freq_num_col] == 4) & (df[tdf.trend_month_col]%3 == 2),
                (df[tdf.freq_num_col] == 4) & (df[tdf.trend_month_col]%3 == 0),
                
                # Use similar logic for triannual data
                (df[tdf.freq_num_col] == 3) & (df[tdf.trend_month_col]%4 == 1),
                (df[tdf.freq_num_col] == 3) & (df[tdf.trend_month_col]%4 == 2),
                (df[tdf.freq_num_col] == 3) & (df[tdf.trend_month_col]%4 == 3),
                (df[tdf.freq_num_col] == 3) & (df[tdf.trend_month_col]%4 == 0),
                
                # Use similar logic for semi-annual data
                (df[tdf.freq_num_col] == 2) & (df[tdf.trend_month_col]%6 == 1),
                (df[tdf.freq_num_col] == 2) & (df[tdf.trend_month_col]%6 == 2),
                (df[tdf.freq_num_col] == 2) & (df[tdf.trend_month_col]%6 == 3),
                (df[tdf.freq_num_col] == 2) & (df[tdf.trend_month_col]%6 == 4),
                (df[tdf.freq_num_col] == 2) & (df[tdf.trend_month_col]%6 == 5),
                (df[tdf.freq_num_col] == 2) & (df[tdf.trend_month_col]%6 == 0),
                
               # Use similar logic for annual data
               (df[tdf.freq_num_col] == 1) & (df[tdf.trend_month_col]%12 == 1),
               (df[tdf.freq_num_col] == 1) & (df[tdf.trend_month_col]%12 == 2),
               (df[tdf.freq_num_col] == 1) & (df[tdf.trend_month_col]%12 == 3),
               (df[tdf.freq_num_col] == 1) & (df[tdf.trend_month_col]%12 == 4),
               (df[tdf.freq_num_col] == 1) & (df[tdf.trend_month_col]%12 == 5),
               (df[tdf.freq_num_col] == 1) & (df[tdf.trend_month_col]%12 == 6),
               (df[tdf.freq_num_col] == 1) & (df[tdf.trend_month_col]%12 == 7),
               (df[tdf.freq_num_col] == 1) & (df[tdf.trend_month_col]%12 == 8),
               (df[tdf.freq_num_col] == 1) & (df[tdf.trend_month_col]%12 == 9),
               (df[tdf.freq_num_col] == 1) & (df[tdf.trend_month_col]%12 ==10),
               (df[tdf.freq_num_col] == 1) & (df[tdf.trend_month_col]%12 ==11),
               (df[tdf.freq_num_col] == 1) & (df[tdf.trend_month_col]%12 == 0)
                ]
            
            midpoint_month = [
                df[tdf.month_col],
                
                (df[tdf.date_col] + pd.DateOffset(months=1)).dt.month,
                df[tdf.month_col],
                
                (df[tdf.date_col] + pd.DateOffset(months=1)).dt.month,
                df[tdf.month_col],
                (df[tdf.date_col] + pd.DateOffset(months=-1)).dt.month,
                
                (df[tdf.date_col] + pd.DateOffset(months=2)).dt.month,
                (df[tdf.date_col] + pd.DateOffset(months=1)).dt.month,
                df[tdf.month_col],
                (df[tdf.date_col] + pd.DateOffset(months=-1)).dt.month,
                
                (df[tdf.date_col] + pd.DateOffset(months=3)).dt.month,
                (df[tdf.date_col] + pd.DateOffset(months=2)).dt.month,
                (df[tdf.date_col] + pd.DateOffset(months=1)).dt.month,
                df[tdf.month_col],
                (df[tdf.date_col] + pd.DateOffset(months=-1)).dt.month,
                (df[tdf.date_col] + pd.DateOffset(months=-2)).dt.month,
                
                (df[tdf.date_col] + pd.DateOffset(
                    months=6 + trend_annual_midpoint_month_diff)
                    ).dt.month,
                (df[tdf.date_col] + pd.DateOffset(
                    months=5 + trend_annual_midpoint_month_diff)
                    ).dt.month,
                (df[tdf.date_col] + pd.DateOffset(
                    months=4 + trend_annual_midpoint_month_diff)
                    ).dt.month,
                (df[tdf.date_col] + pd.DateOffset(
                    months=3 + trend_annual_midpoint_month_diff)
                    ).dt.month,
                (df[tdf.date_col] + pd.DateOffset(
                    months=2 + trend_annual_midpoint_month_diff)
                    ).dt.month,
                (df[tdf.date_col] + pd.DateOffset(
                    months=1 + trend_annual_midpoint_month_diff)
                    ).dt.month,
                (df[tdf.date_col] + pd.DateOffset(
                    months=0 + trend_annual_midpoint_month_diff)
                    ).dt.month,
                (df[tdf.date_col] + pd.DateOffset(
                    months=-1 + trend_annual_midpoint_month_diff)
                    ).dt.month,
                (df[tdf.date_col] + pd.DateOffset(
                    months=-2 + trend_annual_midpoint_month_diff)
                    ).dt.month,
                (df[tdf.date_col] + pd.DateOffset(
                    months=-3 + trend_annual_midpoint_month_diff)
                    ).dt.month,
                (df[tdf.date_col] + pd.DateOffset(
                    months=-4 + trend_annual_midpoint_month_diff)
                    ).dt.month,
                (df[tdf.date_col] + pd.DateOffset(
                    months=-5 + trend_annual_midpoint_month_diff)
                    ).dt.month
                
                ]
            
            midpoint_year = [
                df[tdf.year_col],
                
                (df[tdf.date_col] + pd.DateOffset(months=1)).dt.year,
                df[tdf.year_col],
                
                (df[tdf.date_col] + pd.DateOffset(months=1)).dt.year,
                df[tdf.year_col],
                (df[tdf.date_col] + pd.DateOffset(months=-1)).dt.year,
                
                (df[tdf.date_col] + pd.DateOffset(months=2)).dt.year,
                (df[tdf.date_col] + pd.DateOffset(months=1)).dt.year,
                df[tdf.year_col],
                (df[tdf.date_col] + pd.DateOffset(months=-1)).dt.year,
                
                (df[tdf.date_col] + pd.DateOffset(months=3)).dt.year,
                (df[tdf.date_col] + pd.DateOffset(months=2)).dt.year,
                (df[tdf.date_col] + pd.DateOffset(months=1)).dt.year,
                df[tdf.year_col],
                (df[tdf.date_col] + pd.DateOffset(months=-1)).dt.year,
                (df[tdf.date_col] + pd.DateOffset(months=-2)).dt.year,
                
                (df[tdf.date_col] + pd.DateOffset(
                    months=6 + trend_annual_midpoint_month_diff)
                    ).dt.year,
                (df[tdf.date_col] + pd.DateOffset(
                    months=5 + trend_annual_midpoint_month_diff)
                    ).dt.year,
                (df[tdf.date_col] + pd.DateOffset(
                    months=4 + trend_annual_midpoint_month_diff)
                    ).dt.year,
                (df[tdf.date_col] + pd.DateOffset(
                    months=3 + trend_annual_midpoint_month_diff)
                    ).dt.year,
                (df[tdf.date_col] + pd.DateOffset(
                    months=2 + trend_annual_midpoint_month_diff)
                    ).dt.year,
                (df[tdf.date_col] + pd.DateOffset(
                    months=1 + trend_annual_midpoint_month_diff)
                    ).dt.year,
                (df[tdf.date_col] + pd.DateOffset(
                    months=0 + trend_annual_midpoint_month_diff)
                    ).dt.year,
                (df[tdf.date_col] + pd.DateOffset(
                    months=-1 + trend_annual_midpoint_month_diff)
                    ).dt.year,
                (df[tdf.date_col] + pd.DateOffset(
                    months=-2 + trend_annual_midpoint_month_diff)
                    ).dt.year,
                (df[tdf.date_col] + pd.DateOffset(
                    months=-3 + trend_annual_midpoint_month_diff)
                    ).dt.year,
                (df[tdf.date_col] + pd.DateOffset(
                    months=-4 + trend_annual_midpoint_month_diff)
                    ).dt.year,
                (df[tdf.date_col] + pd.DateOffset(
                    months=-5 + trend_annual_midpoint_month_diff)
                    ).dt.year
                ]
            
        else:
            raise ValueError(f'The {reduction_method} reduction method is not '
                'available for year/month data at this time. Choose another '
                'reduction method or reduce the data to a single result per '
                'season without this package.')
            
        # Use conditions to get month and year for midpoint
        df['__MidpointMonth__'] = np.select(conditions,
                                           midpoint_month,
                                           np.nan)
        df['__MidpointYear__'] = np.select(conditions,
                                           midpoint_year,
                                           np.nan)
        # Construct midpoint date
        df['__Midpoint__'] = pd.to_datetime(dict(year=df['__MidpointYear__'],
                                                 month=df['__MidpointMonth__'],
                                                 day=df['__MidpointDay__']))
        # Determine proximity of result to midpoint
        df['__Proximity__'] = abs(df[tdf.date_col] - df['__Midpoint__'])
        
        # Sort by proximity and use earlier result if ties
        df = df.sort_values(by=['__Proximity__', tdf.date_col])
        # Keep single result closest to midpoint date
        df = df.drop_duplicates(subset=reduction_groupby_cols)
        # Drop temporary column
        df = df.drop(columns=['__MidpointDay__', '__MidpointMonth__',
                    '__MidpointYear__', '__Midpoint__', '__Proximity__'])
    
    return df

def _trend_periods(tdf,
                   df,
                   trend_end_month,
                   trend_lengths,
                   end_years):

    
    
    # Initiate output DataFrame
    output_df = []
    
    # Loop through end dates
    for end_year in end_years:
        
        # Copy data
        data = df.copy()
        
        # Only consider data up through the end year
        data = data[data[tdf.trend_year_col] <= end_year]
        
        # Determine DateTime for the end of the trend period
        end = (pd.to_datetime('{}-{}-1'.format(end_year, trend_end_month))
                   + pd.DateOffset(months=1)
                   + pd.Timedelta(seconds=-1))
        
        # Add a column for the end of the trend, if output format specified
        if tdf.output_format_trend_end:
            data[tdf.trend_end_col] = end.strftime(tdf.output_format_trend_end)
        
        # Sort trend lengths in descending order
        # This allows the use of the same data variable for each trend length
        trend_lengths.sort(reverse=True)
        
        # Loop through trend lengths for each end date
        for trend_length in trend_lengths:
            
            # Determine the start year (the difference between the first year
            # and last year in a trend period is 1 less than the trend length)
            start_year = end_year - (trend_length - 1)
            
            # Only keep years after and including the start year
            data = data[data[tdf.trend_year_col] >= start_year]
            
            # Add a column for the trend period, if output format specified
            if tdf.output_format_trend_period:
                # Determine DateTime for the start of the trend period
                start = (pd.to_datetime('{}-{}-1'.format(end_year,
                                                         trend_end_month))
                            + pd.DateOffset(months=1)
                            + pd.DateOffset(years=-trend_length))
                
                data[tdf.trend_period_col] = (
                    start.strftime(tdf.output_format_trend_period) + ' to '
                    + end.strftime(tdf.output_format_trend_period))
            
            # Add additional information columns
            data[tdf.trend_len_col] = trend_length
            
            # Append trend data to output
            output_df.append(data.copy())
        
    # Combine dataframes
    output_df = pd.concat(output_df).reset_index(drop=True)
    
    return output_df
