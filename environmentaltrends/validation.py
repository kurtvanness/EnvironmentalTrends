'''
These functions validate user input values.

'''

import pandas as pd

#%% Validate TrendData

def _validate_tdf(tdf):
    
    #%% Check data input
    
    # Check that supplied data is a DataFrame
    if not isinstance(tdf.data, pd.core.frame.DataFrame):
        raise ValueError('The data supplied to TrendData() must be a '
            'pandas DataFrame. Instead, an object was passed with type: '
            f'{type(tdf.data).__name__}')
    
    # Ensure that provided data is a copy and not indexed
    tdf.data = tdf.data.copy().reset_index()
    
    #%% Check value_col exists as column in data
    
    if tdf.value_col not in tdf.data.columns:
        raise ValueError('The value column supplied to TrendData() '
            'was not found as a column in the data. Columns include '
            f'{tdf.data.columns.to_list()} which does not include '
            f'{repr(tdf.value_col)}.')
    
    #%% Check that there are no nan values or empty strings
    if ((tdf.data[tdf.value_col].isnull().any()) |
        (tdf.data[tdf.value_col].astype(str).str.len().min() == 0)):
        raise ValueError('Missing values need to be removed from the data '
            'before it can be analysed.')
    
    #%% Check year, month, date columns
    if (((tdf.date_col == None) &
        ((tdf.year_col == None) | (tdf.month_col == None))) |
        ((tdf.date_col != None) &
        ((tdf.year_col != None) | (tdf.month_col != None)))):
        raise ValueError('Either specify date_col or specify both year_col '
            'and month_col.')
    
    # Check that date column is datetime format
    if tdf.date_col != None:
        if not (pd.api.types.is_datetime64_any_dtype(
                                    tdf.data[tdf.date_col].dtype)):
            raise ValueError('The values provided in the column '
                f'{tdf.date_col} must be datetime data type. Try using '
                'pd.to_datetime() to convert values.')
    
    # Check that year month columns are integer and month between 1 and 12
    if tdf.date_col == None:
        if not pd.api.types.is_integer_dtype(tdf.data[tdf.year_col]):
            raise ValueError('The values provided in the column '
                f'{tdf.year_col} must be integers.')
        if not pd.api.types.is_integer_dtype(tdf.data[tdf.month_col]):
            raise ValueError('The values provided in the column '
                f'{tdf.month_col} must be integers.')
        if ((tdf.data[tdf.month_col].max() > 12) |
            (tdf.data[tdf.month_col].min() < 1)):
            raise ValueError('The values provided in the column '
                f'{tdf.month_col} must be between 1 and 12.')
    
    #%% Check the input settings
    
    setting_mode = getattr(tdf, 'censored_values')
    if not isinstance(setting_mode, bool):
        raise ValueError(f'The value supplied to censored_values must be '
            'True or False. Instead, the value supplied was '
            f'{repr(setting_mode)}.')
        
    
    #%% Check column names for conflicts
    
    # Create list of column names that may be created
    created_columns = [attribute for attribute in tdf.__annotations__
                       if attribute.endswith('_col') and
                           attribute not in ['value_col',
                                             'year_col',
                                             'month_col',
                                             'date_col']]
    
    # Check the list for duplicates
    if len(created_columns) != len(set(created_columns)):
        raise ValueError('Two default column names have been set to the '
            'same name. Check that assigned values are unique and not '
            'identical to any default values.')
    
    
    for col in created_columns:
        col_name = getattr(tdf, col)
        # Check that columns created within methods don't conflict with column
        # names in provided data
        if (col_name in tdf.data.columns) & (col_name != tdf.value_col):
            raise ValueError('The data contains a column named '
                f'"{col_name}" which may be used in the output. Either '
                'rename this column or provide an alternative value for '
                f'{col} when initialising TrendData object.')
        # Check that new column names are string values
        if not isinstance(col_name, str):
            raise ValueError('Text values are required for column names. '
                f'The column name {col_name} should be changed to a text '
                f'value for {col}.')
    
    # Check that the data has no columns starting with double underscore
    dunder_cols = [col for col in tdf.data if col.startswith('__')]
    if len(dunder_cols) != 0:
        raise ValueError('Columns starting with "__" are used within '
            'the stat methods. Rename the following columns: '
            f'{dunder_cols}')
    
    #%% Check conversion factors
    if tdf.censored_values:
        # Check that the upper conversion factor is greater than 1
        if tdf.upper_conversion_factor <= 1:
            raise ValueError('The upper conversion factor needs to be greater '
                f'than 1. {tdf.upper_conversion_factor} was passed.')
        # Check that the lower conversion factor is less than 1
        if tdf.lower_conversion_factor >= 1:
            raise ValueError('The lower conversion factor needs to be less '
                f'than 1. {tdf.lower_conversion_factor} was passed.')

#%% Validate trend_end_month
def _validate_trend_end_month(end_month):
    
    # Check that end month is an integer
    if not isinstance(end_month, int):
        raise ValueError('The value supplied to trend_end_month must be '
            f'an integer. The value supplied was {end_month}')
    
    # Check that the end month provided is between 1 and 12
    if end_month - 1 not in range(12):
        raise ValueError('The value for trend_end_month must be between 1 and '
            f'12. The value provided was {end_month}')
        
#%% Validate seasons_per_year
def _validate_seasons_per_year(seasons_per_year):
    
    # Check that seasons_per_year is an integer
    if not isinstance(seasons_per_year, int):
        raise ValueError('The value supplied to seasons_per_year must be '
            f'an integer. The value supplied was {seasons_per_year}')
    
    # Check that the end month provided is between 1 and 12
    if seasons_per_year not in [1,2,3,4,6,12]:
        raise ValueError('The value for seasons_per_year must be a factor of '
            f'12. The value provided was {seasons_per_year}')

#%% Validate groupby columns

def _validate_groupby_cols(tdf, groupby_cols):
    
    # Skip None case
    if groupby_cols == None:
        pass
    else:
        # Check that groupby_cols is a list
        if not isinstance(groupby_cols, list):
            raise ValueError('groupby_cols needs to be a list. '
                f'The provided input was {groupby_cols}')
        
        # Check that column names are included in data
        if not (all(name in tdf.data.columns for name in groupby_cols)):
            raise ValueError('Columns used in groupby_cols were not found in '
                'the DataFrame. Names not found include: '
                f'{[x for x in groupby_cols if x not in tdf.data.columns]}')
        
        # Check for null values in groupby columns
        if tdf.data[groupby_cols].isnull().values.any():
            raise ValueError('Null values found in one of the columns used '
                'for grouping.')

#%% Validate annual midpoint date

def _validate_annual_midpoint_date(annual_midpoint_date):
    
    # Skip None case
    if annual_midpoint_date == None:
        pass
    else:
        # Check that annual_midpoint_date is a dictionary
        if not isinstance(annual_midpoint_date, dict):
            raise ValueError('annual_midpoint_date needs to be a dictionary. '
                f'The provided input was {annual_midpoint_date}')
        
        # Check that key names are only month and day
        if list(annual_midpoint_date.keys()) != ['month','day']:
            raise ValueError('annual_midpoint_date requires a dictionary with '
                'month and day provided as keys. The provided keys were '
                f'{list(annual_midpoint_date.keys())}')
        
        # Check that month value is integer between 1 and 12
        if ((not isinstance(annual_midpoint_date['month'], int)) |
                (annual_midpoint_date['month']-1 not in range(12))):
            raise ValueError('The value supplied as the month in '
                'annual_midpoint_date must be an integer between 1 and 12.'
                f'The value supplied was {annual_midpoint_date["month"]}')
        
        # Check that day value is integer between 1 and 31
        if ((not isinstance(annual_midpoint_date['day'], int)) |
                (annual_midpoint_date['day']-1 not in range(31))):
            raise ValueError('The value supplied as the month in '
                'annual_midpoint_date must be an integer between 1 and 31.'
                f'The value supplied was {annual_midpoint_date["day"]}')

#%% Validate trend lengths and end years

def _validate_trend_lengths_and_end_years(trend_lengths, end_years):
    
    # Check that trend lengths input is a list
    if not isinstance(trend_lengths, list):
        raise ValueError('trend_lengths needs to be a list. '
            f'The provided input was {trend_lengths}')
    # Check that the provided values are integers
    if not (all(isinstance(length, int) for length in trend_lengths)):
        raise ValueError('All trend lengths need to be integers. '
            f'The provided trend lengths were {trend_lengths}')
    # Check that trend_lengths are positive
    if not (all(length > 0 for length in trend_lengths)):
        raise ValueError('All trend lengths need to be positive. '
            f'The provided trend lengths were {trend_lengths}')
    
    # Check that end years input is a list
    if not isinstance(end_years, list):
        raise ValueError('end_years needs to be a list. '
            f'The provided input was {end_years}')
    # Check that the provided values are integers
    if not (all(isinstance(year, int) for year in end_years)):
        raise ValueError('All end years need to be integers. '
            f'The provided end years were {end_years}') 