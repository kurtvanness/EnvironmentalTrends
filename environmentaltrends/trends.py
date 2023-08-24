# -*- coding: utf-8 -*-
"""
Created on Sun Jun 25 11:28:27 2023

@author: KurtV
"""

import pandas as pd
from dataclasses import dataclass, field
from environmentaltrends.validation import (
    _validate_tdf,
    _validate_trend_end_month,
    _validate_seasons_per_year,
    _validate_groupby_cols,
    _validate_annual_midpoint_date,
    _validate_trend_lengths_and_end_years)
from environmentaltrends.data_prep import (
    _define_seasons,
    _season_reduction,
    _trend_periods)
from environmentaltrends.single_trend_analysis import trend_analysis


@dataclass
class TrendData:
    '''
    This dataclass validates and pre-processes a dataframe for generating
    trends.

    Parameters
    ----------
    data : DataFrame
        A pandas dataframe containing time series data
    value_col : string
        The column name for the column with censored results
    date_col : string
        The column name for the column containing DateTime information
    year_col : string
        Alternative option for a DateTime column. The column name for the
        column containing the year.
    month_col : string
        Alternative option for a DateTime column. The column name for the
        column containing the month as an integer.
    seasonality_alpha : float
        The p-value threshold for determining seasonality using the Kruskall-
        Wallis test. The default is 0.05.
    confidence_categories : dictionary of float keys and string values
        The cutoffs used to establish trend likelihood categories. The default
        is {0.90: 'Very likely', 0.67: 'Likely'}, meaning 90%+ confidence is
        considered 'very likely' and 67%+ is consider 'likely'.
    neutral_category : string
        The category name for trends that do not meet the minimum threshold
        set in confidence_categories. The default is 'Indeterminate'.
    confidence_interval : float
        The confidence interval to use for determining the upper and lower
        slope estimates. The default is 90.
    percentile_method : string
        The percentile method to use for setting the confidence interval for
        slopes. The default is 'hazen'. Additional options include 'weiball',
        'tukey', 'blom', 'excel'.
    output_format_trend_end : string
        The format to use in the output column containing the end of the
        trend period. The default is '%Y' which produces the year only.
    output_format_trend_period : string
        The format to use in the output column containing the entire trend
        period. The default is '%b %Y' to output month and year for the start
        and end of the trend period.
    censored_values: boolean
        An indicator for whether censored values are expected in the value_col.
        The default is False.
    censored_kwargs : dictionary
        A dictionary of keywords and values to use for generation of a
        CensoredData object. Users should be familiar with the Python package
        censoredsummarystats.
    lower_conversion_factor : float
        The conversion factor to use for left censored values.
        The default is 0.5 which assumes positive values only.
    upper_conversion_factor : float
        The conversion factor to use for right censored values.
        The default is 1.1 which assumes positive values only.
    '''
    data: pd.core.frame.DataFrame
    value_col: any
    date_col: str = None
    year_col: str = None
    month_col: str = None
    seasonality_alpha: float = 0.05
    confidence_categories: dict[float,str] = field(
        default_factory=lambda: {0.90: 'Very likely', 0.67: 'Likely'}
    )
    neutral_category: str = 'Indeterminate'
    confidence_interval: float = 90
    percentile_method: str = 'hazen'
    output_format_trend_end: str = '%Y'
    output_format_trend_period: str = '%b %Y'
    censored_values: bool = False
    censored_kwargs: dict = field(default_factory=dict)
    lower_conversion_factor: float = 0.5
    upper_conversion_factor: float = 1.1
    freq_col: str = 'Frequency'
    freq_num_col: str = 'SeasonsPeryear'
    trend_year_col:str = 'TrendYear'
    trend_month_col:str = 'TrendMonth'
    trend_season_col:str = 'TrendSeason'
    trend_end_col: str = 'TrendEnd'
    trend_len_col: str = 'TrendLength'
    trend_period_col: str = 'TrendPeriod'
    count_col: str = 'ValueCount'
    minimum_col: str = 'Minimum'
    median_col: str = 'Median'
    average_col: str = 'Average'
    maximum_col: str = 'Maximum'
    left_censored_count_col: str = 'CountBelowDetect'
    left_censored_min_col: str = 'DetectLimitMin'
    left_censored_max_col: str = 'DetectLimitMax'
    right_censored_count_col: str = 'CountAboveQuant'
    right_censored_min_col: str = 'QuantLimitMin'
    right_censored_max_col: str = 'QuantLimitMax'
    years_in_trend_col: str = 'YearsInPeriod'
    seasons_in_trend_col: str = 'SeasonsInPeriod'
    percent_of_years_col: str = 'PercentOfYears'
    percent_of_seasons_col: str = 'PercentOfSeasons'
    kw_pvalue_col: str = 'KW-pValue'
    kw_seasonality_col: str = 'Seasonality'
    applied_seasonality_col: str = 'AppliedSeasonality'
    mk_svalue_col: str = 'MK-S'
    mk_variance_col: str = 'MK-Variance'
    mk_pvalue_col: str = 'MK-pvalue'
    confidence_col: str = 'IncreasingLikelihood'
    trend_category_col: str = 'TrendDirection'
    median_slope_col: str = 'SenSlope'
    lower_slope_col: str = 'LowerSlope'
    upper_slope_col: str = 'UpperSlope'
    _midpointday_col: str = field(default='__tempMidPointDay__', repr=False)
    _midpointmonth_col: str = field(
        default='__tempMidPointMonth__', repr=False)
    _midpointyear_col: str = field(default='__tempMidPointYear__', repr=False)
    _midpointdate_col: str = field(default='__tempMidPointDate__', repr=False)
    _midpointproximity_col: str = field(
        default='__tempMidPointProximity__', repr=False)
    
    
    def __post_init__(self):
        
        #%% Validate inputs
        
        _validate_tdf(self)
        
        #%% Convert values to float if censored_values is False
        if not self.censored_values:
            try:
                self.data[self.value_col] = (self.data[self.value_col]
                                                 .astype(float))
            except:
                raise ValueError('Values could not be converted to a '
                    'float data type. If values contain censor characters '
                    'such as < or > then set censored_values = True')
        
        #%% Complete year and month info
        if self.date_col != None:
            self.year_col = '__tempCalendarYear__'
            self.month_col = '__tempCalendarMonth__'
            self.data[self.year_col] = self.data[self.date_col].dt.year
            self.data[self.month_col] = self.data[self.date_col].dt.month
        
    #%% Trend methods
    
    def trends(self,
               seasons_per_year,
               trend_lengths,
               end_years,
               groupby_cols=None,
               seasonal_test=None,
               trend_end_month=6,
               reduction_method=None,
               annual_midpoint_date=None):
        '''
        This method outputs a DataFrame containing trend results which include
        likelihood of trend direction, trend slope, and a confidence interval
        for the trend slope.

        Parameters
        ----------
        seasons_per_year : int
            The number of seasons that should be analysed within a given year.
            Accepted values are 1,2,3,4,6,12, where a value of 1 would generate
            annual trends and 12 would generate monthly trends.
        trend_lengths : list of integers
            A list of the desired trend lengths. Lists can include a single
            value or multiple values.
        end_years : list of integers
            A list of years that describe when the trend period ends.
        groupby_cols : list
            A list of columns that should be used to differentiate time series.
            The default is None.
        seasonal_test : boolean, optional
            Specify whether seasonal or non-seasonal trend analyses should be
            forced regardless of the seasonality test results. The default is
            None which will apply the trend analyses indicated by the
            seasonality test.
        trend_end_month : int, optional
            Trends don't have to align with the calendar year. This setting
            specifies the last month in the trend period.
            The default is 6 (June) to align with water years.
        reduction_method : string, optional
            When multiple results exist within a single season, trend analyses
            require that they are reduced to a single value. Options for this
            include using a statistical result ('median','mean','average',
            'maximum','minimum') or a temporal result ('first','last',
            'midpoint'), where midpoint is the result closest to the midpoint
            of the season. The default is None.
        annual_midpoint_date : dictionary with month and day keywords, optional
            If the reduction_method is 'midpoint' and seasons_per_year is 1,
            then an alternative target date can be used for reducing annual
            trend data. For example, to target values closest to the middle
            of New Zealand spring, use: {'month':10, 'day':16}.
            The default is None.
        
        Returns
        -------
        DataFrame
            Contains a DataFrame of trend results.

        '''
        
        # Validation
        _validate_trend_end_month(trend_end_month)
        _validate_seasons_per_year(seasons_per_year)
        _validate_groupby_cols(self, groupby_cols)
        _validate_annual_midpoint_date(annual_midpoint_date)
        _validate_trend_lengths_and_end_years(trend_lengths, end_years)
        
        # Classify results into a trend season based on season frequency and trend year
        df = _define_seasons(self, seasons_per_year, trend_end_month)
        
        # Create empty list if no groups
        if groupby_cols == None:
            groupby_cols = []
        
        # Reduce multiple results within a season to a single result
        if df.duplicated(subset = groupby_cols +
                                   [self.trend_year_col,
                                    self.trend_season_col]).any():
            df = _season_reduction(self, df, trend_end_month, groupby_cols,
                                   reduction_method, annual_midpoint_date)
            
        # Create dataset for each trend period
        df = _trend_periods(self,
                            df,
                            trend_end_month,
                            trend_lengths,
                            end_years)
        
        # Create list of columns to group by
        trend_groups = groupby_cols + [self.freq_col,
                                          self.freq_num_col,
                                          self.trend_len_col]
        if self.output_format_trend_end != None:
            trend_groups += [self.trend_end_col]
        if self.output_format_trend_period != None:
            trend_groups += [self.trend_period_col]
        
        # Analyse each dataset
        df = df.groupby(trend_groups).apply(trend_analysis,
                                            self,
                                            seasons_per_year,
                                            seasonal_test)
        
        # Reset index
        df = df.reset_index()
        
        return df