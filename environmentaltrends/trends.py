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
    kw_pvalue_col: str = 'KW-pValue'
    kw_seasonality_col: str = 'Seasonality'
    mk_svalue_col: str = 'MK-S'
    mk_variance_col: str = 'MK-Variance'
    applied_seasonality_col: str = 'AppliedSeasonality'
    mk_pvalue_col: str = 'MK-pvalue'
    confidence_col: str = 'IncreasingLikelihood'
    trend_category_col: str = 'TrendDirection'
    median_slope_col: str = 'SenSlope'
    lower_slope_col: str = 'LowerSlope'
    upper_slope_col: str = 'UpperSlope'
    
    
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
            self.year_col = '_CalendarYear_'
            self.month_col = '_CalendarMonth_'
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
        
        # Validation
        _validate_trend_end_month(trend_end_month)
        _validate_seasons_per_year(seasons_per_year)
        _validate_groupby_cols(self, groupby_cols)
        _validate_annual_midpoint_date(annual_midpoint_date)
        _validate_trend_lengths_and_end_years(trend_lengths, end_years)
        
        # Classify results into a trend season based on season frequency and trend year
        df = _define_seasons(self, seasons_per_year, trend_end_month)
        
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