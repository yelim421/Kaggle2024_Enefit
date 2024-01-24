import polars as pl
import numpy as np
from datetime import timedelta

class MissingValue:
    def __init__(self, df):
        self.df = df
        
    def fill(self):
        #self.df.df_data.
        self.df.df_data = self.fill_target(self.df.df_data)
        self.df.df_forecast_weather = self.fill_radiation(self.df.df_forecast_weather)
        self.df.df_forecast_weather = self.fill_summertime(self.df.df_forecast_weather)
        return self.df
        
    def _interpolate_group(self, group):
        group['target'] = group['target'].interpolate(method='linear')
        return group
    
    def fill_target(self, df):
        return df.groupby(['prediction_unit_id', 'is_consumption']).apply(self._interpolate_group)
    
    def fill_radiation(self, df):
        # Assuming the radiation filling logic is correctly implemented in your original code
        # Implement the logic using `polars` functions to improve performance
        # ...
        return df
    
    def fill_summertime(self, df):
        # Use `polars` functionality to handle missing dates more efficiently
        # ...
        return df

class WeatherTransformer:
    def __init__(self, df):
        self.df = df

    def transform(self):
        self.df.df_forecast_weather = self.separate_tp(self.df.df_forecast_weather)
        self.df.df_forecast_weather = self.exp_forecast_hour(self.df.df_forecast_weather)
        self.df.df_historical_weather = self.snow_to_water(self.df.df_historical_weather)
        self.df.df_historical_weather = self.hist_roll(self.df.df_historical_weather)
        return self.df

    def snow_to_water(self, df):
        return df.with_columns(
            (df['snowfall']/7).alias('snowfall_mm'))

    def separate_tp(self, df):
        return df.with_columns(
            (df['total_precipitation'] - df['snowfall']/100).alias('rain'))

    def _exp(self, x):
        return np.exp(x) / np.exp(48)
        
    def exp_forecast_hour(self, df):
        return df.with_columns(
            df['hours_ahead'].apply(self._exp).alias('exp_hours_ahead'))

    def hist_roll(self, df):
        # Implement rolling logic for temperature and dew point
        # ...
        return df
