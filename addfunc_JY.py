import numpy as np #wind  
import pandas as pd

#train_dataset data 변환 (weekday -> weekend)
class TrainDataTransform:
    def __init__(self, df):
        self.df = df

    def transform(self):
        self.is_weekend()
        return self.df

    #weekend 판별 함수
    def is_weekend(self):
        self.df['is_weekend'] = np.where(self.df['weekday'] > 4, 1, 0)


#DataStorage data 변환 (electricity price, wind)
class DataStorageTransform:
    def __init__(self, df):
        self.df = df

    def transform(self):
        self.electricity_prices_to_positive()
        self.wind_data_to_UV()
        return self.df
    
    #elecrricity prices 음수값 양수로 바꾸는 함수
    def electricity_prices_to_positive(self):
        self.df.df_electricity_prices = self.df.df_electricity_prices.with_columns(
            self.df.df_electricity_prices['euros_per_mwh'].abs().alias('euros_per_mwh'))
        

    #historical wind speend & direction data to U10, V10
    def wind_data_to_UV(self):
        self.df.df_historical_weather = self.df.df_historical_weather.with_columns(
            (self.df.df_historical_weather['windspeed_10m'] * np.cos(np.radians(270 - self.df.df_historical_weather['winddirection_10m']))).alias('U10'),
            (self.df.df_historical_weather['windspeed_10m'] * np.sin(np.radians(270 - self.df.df_historical_weather['winddirection_10m']))).alias('V10')
            )
