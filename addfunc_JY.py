import numpy as np #wind  
import pandas as pd

#train_dataset data 변환 (weekday -> weekend, wind dir, speed -> U10, V10)
class TrainDataTransform:
    def __init__(self, df):
        self.df = df

    def transform(self):
        self.is_weekend()
        self.wind_data_to_UV()
        return self.df

    #weekend 판별 함수
    def is_weekend(self):
        self.df['is_weekend'] = np.where(self.df['weekday'] > 4, 1, 0)
        
    def wind_data_to_UV(self):
        self.df['U10'] = self.df['windspeed_10m'] * np.cos(np.radians(270 - self.df['winddirection_10m']))
        self.df['V10'] = self.df['windspeed_10m'] * np.sin(np.radians(270 - self.df['winddirection_10m']))
        
        