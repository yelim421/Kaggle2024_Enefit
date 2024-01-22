import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from statsmodels.tsa.stattools import acf, pacf
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)


class DataTransformer:
    def __init__(self, df):
        self.df = df

    def transform(self):
        self.add_season()
        self.add_daypart()
        self.add_feels_like_temperature()
        self.add_total_precipitation()
        self.add_energy_usage_trend()
        self.add_temp_change()
        self.add_prec_change()
        self.add_autocorr_features()
        self.add_energy_price_volatility_and_trend()
        self.perform_clustering()
        self.analyze_transit_and_charging_access()
        return self.df

    def add_season(self):
        def get_season(month):
            if month in [3, 4, 5]:
                return 1 #spring
            elif month in [6, 7, 8]:
                return 2 #summer
            elif month in [9, 10, 11]:
                return 3 #fall
            else:
                return 4 #winter
        
        self.df['season'] = self.df['month'].apply(get_season)

    def add_daypart(self):
        def get_daypart(hour):
            if 5 <= hour < 12:
                return 1 #morning
            elif 12 <= hour < 17:
                return 2 #'Afternoon'
            elif 17 <= hour < 21:
                return 3 #Evening'
            else:
                return 4 #'Night'
        
        self.df['daypart'] = self.df['hour'].apply(get_daypart)

    def add_feels_like_temperature(self):
        def calculate_feels_like(T, u, v):
            wind_speed = (u**2 + v**2)**0.5
            if wind_speed < 4.8:
                return T
            else:
                return 13.12 + 0.6215 * T - 11.37 * (wind_speed ** 0.16) + 0.3965 * T * (wind_speed ** 0.16)

        self.df['feels_like_temp'] = self.df.apply(lambda row: calculate_feels_like(row['temperature'], row['10_metre_u_wind_component'], row['10_metre_v_wind_component']), axis=1)

    def add_total_precipitation(self):
        self.df['total_precipitation'] = self.df['snow'] + self.df['rain'] + self.df['preci']

    def add_energy_usage_trend(self, period = 7):
        self.df['energy_trend'] = self.df['target'].rolling(window = period).mean()

    def add_temp_change(self, interval = 24):
        self.df['temp_change'] = self.df['temperature'].diff(periods = interval)

    def add_prec_change(self):
        self.df['precipitation_change'] = self.df['total_precipitation'].diff()

    def add_autocorr_features(self, lags = 10):
        acf_values = acf(self.df['target'], nlags = lags)
        pacf_values = pacf(self.df['target'], nlags = lags)
        for i in range(lags+1):
            self.df[f'acf_lag_{i}'] = acf_values[i]
            self.df[f'pacf_lag{i}'] = pacf_values[i]

    def add_energy_price_volatility_and_trend(self, window = 7):
        self.df['energy_price_volatility'] = self.df['target'].rolling(window = window).std()

    def perform_clustering(self, n_clusters = 3, features = None):
        if features is None:
            features = ['target_24h', 'target_48h', 'temperature', 'cloudcover_total']
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(self.df[features])

        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.df['cluster'] = kmeans.fit_predict(scaled_data)

    def analyze_transit_and_charging_access(self):

        results = []
        for consumption_status in [0, 1]:
            subset = self.df[self.df['is_consumption'] == consumption_status]

            # 대중교통 이용률 분석
            business_hours_energy = subset[subset['is_business'] == 1]['target'].mean()
            non_business_hours_energy = subset[subset['is_business'] == 0]['target'].mean()
            transit_usage_estimate = business_hours_energy - non_business_hours_energy

            # 전기차 충전소 접근성 분석
            high_capacity_energy = subset[subset['installed_capacity'] > subset['installed_capacity'].median()]['target'].mean()
            low_capacity_energy = subset[subset['installed_capacity'] <= subset['installed_capacity'].median()]['target'].mean()
            charging_access_estimate = high_capacity_energy - low_capacity_energy

            results.append((consumption_status, transit_usage_estimate, charging_access_estimate))

        # 결과를 하나의 컬럼으로 합침
        for consumption_status, transit_estimate, charging_estimate in results:
            self.df[f'transit_usage_estimate_{consumption_status}'] = transit_estimate
            self.df[f'charging_access_estimate_{consumption_status}'] = charging_estimate

