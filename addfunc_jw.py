class MissingValue:
    def __init__(self, df):
        self.df = df
        
    def fill(self):
        self.df.df_data = self.fill_target(self.df.df_data)
        self.df.df_forecast_weather = self.fill_radiation(self.df.df_forecast_weather)
        self.df.df_forecast_weather = self.fill_summertime(self.df.df_forecast_weather)
        return self.df
        
    def _interpolate_group(self, group):
        group['target'] = group['target'].interpolate(method='linear')
        return group
    
    def fill_target(self, df):
        return pl.DataFrame(df.to_pandas().groupby(['prediction_unit_id', 'is_consumption']).apply(interpolate_group))
    
    def fill_radiation(self, df):
        rad = df.to_pandas()['surface_solar_radiation_downwards'].values
        idx = df['surface_solar_radiation_downwards'].is_null().to_numpy().nonzero()[0]
        for i, ind in enumerate(idx):
            tmp = df[idx][i]
            df_b1 = df.filter(
                pl.col('latitude')==tmp['latitude'], pl.col('longitude')==tmp['longitude'],
                abs(pl.col('forecast_datetime') - tmp['forecast_datetime']) < timedelta(days=2),
                pl.col('forecast_datetime').dt.hour() == tmp['forecast_datetime'].dt.hour(),
                pl.col('hours_ahead') == tmp['hours_ahead'])
            fillValue = df_b1['direct_solar_radiation'][1] / ((np.divide(df_b1['direct_solar_radiation'][0], df_b1['surface_solar_radiation_downwards'][0]) +
                np.divide(df_b1['direct_solar_radiation'][2], df_b1['surface_solar_radiation_downwards'][2]))/2)
            rad[ind] = fillValue
        df.replace('surface_solar_radiation_downwards', pl.Series(rad))
        return df
    
    def fill_summertime(self, df):
        missingDate = list(set(pd.date_range('2021-09-01', '2023-06-02', freq='h')[3:-22]) - set(df.to_pandas()['forecast_datetime'].unique()))
        hrs_ahead = 2
        add_df = pd.DataFrame()
        for date in missingDate:
            tmp = df.filter(abs(pl.col('forecast_datetime') - date) < timedelta(hours=2),
                            pl.col('hours_ahead') <= 2).sort('latitude', 'longitude').to_pandas()
            for index, row in tmp.iterrows():
                if row['hours_ahead'] == 1:
                    index_1 = index
                    values_1 = row
                elif row['hours_ahead'] == 2:
                    index_2 = index
                    values_2 = row
                    average_values = pd.Series([(v1+v2)/2 if c != 'forecast_datetime' else date for (v1,v2,c) in zip(values_1,values_2,values_2.keys())],
                                              index=values_2.keys())
                    average_values['hours_ahead'] = hrs_ahead
                    add_df = pd.concat([add_df, average_values.to_frame().T]).reset_index(drop=True)
        return pl.DataFrame(pd.concat([df.to_pandas(), add_df]).reset_index(drop=True))

