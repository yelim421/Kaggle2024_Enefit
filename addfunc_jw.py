### fill train.csv --- [target]
### input: data_storage.df_data
def _interpolate_group(group):
    group['target'] = group['target'].interpolate(method='linear')
    return group

def fill_target(df):
    return pl.DataFrame(df.to_pandas().groupby(['prediction_unit_id', 'is_consumption']).apply(interpolate_group))


### fill forecast_weather.csv --- [surface_solar_radiation_downwards]
### input: data_storage.df_forecast_weather
def fill_radiation(df):
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
