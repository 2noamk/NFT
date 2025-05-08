import pandas as pd

df = pd.read_pickle('/home/../multiTS/NFT/data/air_quality/air_quality.pkl')

p1 = df[['T', 'NOx(GT)']]
p2 = df[['C6H6(GT)', 'AH']]
p3 = df[['CO(GT)', 'NO2(GT)']]
p4 = df[['PT08.S2(NMHC)', 'PT08.S4(NO2)']]
p5 = df[['PT08.S5(O3)', 'PT08.S3(NOx)']]
p6 = df[['PT08.S1(CO)', 'NMHC(GT)']]

p1.to_pickle('/home/../multiTS/NFT/data/air_quality_seasonal_2_var/air_quality_seasonality_47_2_var.pkl')
p2.to_pickle('/home/../multiTS/NFT/data/air_quality_seasonal_2_var/air_quality_seasonality_45_2_var.pkl')
p3.to_pickle('/home/../multiTS/NFT/data/air_quality_seasonal_2_var/air_quality_seasonality_41_2_var.pkl')
p4.to_pickle('/home/../multiTS/NFT/data/air_quality_seasonal_2_var/air_quality_seasonality_38_2_var.pkl')
p5.to_pickle('/home/../multiTS/NFT/data/air_quality_seasonal_2_var/air_quality_seasonality_35_2_var.pkl')
p6.to_pickle('/home/../multiTS/NFT/data/air_quality_seasonal_2_var/air_quality_seasonality_00_2_var.pkl')