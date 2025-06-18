# This script converts UTM coordinates to latitude and longitude
import pandas as pd
import utm
df = pd.read_csv('dataset/features_70m_with_lithology.csv')

zone_number = 43
zone_letter = 'N'

def convert_utm_to_latlon(row):
    lat, lon = utm.to_latlon(row['x'], row['y'], zone_number, zone_letter, strict=False)
    return pd.Series({'latitude': lat, 'longitude': lon})

df[['latitude', 'longitude']] = df.apply(convert_utm_to_latlon, axis=1)

df.to_csv("output_with_latlon.csv", index=False)

print("Conversion complete. Output saved to 'output_with_latlon.csv'")
