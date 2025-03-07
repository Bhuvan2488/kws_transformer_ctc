# Module: data_generation.py
import numpy as np
import pandas as pd

def generate_synthetic_data():
    np.random.seed(42)
    date_range = pd.date_range(start='2023-01-01', periods=60*24, freq='H')
    aqi = np.random.normal(loc=100, scale=20, size=len(date_range))
    mode_data = {
        'Bus': np.random.randint(20, 100, size=len(date_range)),
        'Bike': np.random.randint(5, 50, size=len(date_range)),
        'Walk': np.random.randint(10, 60, size=len(date_range)),
        'Public Transport': np.random.randint(50, 150, size=len(date_range))
    }
    engagement_counts = np.random.randint(0, 500, size=len(date_range))
    data = pd.DataFrame({
        'Date': date_range,
        'AQI': np.round(aqi, 2),
        'Bus Commuter Count': mode_data['Bus'],
        'Bike Commuter Count': mode_data['Bike'],
        'Walk Commuter Count': mode_data['Walk'],
        'Public Transport Commuter Count': mode_data['Public Transport'],
        'Engagement Count': engagement_counts
    })
    data.to_csv('synthetic_data.csv', index=False)
    return data