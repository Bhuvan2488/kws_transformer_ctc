  # Module: statistical_testing.py
from scipy import stats

def perform_statistical_testing(data):
    high_aqi = data[data['AQI'] > 100]['Engagement Count']
    low_aqi = data[data['AQI'] <= 100]['Engagement Count']
    t_stat, p_value = stats.ttest_ind(high_aqi, low_aqi)
    print(f"T-Statistic: {t_stat}, P-Value: {p_value}")

# Main Script: main.py
from data_generation import generate_synthetic_data
from data_validation import validate_data
from exploratory_data_analysis import perform_eda
from statistical_testing import perform_statistical_testing

data = generate_synthetic_data()
validate_data(data)
perform_eda(data)
perform_statistical_testing(data)