from data_generation import generate_synthetic_data
from data_validation import validate_data
from exploratory_data_analysis import perform_eda
from statistical_testing import perform_statistical_testing

# Generate synthetic data
data = generate_synthetic_data()

# Validate the data
validate_data(data)

# Perform Exploratory Data Analysis
perform_eda(data)

# Perform Statistical Testing
perform_statistical_testing(data)