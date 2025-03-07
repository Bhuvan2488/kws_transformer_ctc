# Module: data_validation.py
def validate_data(data):
    print("Data Info:")
    print(data.info())
    print("\nData Summary:")
    print(data.describe())