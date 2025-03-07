# Module: exploratory_data_analysis.py
import matplotlib.pyplot as plt
import seaborn as sns

def perform_eda(data):
    sns.set(style='whitegrid')

    # 1. AQI Over Time
    plt.figure(figsize=(12, 6))
    plt.plot(data['Date'], data['AQI'], label='AQI', color='blue')
    plt.xlabel('Date')
    plt.ylabel('Air Quality Index')
    plt.title('Air Quality Index Over Time')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # 2. Engagement vs. AQI
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=data['AQI'], y=data['Engagement Count'], alpha=0.6)
    plt.xlabel('Air Quality Index (AQI)')
    plt.ylabel('Engagement Count')
    plt.title('Engagement vs. AQI')
    plt.show()

    # 3. Transportation Mode Trends
    plt.figure(figsize=(10, 6))
    data[['Date', 'Bus Commuter Count', 'Bike Commuter Count', 'Walk Commuter Count', 'Public Transport Commuter Count']].set_index('Date').plot()
    plt.xlabel('Date')
    plt.ylabel('Commuter Counts')
    plt.title('Commuter Counts by Transportation Mode Over Time')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # 4. Engagement Distribution
    plt.figure(figsize=(8, 6))
    sns.histplot(data['Engagement Count'], kde=True, bins=30, color='purple')
    plt.xlabel('Engagement Count')
    plt.title('Distribution of Engagement Count')
    plt.show()

    # 5. Correlation Heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Heatmap')
    plt.show()