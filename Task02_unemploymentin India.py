import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('C:\\Internship CodeAlpha\\Unemployment in India.csv')

print(df.head())
print("\nData Summary:\n", df.info())
print("\nMissing Values:\n", df.isnull().sum())

df.columns = df.columns.str.strip()

plt.figure(figsize=(12,6))
sns.barplot(x='Region', y='Estimated Unemployment Rate (%)', data=df)
plt.xticks(rotation=90)
plt.title('Unemployment Rate by Region')
plt.show()

df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y', errors='coerce')
plt.figure(figsize=(14,6))
sns.lineplot(x='Date', y='Estimated Unemployment Rate (%)', hue='Region', data=df)
plt.title('Unemployment Rate Over Time by Region')
plt.show()
