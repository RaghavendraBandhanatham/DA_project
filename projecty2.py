import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import *
#load the dataset
df=pd.read_csv('owid-covid-data.csv')
#display dataset information
print(df.info())
#keep only relevent columns
columns=['location','date','total_cases','new_cases','total_deaths','new_deaths','population']
df=df[columns]
#convert date column to datetime
df['date']=pd.to_datetime(df['date'])
#Handling missing values
df.fillna(0,inplace=True)
#Add a column for case for million
df['cases_per_million']=(df['total_cases']/df['population'])
#filter the specific countries
countries = ['United States', 'India', 'Brazil', 'Germany', 'South Africa']
filtered_df = df[df['location'].isin(countries)]
#Global summary statitics
global_summary=df.groupby('date')[['total_cases','total_deaths']].sum().reset_index()
#plot global trends
plt.figure(figsize=(12,6))
#plt.plot(global_summary['date'])
plt.plot(global_summary['date'],global_summary['total_cases'],label='Total cases')
plt.plot(global_summary['date'],global_summary['total_deaths'],label='Total deaths')
plt.title('Global COVID-19 Trends')
plt.xlabel('Date')
plt.ylabel('Count')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
plt.figure(figsize=(12,6))
for country in countries:
    country_data=filtered_df[filtered_df['location']==country]
    plt.plot(country_data['date'],country_data['cases_per_million'],label=country)
plt.title('Cases For Milion Over Time')
plt.xlabel('date')
plt.ylabel('Cases Per Milion')
plt.legend()
plt.grid(True)
plt.show()
# Summary statistics for selected countries
print(filtered_df.groupby('location')[['total_cases', 'total_deaths']].sum())
#correlation analysis
correlation_matrix=filtered_df[['total_cases','new_cases','total_deaths','new_deaths']]
sns.heatmap(correlation_matrix,annot=True,cmap='coolwarm')
plt.title('correlation Between Metrix') 
plt.show()
#treand analysis
India_data=filtered_df[filtered_df['location']=='India']
slope,intercept,r_value,p_value,std_err=linregress(India_data['date'].apply(lambda x: x.toordinal()),India_data['total_cases'])

print(f"slope:{slope},R-squared: {r_value**2}")
filtered_df.to_csv('cleaned_covid_data.csv',index=True)