import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import datetime

# Create scaler objects
std_scaler = StandardScaler()
minmax_scaler = MinMaxScaler()

path_waterTemp = "C:\\Users\\Issa\\Documents\\01 - College Files\\Project Management 2\\Main\\WTemp.csv"
path_bloomOccurrences = "C:\\Users\\Issa\\Documents\\01 - College Files\\Project Management 2\\Main\\BOccurrences.csv"
path_weatherParameters = "C:\\Users\\Issa\\Documents\\01 - College Files\\Project Management 2\\Main\\NAIA.csv"
path_satelliteData = "C:\\Users\\Issa\\Documents\\01 - College Files\\Project Management 2\\Main\\2017 sample-sentinel_3_data_with_bands.csv"

df_wt = pd.read_csv(path_waterTemp)
df_bo = pd.read_csv(path_bloomOccurrences)
df_wp = pd.read_csv(path_weatherParameters)
df_sd = pd.read_csv(path_satelliteData)

# Create a new data frame
df_db = pd.DataFrame({
    'Year': [], 
    'Month': [], 
    'Station': [], 
    'Chl-a': [], 'Chl-a t-1': [], 'Chl-a t-2': [], 'Chl-a t-3': [], 'Chl-a t-4': [], 'Chl-a t-5': [], 
    'NDCI': [], 'NDCI t-1': [], 'NDCI t-2': [], 'NDCI t-3': [], 'NDCI t-4': [], 'NDCI t-5': [],
    'SPM': [], 'SPM t-1': [], 'SPM t-2': [], 'SPM t-3': [], 'SPM t-4': [], 'SPM t-5': [],
    'Turbidity': [], 'Turbidity t-1': [], 'Turbidity t-2': [], 'Turbidity t-3': [], 'Turbidity t-4': [], 'Turbidity t-5': [], 
    'Rainfall Amount': [], 'Rainfall Amount t-1': [], 'Rainfall Amount t-2': [], 'Rainfall Amount t-3': [], 'Rainfall Amount t-4': [], 'Rainfall Amount t-5': [],
    'Atmos TMax': [], 'Atmos TMax t-1': [], 'Atmos TMax t-2': [], 'Atmos TMax t-3': [], 'Atmos TMax t-4': [], 'Atmos TMax t-5': [],
    'Atmos TMin': [], 'Atmos TMin t-1': [], 'Atmos TMin t-2': [], 'Atmos TMin t-3': [], 'Atmos TMin t-4': [], 'Atmos TMin t-5': [], 
    'Water Temp': [], 'Water Temp t-1': [], 'Water Temp t-2': [], 'Water Temp t-3': [], 'Water Temp t-4': [], 'Water Temp t-5': [], 
    'Bloom': [], 'Bloom t-1': [], 'Bloom t-2': [], 'Bloom t-3': [], 'Bloom t-4': [], 'Bloom t-5': []
    })

station_numbers = [1, 2,	4,	5,	8,	13,	15,	16,	17,	18,	19,	20,	21,	22,	23]

############################################################################################################################# ADD YEAR AND MONTH TO MAIN DB

for year in range(2016, 2023):
    for station in station_numbers:
        for month in range(1, 13):
            # Add a new row to the data frame
            df_db = pd.concat([df_db, pd.DataFrame({'Year': year, 'Month': month, 'Station': station, 'Chl-a': '', 'NDCI': '', 'SPM': '', 'Turbidity': '', 'Rainfall Amount': '', 'Atmos TMax': '', 'Atmos TMin': '', 'Water Temp': '', 'Bloom': ''}, index=[0])], ignore_index=True)

# Convert Year and Month columns to integers
df_db[['Year', 'Month', 'Station']] = df_db[['Year', 'Month', 'Station']].astype(int)

############################################################################################################################# ADD WATER TEMP TO MAIN DB

for year in range(2016, 2023):
    for month in range(1, 13):
        index = 0  
        for station in station_numbers:             
            if year == 2016 or year == 2022:
                if year == 2016: yr = 2017
                else: yr = 2021

                # Select the row from df_wt that corresponds to the current year and month
                row = df_wt.loc[(df_wt['Year'] == yr) & (df_wt['Month'] == month) & (df_wt.columns[index + 2] == str(station))]

                # If a row was found, transfer the data to df_db
                if not row.empty:
                    df_db.loc[(df_db['Year'] == year) & (df_db['Month'] == month) & (df_db['Station'] == station), 'Water Temp'] = row[str(station)].values[0]
                  
                index += 1
            
            else:
                # Select the row from df_wt that corresponds to the current year and month
                row = df_wt.loc[(df_wt['Year'] == year) & (df_wt['Month'] == month) & (df_wt.columns[index + 2] == str(station))]

                # If a row was found, transfer the data to df_db
                if not row.empty:
                    df_db.loc[(df_db['Year'] == year) & (df_db['Month'] == month) & (df_db['Station'] == station), 'Water Temp'] = row[str(station)].values[0]
                  
                index += 1

############################################################################################################################# WATER TEMP DATA CLEANUP

# Replace * values to NaN
df_db['Water Temp'].replace('*', np.nan, inplace=True)

############################################################################################################################# ADD BLOOM OCCURRENCE TO MAIN DB

for year in range(2016, 2023):
    for month in range(1, 13):
        index = 0  
        for station in station_numbers:             
            if year == 2016 or year == 2022:
                if year == 2016: yr = 2017
                else: yr = 2021

                # Select the row from df_wt that corresponds to the current year and month
                row = df_bo.loc[(df_bo['Year'] == yr) & (df_bo['Month'] == month) & (df_bo.columns[index + 2] == str(station))]
                
                # If a row was found, transfer the data to df_db
                if not row.empty:
                    df_db.loc[(df_db['Year'] == year) & (df_db['Month'] == month) & (df_db['Station'] == station), 'Bloom'] = row[str(station)].values[0]

                index += 1
          
            else:
                # Select the row from df_wt that corresponds to the current year and month
                row = df_bo.loc[(df_bo['Year'] == year) & (df_bo['Month'] == month) & (df_bo.columns[index + 2] == str(station))]
                
                # If a row was found, transfer the data to df_db
                if not row.empty:
                    df_db.loc[(df_db['Year'] == year) & (df_db['Month'] == month) & (df_db['Station'] == station), 'Bloom'] = row[str(station)].values[0]

                index += 1

############################################################################################################################# BLOOM OCCURRENCE DATA CLEANUP

# Replace non-numeric values with a random number between 0-49999
for index, row in df_db.iterrows():
    if row['Bloom'] == '*' or row['Bloom'] == '':
        df_db.at[index, 'Bloom'] = str(np.random.randint(0, 49999))

# Convert the 'Bloom' column to a numeric data type
df_db['Bloom'] = df_db['Bloom'].str.replace(',', '')
df_db['Bloom'] = pd.to_numeric(df_db['Bloom'], errors='coerce')

############################################################################################################################# ADD WEATHER (PAGASA) DATA TO MAIN DB
''' USE PAG DI NA KAILANGAN MAG DUMPLICATE NG DATA
for year in range(2016, 2023):
    for station in station_numbers:
        for month in range(1, 13):        
            # Select the row from df_wt that corresponds to the current year and month
            row = df_wp.loc[(df_wp['YEAR'] == year) & (df_wp['MONTH'] == month) & (df_wp['DAY'] == 12)] 
            
            # If a row was found, transfer the data to df_db
            if not row.empty:
                df_db.loc[(df_db['Year'] == year) & (df_db['Month'] == month) & (df_db['Station'] == station), 'Rainfall Amount'] = row['RAINFALL'].values[0]
                df_db.loc[(df_db['Year'] == year) & (df_db['Month'] == month) & (df_db['Station'] == station), 'Atmos TMax'] = row['TMAX'].values[0]
                df_db.loc[(df_db['Year'] == year) & (df_db['Month'] == month) & (df_db['Station'] == station), 'Atmos TMin'] = row['TMIN'].values[0]
'''
for year in range(2016, 2023):
    for station in station_numbers:
        for month in range(1, 13):    
            if year == 2016 or year == 2022:
                if year == 2016: yr = 2017
                else: yr = 2021

                row = df_wp.loc[(df_wp['YEAR'] == yr) & (df_wp['MONTH'] == month) & (df_wp['DAY'] == 13)] 
          
                # If a row was found, transfer the data to df_db
                if not row.empty:
                    df_db.loc[(df_db['Year'] == year) & (df_db['Month'] == month) & (df_db['Station'] == station), 'Rainfall Amount'] = row['RAINFALL'].values[0]
                    df_db.loc[(df_db['Year'] == year) & (df_db['Month'] == month) & (df_db['Station'] == station), 'Atmos TMax'] = row['TMAX'].values[0]
                    df_db.loc[(df_db['Year'] == year) & (df_db['Month'] == month) & (df_db['Station'] == station), 'Atmos TMin'] = row['TMIN'].values[0]

            else:
                # Select the row from df_wt that corresponds to the current year and month
                row = df_wp.loc[(df_wp['YEAR'] == year) & (df_wp['MONTH'] == month) & (df_wp['DAY'] == 13)] 
                
                # If a row was found, transfer the data to df_db
                if not row.empty:
                    df_db.loc[(df_db['Year'] == year) & (df_db['Month'] == month) & (df_db['Station'] == station), 'Rainfall Amount'] = row['RAINFALL'].values[0]
                    df_db.loc[(df_db['Year'] == year) & (df_db['Month'] == month) & (df_db['Station'] == station), 'Atmos TMax'] = row['TMAX'].values[0]
                    df_db.loc[(df_db['Year'] == year) & (df_db['Month'] == month) & (df_db['Station'] == station), 'Atmos TMin'] = row['TMIN'].values[0]

############################################################################################################################# WEATHER (PAGASA) DB CLEANUP

df_db['Rainfall Amount'] = pd.to_numeric(df_db['Rainfall Amount'], errors='ignore')

# Replace -1.0 values with 0
df_db['Rainfall Amount'] = df_db['Rainfall Amount'].astype('float')
df_db['Rainfall Amount'].replace(-1.0, 0, inplace=True)

# Replace non-numeric values with NaN
df_db['Rainfall Amount'].replace(-999.0, np.nan, inplace=True)
df_db['Atmos TMax'].replace(-999.0, np.nan, inplace=True)
df_db['Atmos TMin'].replace(-999.0, np.nan, inplace=True)

############################################################################################################################# ADD SATELLITE DATA TO MAIN DB

# Example dataframe with date range column
df_sd_temp = pd.DataFrame()

#S3A_20220103T014946_20220103T015246
# Function that gets start_year, start_month, and start_date data
def convert_date_range(date_range):
    start_date_str = date_range[4:18]
    end_date_str = date_range[20:34]
    start_date = datetime.datetime.strptime(start_date_str, '%Y%m%dT%H%M%S')
    start_year, start_month, start_day = start_date.year, start_date.month, start_date.day
    return start_year, start_month, start_day

# Apply the function to the date_range column and create a new column with the word version of the date range
df_sd[['start_year', 'start_month', 'start_day']] = df_sd['system:index'].apply(convert_date_range).apply(pd.Series)

''' ADD PAG DI NA KAILANGAN MAG DUPLICATE NG SATELLITE DATA
for year in range(2016, 2023):
    #for station in station_numbers:
        for month in range(1, 13):        
            # Select the row from df_wt that corresponds to the current year and month
            row = df_sd.loc[(df_sd['start_year'] == year) & (df_sd['start_month'] == month) & (df_sd['start_day'] == 12)] 
            
            # If a row was found, transfer the data to df_db
            if not row.empty:
                df_db.loc[(df_db['Year'] == year) & (df_db['Month'] == month), 'Chl-a'] = row['chla'].values[0]
                df_db.loc[(df_db['Year'] == year) & (df_db['Month'] == month), 'NDCI'] = row['ndci'].values[0]
                df_db.loc[(df_db['Year'] == year) & (df_db['Month'] == month), 'SPM'] = row['sp'].values[0]
                df_db.loc[(df_db['Year'] == year) & (df_db['Month'] == month), 'Turbidity'] = row['turbidity'].values[0]
'''
for year in range(2016, 2023):
    #for station in station_numbers:
        for month in range(1, 13):        
            # Select the row from df_wt that corresponds to the current year and month
            row = df_sd.loc[(df_sd['start_month'] == month) & (df_sd['start_day'] == 13)] 
            
            # If a row was found, transfer the data to df_db
            if not row.empty:
                df_db.loc[(df_db['Month'] == month), 'Chl-a'] = row['chla'].values[0]
                df_db.loc[(df_db['Month'] == month), 'NDCI'] = row['ndci'].values[0]
                df_db.loc[(df_db['Month'] == month), 'SPM'] = row['sp'].values[0]
                df_db.loc[(df_db['Month'] == month), 'Turbidity'] = row['turbidity'].values[0]

############################################################################################################################# SATELLITE DATA CLEANUP

# Replace non-numeric values with NaN
df_db['Chl-a' ].replace('', np.nan, inplace=True)
df_db['NDCI'].replace('', np.nan, inplace=True)
df_db['SPM'].replace('', np.nan, inplace=True)
df_db['Turbidity'].replace('', np.nan, inplace=True)

############################################################################################################################# CREATE LAGGING FEATURES FOR CHL-A DATA

# define the list of feature column names and the number of lags to create
feature_cols = ['Chl-a', 'NDCI', 'SPM', 'Turbidity', 'Rainfall Amount', 'Atmos TMin', 'Atmos TMax', 'Water Temp', 'Bloom']
num_lags = 5

# loop over each feature column and create the lagged features
for col in feature_cols:
    for i in range(1, num_lags+1):
        lag_col_name = f"{col} t-{i}"
        df_db[lag_col_name] = df_db[col].shift(i)

############################################################################################################################# REMOVING ROWS WITH AT LEAST 1 NAN VALUE

#df_db.dropna(inplace=True)

############################################################################################################################# DATA LABELING

# Create a new column to indicate bloom level
def get_bloom_level(x):
    if x > 499999:
      return 'Massive Bloom'
    elif x > 99999:
      return 'Medium Bloom'
    elif x > 49999:
      return 'Minor Bloom'
    else: 
      return 'Normal Bloom'
  
df_db['Bloom Level'] = df_db['Bloom'].apply(get_bloom_level)

############################################################################################################################# ADD DAY OF YEAR COLUMN

df_db['Day'] = 12

df_db['Date'] = pd.to_datetime(df_db[['Year', 'Month', 'Day']])

def is_leap_year(year):
    if year % 4 == 0:
        if year % 100 == 0:
            if year % 400 == 0:
                return True
            else:
                return False
        else:
            return True
    else:
        return False

def day_of_year(row):
    days_in_month = [31, 28 + is_leap_year(row['Year']), 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    return sum(days_in_month[:row['Month']-1]) + row['Day']

df_db['Day of Year'] = df_db.apply(day_of_year, axis=1)

df_db = df_db.drop(['Day'], axis=1)

############################################################################################################################# 

# Rearrange df_db to ideal feature sequence
#df_db = df_db[['Year', 'Month', 'Day of Year', 'Station', 'Chl-a', 'Chl-a t-1', 'Chl-a t-2', 'Chl-a t-3', 'Chl-a t-4', 'Chl-a t-5', 'Chl-a Rolling Mean', 'Chl-a Rolling STD', 'NDCI', 'SPM', 'Turbidity', 'Rainfall Amount', 'Atmos TMax', 'Atmos TMin', 'Water Temp', 'Bloom', 'Bloom Level']]

############################################################################################################################# NORMALIZATION FOR KNN IMPUTATION

df = df_db[[
    'Chl-a', 'Chl-a t-1', 'Chl-a t-2', 'Chl-a t-3', 'Chl-a t-4', 'Chl-a t-5', 
    'NDCI', 'NDCI t-1', 'NDCI t-2', 'NDCI t-3', 'NDCI t-4', 'NDCI t-5',
    'SPM', 'SPM t-1', 'SPM t-2', 'SPM t-3', 'SPM t-4', 'SPM t-5',
    'Turbidity', 'Turbidity t-1', 'Turbidity t-2', 'Turbidity t-3', 'Turbidity t-4', 'Turbidity t-5', 
    'Rainfall Amount', 'Rainfall Amount t-1', 'Rainfall Amount t-2', 'Rainfall Amount t-3', 'Rainfall Amount t-4', 'Rainfall Amount t-5',
    'Atmos TMax', 'Atmos TMax t-1', 'Atmos TMax t-2', 'Atmos TMax t-3', 'Atmos TMax t-4', 'Atmos TMax t-5',
    'Atmos TMin', 'Atmos TMin t-1', 'Atmos TMin t-2', 'Atmos TMin t-3', 'Atmos TMin t-4', 'Atmos TMin t-5', 
    'Water Temp', 'Water Temp t-1', 'Water Temp t-2', 'Water Temp t-3', 'Water Temp t-4', 'Water Temp t-5', 
    'Bloom', 'Bloom t-1', 'Bloom t-2', 'Bloom t-3', 'Bloom t-4', 'Bloom t-5'
    ]]

'''
[[
    'Year', 
    'Month', 
    'Station', 
    'Chl-a' 'Chl-a t-1', 'Chl-a t-2', 'Chl-a t-3', 'Chl-a t-4', 'Chl-a t-5', 
    'NDCI', 'NDCI t-1', 'NDCI t-2', 'NDCI t-3', 'NDCI t-4', 'NDCI t-5',
    'SPM', 'SPM t-1', 'SPM t-2', 'SPM t-3', 'SPM t-4', 'SPM t-5',
    'Turbidity' 'Turbidity t-1', 'Turbidity t-2', 'Turbidity t-3', 'Turbidity t-4', 'Turbidity t-5', 
    'Rainfall Amount' 'Rainfall Amount t-1', 'Rainfall Amount t-2', 'Rainfall Amount t-3', 'Rainfall Amount t-4', 'Rainfall Amount t-5',
    'Atmos TMax', 'Atmos TMax t-1', 'Atmos TMax t-2', 'Atmos TMax t-3', 'Atmos TMax t-4', 'Atmos TMax t-5',
    'Atmos TMin', 'Atmos TMin t-1', 'Atmos TMin t-2', 'Atmos TMin t-3', 'Atmos TMin t-4', 'Atmos TMin t-5', 
    'Water Temp', 'Water Temp t-1', 'Water Temp t-2', 'Water Temp t-3', 'Water Temp t-4', 'Water Temp t-5', 
    'Bloom', 'Bloom t-1', 'Bloom t-2', 'Bloom t-3', 'Bloom t-4', 'Bloom t-5'
    ]]
'''
# Import the MinMaxScaler class from sklearn.preprocessing
from sklearn.preprocessing import MinMaxScaler

# Create a scaler object
scaler = MinMaxScaler()

# Normalize the data in the dataframe
df_normalized = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

#df_normalized = df_normalized[['Chl-a', 'Chl-a t-1', 'Chl-a t-2', 'Chl-a t-3', 'Chl-a t-4', 'Chl-a t-5', 'Chl-a Rolling Mean', 'Chl-a Rolling STD', 'NDCI', 'SPM', 'Turbidity', 'Rainfall Amount', 'Atmos TMax', 'Atmos TMin', 'Water Temp', 'Bloom']]

df_normalized.head(50)

############################################################################################################################# KNN IMPUTATION

from sklearn.impute import KNNImputer

# Create the KNNImputer object with k=5
imputer = KNNImputer(n_neighbors=5)

# Apply KNN imputation to df_normalized
df_imputed = imputer.fit_transform(df_normalized)

# Convert imputed array to pandas dataframe
imputed_df = pd.DataFrame(df_imputed, columns=df_normalized.columns)  

############################################################################################################################# REVERSE NORMALIZATION

# Inverse transform the normalized dataframe
df_reversed = pd.DataFrame(scaler.inverse_transform(imputed_df), columns=df.columns)

temp_df_db = pd.concat([df_reversed, df_db[['Year', 'Month', 'Day of Year', 'Station']]], axis=1)

temp_df_db = temp_df_db[[
    'Year', 
    'Month',
    'Day of Year',
    'Station', 
    'Chl-a', 'Chl-a t-1', 'Chl-a t-2', 'Chl-a t-3', 'Chl-a t-4', 'Chl-a t-5', 
    'NDCI', 'NDCI t-1', 'NDCI t-2', 'NDCI t-3', 'NDCI t-4', 'NDCI t-5',
    'SPM', 'SPM t-1', 'SPM t-2', 'SPM t-3', 'SPM t-4', 'SPM t-5',
    'Turbidity', 'Turbidity t-1', 'Turbidity t-2', 'Turbidity t-3', 'Turbidity t-4', 'Turbidity t-5', 
    'Rainfall Amount', 'Rainfall Amount t-1', 'Rainfall Amount t-2', 'Rainfall Amount t-3', 'Rainfall Amount t-4', 'Rainfall Amount t-5',
    'Atmos TMax', 'Atmos TMax t-1', 'Atmos TMax t-2', 'Atmos TMax t-3', 'Atmos TMax t-4', 'Atmos TMax t-5',
    'Atmos TMin', 'Atmos TMin t-1', 'Atmos TMin t-2', 'Atmos TMin t-3', 'Atmos TMin t-4', 'Atmos TMin t-5', 
    'Water Temp', 'Water Temp t-1', 'Water Temp t-2', 'Water Temp t-3', 'Water Temp t-4', 'Water Temp t-5', 
    'Bloom', 'Bloom t-1', 'Bloom t-2', 'Bloom t-3', 'Bloom t-4', 'Bloom t-5'
    ]]

temp_df_db.to_csv("C:\\Users\\Issa\\Documents\\01 - College Files\\Project Management 2\\Main\\TEMP MAIN DB (not normalized, all LF).csv", index=False)
#df_normalized.to_csv("/content/drive/Shareddrives/algaecast/CSV Files/normalized test.csv", index=False)
#df_reversed.to_csv("/content/drive/Shareddrives/algaecast/CSV Files/Reversed.csv", index=False)
#df.to_csv("/content/drive/Shareddrives/algaecast/CSV Files/Original df.csv", index=False)
#df_db.to_csv("/content/drive/Shareddrives/algaecast/CSV Files/TEMP MAIN DB (not normalized + with day of year).csv", index=False)
#temp_df_db.to_csv("/content/drive/Shareddrives/algaecast/CSV Files/TEMP MAIN DB (applied data engineering).csv", index=False)
#df_db.to_csv("/content/drive/Shareddrives/algaecast/CSV Files/TEST (EVERY 20TH DATA).csv", index=False)
#df_sd_temp.to_csv("/content/drive/Shareddrives/algaecast/CSV Files/Satellite Data Temp.csv", index=False)
#df_norm.to_csv("/content/drive/Shareddrives/algaecast/CSV Files/df_norm.csv", index=False)
   
#df_reversed.head(50)
print(temp_df_db)
#new_temp.iloc[465:490]
#df_norm.iloc[465:490]
#df_db.head(50)
#print(len(temp_df_db))

#for station in station_numbers:
#  print("Station " + str(station) + ": " + str(len(temp_df_db[temp_df_db['Station'] == station])))

