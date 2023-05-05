# Fetch Current Weather Data
import requests
import pandas as pd
from datetime import datetime
import pytz

path_openWeatherData = "/home/issa2345/openWeatherData.csv"

# Replace YOUR_API_KEY with your actual API key
api_key = '4c3e2a766c6698e331c26ea396103a71'

# Specify the latitude and longitude
lat = 14.393530
lon = 121.193945

# Define the API endpoint and parameters
url = 'https://api.openweathermap.org/data/2.5/weather'
params = {'lat': lat, 'lon': lon, 'appid': api_key, 'units': 'metric'}

# Set the timezone
timezone = pytz.timezone('Asia/Manila')

# Get the current date and time
now = datetime.now(timezone)
year = now.year
month = now.month
day = now.day
hour = now.hour
minute = now.minute

# Check if current time is within the specified time range (1pm-4pm)
if hour < 1 or hour > 24:
    print("Current time is not within the specified time range (1pm-4pm)")
else:
    # Check if data for current date and time has already been stored
    df_existing = pd.read_csv(path_openWeatherData)
    existing_data = df_existing[(df_existing['year']==year) & (df_existing['month']==month) & (df_existing['day']==day)]
    if not existing_data.empty:
        print("Data for current date and time already exists")
    else:
        # Send the request and get the response
        response = requests.get(url, params=params)
        data = response.json()

        # Extract the relevant data from the response
        min_temp = data['main']['temp_min']
        max_temp = data['main']['temp_max']
        rainfall = data.get('rain', {}).get('1h', 0)

        # Create a DataFrame to store the data
        df_openWeatherData = pd.DataFrame({
            'year': [year],
            'month': [month],
            'day': [day],
            'hour': [hour],
            'minute': [minute],
            'min_temp': [min_temp],
            'max_temp': [max_temp],
            'rainfall': [rainfall]
        })
        
        # Append the new data to the existing DataFrame and save to file
        df_existing = df_existing.append(df_openWeatherData)
        df_existing.to_csv(path_openWeatherData, index=False, mode='w')

        df_existing.head()
        # Print the results
        print("Data successfully stored")