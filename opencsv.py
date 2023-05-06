import csv 

with open('openWeatherData.csv', 'r') as csv_file:
    reader = csv.reader(csv_file)
    num_rows = len(list(reader))
    print(f"The CSV file has {num_rows} rows.")