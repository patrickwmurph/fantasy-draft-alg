from helpers.functions import player_csv, clean_data, adp_parser
import pandas as pd


# Input Years
years_list = [2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020,2021,2022]

player_csv(years_list) # downloads years


# Generate File Name

while True :
    file_name = ''
    
    for year in years_list :
        file_name += f'{year}-'
        
    file_name += 'playerstats.csv'
    break


# Read-in Data

file_path = f'data/{file_name}'
df = pd.read_csv(file_path)

clean_data(df, file_name)


# Load average picks

adp_parser()


