from urllib.request import urlopen
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import re
import requests


# Download Player Data

def player_csv(years_list):
    dir_name = 'data'
    years = ''
    dataframes = []
    
    for year in years_list:
        url = (f"https://www.pro-football-reference.com/years/{year}/fantasy.htm")
        html = urlopen(url)
        soup = BeautifulSoup(html, features = 'lxml')

        headers = [th.getText() for th in soup.findAll('tr')[1].findAll('th')]
        headers = headers[1:]

        rows = soup.findAll('tr', class_=lambda table_rows: table_rows != "thead")
        player_stats = [[td.getText() for td in rows[i].findAll('td')]
                        for i in range(len(rows))]
        player_stats = player_stats[2:]

        stats = pd.DataFrame(player_stats, columns=headers)

        stats = stats.replace(r'', 'N/A', regex=True)
        stats['Year'] = year
        
        print(f'Year {year} data retrived')
        
        dataframes.append(stats)
        years += (f'{year}-')
    
    print('Combining Dataframes')    
    
    combined_df = pd.concat(dataframes, ignore_index= True)
    
    file_path = (f'{dir_name}/{years}playerstats.csv')
        
    combined_df.to_csv(file_path)
    
    print(f"Player data for the years {years.strip('-')} has been created.")
    
    
    

# Clean Data

def clean_data(df, file_name) :

    df = df.drop(axis = 1, columns= 'Unnamed: 0')

    pattern = re.compile('[*+]')
    df['Player'] = [pattern.sub('',i) for i in df['Player']] 
    
    df.rename(columns = {
        'G' : 'GamesPlay',
        'GS' : 'GamesStart',
        
        'Cmp' : 'PassCmp',
        'Att' : 'PassAtt',
        'Yds' : 'PassYds',
        'TD' : 'PassTD',
        'Int' : 'PassInt',
        
        'Att.1' : 'RushAtt',
        'Yds.1' : 'RushYds',
        'Y/A' : 'RushYperA',
        'TD.1' : 'RushTD',
        
        'Fmb' : 'FumbleCount',
        'FL' : 'FumbleLost',
        
        'Tgt' : 'RecTgt',
        'Rec' : 'RecRecp',
        'Yds.2' : 'RecYds',
        'Y/R' : 'RecYperR',
        'TD.2' : 'RecTD',
        
        'TD.3' : 'ScoreTD',
        '2PM' : 'Score2PM',
        '2PP' : 'Score2PP'
    }, inplace = True)
    
    df = df.fillna(0)
    
    df['FantPos'] = df['FantPos'].replace(0, 'Unknown')
    
    df['Player'] = df['Player'].apply(lambda x: re.sub(r'\.', '', x))
    
    df.to_csv(f'data/clean_{file_name}')



# Pull AVG Pick from fantasy pros

def adp_parser() :

    '''
    Creates df of 2023 players and their average pick from SLEEPER and RTSPORTS
    '''

    #Parse and Create df

    url = (f"https://www.fantasypros.com/nfl/adp/overall.php")

    response = requests.get(url)
    soup = BeautifulSoup(response.text, features = 'html.parser')

    table_body = soup.find('tbody')

    rows = table_body.find_all('tr')

    data = []

    for row in rows:
        cols = row.find_all('td')
        
        cols_text = [col.text for col in cols]

        data.append(cols_text)

    # Clean Data

    df = pd.DataFrame(data, columns=['Rank', 'Player', 'POS', 'CBS', 'Sleeper', 'RTSports', 'AVG'])
    df.drop(columns = ['POS'], inplace = True)
    df['Player'] = df['Player'].apply(lambda x: re.sub(r' [A-Z]+\s*\(\d+\)', '', x)).apply(str.strip)
    df['Player'] = df['Player'].apply(lambda x: re.sub(r'\.', '', x))

    df.to_csv('export/2023-ave-draft-pick.csv')

