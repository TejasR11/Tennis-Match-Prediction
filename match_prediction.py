#!/usr/bin/env python
# coding: utf-8

# In[1]:


#imports needed for this project
import pandas as pd
import os 
import numpy as np
from numpy import random
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import plotly.express as px
from scipy.stats import boxcox
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split


# In[2]:


#opens the file with match data
os.listdir("match_statistics")


# In[3]:



def load_year(loaded_year):
    '''function to load statistics from a year of choice'''
    
    file = 'match_statistics/atp_matches_' + str(loaded_year) + '.csv' 
    data = pd.read_csv(file)
      
    data_columns = ['winner_id', 'winner_seed',
       'winner_name', 'winner_hand', 'winner_ht', 'winner_age', 'w_ace', 'w_df', 'w_svpt', 'w_1stIn', 'w_1stWon', 'w_2ndWon',
       'w_SvGms', 'w_bpSaved', 'w_bpFaced','winner_rank', 'winner_rank_points','loser_id', 'loser_seed', 'loser_name', 'loser_hand',
       'loser_ht', 'loser_age', 'l_ace', 'l_df', 'l_svpt', 'l_1stIn', 'l_1stWon', 'l_2ndWon', 'l_SvGms', 'l_bpSaved', 'l_bpFaced', 'loser_rank', 'loser_rank_points']

    column_map = {x:"_".join(x.split("_")[1:]) for x in data_columns}

    win_data = data[[ 'draw_size', 'tourney_level', 'winner_id',
       'winner_name', 'winner_hand', 'winner_ht', 'winner_age','score', 'best_of', 'round',
       'minutes', 'w_ace', 'w_df', 'w_svpt', 'w_1stIn', 'w_1stWon', 'w_2ndWon',
       'w_SvGms', 'w_bpSaved', 'w_bpFaced','winner_rank']].groupby(["winner_id"]).mean()

    lose_data = data[['draw_size', 'tourney_level',
       'loser_id','loser_name', 'loser_hand',
       'loser_ht', 'loser_age', 'score', 'best_of', 'round',
       'minutes', 'l_ace', 'l_df', 'l_svpt',
       'l_1stIn', 'l_1stWon', 'l_2ndWon', 'l_SvGms', 'l_bpSaved', 'l_bpFaced', 'loser_rank']].groupby(["loser_id"]).mean()

    combined_data = (win_data.rename(columns = column_map) + lose_data.rename(columns = column_map))/2
    combined_data = combined_data[combined_data.isnull().sum(axis = 1)<=0]
    combined_data = combined_data[~combined_data.ht.isnull()]
    combined_data['rank'] = boxcox(combined_data['rank'], 0)
    scl = StandardScaler()
    names = combined_data.columns
    combined_data[names] = scl.fit_transform(combined_data)
    combined_data.columns = names
    combined_data['player_id'] = combined_data.index

    return combined_data


# In[4]:


def read_year(year_predicted):
    '''splits and compares winners and losers for testing by using data until year_predicted'''
    
    combined_years = None
    for year in range(2000, year_predicted):
        year_data = load_year(year)
        year_data["years_played"] = 1
        if combined_years is None:
            combined_years = year_data
        else: 
            combined_years = pd.concat([combined_years, year_data], ignore_index = True)
    combined_years = combined_years.groupby(['player_id'], as_index = 0).sum()        
    combined_years.index = combined_years.player_id
    combined_years = combined_years.div(combined_years.years_played, axis = 0).drop('years_played', axis = 1)
    
    
    testing_data = pd.read_csv('match_statistics/atp_matches_' + str(year_predicted) + '.csv')
    win_loss = testing_data[["winner_id", "loser_id"]].values
    x = []
    y = []
    for match in win_loss:
        player_1 = match[0]
        player_2 = match[1]
        try: 
            winner = combined_years.loc[player_1].values
            loser = combined_years.loc[player_2].values
        
            if np.random.random()>0.5:
                x.append(winner - loser) 
                y.append(0)
            else:
                x.append(loser - winner)
                y.append(1)
            

        except:
            continue
   
    x = np.stack(x) 
    
    
    
    return x, y
    


# In[5]:


#fits the logistics regression model on training data from 2015 to 2019
reg = LogisticRegression()
for year in range(2015, 2019):
    x, y = read_year(year)
    reg.fit(x, y)


# In[6]:


#converts player name to player id for use in dataset
def name_convert(full_name):
    atp_names = pd.read_csv('match_statistics/atp_players.csv')
    atp_names['name_full'] = atp_names.name_first + ' ' + atp_names.name_last
    return atp_names[atp_names.name_full == full_name].player_id.values[0]
    


# In[7]:


#converting player name to player id example
name_convert("Gardnar Mulloy")


# In[13]:


#iterates through and compares career match data from 2000 to 2022 for player_1 and player_2
def get_stats(player_1, player_2):
   player_1 = name_convert(player_1)
   player_2 = name_convert(player_2)
   combined_years = None
   for year in range(2000, 2022):
       year_data = load_year(year)
       year_data["years_played"] = 1
       if combined_years is None:
           combined_years = year_data
       else: 
           combined_years = pd.concat([combined_years, year_data], ignore_index = True)
   combined_years = combined_years.groupby(['player_id'], as_index = 0).sum()        
   combined_years.index = combined_years.player_id
   combined_years = combined_years.div(combined_years.years_played, axis = 0).drop('years_played', axis = 1)
   
   winner = combined_years.loc[player_1].values
   loser = combined_years.loc[player_2].values
   return (winner - loser).reshape (1,-1)


# In[14]:


reg.predict(get_stats("Novak Djokovic", "Rafael Nadal"))


# In[20]:


def predict_winner(player_1, player_2):
    '''predicts winner between player_1 and player_2 using reg model'''
    
    prediction = reg.predict(get_stats(player_1, player_2))[0]
    if prediction == 0:
        print(player_1)
    elif prediction == 1: 
        print(player_2)
    prediction = reg.predict_proba(get_stats(player_1, player_2))[0]
    print(max(prediction))


# In[26]:


#example of predicting winner
#probability not incredibly confident, but typically produces an accurate winner
predict_winner("Roger Federer", "Taylor Fritz")


# In[24]:


#Make sure to spell the name correctly!
Player1 = input("Enter the first player name!")
Player2 = input("Enter the second player name!")
predict_winner(Player1.title(), Player2.title())


# In[ ]:




