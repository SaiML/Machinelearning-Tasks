# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 14:51:28 2019

@author: Sainath.Reddy
"""
import pandas as pd 
import numpy as np 
from math import sqrt
import matplotlib.pyplot as plt 
from pandas import concat
import datetime
import warnings
import ast
warnings.filterwarnings("ignore")


TED_df=pd.read_csv("ted_main.csv")
# Converting in to Timestamp date format

TED_df['film_date'] = pd.to_datetime(TED_df['film_date'], unit='s')
TED_df['published_date'] = pd.to_datetime(TED_df['published_date'],unit='s')

# A

TED_df['Ratio_0f_Discussions'] = TED_df['comments']/TED_df['views']
TED_df[['title', 'main_speaker','views', 'comments', 'Ratio_0f_Discussions', 'film_date']].sort_values('Ratio_0f_Discussions', ascending=False).head(1)
    
# The case for same-sex marriage is most Disscused TED show

# B
def Outliers(column,x,range1,column1):
    
    # Distribution graphs
    sns.distplot(column[x])
    plt.show()
    sns.distplot(column[column[x] > range1][x])
    plt.show()
    sns.distplot(column[column[x] < range1][x])
    
    # Correlation plot
    fig=plt.figure()
    axes=fig.add_axes([0,0,1,1])
    axes.set_xlabel("views")
    axes.set_ylabel('x')
    axes.plot(column1,column[x],ls="",marker=".")
    plt.show()
    
    # Box plot 
    df.boxplot(column = x)
    column[x].quantile(0.9)

Outliers(TED_df,"comments",1000,TED_df['views'])

# From the plots above, we can see that most of  comments  of the talks have fewer than 500 comments.
#Some true outliers in views as well as comments,


# c

published_date_year = TED_df.groupby(TED_df['published_date'].dt.year)['published_date'].count()

def yearanalysis(x,Data):
    
    # plotting graph
    bar_labels = x.keys()
    x_pos = list(range(len(x)))
    plt.bar(x_pos,x,align='center',color='#FFC222')
    plt.xticks(x_pos, bar_labels, rotation='vertical')
    plt.ylabel('Count')
    plt.title('Number of talks per year')
    plt.show()

    # Finiding Delay in delay time and published year
    df1=Data[['film_date','published_date']].head()
    df1['delay'] = Data['published_date']-Data['film_date']
    return df1.head()

yearanalysis(published_date_year,TED_df)


# E Unpacking the ratings columns

TED_df.ratings=TED_df.ratings.apply(lambda x:ast.literal_eval(x))
extrct=pd.DataFrame(TED_df.ratings[0])
extrct1=pd.DataFrame(TED_df.ratings.sum())
extrct2=extrct1.groupby("name").sum()["count"].sort_values(ascending=False)
print(extrct2)

# D
Categories=["Inspiring","Informative","Fascinating","Persuasive","Beautiful","Courageous",
            "Funny","Ingenious","Jaw-dropping","OK","Unconvincing","Longwinded","Obnoxious","Confusing"]

def Percent_count_of(i,x):
    for count in range(14):
        if pd.DataFrame(x)["name"][count]==Categories[i]:
            return pd.DataFrame(x).loc[count,"count"]
def total_count(x):
    total_votes=0
    for count in range(14):
        total_votes+=pd.DataFrame(x).loc[count,"count"]
    return total_votes
    
TED_df["Total_Votes"]=TED_df["ratings"].apply(lambda x:total_count(x))
        
for i in range(14):
    TED_df[Categories[i]]=TED_df["ratings"].apply(lambda x:Percent_count_of(i,x))/TED_df["Total_Votes"]
    
print(TED_df.head(2))


# best Inspiring show is Susan Lim: Transplant cells, not organs
print(TED_df[TED_df["Inspiring"]==TED_df["Inspiring"].max()])

# best funny show is Julia Sweeney: It's time for "The Talk"
TED_df[TED_df["Funny"]==TED_df["Funny"].max()]


# best informative show is Ramanan Laxminarayan: The coming crisis in ant
TED_df[TED_df["Informative"]==TED_df["Informative"].max()]

# taken the percent count of indivudual rating categories and divided with total Counts
# and by taking maximum percentage of each column we can say that they are best in specified column


# =============================================================================
# f calculate the percentage of ratings that were negative
# =============================================================================

df_negative= TED_df[['name','description','event','Unconvincing','Longwinded','Obnoxious','Confusing']]
z = [ (row.Unconvincing + row.Longwinded+row.Obnoxious+row.Confusing) *100 for index, row in TED_df.iterrows() ]
df_negative['Perecent_negative'] =z
df_negative.head()

# f(ii)
df1=TED_df[['name','Total_Votes','published_date']]
date = datetime.date(2019,1,1)
df1['datetime'] = pd.to_datetime(date)
df1['noofdays'] = (df1['datetime']- df1['published_date']).dt.days
df1['Avergae_no_votes']=df1['Total_Votes']/df1['noofdays']
print(df1.head())

# g

df2 = TED_df[['title', 'main_speaker', 'views','speaker_occupation','Funny']].sort_values('Funny', ascending=False)[:5]
print(df2.head())

# most funny comedy show is by Speaker Ken Robinson and belongs to occupation Author/educator

#g(i)

df3=TED_df[['name','Inspiring', 'Informative','Fascinating', 'Persuasive', 'Beautiful', 'Courageous', 'Funny',
       'Ingenious', 'Jaw-dropping', 'OK', 'Unconvincing', 'Longwinded',
       'Obnoxious', 'Confusing' ]]

df3['recent_rate']=  df3[df3.columns.difference(['name'])].idxmax(axis=1)
print(df3.head())

# g(ii)
# Spliiting the occupation columns

TED_df['single_occupation'] = TED_df.speaker_occupation.str.split('/|;|,').str[0]
print(TED_df.head())
