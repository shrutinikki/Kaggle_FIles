# -*- coding: utf-8 -*-
"""
Author: Shruti Gupta
First Date:05/01/2020
Date: 14/07/2020
Version: 3
File Name: Spotify Top 50
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from scipy import stats
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import seaborn as sns
import sklearn
import warnings
warnings.filterwarnings("ignore")
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler,LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split,cross_val_score, KFold
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans
from wordcloud import WordCloud, STOPWORDS

#%matplotlib inline



#reading the data
music = pd.read_csv('C:/Users/HP/Documents/files for github/top50.csv',encoding='ISO-8859-1')

music.shape
music.info()
music.head(10)
music.describe()
music.drop('Unnamed: 0', axis=1, inplace=True)

music.isnull().sum()
music=music.fillna(0)

music.hist(figsize=(20,20))
plt.show()

music['Genre'].value_counts().plot.bar()
plt.suptitle('Counts for Genre')
plt.show()

music.columns = ['Track', 'Artist', 'Genre', 'BPM', 'Energy', 'Danceability', 'Loudness', 
               'Liveness', 'Valence', 'Length', 'Acousticness', 'Speechiness', 'Popularity']

art_pop = music.groupby('Artist')['Popularity'].mean().sort_values(ascending = False)
print("Artist Popularity")
art_pop

#Finding out the skew for each attribute
skew=music.skew()
print(skew)
# Removing the skew by using the boxcox transformations
transform=np.asarray(music[['Liveness']].values)
music_transform = stats.boxcox(transform)[0]
# Plotting a histogram to show the difference 
plt.hist(music['Liveness'],bins=10)#original data
plt.suptitle('Skewness for Music Liveness')
plt.show()
plt.hist(music_transform,bins=10) #corrected skew data
plt.suptitle('Skewness for Music Liveness after Transformation')
plt.show()

artist_list = music["Artist"].unique().tolist()
artist_list[:2]
artist = " ".join(artist_list)
artist[:100]
# create a word cloud for artist
artist_wordcloud = WordCloud().generate(artist)
# show the created image of word cloud
plt.figure()
plt.imshow(artist_wordcloud)
plt.show()

#word cloud for genre
genre_list = music["Genre"].unique().tolist()
genre_list[:2]
genre = " ".join(genre_list)
genre[:100]
# create a word cloud for artist
genre_wordcloud = WordCloud().generate(genre)
# show the created image of word cloud
plt.figure()
plt.imshow(genre_wordcloud)
plt.show()

#ML
from sklearn.metrics import accuracy_score

le = LabelEncoder()

for col in music.columns.values:
  if music[col].dtypes == 'object':
    le.fit(music[col].values)
    music[col] = le.transform(music[col])

music.head()

# Create test and train dataset
X = music.drop('Loudness', axis=1)
y = music.Loudness

X.drop('Artist', axis=1, inplace=True)

# Creating a test and training dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

#Scaling the data
sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

Lin_reg = LinearRegression()
Lin_reg.fit(X_train, y_train)

y_pred = Lin_reg.predict(X_test)

print(Lin_reg.intercept_, Lin_reg.coef_)
print(mean_squared_error(y_test, y_pred))
#accuracy = accuracy_score(y_test, y_pred)
#accuracy

#SVR
SVR_Reg = SVR(C=0.5)
SVR_Reg.fit(X_train, y_train)

y_pred = SVR_Reg.predict(X_test)
print(mean_squared_error(y_test, y_pred))
#accuracy = accuracy_score(y_test, y_pred)
#accuracy

#Random Forest Tree

from sklearn.ensemble import RandomForestClassifier
#from sklearn.metrics import accuracy_score
clf = RandomForestClassifier(n_estimators=5)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
print(mean_squared_error(y_test, predictions))
accuracy = accuracy_score(y_test, predictions)
accuracy