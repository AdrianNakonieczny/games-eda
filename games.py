#!/usr/bin/env python
# coding: utf-8

# # Video Games EDA from Backloggd app
# wstęp <br>
# **Pytania:**<br>
# - Ile jest gier w bazie aplikacji?
# - Jak oceniają użytkownicy?
# - Jakie są najwyżej oceniane gatunki gier?
# - Jakie są najpopularniejsze gatunki?
# - Jakie są najlepiej i najgorzej oceniane gry?
# - Jakie są najpopularniejsze gry?
# - Kiedy najczęściej wychodzą gry (lata, miesiące)? <br><br>
# Data source: https://www.kaggle.com/datasets/arnabchaki/popular-video-games-1980-2023

# ## Import bibliotek

# In[1]:


import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
plt.ioff()


# ## Wczytanie danych

# In[2]:


games = pd.read_csv("C:\Analiza danych\Datasety\games.csv")


# ## Przygotowanie danych do analizy

# In[6]:


print(f"Dataset size: {games.shape}")
games.info()


# In[8]:


games.describe()


# In[9]:


games.describe(include = ["O"])


# szczegóły datasetu

# In[4]:


games = games.drop(columns = ["Unnamed: 0", "Summary", "Reviews"])
games.columns = games.columns.str.replace(" ", "")


# In[5]:


#games["ReleaseDate"].unique()


# In[6]:


games["ReleaseDate"] = games["ReleaseDate"].str.replace("releases on TBD", "Dec 31, 2023")


# In[7]:


# to date
games["ReleaseDate"] = pd.to_datetime(games["ReleaseDate"], format = "%b %d, %Y")


# In[8]:


# subset
from datetime import datetime

today = datetime.now()
games = games[~(games["ReleaseDate"] > today)]


# In[9]:


games["ReleaseDate"] = games["ReleaseDate"].dt.normalize()


# In[10]:


# remove brackets and quote from developer names and genres
pattern = "|".join(["\[", "'", "\]"])
games["Team"] = games["Team"].str.replace(pattern, "", regex = True)
games["Genres"] = games["Genres"].str.replace(pattern, "", regex = True)


# In[11]:


games.isna().sum()


# In[12]:


games = games.dropna()


# In[13]:


assert games["Rating"].dtype == "float64"


# In[14]:


# function for convert "4.5K" like strings to float
def k_to_values(x):
    return x.replace(r"[K]+$", "", regex = True).astype(float) * \
      x.str.extract(r"[\d\.]+([K]+)", expand = False).fillna(1).replace(["K"], [10**3]).astype(float)


# In[15]:


games[["TimesListed", "NumberofReviews", "Plays", "Playing", "Backlogs", "Wishlist"]] = \
  games[["TimesListed", "NumberofReviews", "Plays", "Playing", "Backlogs", "Wishlist"]].apply(k_to_values)


# In[16]:


games.duplicated("Title").sum()
games[games.duplicated("Title")].sort_values(by = "Title")


# In[17]:


games = games.drop_duplicates("Title", keep = "first").reset_index(drop = True)


# In[18]:


games.info()


# In[19]:


games["TimesListed"].equals(games["NumberofReviews"])


# In[20]:


games = games.drop(columns = ["TimesListed"])


# In[21]:


# reshape and create accurate "Genre" column
games[["MainGenre", "Genre", "RestGenre"]] = games["Genres"].str.split(",", n = 2, expand = True)


# In[22]:


games["Genre"] = games["Genre"].fillna(games["MainGenre"])


# In[23]:


games["Genre"].value_counts(sort = True)


# In[24]:


games["Genre"] = games["Genre"].str.lstrip()


# In[25]:


games.at[885, "Genre"] = "Adventure"
games.at[1048, "Genre"] = "Puzzle"


# In[26]:


games = games.drop(columns = ["Genres", "MainGenre", "RestGenre"])


# In[27]:


games["Genre"].isnull().sum()


# In[28]:


games[["NumberofReviews", "Plays", "Playing", "Backlogs", "Wishlist"]] = \
  games[["NumberofReviews", "Plays", "Playing", "Backlogs", "Wishlist"]].astype(int)


# In[29]:


games["year"] = games["ReleaseDate"].dt.year
games["month"] = games["ReleaseDate"].dt.month


# ## Analiza eksploracyjna

# In[30]:


sns.histplot(x = "Rating", data = games, binwidth = 0.1)
plt.title("Games ratings distribution")
plt.show()


# In[31]:


games["Rating"].describe()


# In[32]:


games["Rating"].mode()


# ### Rozkład ocen gier
# - Histogram ocen gier ma rozkład asymetryczny lewoskośny;
# - Dataset zawiera **1083** rekordy (gry). Najniższa ocena wynosi **0.7**, a najwyższa **4.6**;
# - Zmienna ma średnią ≈ **3.65** o odchyleniu standardowym ≈ **0.54**. Wartością najczęściej występującą jest **3.7**;
# - 50 % gier ma ocenę wyższą niż **3.7**, 75 % ma ocenę wyższą niż **4**, a 25% ma ocenę niższą niż **3.4**;

# In[33]:


games["ReleaseDate"].min(), games["ReleaseDate"].max()


# Najstarsza z obecnych w aplikacji gier została wydana **22-05-1980**, a najnowsza **17-03-2023**.

# In[34]:


plt.figure(figsize = (12, 5))
ax = sns.countplot(x = "year", data = games)
ax.set_xticklabels(ax.get_xticklabels(), rotation = 40, ha = "right")
ax.set_axisbelow(True)
plt.tight_layout()
plt.title("Games released in years")
plt.grid(linewidth = 0.3, alpha = 0.5)
plt.show()


# In[35]:


games["year"].value_counts().head(5)


# **Najwięcej gier obecnych w bazie aplikacji zostało wydanych w latach:**
# - 2022
# - 2021
# - 2019, 2016    
# 
# **Najmniej natomiast w latach 80.**

# In[36]:


ax = sns.countplot(x = "month", data = games)
ax.set_axisbelow(True)
plt.grid(linewidth = 0.3, alpha = 0.5)
plt.title("Games released in months")
plt.show()


# In[37]:


games["month"].value_counts().tail(4)


# **Miesiące z największą ilością premier:**
# - listopad
# - październik
# - wrzesień    
# 
# **Miesiące z najmniejszą ilością premier:**
# - maj
# - styczeń, kwiecień
# - czerwiec

# In[38]:


ax = sns.countplot(y = "Genre", data = games, order = games["Genre"].value_counts().index)
ax.set_axisbelow(True)
plt.grid(linewidth = 0.3, alpha = 0.5)
plt.title("Games by genre")
plt.show()


# In[39]:


games["Genre"].value_counts().tail(5)


# **Najwięcej gier z gatunków:**
# - RPG
# - Platform
# - Shooter   
# 
# **Najmniej gier z gatunków:**
# - MOBA, Quiz/Trivia
# - Card & Board Game, Tactical
# - Real Time Strategy

# In[40]:


games_genre_rating = pd.DataFrame(games.groupby("Genre")["Rating"].agg(np.mean)).reset_index()


# In[41]:


ax = sns.barplot(x = "Rating", y = "Genre", data = games_genre_rating,
                 order = games_genre_rating.sort_values("Rating", ascending = False)["Genre"])
plt.grid(linewidth = 0.3, alpha = 0.5)
plt.margins(0, 0.01)
plt.xlim(0, 5)
plt.title("Mean rating by genre")
plt.show()


# **Najlepiej oceniane gatunki:**
# - Visual Novel
# - Turn Based Strategy
# - Puzzle    
# 
# **Najgorzej oceniane gatunki:**
# - MOBA
# - Quiz/Trivia
# - Fighting

# **Najpopularniejsze gatunki:**

# In[42]:


games_genre_popularity = pd.DataFrame(games.groupby("Genre")["Plays"].agg(sum)).reset_index()
games_genre_popularity.nlargest(3, "Plays")


# In[43]:


games_genre_popularity.nsmallest(3, "Plays")


# In[44]:


games.describe()


# ### Najwyżej/najniżej oceniane gry

# In[45]:


games.nlargest(5, "Rating")


# In[46]:


games.nsmallest(5, "Rating")


# ### Najwięcej/najmniej ocen

# In[47]:


games.nlargest(5, "NumberofReviews")


# In[48]:


games.nsmallest(5, "NumberofReviews")


# ### Najwięcej/najmniej "backlogów"

# In[49]:


games.nlargest(5, "Backlogs")


# In[50]:


games.nsmallest(5, "Backlogs")


# ### Najczęściej/najrzadziej na liście życzeń

# In[51]:


games.nlargest(5, "Wishlist")


# In[52]:


games.nsmallest(5, "Wishlist")


# ### Najbardziej i najmniej popularne

# In[53]:


games.nlargest(5, "Plays")


# In[54]:


games.nsmallest(5, "Plays")


# ### Najbardziej i najmniej popularne w momencie analizy

# In[55]:


games.nlargest(5, "Playing")


# In[56]:


games.nsmallest(5, "Playing")


# ### Średnia ocena w latach

# In[57]:


games_years_rating = pd.DataFrame(games.groupby("year")["Rating"].agg(np.mean)).reset_index()


# In[58]:


plt.figure(figsize = (12, 5))
ax = sns.lineplot(x = "year", y = "Rating", data = games_years_rating)
plt.xlim(1980, 2025)
plt.title("Mean game ratings by year")
plt.grid(linewidth = 0.3, alpha = 0.5)
plt.show()


# In[59]:


games[games["year"] == 1987]


# ### Średnia ocena w miesiącach

# In[60]:


games_months_rating = pd.DataFrame(games.groupby("month")["Rating"].agg(np.mean)).reset_index()


# In[61]:


ax = sns.barplot(x = "month", y = "Rating", data = games_months_rating)
ax.set_axisbelow(True)
plt.grid(linewidth = 0.3, alpha = 0.5)
plt.ylim(0, 5)
plt.title("Mean game ratings by months")
plt.show()


# In[62]:


games_months_rating.describe()


# In[ ]:


#sns.heatmap(games[["Rating", "NumberofReviews", "Plays", "Playing", "Backlogs", "Wishlist"]].corr(), annot = True)
#plt.show()

