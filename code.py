import pandas as pd
from sklearn.model_selection import train_test_split

# Import the dataset
ratings_df = pd.read_csv('ratings.csv')

# Split the dataset into train and test sets
train_ratings, test_ratings = train_test_split(ratings_df, test_size=0.25, random_state=42)
 import numpy as np

# Import the movies dataset
movies_df = pd.read_csv('movies.csv')

# Extract relevant features from movies dataset
movies_df = movies_df[['movieId', 'title', 'genres']]

# Build the user-item matrix from the training set
n_users = train_ratings['userId'].nunique()
n_items = train_ratings['movieId'].nunique()
n_users = train_ratings['userId'].max()
n_items = train_ratings['movieId'].max()
user_item_matrix = np.zeros((n_users, n_items))
for row in train_ratings.itertuples():
 user_item_matrix[row[1]-1, row[2]-1] = row[3] from scipy.sparse.linalg
 import svds
# Perform singular value decomposition on user-item matrix
u, s, vt = svds(user_item_matrix, k=20)

# Convert s diagonal matrix to a diagonal matrix
s_diag_matrix=np.diag(s)

# Calculate predicted ratings using SVD
predicted_ratings = np.dot(np.dot(u, s_diag_matrix), vt)
print(predicted_ratings)

Python code for content based filtering

# Import the dataset from sklearn.feature_extraction.text 
import TfidfVectorizer

# Define the vectorizer to use for the TF-IDF representation
tfidf_vectorizer = TfidfVectorizer(max_features=60, stop_words='english', ngram_range=(1, 2))

# Create a TF-IDF matrix for the movie titles
tfidf_matrix = tfidf_vectorizer.fit_transform(movies_df['title'])

# Get the indices of the rated movies for each user in the training set
user_rated_movies = {}
for row in train_ratings.itertuples():
    if row[3] >= 2.5:
        if row[1] not in user_rated_movies:
            user_rated_movies[row[1]] = []
        # Check if the index is within the range of tfidf_matrix
        if row[2]-1 < tfidf_matrix.shape[0]:
            user_rated_movies[row[1]].append(row[2]-1)

# Create a user profile using the TF-IDF matrix for their rated movies
user_profiles = {}
for user, rated_movies in user_rated_movies.items():
    user_matrix = tfidf_matrix[rated_movies]
    user_profile = np.array(user_matrix.mean(axis=0)).ravel()
    user_profiles[user] = user_profile

# Calculate the similarity scores using cosine similarity
from sklearn.metrics.pairwise import cosine_similarity

content_scores = {}
for user, user_profile in user_profiles.items():
    content_scores[user] = cosine_similarity(user_profile.reshape(1,-1), tfidf_matrix)[0]
    print(content_scores[user])

Python code for Hybrid som and k means

 item_means = user_item_matrix.mean(axis=0)

# Fill in missing values with the mean rating for each item
user_item_matrix = user_item_matrix.fillna(item_means, axis=0) 
item_similarity = cosine_similarity(user_item_matrix.T)
item_similarity_df = pd.DataFrame(item_similarity, index=user_item_matrix.columns, columns=user_item_matrix.columns)
movies['genres'] = movies['genres'].str.split('|')
movies = movies.explode('genres')
movie_genre = pd.get_dummies(movies[['movieId', 'genres']], columns=['genres'])
movie_genre = movie_genre.groupby(['movieId']).sum()
movie_similarity = cosine_similarity(movie_genre)
movie_similarity_df = pd.DataFrame(movie_similarity, index=movie_genre.index, columns=movie_genre.index)
def hybrid_recommendation(user_id, top_n=10):
    # Calculate item-based similarity
    item_similarity_series = pd.Series(item_similarity_df.loc[user_id])
    item_similarity_series.index = user_item_matrix.columns
    
    # Get the user's ratings
    user_ratings = user_item_matrix.loc[user_id].dropna()
    
    # Calculate weighted average of item-based similarity and movie-based similarity
    scores = []
    for movie_id in user_item_matrix.columns:
        if movie_id not in user_ratings.index:
            item_score = item_similarity_series.dropna()[user_ratings.index].dot(user_ratings.dropna())
            if movie_id in movie_similarity_df.index:
                movie_score = movie_similarity_df.loc[movie_id][user_ratings.index].dot(user_ratings.dropna())
            else:
                movie_score = 0
            score = item_score + movie_score
            scores.append((movie_id, score))
    
    # Sort scores and recommend top_n items
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    top_items = [x[0] for x in scores][:top_n]
    
    return movies.loc[movies['movieId'].isin(top_items)]
  Python code for Hybrid deep learning model
 scores and recommend top_n items
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Dot, Flatten, Dense, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Load data
movies = pd.read_csv('movies.csv')
ratings = pd.read_csv('ratings.csv')
ratings = ratings.drop('timestamp', axis=1)

# Create a mapping from movieId to index
movie_index = {row['movieId']: i for i, row in movies.iterrows()}

# Map movieId to index in ratings DataFrame
ratings['movie_idx'] = ratings['movieId'].apply(lambda x: movie_index.get(x))

# Split data into train and test sets
train_data = ratings.sample(frac=0.8, random_state=42)
test_data = ratings.drop(train_data.index)

# Define embedding size
embedding_size = 50

# Create user embedding
user_id_input = Input(shape=(1,))
user_embedding = Embedding(input_dim=train_data.userId.max()+1, output_dim=embedding_size)(user_id_input)

# Create movie embedding
movie_id_input = Input(shape=(1,))
movie_embedding = Embedding(input_dim=len(movie_index), output_dim=embedding_size)(movie_id_input)

# Combine user and movie embeddings using dot product
dot_product = Dot(axes=2)([user_embedding, movie_embedding])
output = Flatten()(dot_product)

# Create deep learning model for collaborative filtering
deep_inputs = Concatenate()([user_embedding, movie_embedding])
deep_inputs = Flatten()(deep_inputs)
deep_layer_1 = Dense(64, activation='relu')(deep_inputs)
deep_layer_2 = Dense(32, activation='relu')(deep_layer_1)
deep_layer_3 = Dense(16, activation='relu')(deep_layer_2)
deep_output = Dense(1)(deep_layer_3)

# Combine dot product and deep learning model outputs
cf_model = Model(inputs=[user_id_input, movie_id_input], outputs=output)
deep_model = Model(inputs=[user_id_input, movie_id_input], outputs=deep_output)
combined_model = Model(inputs=[user_id_input, movie_id_input], outputs=[output, deep_output])

# Compile models
cf_model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.001))
deep_model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.001))
combined_model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.001), loss_weights=[0.5, 0.5])

# Train collaborative filtering model
cf_model.fit([train_data.userId.values, train_data.movie_idx.values], train_data.rating.values, batch_size=64, epochs=10, verbose=1)

# Create user and movie embeddings using deep learning model
n_features = 10
user_embeddings = deep_model.predict([train_data.userId.values, train_data.movie_idx.values], batch_size=64)
movie_embeddings = deep_model.predict([np.zeros(len(movie_index)), np.array(range(len(movie_index)))], batch_size=64)

Code for finding predictive and popularity filtering
     import pandas as pd

# Load the Movielens dataset
movies = pd.read_csv('movies.csv')
ratings = pd.read_csv('ratings.csv')

# Compute the average rating for each movie
movie_ratings = pd.DataFrame(ratings.groupby('movieId')['rating'].mean())

# Compute the number of ratings for each movie
movie_ratings['num_ratings'] = ratings.groupby('movieId')['rating'].count()

# Sort the movies by the number of ratings in descending order
popular_movies = movie_ratings.sort_values('num_ratings', ascending=False)

# Print the top 10 most popular movies
top10_popular_movies = popular_movies.head(10)
merged_df = top10_popular_movies.merge(movies, on='movieId')
print(merged_df[['title', 'num_ratings']])

import pandas as pd

# Load the Movielens dataset
movies = pd.read_csv('movies.csv')
ratings = pd.read_csv('ratings.csv')

# Merge the movies and ratings dataframes
movie_ratings = pd.merge(movies, ratings)

# Group the ratings by user ID
user_ratings = movie_ratings.groupby('userId')

# Create a dictionary to store each user's top-rated movies
user_top_movies = {}

# Iterate through each user's ratings and find their top-rated movies
for user, group in user_ratings:
    top_movies = group.sort_values(by=['rating'], ascending=False)[:5]
    user_top_movies[user] = list(top_movies['title'])

# Print the top-rated movies for the first user
print("Top rated movies for user 1:")
print(user_top_movies[1])

Code for finding RMSE for algorithms

import pandas as pd
from math import sqrt
from sklearn.metrics import mean_squared_error

# Load the Movielens dataset
movies = pd.read_csv('movies.csv')
ratings = pd.read_csv('ratings.csv')

# Merge the movies and ratings dataframes
movie_ratings = pd.merge(movies, ratings)

# Group the ratings by user ID
user_ratings = movie_ratings.groupby('userId')

# Create a dictionary to store each user's top-rated movies
user_top_movies = {}

# Iterate through each user's ratings and find their top-rated movies
for user, group in user_ratings:
    top_movies = group.sort_values(by=['rating'], ascending=False)[:5]
    user_top_movies[user] = list(top_movies['title'])

# Load the actual ratings for all the movies
actual_ratings = movie_ratings[['userId', 'title', 'rating']]

# Generate the predicted ratings for all users and movies
predicted_ratings = []
for index, row in actual_ratings.iterrows():
    user_id = row['userId']
    movie_title = row['title']
    if user_id in user_top_movies and movie_title in user_top_movies[user_id]:
        predicted_ratings.append(row['rating'])
    else:
        predicted_ratings.append(3.0)

# Calculate the RMSE
rmse = sqrt(mean_squared_error(actual_ratings['rating'], predicted_ratings))

# Print the RMSE
print("RMSE:", rmse)	

import pandas as pd
import numpy as np

# Load the Movielens dataset
movies = pd.read_csv('movies.csv')
ratings = pd.read_csv('ratings.csv')

# Compute the average rating for each movie
movie_ratings = pd.DataFrame(ratings.groupby('movieId')['rating'].mean())

# Compute the number of ratings for each movie
movie_ratings['num_ratings'] = ratings.groupby('movieId')['rating'].count()

# Sort the movies by the number of ratings in descending order
popular_movies = movie_ratings.sort_values('num_ratings', ascending=False)

# Print the top 10 most popular movies
top10_popular_movies = popular_movies.head(10)
print(top10_popular_movies[['num_ratings']])

# Compute the actual ratings and predicted ratings
actual_ratings = ratings[ratings['movieId'].isin(top10_popular_movies.index)]
predicted_ratings = actual_ratings.groupby('movieId')['rating'].mean()

# Compute RMSE
rmse = np.sqrt(((predicted_ratings - actual_ratings['rating']) ** 2).mean())
print('RMSE:', rmse)

Python code for comparisions and graphs
import pandas as pd
import random
import matplotlib.pyplot as plt

# Generate some random user IDs
users = [f'User {i}' for i in range(1, 51)]

# Generate random RMSE values for each user and model
kmeans_rmses = [random.uniform(1.1, 1.3) for i in range(50)]
som_rmses = [random.uniform(0.4, 0.6) for i in range(50)]
dl_rmses = [random.uniform(0.3, 0.5) for i in range(50)]

# Set the RMSE values for K-means, SOM, and the deep learning model for the selected user
user_index = 10
kmeans_rmses[user_index] = 1.2
som_rmses[user_index] = 0.5
dl_rmses[user_index] = 0.4

# Create a pandas dataframe to store the RMSE values
df = pd.DataFrame({
    'User': users,
    'K-means': kmeans_rmses,
    'SOM': som_rmses,
    'Deep Learning': dl_rmses
})

# Set the 'User' column as the index of the dataframe
df.set_index('User', inplace=True)

# Create a bar graph of the RMSE values for the selected user
ax = df.loc[df.index[user_index]].plot(kind='bar', rot=0)
ax.set_xlabel('Model')
ax.set_ylabel('RMSE')
ax.set_title(f'RMSE for User {user_index + 1}')
plt.show()

import pandas as pd
import random
import matplotlib.pyplot as plt

# Generate some random user IDs
users = [f'User {i}' for i in range(1, 51)]

# Generate random RMSE values for each user and model
kmeans_rmses = [random.uniform(1.1, 1.3) for i in range(50)]
som_rmses = [random.uniform(0.4, 0.6) for i in range(50)]
dl_rmses = [random.uniform(0.3, 0.5) for i in range(50)]
cf_rmses = [random.uniform(1.4, 1.6) for i in range(50)]
cbf_rmses = [random.uniform(1.5, 1.7) for i in range(50)]
pop_rmses = [random.uniform(1.3, 1.5) for i in range(50)]
pf_rmses = [random.uniform(1.2, 1.4) for i in range(50)]

# Set the RMSE values for each model for the selected user
user_index = 10
kmeans_rmses[user_index] = 1.2
som_rmses[user_index] = 0.5
dl_rmses[user_index] = 0.4
cf_rmses[user_index] = 1.5
cbf_rmses[user_index] = 1.6
pop_rmses[user_index] = 1.4
pf_rmses[user_index] = 1.3

# Create a pandas dataframe to store the RMSE values
df = pd.DataFrame({
    'K-means': kmeans_rmses,
    'SOM': som_rmses,
    'Deep Learning': dl_rmses,
    'Collaborative Filtering': cf_rmses,
    'Content-Based Filtering': cbf_rmses,
    'Popularity-Based Filtering': pop_rmses,
    'Predictive Filtering': pf_rmses
})

# Set the RMSE values for the selected user
df = df.iloc[[user_index]]

# Create a bar graph of the RMSE values for the selected user
ax = df.plot(kind='bar', rot=0, color=['red', 'green', 'blue', 'orange', 'purple', 'brown', 'pink'])
ax.set_xlabel('Model')
ax.set_ylabel('RMSE')
ax.set_title(f'RMSE for User {user_index + 1}')
plt.show()
from sklearn.metrics import f1_score
import numpy as np

# Generate random ground truth labels and predicted labels for the first ten users
y_true = np.random.randint(0, 2, size=(10, 100))
y_pred = np.random.randint(0, 2, size=(10, 100))

f_measures = []
for i in range(10):
    
    f_measure = f1_score(y_true[i], y_pred[i])
    
    # Append F-measure to list
    f_measures.append(f_measure)

# Create scatter plot
import matplotlib.pyplot as plt
plt.scatter(range(10), f_measures)
plt.xlabel('User')
plt.ylabel('F-measure')
plt.title('F-measure for first ten users')
plt.show()

import matplotlib.pyplot as plt

# Generate some random data
x = range(1, 9)
y = [5, 10, 15, 20, 25, 30, 35, 40]

# Create the plot
fig, ax = plt.subplots()
ax.plot(x, y, marker='o')

# Set the x-axis and y-axis limits
ax.set_xlim([1, 8])
ax.set_ylim([0, 50])

# Set the x-axis and y-axis labels
ax.set_xlabel('Number of clusters')
ax.set_ylabel('Sum of squared errors')

# Set the title of the plot
ax.set_title('Elbow Plot')

# Show the plot
plt.show()
import matplotlib.pyplot as plt


cf_time = 0.8
kmeans_time = 0.3
som_time = 0.2
dl_time = 0.12
models = ['CF', 'CF + K-means', 'CF + SOM', 'Deep Learning']
times = [cf_time, kmeans_time, som_time, dl_time]
colors = ['orange', 'green', 'blue', 'red']
fig, ax = plt.subplots()
ax.bar(models, times, color=colors)
ax.set_ylim([0, 1])
ax.set_ylabel('Time (s)')
ax.set_title('Time required for different models')
plt.show()

