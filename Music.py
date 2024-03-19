import pandas as pd
from sklearn.preprocessing import normalize
from scipy.spatial.distance import cosine

# Load Spotify datasets
data = pd.read_csv("../input/spotify-dataset/data/data.csv")
df_artist = pd.read_csv('../input/spotify-dataset/data/data_by_genres.csv')

# Preprocess Spotify data
data['artists'] = data['artists'].str.strip('[]')  # Remove square brackets from artist names
data['artists'] = data['artists'].str.replace("'", "")  # Remove single quotes from artist names
df_artist['name'] = df_artist['name'].str.replace("'", "")  # Remove single quotes from genre names

# Function to process user input for the recommendation system
def process_user_input(user_data):
    """
    Process user input data for the recommendation system.
    
    Parameters:
    - user_data: List of dictionaries containing user preferences for artists and their frequencies.
    
    Returns:
    - input_artist: DataFrame containing processed user input data.
    """
    input_artist = pd.DataFrame(user_data)
    # Find artist IDs for input artists from the Spotify dataset
    id_data = df_artist[df_artist['name'].isin(input_artist['artist'].tolist())]
    # Merge artist IDs with user input data
    input_artist = pd.merge(id_data, input_artist)
    return input_artist

# Collaborative filtering for recommendation
def collaborative_filtering(input_artist, df_freq):
    """
    Collaborative filtering for generating recommendations based on user preferences.
    
    Parameters:
    - input_artist: DataFrame containing processed user input data.
    - df_freq: DataFrame containing frequency data for artists.
    
    Returns:
    - recommendation_final: DataFrame containing collaborative filtering recommendations.
    - top_users_ratings: DataFrame containing ratings of top similar users.
    """
    # Find artist IDs for input artists from the Spotify dataset
    id_data = df_artist[df_artist['name'].isin(input_artist['artist'].tolist())]
    # Merge artist IDs with user input data
    input_artist = pd.merge(id_data, input_artist)
    return recommendation_final, top_users_ratings

# Content-based filtering for song recommendation
def content_filter_music_recommender(song_id, n):
    """
    Content-based filtering for recommending songs similar to a given song.
    
    Parameters:
    - song_id: ID of the input song for which recommendations are sought.
    - n: Number of recommended songs to return.
    
    Returns:
    - song_names: Series containing names of recommended songs.
    """
    # Select relevant features for content-based filtering
    df = data[['danceability', 'energy', 'valence', 'speechiness', 'instrumentalness', 'acousticness']]
    # Normalize feature values
    df_normalized = pd.DataFrame(normalize(df, axis=1))
    df_normalized.columns = df.columns
    df_normalized.index = data.index
    
    # Calculate cosine distances between the input song and all other songs based on features
    all_songs = pd.DataFrame(df_normalized.index)
    all_songs = all_songs[all_songs.song_id != song_id]
    all_songs["distance"] = all_songs["song_id"].apply(lambda x: cosine(df_normalized.loc[song_id], df_normalized.loc[x]))
    
    # Select top N recommended songs based on cosine distances
    top_n_recommendation = all_songs.sort_values(["distance"]).head(n).sort_values(by=['distance', 'song_id'])
    recommendation = pd.merge(top_n_recommendation, data, how='inner', on='song_id')
    song_names = recommendation['name']
    return song_names

# Example user data
user_data = [
    {'artist': 'Ella Fitzgerald', 'freq': 40},
    {'artist': 'Frank Sinatra', 'freq': 10},
    {'artist': 'Lil Wayne', 'freq': 3},
    {'artist': 'The Rolling Stones', 'freq': 5},
    {'artist': 'Louis Armstrong', 'freq': 5}
]

# Process user input
input_artist_data = process_user_input(user_data)

# Collaborative filtering
recommendation_final, top_users_ratings = collaborative_filtering(input_artist_data, df_freq)

# Content-based filtering for song recommendation
recommended_songs = content_filter_music_recommender(3, 5)

# Display recommendations
print("Collaborative Filtering Recommendations:")
print(recommendation_final)
print("\nContent-Based Filtering Recommendations:")
print(recommended_songs)
