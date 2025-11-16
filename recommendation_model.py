import pandas as pd
import numpy as np
import ast # For safely evaluating string representations of lists/dicts
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os

print("Starting recommendation model build...")

# --- 1. LOAD DATA ---
# Ensure the CSV files are in the same directory as this script
movies_path = 'tmdb_5000_movies.csv'
credits_path = 'tmdb_5000_credits.csv'

if not (os.path.exists(movies_path) and os.path.exists(credits_path)):
    print(f"Error: Make sure '{movies_path}' and '{credits_path}' are in the same folder as this script.")
    exit()

try:
    movies = pd.read_csv(movies_path)
    credits = pd.read_csv(credits_path)
except Exception as e:
    print(f"Error reading CSV files: {e}")
    exit()

print("Loaded CSV files successfully.")

# --- 2. MERGE AND CLEAN DATA ---
# Merge the two dataframes on the 'id' (for movies) and 'movie_id' (for credits)
movies = movies.merge(credits.rename(columns={'movie_id': 'id'}), on='id')

# We only need a few columns for a content-based recommender
movies = movies[['id', 'title_x', 'overview', 'genres', 'keywords', 'cast', 'crew']]
movies.rename(columns={'title_x': 'title'}, inplace=True)

# Drop any rows where essential data is missing
movies.dropna(inplace=True)

print("Merged and cleaned data.")

# --- 3. FEATURE ENGINEERING ---
# The 'genres', 'keywords', 'cast', and 'crew' columns are strings
# containing JSON-like structures. We need to parse them.

def safe_literal_eval(s):
    """Safely evaluate a string that looks like a Python literal."""
    try:
        # ast.literal_eval is safer than eval()
        return ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return []

# Helper function to get a list of names from 'genres' or 'keywords'
def get_names(obj_string):
    obj_list = safe_literal_eval(obj_string)
    if not isinstance(obj_list, list):
        return []
    return [item['name'].replace(" ", "") for item in obj_list if 'name' in item]

# Helper function to get the top 3 actors
def get_actors(obj_string):
    obj_list = safe_literal_eval(obj_string)
    if not isinstance(obj_list, list):
        return []
    actors = [item['name'].replace(" ", "") for item in obj_list if 'name' in item]
    return actors[:3] # Return only the top 3 actors

# Helper function to get the director
def get_director(obj_string):
    obj_list = safe_literal_eval(obj_string)
    if not isinstance(obj_list, list):
        return [] # Return an empty list on failure
    for item in obj_list:
        if 'job' in item and item['job'] == 'Director' and 'name' in item:
            return [item['name'].replace(" ", "")] # Return a list with the director's name
    return [] # Return an empty list if no director is found

print("Applying feature engineering...")

# Apply the helper functions to extract the features
movies['genres'] = movies['genres'].apply(get_names)
movies['keywords'] = movies['keywords'].apply(get_names) # <-- This is the corrected line
movies['cast'] = movies['cast'].apply(get_actors)
movies['director'] = movies['crew'].apply(get_director)

# We don't need the 'crew' column anymore
movies = movies.drop('crew', axis=1)

# Create the 'tags' column
# We'll split the overview into a list of words
movies['overview'] = movies['overview'].apply(lambda x: x.split())

# Combine all features into a single 'tags' list
# All columns now contain lists, so we can add them directly
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['director']

# Convert the 'tags' list back to a single space-separated string
movies['tags'] = movies['tags'].apply(lambda x: " ".join(x).lower())

# Create the final dataframe we'll use
final_movies = movies[['id', 'title', 'tags']]

print("Feature engineering complete. 'tags' column created.")

# --- 4. MODEL BUILDING (VECTORIZATION) ---
# Use CountVectorizer to convert the text 'tags' into a matrix of word counts
# max_features=5000: We'll use the 5000 most common words
# stop_words='english': We'll ignore common English words (like 'the', 'a', 'is')
cv = CountVectorizer(max_features=5000, stop_words='english')

# Fit and transform the 'tags' column
vectors = cv.fit_transform(final_movies['tags']).toarray()

print("Vectorization complete.")

# --- 5. CALCULATE SIMILARITY ---
# Calculate the cosine similarity between all movie vectors
similarity = cosine_similarity(vectors)

print("Cosine similarity calculated.")

# --- 6. SAVE THE MODEL ---
# We save the processed 'final_movies' dataframe and the 'similarity' matrix
# 'pickle' is used to serialize Python objects into files

try:
    # Save the movie list (as a list of dictionaries for easier use in the API)
    with open('movies.pkl', 'wb') as f:
        pickle.dump(final_movies.to_dict('records'), f)
        
    # Save the similarity matrix
    with open('similarity.pkl', 'wb') as f:
        pickle.dump(similarity, f)
        
    print("\nSUCCESS!")
    print("Model built and saved as 'movies.pkl' and 'similarity.pkl'.")
    
    # --- 7. EXAMPLE USAGE ---
    print("\n--- EXAMPLE RECOMMENDATION ---")
    
    def recommend(movie_title):
        try:
            # Find the index of the movie
            movie_index = final_movies[final_movies['title'] == movie_title].index[0]
            # Get the similarity scores for that movie
            distances = similarity[movie_index]
            # Sort the movies based on similarity (top 5)
            movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
            
            print(f"Recommendations for '{movie_title}':")
            for i in movies_list:
                print(final_movies.iloc[i[0]].title)
        except (IndexError, KeyError):
            print(f"Movie '{movie_title}' not found in the dataset.")

    # Test the recommendation function
    recommend('Avatar')
    recommend('The Dark Knight Rises')

except Exception as e:
    print(f"Error saving files: {e}")


