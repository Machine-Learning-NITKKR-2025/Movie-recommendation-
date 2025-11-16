import pickle
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import sys

# Initialize the Flask application
app = Flask(__name__)
# Enable CORS (Cross-Origin Resource Sharing) to allow our HTML page
# to make requests to this server
CORS(app)

# --- LOAD THE MODEL FILES ---
# Define paths to the model files
movies_path = 'movies.pkl'
similarity_path = 'similarity.pkl'

# Check if the model files exist
if not (os.path.exists(movies_path) and os.path.exists(similarity_path)):
    print("--------------------------------------------------", file=sys.stderr)
    print(f"Error: Model files not found.", file=sys.stderr)
    print(f"Please run 'recommendation_model.py' first to create 'movies.pkl' and 'similarity.pkl'.", file=sys.stderr)
    print("--------------------------------------------------", file=sys.stderr)
    sys.exit(1) # Exit the script if files are missing

try:
    # Load the processed movie data (a list of dictionaries)
    with open(movies_path, 'rb') as f:
        movies_list = pickle.load(f)
    
    # Load the cosine similarity matrix
    with open(similarity_path, 'rb') as f:
        similarity = pickle.load(f)
        
    # Create a simpler list of just movie titles for the dropdown
    movie_titles = [movie['title'] for movie in movies_list]
    
    # Create a mapping from title to its index in the similarity matrix
    # This is crucial for fast lookups
    title_to_index = {movie['title']: i for i, movie in enumerate(movies_list)}

    print("Model files loaded successfully.", file=sys.stderr)

except Exception as e:
    print(f"Error loading model files: {e}", file=sys.stderr)
    sys.exit(1)


# --- API ENDPOINTS ---

@app.route('/movies', methods=['GET'])
def get_movies():
    """
    Endpoint to get the list of all movie titles.
    This populates the dropdown on the frontend.
    """
    # Sort the titles alphabetically for a user-friendly dropdown
    sorted_titles = sorted(movie_titles)
    return jsonify(sorted_titles)

@app.route('/recommend', methods=['GET'])
def recommend():
    """
    Endpoint to get recommendations for a specific movie.
    Expects a query parameter: ?movie=Movie Title
    """
    # Get the movie title from the request's query parameters
    movie_title = request.args.get('movie')
    
    if not movie_title:
        # Return an error if the 'movie' parameter is missing
        return jsonify({'error': 'Movie title parameter is required.'}), 400
        
    if movie_title not in title_to_index:
        # Return an error if the movie title is not in our dataset
        return jsonify({'error': 'Movie not found.'}), 404

    try:
        # --- THIS IS THE FIX ---
        # Find the index of the movie using our lookup map
        movie_index = title_to_index[movie_title]
        # ---------------------
        
        # Get the similarity scores for that movie from the matrix
        distances = similarity[movie_index]
        
        # Sort the movies based on similarity (top 5)
        # We get indices [1:6] to skip the movie itself (which has a score of 1.0)
        movies_list_indices = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
        
        # Get the titles from the indices
        recommended_movies = []
        for i in movies_list_indices:
            movie_idx = i[0] # The index of the recommended movie
            recommended_movies.append(movies_list[movie_idx]['title'])
            
        # Return the list of 5 movie titles as JSON
        return jsonify(recommended_movies)
        
    except Exception as e:
        print(f"Error getting recommendation: {e}", file=sys.stderr)
        return jsonify({'error': 'Internal server error.'}), 500

# --- RUN THE APP ---
if __name__ == '__main__':
    print("Starting Flask server at http://127.0.0.1:5000", file=sys.stderr)
    # Runs the server in debug mode for development
    app.run(debug=True, port=5000)

