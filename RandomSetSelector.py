# The purpose of this code is to select 100 random music tracks from each of the 8 different genres
# and put them in a folder with the genre names.

import os
import random
import shutil
import pandas as pd

# Define the path to the dataset
dataset_path = 'fma_small'
# Define the path to the output folder
output_path = 'random_dataset'
# Define the path to the metadata file
metadata_path = 'fma_metadata/genres.csv'
# Define the path to the tracks metadata file
tracks_metadata_path = 'fma_metadata/tracks.csv'

# Load the metadata
genres_df = pd.read_csv(metadata_path)
# Load the tracks metadata
tracks_df = pd.read_csv(tracks_metadata_path, index_col=0, header=[0, 1])

# Extract the genre and track ID mapping
track_genre_mapping = tracks_df[('track', 'genre_top')]

# Dynamically extract all unique genres from the metadata
unique_genres = track_genre_mapping.dropna().unique()

# Ensure the output directory exists
os.makedirs(output_path, exist_ok=True)

# Create a dictionary to store genre-to-filepath mapping
genre_filepaths = {genre: [] for genre in unique_genres}

# Sanitize genre names to remove invalid characters for folder names
def sanitize_genre_name(genre):
    return genre.replace('/', '_').replace(' ', '_')  # Replace invalid characters with underscores

# Iterate through all files in the fma_small directory
for root, _, files in os.walk(dataset_path):
    for file in files:
        if file.endswith('.mp3'):
            # Extract the track ID from the file name
            track_id = int(file.split('.')[0])

            # Check if the track ID exists in the metadata
            if track_id in track_genre_mapping.index:
                genre = track_genre_mapping.loc[track_id]

                # Add the file path to the corresponding genre
                if genre in genre_filepaths:
                    genre_filepaths[genre].append(os.path.join(root, file))

# Process each genre and select 10% of tracks
for genre, filepaths in genre_filepaths.items():
    # Sanitize the genre name for folder creation
    sanitized_genre = sanitize_genre_name(genre)

    # Calculate 10% of the total tracks for the genre
    num_to_select = min(len(filepaths), max(1, len(filepaths) // 10))  # Ensure at least 1 track is selected

    # Randomly select the calculated number of tracks
    selected_filepaths = random.sample(filepaths, num_to_select)

    # Create a folder for the genre in the output directory
    genre_output_path = os.path.join(output_path, sanitized_genre)
    os.makedirs(genre_output_path, exist_ok=True)

    # Copy the selected tracks to the genre folder
    for filepath in selected_filepaths:
        destination_path = os.path.join(genre_output_path, os.path.basename(filepath))
        try:
            shutil.copy(filepath, destination_path)
        except PermissionError as e:
            print(f"PermissionError: Could not copy {filepath} to {destination_path}. {e}")
        except Exception as e:
            print(f"Error: Could not copy {filepath} to {destination_path}. {e}")

print("10% selection completed for all genres.")
