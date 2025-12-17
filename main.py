#Import all necessary libraries
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fastapi.middleware.cors import CORSMiddleware
import difflib

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#Load CSV
songs_data = pd.read_csv("songs.csv")

#Select important features
selected_features = ["song", "artist", "text"]

#Fill missing values
for feature in selected_features:
    songs_data[feature] = songs_data[feature].fillna("")

#Combine features
combined_features = (
    songs_data["song"] + " " +
    songs_data["artist"] + " " +
    songs_data["text"]
)

#Initialize vectorizer
vectorizer = TfidfVectorizer(stop_words="english")

#Generate vectors
feature_vector = vectorizer.fit_transform(combined_features)

#Song list
list_of_all_song = songs_data["song"].to_list()

class SongRequest(BaseModel):
    song: str

@app.get("/")
def root():
    return {"message": "Welcome to the song recommendation API!"}

@app.post("/recommendation")
def recommend(request: SongRequest):
    song_name = request.song

    find_close_match = difflib.get_close_matches(
        song_name,
        list_of_all_song,
        cutoff=0.4
    )

    if not find_close_match:
        return {"error": "Song not found"}

    close_match = find_close_match[0]

    index_of_the_song = songs_data[songs_data['song'] == close_match].index[0]

    #  Compute similarity ONLY for one song
    similarity_score = cosine_similarity(
        feature_vector[index_of_the_song],
        feature_vector
    )[0]

    similarity_score = list(enumerate(similarity_score))

    sorted_similar_songs = sorted(
        similarity_score,
        key=lambda x: x[1],
        reverse=True
    )

    recommendations = []
    for song in sorted_similar_songs[1:20]:
        index = song[0]
        title = songs_data.loc[index, 'song']
        link = songs_data.loc[index, 'link'] 
        recommendations.append({"song": title, "link": link})

    return {
        "matched_song": close_match,
        "recommendations": recommendations
    }
