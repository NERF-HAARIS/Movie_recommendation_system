import streamlit as st
import pickle
import pandas as pd
import base64
from huggingface_hub import hf_hub_download, HfApi
import os

# Streamlit Config - Must be first Streamlit command
st.set_page_config(page_title="Movie Recommender", layout="centered")

# Set Background Image
def set_background(image_file, opacity=1):
    with open(image_file, "rb") as image:
        encoded = base64.b64encode(image.read()).decode()
    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{encoded}");
        background-size: cover;
        background-position: center;
        opacity: {opacity};
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

set_background('movieback2.png', opacity=1)

# HuggingFace API Setup
api = HfApi()

try:
    repo_files = api.list_repo_files("NERF-HAARIS/movie-recommender-files")
    if "similarity.pkl" not in repo_files:
        st.error("similarity.pkl not found in HuggingFace repo.")
        st.stop()

    similarity_path = hf_hub_download(
        repo_id="NERF-HAARIS/movie-recommender-files",
        filename="similarity.pkl"
    )
    similarity = pickle.load(open(similarity_path, 'rb'))

except Exception as e:
    st.error("Failed to load similarity.pkl from HuggingFace.")
    st.text(f"Error: {e}")
    st.stop()

# Load Movie Data
movie_dict = pickle.load(open('movies_dict.pkl', 'rb'))
movies = pd.DataFrame(movie_dict)

# Recommend Movies
def recommend(movie):
    if movie not in movies['title'].values:
        return []
    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]
    movie_list = sorted(enumerate(distances), reverse=True, key=lambda x: x[1])[1:6]
    return [movies.iloc[i[0]].title for i in movie_list]

# UI
st.markdown("<h1 style='text-align: center; color: white;'>Movie Recommendation System 🎬</h1>", unsafe_allow_html=True)
st.markdown("---")

selected_movie = st.selectbox('Select a movie to get recommendations:', movies['title'].values)

if st.button('Recommend'):
    recommendations = recommend(selected_movie)
    st.markdown("### Recommended Movies:")
    for movie in recommendations:
        st.markdown(f"""
        <div style="background-color: rgba(0.1,255,255,0.1); padding: 10px; border-radius: 10px; margin:10px 0; color: White;">
            {movie}
        </div>
        """, unsafe_allow_html=True)
