import streamlit as st
import pickle
import pandas as pd
import base64
from huggingface_hub import hf_hub_download, HfApi
import os

# Background Image
def set_background(image_file, opacity=1):
    with open(image_file, "rb") as image:
        encoded = base64.b64encode(image.read()).decode()
    st.markdown(f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{encoded}");
        background-size: cover;
        background-position: center;
        opacity: {opacity};
    }}
    </style>
    """, unsafe_allow_html=True)

# Recommend Function
def recommend(movie):
    if movie not in movies['title'].values:
        return []
    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]
    movie_list = sorted(enumerate(distances), reverse=True, key=lambda x: x[1])[1:6]
    return [movies.iloc[i[0]].title for i in movie_list]

# Load Movie Data
movie_dict = pickle.load(open('movies_dict.pkl', 'rb'))
movies = pd.DataFrame(movie_dict)

# HuggingFace Setup
api = HfApi()
SIMILARITY_FILE = "similarity.pkl"
LOCAL_PATH = f"./{SIMILARITY_FILE}"

try:
    if os.path.exists(LOCAL_PATH):
        st.info("Loading similarity.pkl from local cache...")
        with open(LOCAL_PATH, "rb") as f:
            similarity = pickle.load(f)
    else:
        st.info("Downloading similarity.pkl from HuggingFace...")
        similarity_path = hf_hub_download(
            repo_id="NERF-HAARIS/movie-recommender-files",
            filename=SIMILARITY_FILE
        )
        similarity = pickle.load(open(similarity_path, 'rb'))
        # Save a local copy
        with open(LOCAL_PATH, "wb") as f:
            pickle.dump(similarity, f)

except Exception as e:
    st.error("Error loading similarity.pkl")
    st.text(f"{e}")
    st.stop()

# Streamlit Config
st.set_page_config(page_title="Movie Recommender", layout="centered")
set_background('movieback2.png', opacity=1)

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
