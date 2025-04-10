import streamlit as st
import pickle
import pandas as pd
import base64
from huggingface_hub import hf_hub_download

# Function to set background image with opacity
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

def recommend(movie):
    movie_index = movies[movies['title'] == movie].index[0]
    similarity_array = similarity[movie_index]
    movie_list = sorted(enumerate(similarity_array), reverse=True, key=lambda x: x[1])[1:6]
    return [movies.iloc[i[0]].title for i in movie_list]

# Load movie_dict locally
movie_dict = pickle.load(open('movies_dict.pkl', 'rb'))
movies = pd.DataFrame(movie_dict)

# Load similarity.pkl from HuggingFace
similarity_path = hf_hub_download(
    repo_id="NERF-HAARIS/movie-recommender-files",
    filename="similarity.pkl"
)

with open(similarity_path, 'rb') as f:
    similarity = pickle.load(f)

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
