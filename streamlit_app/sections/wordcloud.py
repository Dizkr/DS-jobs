import streamlit as st
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from src.utils import load_latest_file
import pandas as pd

INPUT_DIR = "data/processed"

# Sidebar selectors
categories = ['Data Analysis & BI', 'Data Engineering', 
'Data Science', 'Database Administration']


def render():
    cat_opt = st.segmented_control("Choose category", categories, selection_mode="single")

    if cat_opt is None:
        cat_opt = 'All'

    df = load_latest_file(f"justjoinit_{cat_opt}_word_freq", INPUT_DIR)

    # Ensure proper column names if unsure
    df.columns = [col.strip().lower() for col in df.columns]

    # Convert counts to numeric
    df['count'] = pd.to_numeric(df['count'], errors='coerce')

    # Drop non-numeric or missing entries
    df = df.dropna(subset=['count'])

    # Convert to proper format
    freq = {str(row['word']): float(row['count']) for _, row in df.iterrows()}

    # Create the wordcloud object
    wordcloud = WordCloud(width=1600, height=500, margin=0).generate_from_frequencies(freq)

    # Display the generated image:
    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)