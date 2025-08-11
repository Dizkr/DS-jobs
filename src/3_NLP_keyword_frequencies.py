#%%
import stopwords
from datetime import datetime
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from src.utils import load_latest_file

DIR = "data/processed"

jobs_df = load_latest_file("justjoinit_jobs", DIR)

english_stopwords = set(stopwords.get_stopwords('english')).union(set(ENGLISH_STOP_WORDS))
polish_stopwords = set(stopwords.get_stopwords('polish') )
additional_stopwords = set(["oraz", "danych", "english", "hands", "znajomość", 
"pracy", "experience", "skills", "doświadczenie", 
"rozwiązań", "modeli", "role", "years", "best"])

combined_stopwords = list(english_stopwords.union(polish_stopwords).union(additional_stopwords))

vectorizer = CountVectorizer(stop_words=combined_stopwords, ngram_range=(2,3), max_features=80)

freq_dict = {}

X_all = vectorizer.fit_transform(jobs_df["Job Description"].fillna(""))
all_freqs = dict(zip(vectorizer.get_feature_names_out(), X_all.toarray().sum(axis=0)))
freq_dict["All"] = all_freqs

categories = jobs_df["Category"].unique()
categories = categories[categories != "Unclassified"]

for category in categories:
    subset = jobs_df[jobs_df["Category"] == category]["Job Description"].fillna("")
    X_de = vectorizer.fit_transform(subset)
    de_freqs = dict(zip(vectorizer.get_feature_names_out(), X_de.toarray().sum(axis=0)))
    freq_dict[category] = de_freqs

date = datetime.today().strftime('%Y_%m_%d')

for name, freqs in freq_dict.items():
    df = pd.DataFrame(list(freqs.items()), columns=["word", "count"])
    df.to_csv(f"{DIR}/justjoinit_{name}_word_freq_{date}.csv", index=False)

