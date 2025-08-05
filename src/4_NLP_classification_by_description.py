#%%
import stopwords
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfTransformer, CountVectorizer
from src.utils import load_latest_file
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException
import string
from sklearn.cluster import KMeans
import pandas as pd

english_stopwords = set(stopwords.get_stopwords('english')).union(set(ENGLISH_STOP_WORDS))
polish_stopwords = set(stopwords.get_stopwords('polish') )
additional_stopwords = set(["oraz", "danych", "english", "hands", "znajomość", 
"pracy", "experience", "skills", "doświadczenie", 
"rozwiązań", "modeli", "role", "years"])
combined_stopwords = list(english_stopwords.union(polish_stopwords).union(additional_stopwords))

DIR = "data/processed"


#%%
jobs_df = load_latest_file("justjoinit_jobs", DIR)

jobs_df = jobs_df.drop(
    jobs_df[(jobs_df["Category"]=="Database Administration") | (jobs_df["Category"]=="Unclassified") | (jobs_df["Category"]=="Specialist / Other")].index)

def detect_language_safe(text):
    try:
        return detect(text) if isinstance(text, str) and text.strip() else "unknown"
    except LangDetectException:
        return "unknown"



jobs_df["Language"] = jobs_df["Job Description"].apply(detect_language_safe)

def text_process(text):
    """
    Takes in a string of text, then performs the following:
    1. Remove all punctuation
    2. Remove all stopwords
    3. Returns a list of the cleaned text
    """
    # Check characters to see if they are in punctuation
    nopunc = [char for char in text if char not in string.punctuation]

    # Join the characters again to form the string.
    nopunc = ''.join(nopunc)
    
    # Now just remove any stopwords
    return [word for word in nopunc.split() if word.lower() not in combined_stopwords]

polish_jobs = jobs_df[jobs_df["Language"] == "pl"]
english_jobs = jobs_df[jobs_df["Language"] == "en"]

#%%
bow_transformer_en = CountVectorizer(analyzer=text_process).fit(english_jobs["Job Description"])

descriptions_bow_en = bow_transformer_en.transform(english_jobs["Job Description"])
#%%
tfidf_transformer_en = TfidfTransformer().fit(descriptions_bow_en)

descriptions_tfidf_en = tfidf_transformer_en.transform(descriptions_bow_en)

#%%
kmeans = KMeans(n_clusters=3)
kmeans.fit(descriptions_tfidf_en)

#%%
english_jobs["Cluster"] = kmeans.predict(descriptions_tfidf_en)

#%%

cluster_category_pct = pd.crosstab(
    english_jobs["Category"],
    english_jobs["Cluster"],
    normalize='index'  # normalize over rows
)

plt.figure(figsize=(10, 6))
sns.heatmap(cluster_category_pct, annot=True, fmt=".1%", cmap="YlGnBu")
plt.title("Percentage of Category per Cluster")
plt.ylabel("Job Category")
plt.xlabel("Cluster")
plt.tight_layout()
plt.show()

#%%

from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(english_jobs["Job Description"].tolist())

import umap

umap_embeddings = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine').fit_transform(embeddings)

# Then cluster on these
kmeans = KMeans(n_clusters=3)
clusters = kmeans.fit_predict(umap_embeddings)
english_jobs["Cluster"] = clusters

#%%

cluster_category_pct = pd.crosstab(
    english_jobs["Category"],
    english_jobs["Cluster"],
    normalize='index'  # normalize over rows
)

plt.figure(figsize=(10, 6))
sns.heatmap(cluster_category_pct, annot=True, fmt=".1%", cmap="YlGnBu")
plt.title("Percentage of Category per Cluster")
plt.ylabel("Job Category")
plt.xlabel("Cluster")
plt.tight_layout()
plt.show()



#%%

import spacy
from spacy.matcher import PhraseMatcher
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load model
nlp = spacy.load("en_core_web_sm")

# Example input lists
tech_list = ["Python", "PostgreSQL", "Power BI", "Kubernetes", "Pandas", "Scikit-learn", "AWS"]
tech_mapping = {
    "postgres": "PostgreSQL",
    "plpgsql": "PostgreSQL",
    "powerbi": "Power BI",
    "scikit learn": "Scikit-learn",
    "scikitlearn": "Scikit-learn"
    # Add more as needed
}

# Build matcher
matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
patterns = [nlp.make_doc(skill) for skill in tech_list]
matcher.add("TECH", patterns)

# Function to extract and normalize
def extract_normalized_skills(text):
    doc = nlp(text)
    matches = matcher(doc)
    raw_skills = [doc[start:end].text.lower() for _, start, end in matches]
    return list(set(tech_mapping.get(skill.replace(" ", ""), skill.title()) for skill in raw_skills))

# Apply to job descriptions
jobs_df["Matched Skills"] = jobs_df["Job Description"].apply(extract_normalized_skills)