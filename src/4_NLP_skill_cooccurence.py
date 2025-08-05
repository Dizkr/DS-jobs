import re
import pandas as pd
from collections import defaultdict
from src.utils import synonym_map, load_latest_file
from itertools import combinations
from collections import Counter
from datetime import datetime

DIR = "data/processed"
date = datetime.today().strftime('%Y_%m_%d')

jobs_df = load_latest_file("justjoinit_jobs", DIR)

categories = jobs_df["Category"].unique()
categories = categories[categories != "Unclassified"]
categories = list(categories) + ["All"]

# Flatten keys
flat_skill_map = {k.lower(): v for k, v in synonym_map.items()}

def extract_skills(text, flat_map):
    """Extract normalized skills from job description text using whole word matching."""
    found_skills = set()
    lowered_text = text.lower()

    for synonym, norm_skills in flat_map.items():
        # Escape and match only whole words or phrases
        pattern = rf'\b{re.escape(synonym)}\b'
        if re.search(pattern, lowered_text):
            found_skills.update(norm_skills)

    return list(found_skills)

jobs_df["skills_found"] = jobs_df["Job Description"].apply(lambda x: extract_skills(x, flat_skill_map))

def compute_cooccurrence_for_category(category, jobs_df, flat_skill_map, out_dir, date):
    if category == "All":
        category_df = jobs_df 
    else:
        category_df = jobs_df[jobs_df["Category"]==category]

    # Count co-occurrences
    co_occurrence = defaultdict(int)

    for skills in category_df["skills_found"]:
        for s1, s2 in combinations(sorted(set(skills)), 2):
            co_occurrence[(s1, s2)] += 1

    # Get unique skills
    unique_skills = sorted(set([s for pair in co_occurrence for s in pair]))

    # Create empty matrix
    co_matrix = pd.DataFrame(0, index=unique_skills, columns=unique_skills)

    # Fill matrix
    for (s1, s2), count in co_occurrence.items():
        co_matrix.loc[s1, s2] = count
        co_matrix.loc[s2, s1] = count  # mirror

    # Flatten all found skills from all postings into one list
    all_skills = [skill for skills_list in category_df["skills_found"] for skill in skills_list]

    skill_counts = Counter(all_skills)

    top_skills = [skill for skill, _ in skill_counts.most_common(10)]

    co_matrix_pct = co_matrix.copy().astype(float)

    for s1 in co_matrix_pct.index:
        count_s1 = skill_counts[s1]
        if count_s1 > 0:
            co_matrix_pct.loc[s1] = (co_matrix_pct.loc[s1] / count_s1) * 100
        else:
            co_matrix_pct.loc[s1] = 0  # avoid division by zero

    co_matrix_pct_top = co_matrix_pct.loc[top_skills, top_skills]

    co_matrix_pct_top.to_csv(f"{DIR}/justjoinit_{category}_cooccurence_{date}.csv")

for category in categories:
    compute_cooccurrence_for_category(category, jobs_df, flat_skill_map, DIR, date)

