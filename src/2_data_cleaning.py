
import pandas as pd
import json
from datetime import datetime
from src.utils import load_latest_file, clean_and_map_skills, classify_job

INPUT_DIR = "data/raw"
OUTPUT_DIR = "data/processed"

# Load the most recent job and skill CSVs
jobs_df = load_latest_file("justjoinit_jobs", INPUT_DIR)
skills_df = load_latest_file("justjoinit_skills", INPUT_DIR)

# Keep manual classification in a new column
jobs_df['Category_manual'] = jobs_df['Job Title'].apply(classify_job)

# Also keep 'Category' for backward compatibility or for model training convenience
jobs_df['Category'] = jobs_df['Category_manual']

contract_map = {
    'B2B': 'B2B',
    'Permanent': 'Permanent',
    'B2B, Permanent': 'Permanent or B2B',
    'Permanent, B2B': 'Permanent or B2B',
    'Specific-task': 'Specific-task',
    'Mandate': 'Mandate',
    'Mandate, B2B': 'Mandate',
    'Any': 'Any',
    'Internship': 'Internship'
}

# Normalize employment types into cleaner labels
jobs_df['Employment Type'] = jobs_df['Employment Type'].map(contract_map).fillna('Other')

language_levels = ['A1', 'A2', 'B1', 'B2', 'C1', 'C2']

# Extract language proficiency entries (A1–C2) into a separate DataFrame
languages_df = skills_df[skills_df['Experience Level'].isin(language_levels)].copy()
languages_df = languages_df.rename(columns={'Technology':'Language','Experience Level':'Level'})

# Keep only skills (excluding language entries)
pure_skills_df = skills_df[~skills_df['Experience Level'].isin(language_levels)]

pure_skills_df = clean_and_map_skills(pure_skills_df, skill_col='Technology')

# Add  'Category' and 'Experience' to skills DataFrame
pure_skills_df = pure_skills_df.join(jobs_df[['Job ID', 'Category', 'Experience']].set_index('Job ID'), on='Job ID', how='left')

# Extract salaries
salaries_df = jobs_df.join(jobs_df["Salaries"].apply(json.loads).apply(pd.Series))
salaries_df = salaries_df[salaries_df[0].notna()]

rows_to_expand = salaries_df[salaries_df[1].notnull()].copy()
rows_to_expand[0] = rows_to_expand[1]
df_expanded = pd.concat([salaries_df, rows_to_expand], ignore_index=True).drop(columns=1)

salaries_df = pd.concat([df_expanded.drop(columns=[0]), df_expanded[0].apply(pd.Series)], axis=1)

# Save cleaned outputs to disk
date = datetime.today().strftime('%Y_%m_%d')
jobs_df.to_csv(f"{OUTPUT_DIR}/justjoinit_jobs_{date}.csv", index=False)
pure_skills_df.to_csv(f"{OUTPUT_DIR}/justjoinit_skills_{date}.csv", index=False)
languages_df.to_csv(f"{OUTPUT_DIR}/justjoinit_languages_{date}.csv", index=False)
salaries_df.to_csv(f"{OUTPUT_DIR}/justjoinit_salaries_{date}.csv", index=False)

# Summary of records processed
print(f"Processed {len(jobs_df)} job postings and {len(pure_skills_df)} skills (including {len(languages_df)} languages).")