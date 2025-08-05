
import pandas as pd
import json
import re
from datetime import datetime
from src.utils import load_latest_file, clean_and_map_skills

INPUT_DIR = "data/raw"
OUTPUT_DIR = "data/processed"

# Load the most recent job and skill CSVs
jobs_df = load_latest_file("justjoinit_jobs", INPUT_DIR)
skills_df = load_latest_file("justjoinit_skills", INPUT_DIR)

def matches_keyword(text, keywords):
    return any(re.search(rf"\b{re.escape(k)}\b", text) for k in keywords)

def classify_job(title):
    # Define keywords for each merged category
    data_science_keywords = [
        'scientist', 'machine learning', 'ai', 'nlp', 'deep learning', 'science', 'llm',
        'modeler', 'modeller', 'ml', 'bot detection', 'fraud detection', 'mlops'
    ]

    data_engineering_keywords = [
        'engineer', 'engineering', 'etl', 'pipeline', 'snowflake', 'databricks', 'cloud', 'developer',
        'database', 'integration', 'azure', 'dataops',
        'master data', 'hurtowni danych', 'dwh',
        'infrastructure', 'mulesoft', 'system integration', 'inżynier'
    ]

    data_analysis_keywords = [
        'analyst', 'reporting', 'dashboard', 'power bi', 'analytics', 'business intelligence',
        'analityk', 'process mining', 'celonis', 'conversion specialist', 'specjalista ds. analiz'
    ]

    database_administration_keywords = [
        'administrator', 'admin', 'db administrator', 'sql server production', 'młodszy administrator',
        'programista baz danych', 'database administrator', 'ms-sql', 'pl/sql', 'baz danych', 'sql'
    ]

    data_architect_keywords = [
        'data architect', 'solution architect', 'enterprise architect', 'architecture',
        'dimensional modeling', 'information architecture',
        'azure architecture', 'cloud architecture', 'data vault', 'architect',
        'architekt', 'database design', 'schema design'
    ]

    # Normalize title for case-insensitive matching
    title_lower = title.lower()

    # Check keywords by category
    if matches_keyword(title_lower, database_administration_keywords):
        return "Database Administration"
    elif matches_keyword(title_lower, data_architect_keywords):
        return "Data Architecture"
    elif matches_keyword(title_lower, data_science_keywords):
        return "Data Science"
    elif matches_keyword(title_lower, data_analysis_keywords):
        return "Data Analysis & BI"
    elif matches_keyword(title_lower, data_engineering_keywords):
        return "Data Engineering"
    else:
        return "Unclassified"

# Classify job titles into predefined categories
jobs_df['Category'] = jobs_df['Job Title'].apply(classify_job)

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