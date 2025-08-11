import pandas as pd
import os
import re
from datetime import datetime
import joblib

MODEL_DIR = "models"

synonym_map = {
    'aws or azure': ['AWS', 'Azure'],
    'apache airflow Or hadoop': ['Apache Airflow', 'Hadoop'],
    'aws/gcp/azure': ['AWS', 'GCP', 'Azure'],
    'big query' : ['Bigquery'],
    'docker swarm': ['Docker'],
    'restful api': ['REST API'],
    'rest api': ['REST API'],
    'python 3.x': ['Python'],
    'sql server data tools': ['SQL Server'],
    'sql prompt': ['SQL Prompt'],
    'github actions': ['GitHub Actions'],
    'flask': ['Flask'],
    'dremio': ['Dremio'],
    'deep learning': ['Deep Learning'],
    'ml/ai': ['Machine Learning', 'Artificial Intelligence'],
    'ai/ml': ['Machine Learning', 'Artificial Intelligence'],
    'ai & machine learning': ['Machine Learning', 'Artificial Intelligence'],
    'ml': ['Machine Learning'],
    'ai': ['Artificial Intelligence'],
    'databrics spark': ['Databricks Spark'],
    'powerquery': ['Power Query'],
    'etl/elt': ['ETL', 'ELT'],
    'etl/elt processes': ['ETL', 'ELT'],
    'python(pandas)': ['Python', 'Pandas'],
    '(py)spark': ['PySpark', 'Spark'],
    'databricks/snowflake/bigquery/redshift': ['Databricks', 'Snowflake', 'BigQuery', 'Redshift'],
    'sql server': ['SQL Server'],
    'azure functions': ['Azure Functions'],
    'powerapps': ['PowerApps'],
    'powerautomate': ['Power Automate'],
    'microsoft office': ['Microsoft Office'],
    'ms office': ['Microsoft Office'],
    'ms azure': ['Microsoft Azure'],
    'microsoft 365': ['Microsoft 365'],
    'sql serve': ['SQL Server'],
    'azure db': ['Azure Database'],
    'azure data brics': ['Azure Databricks'],
    'azure synapse analytics': ['Azure Synapse Analytics'],
    'microsoft dynamics crm': ['Microsoft Dynamics CRM'],
    'azure logic apps': ['Azure Logic Apps'],
    'databricks spark': ['Databricks Spark'],
    'microsoft d365': ['Microsoft Dynamics 365'],
    'microsoft bi': ['Microsoft BI'],
    'rds sql database': ['Amazon RDS SQL Database'],
    'azure data storage': ['Azure Data Storage'],
    'snowflake cloud data platform': ['Snowflake'],
    'google big query': ['Google BigQuery'],
    'aws aurora': ['AWS Aurora'],
    'microsoft planner': ['Microsoft Planner'],
    'microsoft access': ['Microsoft Access'],
    'powerpoint': ['PowerPoint'],
    'dax studio': ['DAX Studio'],
    'jquexy': ['jQuery'],
    'ms fabric': ['Microsoft Fabric'],
    'sap cdc': ['SAP CDC'],
    'python/pandas': ['Python', 'Pandas'],
    'apache beam': ['Apache Beam'],
    'google cloud services': ['Google Cloud Platform'],
    'google (google cloud platform, looker studio)': ['Google Cloud Platform', 'Looker Studio'],
    'gitflow': ['GitFlow'],
    'teradata': ['Teradata'],
    'sas': ['SAS'],
    'sinequa': ['Sinequa'],
    'idmc': ['IDMC'],
    'kubernetes engine': ['Kubernetes'],
    'google kubernetes engine': ['Google Kubernetes Engine'],
    'active directory': ['Active Directory'],
    'dns': ['DNS'],
    'dhcp': ['DHCP'],
    'rest': ['REST API'],
    'graphql': ['GraphQL'],
    'html5': ['HTML5'],
    'html': ['HTML'],
    'css': ['CSS'],
    'node.js': ['Node.js'],
    'golang': ['Go'],
    'go': ['Go'],

    # PostgreSQL
    'postresql': ['PostgreSQL'],
    'postgresql': ['PostgreSQL'],
    'postgres': ['PostgreSQL'],
    'pl/pgsql': ['PostgreSQL'],

    # MySQL
    'mysql': ['MySQL'],

    # SQL Server
    'mssql': ['SQL Server'],
    'ms sql': ['SQL Server'],
    'ms sql server': ['SQL Server'],
    'microsoft sql': ['SQL Server'],
    'microsoft sql server': ['SQL Server'],

    # Oracle
    'oracle db': ['Oracle'],
    'oracle sql': ['Oracle'],
    'plsql': ['Oracle'],
    'plsql (oracle)': ['Oracle'],
    'oracle pl/sql': ['Oracle'],
    'pl/sql': ['Oracle'],
    'pl/sql/pl/pqsql': ['Oracle'],
    'oracle/sql': ['Oracle'],
    'oracle dba': ['Oracle'],

    # Azure
    'microsoft azure': ['Azure'],
    'azure cloud': ['Azure'],
    'azure services': ['Azure'],
    'microsoft azure cloud': ['Azure'],
    'azure synapse': ['Azure Synapse Analytics'],
    'azure data lake': ['Azure Data Lake'],
    'azure data lake gen2': ['Azure Data Lake'],
    'azure data factory': ['Azure Data Factory'],
    'azure analysis services': ['Azure Analysis Services'],
    'azure sql': ['Azure SQL'],

    # AWS
    'amazon aws': ['AWS'],
    'aws': ['AWS'],
    'aws/azure/gcp/snowflake': ['AWS', 'Azure', 'GCP', 'Snowflake'],
    'amazon redshift': ['Redshift'],

    # GCP
    'google cloud': ['GCP'],
    'google cloud platform': ['GCP'],
    'google cloud platform (gcp)': ['GCP'],

    # Databricks
    'databrics': ['Databricks'],
    '•\tdatabricks': ['Databricks'],

    # DBT
    'dbt core': ['DBT'],
    'dbt cloud': ['DBT'],

    # BI Tools
    'power bi': ['Power BI'],
    'powerbi': ['Power BI'],
    'microsoft power bi': ['Power BI'],
    'power platform': ['Power Platform'],
    'power apps': ['Power Apps'],
    'power automate': ['Power Automate'],
    'power automate desktop': ['Power Automate'],
    'power query': ['Power Query'],
    'power pages': ['Power Pages'],

    # Tableau / Looker
    'tableau or looker': ['Tableau', 'Looker'],
    'tableau desktop': ['Tableau'],
    'tableau server': ['Tableau'],
    'looker studio': ['Looker'],
    'lookml': ['Looker'],

    # PySpark
    'pyspark/pandas': ['PySpark', 'Pandas'],
    'pyspark': ['PySpark'],

    # Pandas + other combos
    'pandas / numpy / scikit-learn': ['Pandas', 'Numpy', 'Scikit-Learn'],

    # Python (cleanup)
    'python,': ['Python'],

    # SQL/NoSQL Combos
    'sql and nosql databases': ['SQL', 'NoSQL'],
    'sql & database skill': ['SQL'],

    'sql': ['SQL'],
    't-sql': ['T-SQL'],
    'transact-sql': ['T-SQL'],
    'sql source control': ['SQL Source Control'],

    # Microsoft Excel
    'ms excel': ['Microsoft Excel'],
    'microsoft office excel': ['Microsoft Excel'],

    # Data Science Keywords
    'data analysys': ['Data Analysis'],
    'machine learning': ['Machine Learning'],
    'natural language processing': ['NLP'],
    'nlp': ['NLP'],
    'llms': ['LLMs'],
    'llm': ['LLMs'],

    # Jupyter
    'jupyter notebook': ['Jupyter'],

    # Git
    'github': ['GitHub'],
    'gitlab': ['GitLab'],

    # Cloud
    'cloud data platform': ['Cloud'],
    'cloud storage': ['Cloud'],
    'cloud development kit': ['Cloud Dev Kit'],
    'cloud platforms': ['Cloud'],

    # Typos / Format Cleanup
    'datya modelinig': ['Data Modeling'],
    'bazy danych': ['Databases'],
    'hortownia danych': ['Data Warehouse'],
    'dhw': ['Data Warehouse'],
    'datawarehouse': ['Data Warehouse'],
    'data centre': ['Data Center'],
    'dataverse': ['Microsoft Dataverse'],
    'data, modeling': ['Data Modeling'],
    'strategia data & ai': ['Data & AI Strategy'],
    'google analytics 4': ['Google Analytics'],
}

def load_latest_file(prefix, folder):
    """
    Find the latest CSV file in the folder with the given prefix and a date in the format YYYY_MM_DD.
    
    Args:
        prefix (str): Prefix of the file name (e.g. "justjoinit_dsjobs").
        folder (str): Directory where files are stored.
    
    Returns:
        str: Path to the latest file matching the prefix.
    """
    pattern = re.compile(rf"{re.escape(prefix)}_(\d{{4}}_\d{{2}}_\d{{2}})\.csv")
    files = [f for f in os.listdir(folder) if pattern.match(f)]
    
    if not files:
        raise FileNotFoundError(f"No files found with prefix '{prefix}' in {folder}")
    
    # Extract dates and sort
    dated_files = [(f, datetime.strptime(pattern.match(f).group(1), "%Y_%m_%d")) for f in files]
    latest_file = max(dated_files, key=lambda x: x[1])[0]
    
    path =  os.path.join(folder, latest_file)

    return pd.read_csv(path)

def clean_and_map_skills(df, skill_col):
    """
    Clean and normalize skills in a DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame containing skills in column skill_col
        skill_col (str): Name of the column with skill names
    
    Returns:
        pd.DataFrame: Cleaned DataFrame with columns:
          - original skill column,
          - 'skill_norm' - normalized lowercase skill,
          - 'skill_mapped' - canonical mapped skill(s) (list of strings),
          - 'skill_final' - human-readable final skill(s) (string, joined if multiple)
    """
    # === Setup drop lists ===
    languages = {'english', 'angielski', 'polish', 'fluent english', 'polish & english'}
    
    soft_skills = {
        'proactivity', 'critical thinking', 'communication skills', 'team player',
        'leadership', 'clean coder', 'product owner', 'troubleshooting', 'storytelling',
        'process automation', 'process mining', 'process', 'task', 'experience',
        'optimization', 'optimization methodologies', 'reading code', 'integration',
        'planning', 'controlling', 'management', 'kanban', 'kaban', 'testplan',
        'implementation specialist', 'security', 'project management', 'agile', 'scrum',
        'collaboration', 'problem solving', 'presentation skills'
    }
    
    # Vague concepts except ML and AI which we keep (map consistently)
    vague_concepts_to_drop = {
        'cloud', 'data', 'analytics', 'engineering', 'software architecture',
        'devops', 'product management', 'testing', 'validation', 'quality assurance',
        'devsecops', 'devops research and assessment', 'planning', 'controlling',
        'optimization', 'optimization methodologies', 'data science', 'analytical thinking',
        'big data', 'business analysis', 'data modeling', 'communication', 'reporting', 
        'e-commerce', 'energy', 'architecture', 'team leadership', 'performance', 
        'programming', 'data, modeling', 'metadata', 'tech lead', 'vendor', 'data center', 
        'data analytics'
    }
    
    # === Synonym mapping (keys lowercase) ===
    # Values are always lists of canonical skill names (for multi-mapping)

    
    # Normalize function
    def normalize_text(text):
        return text.strip().lower()
    
    def should_drop(skill_norm):
        if skill_norm in languages:
            return True
        if skill_norm in soft_skills:
            return True
        if skill_norm in vague_concepts_to_drop:
            return True
        # Drop if contains banned keywords (avoid false positives)
        if re.search(r'\b(process|task|experience|management|automation|validation|planning|controlling|testing|quality assurance|collaboration)\b', skill_norm):
            return True
        return False
    
    def map_synonym(row):
        norm = row['skill_norm']
        original = row[skill_col]
        if norm in synonym_map:
            return synonym_map[norm]
        else:
            return [original]  # use original exactly as-is
    
    # Copy df to avoid side effects
    df_clean = df.copy()
    
    # Normalize skills column
    df_clean['skill_norm'] = df_clean[skill_col].astype(str).apply(normalize_text)
    
    # Drop rows based on should_drop
    df_clean = df_clean[~df_clean['skill_norm'].apply(should_drop)].copy()
    
    # Map synonyms
    df_clean['skill_mapped'] = df_clean.apply(map_synonym, axis=1)

    # Explode rows for multiple mapped skills
    df_clean = df_clean.explode('skill_mapped')

    # Replace original skill column with mapped canonical skill
    df_clean[skill_col] = df_clean['skill_mapped']

    # Drop helper columns
    df_clean = df_clean.drop(columns=['skill_norm', 'skill_mapped'])

    # Reset index and return
    return df_clean.reset_index(drop=True)

def matches_keyword(text, keywords):
    return any(re.search(rf"\b{re.escape(k)}\b", text) for k in keywords)

def manual_classify(title):
    # Define keywords for each merged category
    data_science_keywords = [
        'scientist', 'machine learning', 'ai', 'nlp', 'deep learning', 'science', 'llm',
        'modeler', 'modeller', 'ml', 'bot detection', 'fraud detection', 'mlops'
    ]

    data_engineering_keywords = [
        'data engineer', 'data engineering', 'etl', 'pipeline', 'snowflake', 'databricks', 'cloud', 
        'database', 'integration', 'azure', 'dataops',
        'master data', 'hurtowni danych', 'dwh',
        'infrastructure', 'mulesoft', 'system integration', 'inżynier danych'
    ]

    data_analysis_keywords = [
        'analyst', 'reporting', 'dashboard', 'power bi', 'powerbi', 'analytics', 'business intelligence',
        'analityk', 'process mining', 'celonis', 'conversion specialist', 'specjalista ds. analiz', 'bi'
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

def model_classify(job_title: str, model=None) -> str:
    if model is None:
        model_path = MODEL_DIR / "job_classifier_pipeline-LR.pkl"
        if model_path.exists():
            model = joblib.load(model_path)
        else:
            raise FileNotFoundError(f"No model found at {model_path}")
    return model.predict([job_title])[0]

def classify_job(job_title, use_model=False, model=None):
    if not use_model:
        return manual_classify(job_title)
    else:
        return model.predict([job_title])[0]

