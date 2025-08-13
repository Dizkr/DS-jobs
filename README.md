# Data Analysis and Classification of Job Ads from justjoin.it

This project collects, cleans, and analyzes job advertisements from the Polish job board [justjoin.it](https://justjoin.it/job-offers/all-locations/data).

## Project Overview

The project automates the collection, cleaning, and analysis of job ads to uncover trends in the job market and identify in-demand skills. It combines web scraping, NLP, and machine learning to process the data. The results are presented in an interactive [Streamlit dashboard](https://ds-jobs-odqrhuay6hqx45rvmwrckt.streamlit.app/), allowing users to easily explore job and skill trends.

## Features

* **Data Collection:** Automatically scrape job advertisements from justjoin.it.
* **Data Cleaning:** Normalize and filter job titles, descriptions, and skills for consistent analysis.
* **NLP Analysis:** Extract keyword frequencies and analyze skill co-occurrence patterns.
* **Classification Pipeline:** Detect language and categorize job ads using TF-IDF vectorization and logistic regression.
* **Interactive Visualization:** Explore job trends, salary distributions, and skill networks through a Streamlit dashboard.

## Technologies

* **General:** Python, Pandas, NumPy
* **Web scraping:** Selenium, BeautifulSoup, Requests
* **NLP & Machine Learning:** scikit-learn, langdetect
* **Classification:** scikit-learn (TF-IDF, logistic regression), langdetect
* **Data Visualization:** matplotlib, seaborn, Streamlit

## Project Structure
```
├── data 
│   ├── raw              # Raw scraped job ads
│   └── processed        # Cleaned and processed datasets
├── models               # Saved ML models (pickle files)
├── src                  # Source code for scraping, cleaning, NLP, and classification
│   ├── 1_data_collection.py
│   ├── 2_data_cleaning.py
│   ├── 3_NLP_keyword_frequencies.py
│   ├── 4_NLP_skill_cooccurrence.py
│   ├── 5_NLP_classification.py
│   ├── __init__.py
│   └── utils.py
├── streamlit_app         # Streamlit dashboard application
│   ├── app.py
│   ├── requirements.txt
│   └── sections          # Modular dashboard components
│       ├── jobs_overview.py
│       ├── salaries_overview.py
│       ├── skills_cooccurrence.py
│       ├── skills_overview.py
│       └── wordcloud.py
├── README.md             # Project documentation
└── requirements.txt      # Python dependencies
```

## Installation / Usage
1. Clone this repository:
```git clone https://github.com/yourusername/job-market-analysis.git```

2. Navigate to the project directory:
```cd job-market-analysis```

3. Install dependencies:
```pip install -r requirements.txt```

4. Set up ChromeDriver or GeckoDriver for Selenium (required for scraping).

Data Collection and Cleaning

```
python src/1_data_collection.py
python src/2_data_cleaning.py
```

NLP analysis

```
python src/3_NLP_keyword_frequencies.py
python src/4_NLP_skill_cooccurrence.py
```
Classification Model
```
python src/5_NLP_classification.py

```
Streamlit Dashboard
```
streamlit run streamlit_app/app.py
```

## Data
* All data is scraped from [justjoin.it](https://justjoin.it/job-offers/all-locations/data) website.
* Raw job ads are saved under ```data/raw/```.
* Cleaned and processed datasets are stored in  ```data/processed/```.
* The scraped and processed dataset contains: 
    * Job title, 
    * Job description, 
    * Employer name,
    * Expected experience (junior, mid, senior),
    * Offered employment type (B2B, Permanent, B2B or Permanent, Mandate, Internship, Any),
    * Operating mode (Remote, Hybrid, Office),
    * Offered salary range,
    * Manually assigned category (Data Analysis & BI, Data Architecture, Data Engineering, Data Science, Database Administration),
    * Required skills/technologies, including expected proficiency level (nice to have, junior, regular, advanced, master).

## Data scraping
All data used in this project is collected from [justjoin.it](https://justjoin.it/job-offers/all-locations/data) using Selenium,  BeautifulSoup and Requests. 

The process begins by gathering all links to relevant job advertisements. Selenium handles scrolling on the page, while BeautifulSoup extracts the URLs after each scroll. This approach allows scraping data from dynamically loaded content:
```python
def collect_job_links(driver, scroll_pause_time=0.015):
    """
    Scroll through the job listings page and collect unique job offer URLs.

    The function scrolls down the page until the number of unique job links
    found does not increase for several iterations (stable_iterations).

    Args:
        driver (webdriver): Selenium WebDriver instance.
        scroll_pause_time (float): Time to wait after each scroll (seconds).

    Returns:
        list: List of unique job offer URLs collected from the page.
    """
    url = "https://justjoin.it/job-offers/all-locations/data"
    driver.get(url)
    time.sleep(3) # Wait for page to load

    screen_height = driver.execute_script("return window.innerHeight;")
    seen_links = set()
    iter_count = 0
    prev_count = 0
    stable_iterations = 0

    while stable_iterations < SCROLL_END_THRESHOLD:
        soup = BeautifulSoup(driver.page_source, "html.parser")

        # Extract job offer links from anchor tags with href containing "/job-offer/"
        for link in soup.find_all('a', href=True):
            if "/job-offer/" in link['href']:
                seen_links.add('https://justjoin.it' + link['href'])


        current_count = len(seen_links)
        print(f"Unique links: {current_count} after scroll {iter_count+1}")

        # Scroll down by one screen height
        driver.execute_script("window.scrollBy(0, arguments[0]);", screen_height)
        time.sleep(scroll_pause_time)

        # Check if the number of links has stabilized (no new links found)
        if current_count == prev_count:
            stable_iterations += 1
        else:
            stable_iterations = 0  # Reset if new links appeared
        
        prev_count = current_count  # Update previous count
        iter_count += 1

    print(f"Collected {len(seen_links)} links after {iter_count+1} scrolls")
    return list(seen_links)
```

After collecting the URLs of all job advertisements, each URL is visited and relevant data is extracted:
```python
def scrape_job_data(url, job_id):
    """
    Scrape job details and skills from a given job offer URL.

    Args:
        url (str): URL of the job offer page.
        job_id (int): Unique integer ID assigned to the job.

    Returns:
        tuple: A tuple containing:
            - job_details (dict): Extracted details about the job.
            - skills (list): List of dictionaries with skills and experience levels.
    """
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, "lxml")

    job_details = {}

    # Extract job title from the <meta property="og:title"> tag (SEO metadata)
    meta_title = soup.find('meta', {'property': 'og:title'})
    job_title = meta_title['content'] if meta_title else "Unknown Title"

    # Extract employer name from the <h2> tag with class "MuiTypography-root"
    employer_name = soup.find("h2", class_="MuiTypography-root").text.strip() if soup.find("h2", class_="MuiTypography-root") else "Unknown Employer"


    # Extract job details (Experience, Employment Type, etc.)
    sections = soup.find_all("div", class_="MuiBox-root")
    for section in sections:
        label = section.text.strip()
        if label in ["Experience", "Employment Type", "Operating mode", "Type of work"]:
            value_section = section.find_next("div")
            value = value_section.text.strip() if value_section else "N/A"
            job_details[label] = value

    job_details["Job ID"] = job_id
    job_details["Job Title"] = job_title
    job_details["Employer Name"] = employer_name

    # Extract job description paragraphs or fallback content
    description_parts = []

    desc_paragraphs = soup.find_all("p", class_="editor-paragraph")
    if desc_paragraphs:
        description_parts = [p.get_text(separator=" ", strip=True) for p in desc_paragraphs]
    else:
        desc_container = soup.find("div", class_="MuiBox-root mui-1vqiku9")
        if desc_container:
            ps = desc_container.find_all(["p", "li"])
            description_parts = [tag.get_text(separator=" ", strip=True) for tag in ps]

    description_text = " ".join(description_parts)
    job_details["Job Description"] = clean_description(description_text)

    # Extract salary information from salary blocks
    salaries = []
    seen_salaries = set()
    salary_blocks = soup.find_all("div", class_="MuiBox-root mui-1km0bek")
    
    for block in salary_blocks:
        try:
            # Extract min and max salary values from nested spans
            amount_spans = block.select("span.mui-mrzdjb span")
            if len(amount_spans) < 2:
                continue  # Skip blocks without expected structure
            
            min_val = int(amount_spans[0].text.replace(" ", ""))
            max_val = int(amount_spans[1].text.replace(" ", ""))
            
            # Extract salary type (e.g., "per month", "gross/net") if available
            fallback_span = block.find("span", class_="mui-1waow8k")
            text_node = fallback_span.get_text(strip=True) if fallback_span else ""

            key = (min_val, max_val, text_node)
            if key in seen_salaries:
                continue  # Avoid duplicates

            seen_salaries.add(key)

            salaries.append({
                "min": min_val,
                "max": max_val,
                "type": text_node
            })

        except Exception as e:
            print(f"Error parsing salary block: {e}")
            continue

    job_details["Salaries"] = json.dumps(salaries)

    # Extract required skills and corresponding experience levels
    skills = []
    for tech in soup.select("h4.MuiTypography-subtitle2"):
        tech_name = tech.get_text(strip=True)
        level_span = tech.find_next("span", class_="MuiTypography-subtitle4")
        experience_level = level_span.get_text(strip=True) if level_span else "Unknown"
        skills.append({"Job ID": job_id, "Technology": tech_name, "Experience Level": experience_level})

    return job_details, skills
```
For increased time-efficiency, multiple jobs are scraped in parallel using threads. Collected data on jobs and required technologies are saved into two files: ```f"{OUTPUT_DIR}/justjoinit_jobs_{date}.csv"``` and ```f"{OUTPUT_DIR}/justjoinit_skills_{date}.csv"```

## Data processing
The collected data is then processed in several steps:

1. **Job Categorization:** Job titles are assigned to categories (e.g., Data Science, Data Engineering) using keyword-based matching. This provides a first-pass classification, although it may be imperfect.


``` python
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
```
This manual keyword-based classification is clearly tedious and unreliable, which is why a **machine learning-based classification model** is introduced later for more accurate categorization.

2. **Skill Standardization:** Skills extracted from job postings are mapped to unified values using a synonym dictionary to avoid duplicates and normalize different spellings or representations of the same technology.
```python
synonym_map = {
    'aws or azure': ['AWS', 'Azure'],
    'apache airflow Or hadoop': ['Apache Airflow', 'Hadoop'],
    'aws/gcp/azure': ['AWS', 'GCP', 'Azure'],
    'restful api': ['REST API'],
    'rest api': ['REST API'],
    'ml/ai': ['Machine Learning', 'Artificial Intelligence'],
    'ai/ml': ['Machine Learning', 'Artificial Intelligence'],
    'ai & machine learning': ['Machine Learning', 'Artificial Intelligence'],
    'ml': ['Machine Learning'],
    'ai': ['Artificial Intelligence'],
    'etl/elt': ['ETL', 'ELT'],
    'etl/elt processes': ['ETL', 'ELT'],
    ...
}
```
This ensures that variations or spellings of the same technology are standardized for consistent skill analysis and aggregation.

3. **Language Removal:** Human languages and soft skills are removed from the skill lists to focus on technical requirements.

4. **Salary Extraction:** Salary ranges are parsed and standardized for easier analysis.

## Keyword frequencies

To better understand the most common terms in job descriptions, keywords are extracted and used for building visualizations such as word clouds. The extraction uses the ```CountVectorizer``` from the ```scikit-learn``` package, with preprocessing to remove irrelevant words.

1. **Stopwords:** Both English and Polish stopwords are removed. Additional domain-specific stopwords (e.g., "experience", "pracy", "danych") are manually added to avoid overly common, non-informative terms.
```python
english_stopwords = set(stopwords.get_stopwords('english')).union(set(ENGLISH_STOP_WORDS))
polish_stopwords = set(stopwords.get_stopwords('polish') )
additional_stopwords = set(["oraz", "danych", "english", "hands", "znajomość", 
"pracy", "experience", "skills", "doświadczenie", 
"rozwiązań", "modeli", "role", "years", "best"])

combined_stopwords = list(english_stopwords.union(polish_stopwords).union(additional_stopwords))
```
2. **Keyword Extraction:** N-grams of length 2–3 are extracted, and the top 80 features are kept for analysis.
```python
vectorizer = CountVectorizer(stop_words=combined_stopwords, ngram_range=(2,3), max_features=80)
```
3. **Global Keyword Frequencies:** Keywords are extracted from all job postings combined.
```python
X_all = vectorizer.fit_transform(jobs_df["Job Description"].fillna(""))
all_freqs = dict(zip(vectorizer.get_feature_names_out(), X_all.toarray().sum(axis=0)))
freq_dict["All"] = all_freqs
```
4. **Category-specific Frequencies:** Keywords are also extracted for each job category separately, excluding "Unclassified" jobs.
```python
categories = jobs_df["Category"].unique()
categories = categories[categories != "Unclassified"]

for category in categories:
    subset = jobs_df[jobs_df["Category"] == category]["Job Description"].fillna("")
    X_de = vectorizer.fit_transform(subset)
    de_freqs = dict(zip(vectorizer.get_feature_names_out(), X_de.toarray().sum(axis=0)))
    freq_dict[category] = de_freqs
```

## Skill co-occurence 
To understand which technologies tend to appear together in job postings, a skill co-occurrence analysis was performed. This helps visualize complementary skills and common technology stacks.

1. **Skill Extraction:** 
To capture skills mentioned in job descriptions—even if they were not explicitly listed as required—the text is compared against a ```synonym_map``` defined earlier. This ensures that different variants of the same skill (e.g., "Postgres" vs. "PostgreSQL") are recognized.
```python
def extract_skills(text, synonym_map):
    found_skills = set()
    lowered_text = text.lower()
    
    # Stage 1: exact key matches
    for key, values in synonym_map.items():
        pattern = rf'\b{re.escape(key.lower())}\b'
        if re.search(pattern, lowered_text):
            found_skills.update(values)
    
    # Stage 2: individual value matches
    for values in synonym_map.values():
        for val in values:
            pattern = rf'\b{re.escape(val.lower())}\b'
            if re.search(pattern, lowered_text):
                found_skills.add(val)
    
    return list(found_skills)

jobs_df["skills_found"] = jobs_df["Job Description"].apply(lambda x: extract_skills(x, synonym_map))
```
This adds a new column ```skills_found``` to the dataset, containing all recognized skills per job posting.

2. **Computing Co-occurrence:** Co-occurrence is calculated for each category of jobs as well as for all jobs combined. The procedure includes:
    * Counting how often each pair of skills appears together in the same posting.
    * Creating a symmetric co-occurrence matrix.
    * Normalizing the matrix to percentages relative to the total occurrences of each skill.
    * Extracting a top-10 skill matrix for easier visualization.

```python
def compute_cooccurrence_for_category(category, jobs_df, flat_skill_map, out_dir, date):
    if category == "All":
        category_df = jobs_df 
    else:
        category_df = jobs_df[jobs_df["Category"]==category]

    co_occurrence = defaultdict(int)

    # Count co-occurrences
    for skills in category_df["skills_found"]:
        for s1, s2 in combinations(sorted(set(skills)), 2):
            co_occurrence[(s1, s2)] += 1

    # Create matrix
    unique_skills = sorted(set([s for pair in co_occurrence for s in pair]))
    co_matrix = pd.DataFrame(0, index=unique_skills, columns=unique_skills)
    for (s1, s2), count in co_occurrence.items():
        co_matrix.loc[s1, s2] = count
        co_matrix.loc[s2, s1] = count 

    # Normalize to percentages
    all_skills = [skill for skills_list in category_df["skills_found"] for skill in skills_list]
    skill_counts = Counter(all_skills)
    co_matrix_pct = co_matrix.copy().astype(float)
    for s1 in co_matrix_pct.index:
        count_s1 = skill_counts[s1]
        if count_s1 > 0:
            co_matrix_pct.loc[s1] = (co_matrix_pct.loc[s1] / count_s1) * 100
        else:
            co_matrix_pct.loc[s1] = 0  

    # Top 10 skills
    top_skills = [skill for skill, _ in skill_counts.most_common(10)]
    co_matrix_pct_top = co_matrix_pct.loc[top_skills, top_skills]
    co_matrix_pct_top = co_matrix_pct.loc[top_skills, top_skills]

    co_matrix_pct_top.to_csv(f"{DIR}/justjoinit_{category}_cooccurrence_{date}.csv")
```
The resulting CSV files can be used to create heatmaps showing the percentage of times skills co-occur, helping to identify common technology combinations in job postings.


## Job Classification Model
Manually classifying jobs based on their titles is tedious because unconventional titles are frequently used, requiring a large mapping dictionary. To streamline this, a machine learning pipeline was developed to automatically classify jobs into categories.

### Pipeline Overview

The pipeline performs the following steps:
1. **Language Detection:**
Determines if the job description is in Polish or English, allowing language-specific processing.

2. **Text Cleaning:**
Removes punctuation and stopwords from job titles and descriptions. This ensures the model focuses on meaningful words.

3. **Feature extraction:** Combines features from both job titles and descriptions, giving higher weight to title words. A ```ColumnTransformer``` is used with TF-IDF vectorization and the language flag:

4. **Classification:** A ```LogisticRegression``` model is trained on the extracted features. The full pipeline is:
```python
class LanguageDetector(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        def safe_detect(text):
            try:
                return detect(text) if isinstance(text, str) and text.strip() else "unknown"
            except LangDetectException:
                return "unknown"

        X = X.copy()
        X['is_polish'] = X['Job Description'].apply(lambda txt: 1 if safe_detect(txt) == 'pl' else 0)
        return X

class TextCleaner(BaseEstimator, TransformerMixin):
    def __init__(self, stopwords):
        self.stopwords = stopwords

    def fit(self, X, y=None):
        return self

    def clean_text(self, text):
        if not isinstance(text, str):
            return ""
        # Remove punctuation
        nopunc = ''.join(char for char in text if char not in string.punctuation)
        # Remove stopwords
        filtered_words = [word for word in nopunc.split() if word.lower() not in self.stopwords]
        return ' '.join(filtered_words)

    def transform(self, X):
        X = X.copy()
        X['Job Title'] = X['Job Title'].apply(self.clean_text)
        X['Job Description'] = X['Job Description'].apply(self.clean_text)
        return X

# Feature extraction 
combined_features = ColumnTransformer([
    ('title', TfidfVectorizer(max_features=TITLE_MAX_FEATURES), 'Job Title'),
    ('desc', TfidfVectorizer(max_features=DESC_MAX_FEATURES), 'Job Description'),
    ('lang', 'passthrough', ['is_polish'])
])

# Build pipeline
pipeline = Pipeline([
    ('lang_detect', LanguageDetector()),
    ('text_cleaning', TextCleaner(combined_stopwords)),
    ('features', combined_features),
    ('clf', LogisticRegression(max_iter=LOGREG_MAX_ITER))
])
```
This pipeline allows automatic and scalable classification of job postings, reducing the need for manual labeling and extensive mapping dictionaries.

### Model training

The model was trained on data where ```"Unclassified"``` jobs were removed from the training set to prevent bias:
```python
# Split features & labels
X = jobs_df[['Job Title', 'Job Description']]
y = jobs_df['Category']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, random_state=RANDOM_STATE
)

# Drop "Unclassified" from training set only
train_mask = y_train != "Unclassified"
X_train = X_train[train_mask]
y_train = y_train[train_mask]

# --- Train model ---
model = pipeline.fit(X_train, y_train)
```
### Thresholded Prediction
To handle low-confidence predictions, a probability threshold is applied. Predictions below the threshold are labeled as ```"Unclassified"```:
```python
def threshold_predict(model, X, threshold=THRESHOLD_DEFAULT):
    probs = model.predict_proba(X)
    class_preds = model.classes_[np.argmax(probs, axis=1)]
    max_probs = np.max(probs, axis=1)

    # Replace low-confidence predictions with "Unclassified"
    final_preds = [pred if prob >= threshold else "Unclassified"
                   for pred, prob in zip(class_preds, max_probs)]
    
    return final_preds
```
### Evaluation
Using the thresholded predictions on the test set yields excellent performance:
```sql
Metrics on classified samples only:
                         precision    recall  f1-score   support

     Data Analysis & BI       0.96      0.96      0.96        23
      Data Architecture       1.00      1.00      1.00        10
       Data Engineering       0.97      1.00      0.98        62
           Data Science       1.00      1.00      1.00        20
Database Administration       1.00      0.83      0.91        12

               accuracy                           0.98       127
              macro avg       0.99      0.96      0.97       127
           weighted avg       0.98      0.98      0.98       127

Unclassified overlap:
True Positives (correctly unclassified): 15
False Positives (predicted unclassified but actually classified): 17
False Negatives (manual unclassified but predicted classified): 9
```
This approach ensures high-confidence predictions while still identifying uncertain cases as ```"Unclassified"```, making the model robust for real-world job classification tasks.

## Streamlit app
The dashboard provides an interactive interface to explore job postings. Features include:
* **Overview of job postings:** Number of jobs posted per category, filtered by type of work, experience, employment type, and operating mode.
* **Skill frequency:** Explore how often skills appear across categories and experience levels.
* **Skill co-occurence:** Visualize which skills tend to appear together within each category.
* **Salary range charts:** Compare salaries by salary type, job category, and experience.
* **Word clouds** Quickly identify important keywords for each job category.

Users can interactively filter the data and explore insights in real time.

## License
This project is licensed under the [MIT License](https://opensource.org/license/mit). 

## Contact
Created by Piotr Kapuściński  
Email: piotrek.kapuscinski@gmail.com   
GitHub: https://github.com/Dizkr