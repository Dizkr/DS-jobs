from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import time
import json
import requests
from bs4 import BeautifulSoup
import pandas as pd
from multiprocessing.dummy import Pool as ThreadPool
from datetime import datetime

NUM_THREADS = 4
SCROLL_PAUSE_TIME = 0.015
SCROLL_END_THRESHOLD = 6
OUTPUT_DIR = "data/raw"


# Setup Chrome options
options = webdriver.ChromeOptions()
options.add_argument("--headless")  # Run in headless mode (no GUI)
options.add_argument("--disable-blink-features=AutomationControlled")  # Avoid bot detection


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

    while stable_iterations < 5:
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

    return list(seen_links)

def clean_description(text):
    """
    Clean and normalize job description text.

    - Removes newlines and extra whitespace.
    - Normalizes text around colons to avoid unwanted spaces.

    Args:
        text (str): Raw job description text.

    Returns:
        str: Cleaned and normalized job description.
    """
    text = text.replace("\n", " ").strip()
    text = " ".join(text.split())
    parts = text.split(":")
    return parts[0] + "".join(f": {p.lstrip()}" for p in parts[1:])

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
        description_parts = [p.get_text(strip=True) for p in desc_paragraphs]
    else:
        desc_container = soup.find("div", class_="MuiBox-root mui-1vqiku9")
        if desc_container:
            ps = desc_container.find_all(["p", "li"])
            description_parts = [" ".join(tag.stripped_strings) for tag in ps]

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

def scrape_multiple_jobs(links):
    """
    Scrape job data and skills for multiple job URLs in parallel using threads.

    Args:
        links (list): List of job offer URLs.

    Returns:
        tuple: Two pandas DataFrames:
            - jobs_df: DataFrame containing job details.
            - skills_df: DataFrame containing job skills and experience levels.
    """
    pool = ThreadPool(processes=NUM_THREADS)
        
    results = []

    # Schedule scraping tasks asynchronously
    for i, url in enumerate(links):
        results.append(pool.apply_async(scrape_job_data, (url, i)))

    pool.close()
    pool.join()

    job_details_list = []
    skill_details_list = []

    # Collect results from each thread
    for result in results:
        job_details, skills = result.get() 
        job_details_list.append(job_details)
        skill_details_list.extend(skills)  

    jobs_df = pd.DataFrame(job_details_list)
    skills_df = pd.DataFrame(skill_details_list)
    return jobs_df, skills_df

def main():
    """
    Main function to orchestrate scraping:
    - Launch browser and collect job links.
    - Scrape job details and skills for each link.
    - Save results to CSV files.
    """
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    links = collect_job_links(driver, SCROLL_PAUSE_TIME)
    driver.quit()

    jobs_df, skills_df = scrape_multiple_jobs(links)
    date = datetime.today().strftime('%Y_%m_%d')

    jobs_df.to_csv(f"{OUTPUT_DIR}/justjoinit_jobs_{date}.csv", index=False)
    skills_df.to_csv(f"{OUTPUT_DIR}/justjoinit_skills_{date}.csv", index=False)

if __name__ == "__main__":
    main()




