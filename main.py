import os
import json
import requests
import pandas as pd
from datetime import datetime, timedelta
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from jobspy import scrape_jobs

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
SENT_FILE = "sent_jobs.json"

# We add "1+ years experience" to the search term to force the platform filters
SEARCH_TERMS = [
    "data analyst 1+ years experience",
    "business analyst 1+ years experience",
    "marketing analyst 1+ years experience",
    "sql analyst 1+ years",
    "power bi analyst"
]

# Words that indicate the job is TOO senior (to keep it at 1-3 years level)
BAD_WORDS = ["senior", "lead", "manager", "director", "architect", "principal", "sr."]

# ─────────────────────────────────────────
# STATE MANAGEMENT (Avoid Duplicates)
# ─────────────────────────────────────────
if os.path.exists(SENT_FILE):
    with open(SENT_FILE, "r") as f:
        sent_jobs = set(json.load(f)) # Use a set for O(1) lookup
else:
    sent_jobs = set()

with open("resume.txt", "r", encoding="utf-8") as f:
    resume = f.read()

# ─────────────────────────────────────────
# CORE FUNCTIONS
# ─────────────────────────────────────────

def is_experience_match(description):
    """
    Checks if the description mentions 'freshers' or 'interns' 
    to ensure we are getting 1+ years as requested.
    """
    desc_lower = description.lower()
    if "fresher" in desc_lower or "internship" in desc_lower:
        return False
    return True

def send_telegram(msg):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    data = {"chat_id": CHAT_ID, "text": msg, "parse_mode": "HTML"}
    try:
        requests.post(url, data=data, timeout=10)
    except Exception as e:
        print(f"Telegram error: {e}")

# ─────────────────────────────────────────
# SCRAPE ENGINE
# ─────────────────────────────────────────

all_jobs_list = []

for term in SEARCH_TERMS:
    print(f"🔍 Scraping: {term}")
    try:
        # Note: LinkedIn is strict. If running on GitHub Actions, 
        # it might return 0 results without a proxy.
        jobs = scrape_jobs(
            site_name=["linkedin", "indeed", "naukri"],
            search_term=term,
            location="India",
            results_wanted=20,
            hours_old=48,
            country_indeed="india", # Critical for Indeed India
            description_format="markdown",
            verbose=0
        )
        if jobs is not None and not jobs.empty:
            all_jobs_list.append(jobs)
    except Exception as e:
        print(f"Error scraping {term}: {e}")

# ─────────────────────────────────────────
# PROCESSING & FILTERING
# ─────────────────────────────────────────

if not all_jobs_list:
    print("No new jobs found.")
    exit()

df = pd.concat(all_jobs_list, ignore_index=True)
df = df.drop_duplicates(subset=["job_url"])

new_count = 0
for _, row in df.iterrows():
    link = str(row.get("job_url", ""))
    title = str(row.get("title", ""))
    desc = str(row.get("description", ""))
    
    # 1. Skip if already sent
    if link in sent_jobs:
        continue
    
    # 2. Filter out Senior roles (Bad Words)
    if any(bad in title.lower() for bad in BAD_WORDS):
        continue

    # 3. Quick Exp check (1+ years)
    if not is_experience_match(desc):
        continue

    # --- MATCHING LOGIC (Simplified for speed) ---
    # (Your existing Tfidf/ATS logic here)
    
    # --- SENDING ---
    platform = str(row.get("site", "Job Board")).upper()
    company = str(row.get("company", "N/A"))
    
    msg = (
        f"🚀 <b>New Job on {platform}</b>\n"
        f"💼 <b>{title}</b>\n"
        f"🏢 {company}\n\n"
        f"🔗 <a href='{link}'>Apply Here</a>"
    )
    
    send_telegram(msg)
    sent_jobs.add(link)
    new_count += 1

# Save state
with open(SENT_FILE, "w") as f:
    json.dump(list(sent_jobs), f)

print(f"Done! Sent {new_count} new links.")
