import requests
import os
import json
import re
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

if not BOT_TOKEN or not CHAT_ID:
    raise ValueError("Missing BOT_TOKEN or CHAT_ID in GitHub Secrets")

SENT_FILE = "sent_jobs.json"

SEARCH_TERMS = [
    "data analyst",
    "business analyst",
    "sql analyst",
    "power bi analyst",
    "reporting analyst",
    "bi analyst",
    "marketing analyst"
]

BAD_WORDS = [
    "senior", "lead", "manager", "director", "architect", "5 years", "7 years"
]

IMPORTANT_KEYWORDS = [
    "sql", "python", "power bi", "tableau", "bigquery",
    "excel", "etl", "dashboard", "analytics",
    "kpi", "reporting", "data modeling",
    "forecasting", "statistics", "machine learning",
    "google ads", "meta ads", "looker studio",
    "cloud scheduler", "crm", "marketing analytics"
]

if os.path.exists(SENT_FILE):
    with open(SENT_FILE, "r") as f:
        sent_jobs = json.load(f)
else:
    sent_jobs = []

with open("resume.txt", "r", encoding="utf-8") as f:
    resume = f.read()

headers = {
    "User-Agent": "Mozilla/5.0"
}

jobs = []
cutoff_date = datetime.utcnow() - timedelta(days=2)


def is_recent(posted_text):
    posted_text = posted_text.lower()

    if "today" in posted_text or "just" in posted_text:
        return True

    match = re.search(r'(\d+)', posted_text)

    if match:
        num = int(match.group(1))

        if "hour" in posted_text:
            return num <= 48

        if "day" in posted_text:
            return num <= 2

    return False


def ats_score(resume, jd):
    vectorizer = TfidfVectorizer(stop_words="english")
    vectors = vectorizer.fit_transform([resume, jd])
    score = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
    return int(score * 100)


def analyze_match(resume, jd):
    matched = []
    missing = []

    for word in IMPORTANT_KEYWORDS:
        if word in jd.lower():
            if word in resume.lower():
                matched.append(word)
            else:
                missing.append(word)

    return matched[:5], missing[:5]


def send_telegram(msg):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    data = {"chat_id": CHAT_ID, "text": msg}
    requests.post(url, data=data)


def valid_title(title):
    return not any(bad in title.lower() for bad in BAD_WORDS)


# ---------- RemoteOK ----------

try:
    res = requests.get("https://remoteok.com/api", headers=headers)
    data = res.json()[1:]

    for job in data:
        title = job.get("position", "")
        desc = job.get("description", "")
        link = job.get("url", "")
        date_str = job.get("date", "")

        try:
            posted_date = datetime.fromisoformat(date_str.replace("Z", "+00:00")).replace(tzinfo=None)
        except:
            continue

        if posted_date >= cutoff_date and valid_title(title):
            if any(term in title.lower() for term in SEARCH_TERMS):
                jobs.append(("RemoteOK", title, desc, link))

except:
    pass

# ---------- Remotive ----------

try:
    res = requests.get("https://remotive.io/api/remote-jobs", headers=headers)
    data = res.json()["jobs"]

    for job in data:
        title = job.get("title", "")
        desc = job.get("description", "")
        link = job.get("url", "")
        date_str = job.get("publication_date", "")

        try:
            posted_date = datetime.fromisoformat(date_str.replace("Z", "+00:00")).replace(tzinfo=None)
        except:
            continue

        if posted_date >= cutoff_date and valid_title(title):
            if any(term in title.lower() for term in SEARCH_TERMS):
                jobs.append(("Remotive", title, desc, link))

except:
    pass

# ---------- Generic scraper helper ----------


def scrape_platform(platform, base_url, selector, title_selector=None):
    for term in SEARCH_TERMS:
        try:
            url = base_url.format(term.replace(" ", "+"), term.replace(" ", "-"), term.replace(" ", "%20"))
            page = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(page.text, "html.parser")

            for card in soup.select(selector)[:5]:
                title_tag = card.select_one(title_selector) if title_selector else card.select_one("a")
                title = title_tag.get_text(strip=True) if title_tag else ""
                link = title_tag.get("href") if title_tag else url

                if link and link.startswith("/"):
                    if "indeed" in url:
                        link = "https://in.indeed.com" + link

                if valid_title(title):
                    jobs.append((platform, title, title, link if link else url))

        except:
            pass


# ---------- LinkedIn ----------

scrape_platform(
    "LinkedIn",
    "https://www.linkedin.com/jobs/search/?keywords={2}",
    ".base-search-card",
    ".base-search-card__title"
)

# ---------- Indeed ----------

scrape_platform(
    "Indeed",
    "https://in.indeed.com/jobs?q={0}",
    ".job_seen_beacon",
    "h2 a"
)

# ---------- Naukri ----------

scrape_platform(
    "Naukri",
    "https://www.naukri.com/{1}-jobs",
    "article",
    "a"
)

# ---------- Foundit ----------

scrape_platform(
    "Foundit",
    "https://www.foundit.in/srp/results?query={0}",
    "section",
    "a"
)

# ---------- Glassdoor ----------

scrape_platform(
    "Glassdoor",
    "https://www.glassdoor.co.in/Job/jobs.htm?sc.keyword={0}",
    "li",
    "a"
)

# ---------- Wellfound ----------

scrape_platform(
    "Wellfound",
    "https://wellfound.com/jobs?query={0}",
    "div",
    "a"
)

# ---------- Final send ----------

for platform, title, desc, link in jobs:
    score = ats_score(resume, desc)
    matched, missing = analyze_match(resume, desc)

    reason = f"Matched: {', '.join(matched)}" if matched else "Few direct keyword matches"
    improve = f"Add: {', '.join(missing)}" if missing else "Resume already covers major keywords"

    if link not in sent_jobs and score >= 30:
        msg = f"""🔥 {platform} Job Match ({score}%)

Title: {title}

Why this score:
{reason}

Improve score:
{improve}

Apply:
{link}
"""
        send_telegram(msg[:3500])
        sent_jobs.append(link)

        with open(SENT_FILE, "w") as f:
            json.dump(sent_jobs, f)
