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


# ---------- LinkedIn ----------

for term in SEARCH_TERMS:
    try:
        url = f"https://www.linkedin.com/jobs/search/?keywords={term.replace(' ', '%20')}"
        page = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(page.text, "html.parser")

        for card in soup.select(".base-search-card")[:5]:
            title_tag = card.select_one(".base-search-card__title")
            link_tag = card.select_one("a.base-card__full-link")

            title = title_tag.get_text(strip=True) if title_tag else ""
            link = link_tag["href"] if link_tag else ""

            if valid_title(title) and link:
                jobs.append(("LinkedIn", title, title, link))

    except:
        pass


# ---------- Indeed ----------

for term in SEARCH_TERMS:
    try:
        url = f"https://in.indeed.com/jobs?q={term.replace(' ', '+')}"
        page = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(page.text, "html.parser")

        for card in soup.select(".job_seen_beacon")[:5]:
            title_tag = card.select_one("h2 a")

            title = title_tag.get_text(strip=True) if title_tag else ""
            link = title_tag.get("href") if title_tag else ""

            if link.startswith("/"):
                link = "https://in.indeed.com" + link

            if valid_title(title) and link:
                jobs.append(("Indeed", title, title, link))

    except:
        pass


# ---------- Naukri ----------

for term in SEARCH_TERMS:
    try:
        url = f"https://www.naukri.com/{term.replace(' ', '-')}-jobs"
        page = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(page.text, "html.parser")

        for card in soup.select("article")[:5]:
            title_tag = card.select_one("a.title")

            title = title_tag.get_text(strip=True) if title_tag else ""
            link = title_tag.get("href") if title_tag else ""

            if valid_title(title) and link:
                jobs.append(("Naukri", title, title, link))

    except:
        pass


# ---------- Final send ----------

for platform, title, desc, link in jobs:
    if not link.startswith("http"):
        continue

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
