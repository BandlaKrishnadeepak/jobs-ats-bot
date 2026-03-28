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
    "junior data analyst",
    "associate data analyst",
    "business analyst",
    "reporting analyst",
    "sql analyst",
    "power bi analyst",
    "tableau developer",
    "excel analyst",
    "data insights analyst",
    "bi analyst",
    "research analyst"
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


def missing_keywords(resume, jd):
    important = [
        "sql", "python", "power bi", "tableau", "bigquery",
        "excel", "etl", "dashboard", "analytics",
        "kpi", "reporting", "data modeling",
        "forecasting", "statistics", "machine learning",
        "google ads", "meta ads", "looker studio",
        "cloud scheduler", "crm", "marketing analytics"
    ]

    missing = []

    for word in important:
        if word in jd.lower() and word not in resume.lower():
            missing.append(word)

    return missing[:5]


def send_telegram(msg):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    data = {
        "chat_id": CHAT_ID,
        "text": msg
    }
    requests.post(url, data=data)


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

        if posted_date >= cutoff_date:
            if any(term in title.lower() for term in SEARCH_TERMS):
                jobs.append({
                    "title": title,
                    "desc": desc,
                    "link": link
                })

except Exception:
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

        if posted_date >= cutoff_date:
            if any(term in title.lower() for term in SEARCH_TERMS):
                jobs.append({
                    "title": title,
                    "desc": desc,
                    "link": link
                })

except Exception:
    pass

# ---------- LinkedIn ----------

for term in SEARCH_TERMS:
    try:
        url = f"https://www.linkedin.com/jobs/search/?keywords={term.replace(' ', '%20')}"
        page = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(page.text, "html.parser")

        for card in soup.select(".base-search-card")[:5]:
            title_tag = card.select_one(".base-search-card__title")
            link_tag = card.select_one("a")
            date_tag = card.select_one("time")

            title = title_tag.get_text(strip=True) if title_tag else ""
            link = link_tag["href"] if link_tag else url
            posted = date_tag.get_text(strip=True) if date_tag else ""

            if is_recent(posted):
                jobs.append({
                    "title": title,
                    "desc": title,
                    "link": link
                })

    except Exception:
        pass

# ---------- Indeed ----------

for term in SEARCH_TERMS:
    try:
        url = f"https://in.indeed.com/jobs?q={term.replace(' ', '+')}"
        page = requests.get(url, headers=headers)
        soup = BeautifulSoup(page.text, "html.parser")

        for card in soup.select(".job_seen_beacon")[:5]:
            title = card.get_text(strip=True)
            date_tag = card.select_one(".date")
            posted = date_tag.get_text(strip=True) if date_tag else ""

            if is_recent(posted):
                jobs.append({
                    "title": title,
                    "desc": title,
                    "link": url
                })

    except Exception:
        pass

# ---------- Naukri ----------

for term in SEARCH_TERMS:
    try:
        url = f"https://www.naukri.com/{term.replace(' ', '-')}-jobs"
        page = requests.get(url, headers=headers)
        soup = BeautifulSoup(page.text, "html.parser")

        for card in soup.select("article")[:5]:
            title = card.get_text(strip=True)
            date_tag = card.select_one(".jobTupleFooter")
            posted = date_tag.get_text(strip=True) if date_tag else ""

            if is_recent(posted):
                jobs.append({
                    "title": title,
                    "desc": title,
                    "link": url
                })

    except Exception:
        pass

# ---------- Process Jobs and Send Telegram Notifications ----------

for job in jobs:
    score = ats_score(resume, job["desc"])
    missing = missing_keywords(resume, job["desc"])

    print(job["title"], score)

    if job["link"] not in sent_jobs and score >= 30:
        msg = f"""🔥 Job Match ({score}%)

Title: {job['title']}

Missing Keywords:
{', '.join(missing) if missing else 'None'}

Apply:
{job['link']}
"""
        send_telegram(msg[:3500])
        sent_jobs.append(job["link"])

        with open(SENT_FILE, "w") as f:
            json.dump(sent_jobs, f)
