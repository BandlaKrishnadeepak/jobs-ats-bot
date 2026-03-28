import requests
import os
import json
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

#---------- RemoteOK ----------

try:
res = requests.get("https://remoteok.com/api", headers=headers)
data = res.json()[1:]

for job in data:
    title = job.get("position", "")
    desc = job.get("description", "")
    link = job.get("url", "")

    if any(term in title.lower() for term in SEARCH_TERMS):
        jobs.append({
            "title": title,
            "desc": desc,
            "link": link
        })

except Exception:
pass

#---------- Remotive ----------

try:
res = requests.get("https://remotive.io/api/remote-jobs", headers=headers)
data = res.json()["jobs"]

for job in data:
    title = job.get("title", "")
    desc = job.get("description", "")
    link = job.get("url", "")

    if any(term in title.lower() for term in SEARCH_TERMS):
        jobs.append({
            "title": title,
            "desc": desc,
            "link": link
        })

except Exception:
pass

#---------- LinkedIn ----------

for term in SEARCH_TERMS:
try:
url = f"https://www.linkedin.com/jobs/search/?keywords={term.replace(' ', '%20')}"
page = requests.get(url, headers=headers, timeout=10)
soup = BeautifulSoup(page.text, "html.parser")

    for card in soup.select(".base-search-card")[:5]:
        title_tag = card.select_one(".base-search-card__title")
        link_tag = card.select_one("a")

        title = title_tag.get_text(strip=True) if title_tag else ""
        link = link_tag["href"] if link_tag else url

        jobs.append({
            "title": title,
            "desc": title,
            "link": link
        })

except Exception:
    pass
#---------- Indeed ----------

for term in SEARCH_TERMS:
try:
url = f"https://in.indeed.com/jobs?q={term.replace(' ', '+')}"
page = requests.get(url, headers=headers)
soup = BeautifulSoup(page.text, "html.parser")

    for card in soup.select(".job_seen_beacon")[:5]:
        title = card.get_text(strip=True)

        jobs.append({
            "title": title,
            "desc": title,
            "link": url
        })

except Exception:
    pass
#---------- Naukri ----------

for term in SEARCH_TERMS:
try:
url = f"https://www.naukri.com/{term.replace(' ', '-')}-jobs"
page = requests.get(url, headers=headers)
soup = BeautifulSoup(page.text, "html.parser")

    for card in soup.select("article")[:5]:
        title = card.get_text(strip=True)

        jobs.append({
            "title": title,
            "desc": title,
            "link": url
        })

except Exception:
    pass

def ats_score(resume, jd):
vectorizer = TfidfVectorizer(stop_words="english")
vectors = vectorizer.fit_transform([resume, jd])
score = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
return int(score * 100)

def send_telegram(msg):
url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
data = {"chat_id": CHAT_ID, "text": msg}
requests.post(url, data=data)

for job in jobs:
score = ats_score(resume, job["desc"])

if job["link"] not in sent_jobs and score >= 70:
    msg = f"🔥 Job Match ({score}%)\n\nTitle: {job['title']}\n\nApply: {job['link']}"
    send_telegram(msg[:3500])
    sent_jobs.append(job["link"])

with open(SENT_FILE, "w") as f:
json.dump(sent_jobs, f)
