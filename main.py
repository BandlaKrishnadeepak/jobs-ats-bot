import os
import json
import requests
from datetime import datetime, timedelta
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from jobspy import scrape_jobs

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────

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
    "marketing analyst",
]

BAD_WORDS = [
    "senior", "lead", "manager", "director", "architect",
    "5 years", "7 years", "10 years",
]

IMPORTANT_KEYWORDS = [
    "sql", "python", "power bi", "tableau", "bigquery",
    "excel", "etl", "dashboard", "analytics",
    "kpi", "reporting", "data modeling",
    "forecasting", "statistics", "machine learning",
    "google ads", "meta ads", "looker studio",
    "cloud scheduler", "crm", "marketing analytics",
]

MIN_ATS_SCORE = 30   # Only send jobs with ATS score >= this
HOURS_OLD = 48       # Only fetch jobs posted in the last 48 hours (2 days)

# ─────────────────────────────────────────
# LOAD STATE
# ─────────────────────────────────────────

if os.path.exists(SENT_FILE):
    with open(SENT_FILE, "r") as f:
        sent_jobs = json.load(f)
else:
    sent_jobs = []

with open("resume.txt", "r", encoding="utf-8") as f:
    resume = f.read()

# ─────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────

def ats_score(resume_text, jd):
    if not jd or not jd.strip():
        return 0
    vectorizer = TfidfVectorizer(stop_words="english")
    vectors = vectorizer.fit_transform([resume_text, jd])
    score = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
    return int(score * 100)


def analyze_match(resume_text, jd):
    matched = []
    missing = []
    jd_lower = jd.lower() if jd else ""
    resume_lower = resume_text.lower()
    for word in IMPORTANT_KEYWORDS:
        if word in jd_lower:
            if word in resume_lower:
                matched.append(word)
            else:
                missing.append(word)
    return matched[:5], missing[:5]


def valid_title(title):
    if not title:
        return False
    return not any(bad in title.lower() for bad in BAD_WORDS)


def send_telegram(msg):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    data = {"chat_id": CHAT_ID, "text": msg, "parse_mode": "HTML"}
    try:
        requests.post(url, data=data, timeout=10)
    except Exception as e:
        print(f"Telegram error: {e}")


# ─────────────────────────────────────────
# SCRAPE JOBS USING JOBSPY
# ─────────────────────────────────────────

all_jobs = []
cutoff_date = datetime.utcnow() - timedelta(hours=HOURS_OLD)

for term in SEARCH_TERMS:
    print(f"\n🔍 Searching: {term}")
    try:
        # ── India sites: Naukri + Indeed India + LinkedIn ──
        india_jobs = scrape_jobs(
            site_name=["naukri", "linkedin", "indeed"],
            search_term=term,
            location="India",
            results_wanted=15,
            hours_old=HOURS_OLD,
            country_indeed="India",
            description_format="markdown",
            verbose=0,
        )
        if india_jobs is not None and not india_jobs.empty:
            all_jobs.append(india_jobs)
            print(f"  India: {len(india_jobs)} results")

    except Exception as e:
        print(f"  India scrape error for '{term}': {e}")

    try:
        # ── Remote jobs: Indeed + LinkedIn remote ──
        remote_jobs = scrape_jobs(
            site_name=["indeed", "linkedin"],
            search_term=f"{term} remote",
            location="Remote",
            results_wanted=10,
            hours_old=HOURS_OLD,
            is_remote=True,
            description_format="markdown",
            verbose=0,
        )
        if remote_jobs is not None and not remote_jobs.empty:
            all_jobs.append(remote_jobs)
            print(f"  Remote: {len(remote_jobs)} results")

    except Exception as e:
        print(f"  Remote scrape error for '{term}': {e}")

# ── RemoteOK (free public API) ──
try:
    import pandas as pd
    headers = {"User-Agent": "Mozilla/5.0"}
    res = requests.get("https://remoteok.com/api", headers=headers, timeout=15)
    remoteok_data = res.json()[1:]
    remoteok_rows = []
    for job in remoteok_data:
        title = job.get("position", "")
        if not any(term in title.lower() for term in SEARCH_TERMS):
            continue
        date_str = job.get("date", "")
        try:
            posted_date = datetime.fromisoformat(
                date_str.replace("Z", "+00:00")
            ).replace(tzinfo=None)
        except Exception:
            continue
        if posted_date < cutoff_date:
            continue
        remoteok_rows.append({
            "title": title,
            "company": job.get("company", ""),
            "location": "Remote",
            "job_url": job.get("url", ""),
            "description": job.get("description", ""),
            "site": "remoteok",
            "date_posted": posted_date,
        })
    if remoteok_rows:
        all_jobs.append(pd.DataFrame(remoteok_rows))
        print(f"\nRemoteOK: {len(remoteok_rows)} results")
except Exception as e:
    print(f"RemoteOK error: {e}")

# ── Remotive (free public API) ──
try:
    import pandas as pd
    headers = {"User-Agent": "Mozilla/5.0"}
    res = requests.get("https://remotive.com/api/remote-jobs", headers=headers, timeout=15)
    remotive_data = res.json().get("jobs", [])
    remotive_rows = []
    for job in remotive_data:
        title = job.get("title", "")
        if not any(term in title.lower() for term in SEARCH_TERMS):
            continue
        date_str = job.get("publication_date", "")
        try:
            posted_date = datetime.fromisoformat(
                date_str.replace("Z", "+00:00")
            ).replace(tzinfo=None)
        except Exception:
            continue
        if posted_date < cutoff_date:
            continue
        remotive_rows.append({
            "title": title,
            "company": job.get("company_name", ""),
            "location": "Remote",
            "job_url": job.get("url", ""),
            "description": job.get("description", ""),
            "site": "remotive",
            "date_posted": posted_date,
        })
    if remotive_rows:
        all_jobs.append(pd.DataFrame(remotive_rows))
        print(f"Remotive: {len(remotive_rows)} results")
except Exception as e:
    print(f"Remotive error: {e}")

# ─────────────────────────────────────────
# DEDUPLICATE & FILTER
# ─────────────────────────────────────────

import pandas as pd

if not all_jobs:
    print("No jobs found across all platforms.")
    exit(0)

df = pd.concat(all_jobs, ignore_index=True)

# Drop duplicate URLs
df = df.drop_duplicates(subset=["job_url"])

# Enforce 48-hour cutoff
if "date_posted" in df.columns:
    df["date_posted"] = pd.to_datetime(df["date_posted"], errors="coerce", utc=True)
    df = df[
        df["date_posted"].isna() |
        (df["date_posted"] >= pd.Timestamp(cutoff_date, tz="UTC"))
    ]

# Filter bad titles
df = df[df["title"].apply(valid_title)]

print(f"\n✅ {len(df)} unique recent jobs after filtering\n")

# ─────────────────────────────────────────
# SCORE & SEND
# ─────────────────────────────────────────

new_count = 0

for _, row in df.iterrows():
    link = str(row.get("job_url", "")).strip()
    title = str(row.get("title", "")).strip()
    company = str(row.get("company", "")).strip()
    location = str(row.get("location", "")).strip()
    desc = str(row.get("description", "")).strip()
    platform = str(row.get("site", "Unknown")).strip().title()

    if not link or not link.startswith("http"):
        continue
    if link in sent_jobs:
        continue

    score = ats_score(resume, desc)
    matched, missing = analyze_match(resume, desc)

    reason = f"Matched: {', '.join(matched)}" if matched else "Few direct keyword matches"
    improve = f"Add to resume: {', '.join(missing)}" if missing else "Resume already covers major keywords"

    date_info = ""
    if "date_posted" in row and pd.notna(row["date_posted"]):
        date_info = f"\n📅 Posted: {row['date_posted'].strftime('%d %b %Y')}"

    if score >= MIN_ATS_SCORE:
        msg = (
            f"🔥 <b>{platform}</b> — ATS Match: <b>{score}%</b>\n"
            f"💼 <b>{title}</b>\n"
            f"🏢 {company}  |  📍 {location}"
            f"{date_info}\n\n"
            f"✅ {reason}\n"
            f"📝 {improve}\n\n"
            f"🔗 <a href='{link}'>Apply Now</a>"
        )
        send_telegram(msg[:4096])
        sent_jobs.append(link)
        new_count += 1
        print(f"  Sent: [{score}%] {title} @ {company}")

# ─────────────────────────────────────────
# SAVE STATE
# ─────────────────────────────────────────

with open(SENT_FILE, "w") as f:
    json.dump(sent_jobs, f)

print(f"\n📨 Done. Sent {new_count} new job alerts.")
