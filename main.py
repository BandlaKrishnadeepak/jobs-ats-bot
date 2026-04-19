import os
import json
import hashlib
import requests
import pandas as pd
from datetime import datetime, timezone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from jobspy import scrape_jobs

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
BOT_TOKEN  = os.getenv("BOT_TOKEN")
CHAT_ID    = os.getenv("CHAT_ID")
SENT_FILE  = "sent_jobs.json"

# Minimum resume-match score to send a job (0.0 – 1.0)
MIN_MATCH_SCORE = 0.15

# Only send jobs posted within this many hours
MAX_HOURS_OLD = 24

# How many top matches to send per run (avoid spam)
MAX_JOBS_PER_RUN = 10

# Core search terms — keep them tight; filtering does the heavy lifting
SEARCH_TERMS = [
    "data analyst",
    "business analyst",
    "marketing analyst",
    "sql analyst",
    "power bi analyst",
]

# ── Roles you DO want (title must contain at least one) ──────────────────────
GOOD_TITLE_KEYWORDS = [
    "data analyst", "business analyst", "marketing analyst",
    "sql analyst", "power bi", "bi analyst", "analytics analyst",
    "reporting analyst", "insights analyst",
]

# ── Words that make a job too senior / wrong role ────────────────────────────
BAD_TITLE_WORDS = [
    "senior", "sr.", "lead", "manager", "director", "head of",
    "architect", "principal", "vp ", "vice president", "chief",
    "intern", "internship", "fresher", "trainee", "graduate trainee",
    "lecturer", "professor", "consultant",          # unrelated roles
]

# ── Description red-flags (too much experience demanded) ─────────────────────
BAD_DESC_PATTERNS = [
    "8+ years", "9+ years", "10+ years", "12+ years",
    "minimum 8 years", "minimum 10 years",
    "15 years", "20 years",
]

# ─────────────────────────────────────────
# STATE MANAGEMENT
# ─────────────────────────────────────────
if os.path.exists(SENT_FILE):
    with open(SENT_FILE, "r") as f:
        sent_jobs: set = set(json.load(f))
else:
    sent_jobs: set = set()

with open("resume.txt", "r", encoding="utf-8") as f:
    resume_text = f.read()

# ─────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────

def make_fingerprint(title: str, company: str) -> str:
    """
    Deduplicate by (normalised title + company) so the same job posted
    on LinkedIn AND Indeed is not sent twice.
    """
    raw = f"{title.lower().strip()}|{company.lower().strip()}"
    return hashlib.md5(raw.encode()).hexdigest()


def is_recent(date_posted, max_hours: int = MAX_HOURS_OLD) -> bool:
    """Return True if job was posted within max_hours."""
    if pd.isna(date_posted) or date_posted is None:
        return False  # drop if we have no date at all
    if isinstance(date_posted, str):
        try:
            date_posted = pd.to_datetime(date_posted, utc=True)
        except Exception:
            return False
    if date_posted.tzinfo is None:
        date_posted = date_posted.replace(tzinfo=timezone.utc)
    age = datetime.now(timezone.utc) - date_posted
    return age.total_seconds() / 3600 <= max_hours


def title_is_relevant(title: str) -> bool:
    """Title must match a wanted role AND not contain a bad word."""
    t = title.lower()
    has_good = any(kw in t for kw in GOOD_TITLE_KEYWORDS)
    has_bad  = any(bw in t for bw in BAD_TITLE_WORDS)
    return has_good and not has_bad


def desc_is_ok(description: str) -> bool:
    """Description must not mention excessive years of experience."""
    d = description.lower()
    return not any(pat in d for pat in BAD_DESC_PATTERNS)


def resume_match_score(job_text: str) -> float:
    """TF-IDF cosine similarity between resume and job description+title."""
    if not job_text.strip():
        return 0.0
    try:
        vectorizer = TfidfVectorizer(stop_words="english")
        tfidf = vectorizer.fit_transform([resume_text, job_text])
        score = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
        return round(float(score), 3)
    except Exception:
        return 0.0


def send_telegram(msg: str) -> None:
    url  = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    data = {"chat_id": CHAT_ID, "text": msg, "parse_mode": "HTML",
            "disable_web_page_preview": True}
    try:
        r = requests.post(url, data=data, timeout=10)
        r.raise_for_status()
    except Exception as e:
        print(f"  ⚠️  Telegram error: {e}")


def format_posted_time(date_posted) -> str:
    """Human-readable 'X hours ago' string."""
    if pd.isna(date_posted) or date_posted is None:
        return "unknown"
    if isinstance(date_posted, str):
        try:
            date_posted = pd.to_datetime(date_posted, utc=True)
        except Exception:
            return "unknown"
    if date_posted.tzinfo is None:
        date_posted = date_posted.replace(tzinfo=timezone.utc)
    delta = datetime.now(timezone.utc) - date_posted
    hours = int(delta.total_seconds() // 3600)
    if hours < 1:
        return "just now"
    if hours == 1:
        return "1 hour ago"
    return f"{hours} hours ago"


# ─────────────────────────────────────────
# SCRAPE
# ─────────────────────────────────────────
all_jobs_list = []

for term in SEARCH_TERMS:
    print(f"🔍  Scraping: {term}")
    try:
        jobs = scrape_jobs(
            site_name=["linkedin", "indeed", "naukri"],
            search_term=term,
            location="India",
            results_wanted=30,          # fetch more so filters have room
            hours_old=MAX_HOURS_OLD,    # platform-side pre-filter
            country_indeed="india",
            description_format="markdown",
            verbose=0,
        )
        if jobs is not None and not jobs.empty:
            all_jobs_list.append(jobs)
            print(f"    ✅  {len(jobs)} results")
        else:
            print(f"    ⚠️  0 results")
    except Exception as e:
        print(f"    ❌  Error: {e}")

# ─────────────────────────────────────────
# BUILD & CLEAN DATAFRAME
# ─────────────────────────────────────────
if not all_jobs_list:
    print("No jobs fetched at all. Exiting.")
    exit()

df = pd.concat(all_jobs_list, ignore_index=True)

# Normalise date column
if "date_posted" in df.columns:
    df["date_posted"] = pd.to_datetime(df["date_posted"], utc=True, errors="coerce")

# Drop URL duplicates first (same job, same platform)
df = df.drop_duplicates(subset=["job_url"])

print(f"\n📦  Total raw jobs fetched: {len(df)}")

# ─────────────────────────────────────────
# FILTER PIPELINE
# ─────────────────────────────────────────
candidates = []

for _, row in df.iterrows():
    link    = str(row.get("job_url", "")).strip()
    title   = str(row.get("title", "")).strip()
    company = str(row.get("company", "N/A")).strip()
    desc    = str(row.get("description", "")).strip()
    posted  = row.get("date_posted", None)

    # ── 1. Recency check ────────────────────────────────────────────────────
    if not is_recent(posted):
        continue

    # ── 2. Cross-platform dedupe via fingerprint ────────────────────────────
    fp = make_fingerprint(title, company)
    if link in sent_jobs or fp in sent_jobs:
        continue

    # ── 3. Title relevance (role match + seniority filter) ──────────────────
    if not title_is_relevant(title):
        continue

    # ── 4. Description sanity (no crazy exp requirements) ───────────────────
    if not desc_is_ok(desc):
        continue

    # ── 5. Resume match score ────────────────────────────────────────────────
    combined_text = f"{title} {company} {desc}"
    score = resume_match_score(combined_text)
    if score < MIN_MATCH_SCORE:
        continue

    candidates.append({
        "link":    link,
        "fp":      fp,
        "title":   title,
        "company": company,
        "posted":  posted,
        "score":   score,
        "site":    str(row.get("site", "Job Board")).upper(),
        "location": str(row.get("location", "India")),
    })

print(f"✅  Jobs passing all filters: {len(candidates)}")

# ─────────────────────────────────────────
# SORT: most recent first, then by score
# ─────────────────────────────────────────
candidates.sort(
    key=lambda x: (
        x["posted"].timestamp() if x["posted"] and not pd.isna(x["posted"]) else 0,
        x["score"],
    ),
    reverse=True,
)

# ─────────────────────────────────────────
# SEND TOP MATCHES
# ─────────────────────────────────────────
new_count = 0

for job in candidates[:MAX_JOBS_PER_RUN]:
    score_pct  = int(job["score"] * 100)
    posted_str = format_posted_time(job["posted"])

    # Build score bar (visual)
    filled = score_pct // 10
    bar    = "█" * filled + "░" * (10 - filled)

    msg = (
        f"🚀 <b>New Job — {job['site']}</b>  •  <i>{posted_str}</i>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"💼 <b>{job['title']}</b>\n"
        f"🏢 {job['company']}\n"
        f"📍 {job['location']}\n\n"
        f"🎯 Resume Match: {score_pct}%\n"
        f"<code>[{bar}]</code>\n\n"
        f"🔗 <a href='{job['link']}'>Apply Now →</a>"
    )

    send_telegram(msg)
    sent_jobs.add(job["link"])
    sent_jobs.add(job["fp"])          # also store fingerprint
    new_count += 1
    print(f"  📨  Sent: {job['title']} @ {job['company']}  [{score_pct}%]")

# ─────────────────────────────────────────
# PERSIST STATE
# ─────────────────────────────────────────
with open(SENT_FILE, "w") as f:
    json.dump(list(sent_jobs), f, indent=2)

print(f"\n🏁  Done! Sent {new_count} new job(s).")
