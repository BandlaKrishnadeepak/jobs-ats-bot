import os
import json
import hashlib
import requests
import pandas as pd
from datetime import datetime, timezone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from jobspy import scrape_jobs

# ═══════════════════════════════════════════════════════════
#  CONFIG  — tuned for 1-year Data Analyst, India
# ═══════════════════════════════════════════════════════════

BOT_TOKEN  = os.getenv("BOT_TOKEN")
CHAT_ID    = os.getenv("CHAT_ID")
SENT_FILE  = "sent_jobs.json"

# Lower threshold = more jobs get through.
# 0.08 is the sweet spot for a short resume against full JDs.
MIN_MATCH_SCORE = 0.08

# Allow jobs up to 48 h old so you don't miss anything.
# Reduce to 24 once you're getting consistent results.
MAX_HOURS_OLD = 48

# Cap per run to avoid Telegram spam
MAX_JOBS_PER_RUN = 15

# ── Search terms (broad — filtering handles precision) ─────────────────────
SEARCH_TERMS = [
    "data analyst",
    "junior data analyst",
    "sql analyst",
    "power bi analyst",
    "business analyst",
    "reporting analyst",
    "marketing analyst",
    "mis analyst",
    "data analyst python sql",
    "bi analyst looker",
    "analytics executive",
]

# ── Titles to ACCEPT  (at least one must appear in the job title) ──────────
GOOD_TITLE_KEYWORDS = [
    # Standard analyst titles
    "data analyst", "junior data analyst", "jr data analyst",
    "associate data analyst", "trainee data analyst",
    "sql analyst", "power bi analyst", "bi analyst",
    "business analyst", "reporting analyst", "marketing analyst",
    "analytics analyst", "insights analyst", "growth analyst",
    # Indian market variants
    "mis analyst", "mis executive", "mis associate",
    "data executive", "analytics executive", "reporting executive",
    "business intelligence analyst", "bi developer",
    "data specialist", "analytics specialist",
    # Tools-first titles common on Naukri
    "power bi", "looker studio", "bigquery analyst",
    "python analyst", "etl analyst",
]

# ── Titles to REJECT  (any match = skip) ──────────────────────────────────
BAD_TITLE_WORDS = [
    "senior", "sr.", "sr ", "lead", "principal", "staff",
    "manager", "head of", "director", "vp ", "vice president",
    "chief", "architect", "consultant",
    "data engineer", "data scientist",
    "software engineer", "developer", "devops",
    "intern", "internship", "fresher", "trainee",
    "lecturer", "professor", "teacher",
]

# ── Description red-flags: too much experience demanded ───────────────────
BAD_DESC_EXPERIENCE = [
    "7+ years", "8+ years", "9+ years", "10+ years",
    "12+ years", "15+ years", "20+ years",
    "minimum 7 years", "minimum 8 years", "minimum 10 years",
    "7 years of experience", "8 years of experience",
    "10 years of experience",
]

# ── Your exact skills — used to boost match score ─────────────────────────
YOUR_SKILLS = [
    "sql", "mysql", "bigquery", "python", "pandas", "numpy",
    "power bi", "dax", "power query", "looker studio",
    "excel", "google sheets", "etl", "data modeling",
    "reporting automation", "dashboarding", "data cleaning",
    "api integration", "google cloud", "matplotlib",
    "marketing analytics", "campaign analysis", "kpi",
    "ctr", "cpc", "roi", "lead funnel",
]

# ═══════════════════════════════════════════════════════════
#  STATE
# ═══════════════════════════════════════════════════════════

if os.path.exists(SENT_FILE):
    with open(SENT_FILE, "r") as f:
        sent_jobs: set = set(json.load(f))
else:
    sent_jobs: set = set()

with open("resume.txt", "r", encoding="utf-8") as f:
    resume_text = f.read()

# ═══════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════

def make_fingerprint(title: str, company: str) -> str:
    raw = f"{title.lower().strip()}|{company.lower().strip()}"
    return hashlib.md5(raw.encode()).hexdigest()


def is_recent(date_posted, max_hours: int = MAX_HOURS_OLD) -> bool:
    # Null dates: pass through — many Indian boards don't include them
    if pd.isna(date_posted) or date_posted is None:
        return True
    if isinstance(date_posted, str):
        try:
            date_posted = pd.to_datetime(date_posted, utc=True)
        except Exception:
            return True
    if hasattr(date_posted, "tzinfo") and date_posted.tzinfo is None:
        date_posted = date_posted.replace(tzinfo=timezone.utc)
    try:
        return (datetime.now(timezone.utc) - date_posted).total_seconds() / 3600 <= max_hours
    except Exception:
        return True


def title_is_relevant(title: str) -> bool:
    t = title.lower()
    has_good = any(kw in t for kw in GOOD_TITLE_KEYWORDS)
    has_bad  = any(bw in t for bw in BAD_TITLE_WORDS)
    return has_good and not has_bad


def desc_is_ok(description: str) -> bool:
    if not description or description in ("", "nan"):
        return True   # no description = pass through
    d = description.lower()
    return not any(pat in d for pat in BAD_DESC_EXPERIENCE)


def skills_match_count(text: str) -> int:
    t = text.lower()
    return sum(1 for skill in YOUR_SKILLS if skill in t)


def resume_match_score(job_text: str) -> float:
    if not job_text or not job_text.strip():
        return 0.0
    # Short descriptions: fall back to skill-count scoring
    if len(job_text.strip()) < 100:
        return min(skills_match_count(job_text) * 0.04, 1.0)
    try:
        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        tfidf  = vectorizer.fit_transform([resume_text, job_text])
        score  = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
        # Small bonus per matched skill (rewards jobs mentioning your tools)
        bonus  = skills_match_count(job_text) * 0.01
        return round(min(float(score) + bonus, 1.0), 3)
    except Exception:
        return 0.0


def send_telegram(msg: str) -> bool:
    if not BOT_TOKEN or not CHAT_ID:
        print("  ⚠️  BOT_TOKEN or CHAT_ID missing — set them as environment variables")
        return False
    url  = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    data = {"chat_id": CHAT_ID, "text": msg,
            "parse_mode": "HTML", "disable_web_page_preview": True}
    try:
        r = requests.post(url, data=data, timeout=10)
        r.raise_for_status()
        return True
    except Exception as e:
        print(f"  ⚠️  Telegram error: {e}")
        return False


def format_age(date_posted) -> str:
    if pd.isna(date_posted) or date_posted is None:
        return "recently posted"
    if isinstance(date_posted, str):
        try:
            date_posted = pd.to_datetime(date_posted, utc=True)
        except Exception:
            return "recently posted"
    if hasattr(date_posted, "tzinfo") and date_posted.tzinfo is None:
        date_posted = date_posted.replace(tzinfo=timezone.utc)
    try:
        hours = int((datetime.now(timezone.utc) - date_posted).total_seconds() // 3600)
        if hours < 1:   return "just now"
        if hours < 24:  return f"{hours}h ago"
        return f"{hours // 24}d ago"
    except Exception:
        return "recently posted"


def get_timestamp(date_posted) -> float:
    if pd.isna(date_posted) or date_posted is None:
        return 0.0
    if isinstance(date_posted, str):
        try:
            date_posted = pd.to_datetime(date_posted, utc=True)
        except Exception:
            return 0.0
    if hasattr(date_posted, "tzinfo") and date_posted.tzinfo is None:
        date_posted = date_posted.replace(tzinfo=timezone.utc)
    try:
        return date_posted.timestamp()
    except Exception:
        return 0.0


# ═══════════════════════════════════════════════════════════
#  SCRAPE
# ═══════════════════════════════════════════════════════════

all_jobs_list = []
site_counts   = {"linkedin": 0, "indeed": 0, "naukri": 0}

for term in SEARCH_TERMS:
    print(f"🔍  {term}")
    try:
        jobs = scrape_jobs(
            site_name=["linkedin", "indeed", "naukri"],
            search_term=term,
            location="India",
            results_wanted=25,
            hours_old=MAX_HOURS_OLD,
            country_indeed="india",
            description_format="markdown",
            verbose=0,
        )
        if jobs is not None and not jobs.empty:
            all_jobs_list.append(jobs)
            for site in site_counts:
                site_counts[site] += len(jobs[jobs["site"].str.lower() == site])
            print(f"    → {len(jobs)} results")
        else:
            print(f"    → 0 results")
    except Exception as e:
        print(f"    ❌ Error: {e}")

print(f"\n📊  Per-site totals: {site_counts}")

# ═══════════════════════════════════════════════════════════
#  BUILD DATAFRAME
# ═══════════════════════════════════════════════════════════

if not all_jobs_list:
    print("\n❌  No jobs fetched at all.")
    print("    → If LinkedIn=0: add a residential proxy (jobspy proxy= param)")
    print("    → If all=0: check your internet/network, or jobspy may be rate-limited")
    exit()

df = pd.concat(all_jobs_list, ignore_index=True)

if "date_posted" in df.columns:
    df["date_posted"] = pd.to_datetime(df["date_posted"], utc=True, errors="coerce")

df = df.drop_duplicates(subset=["job_url"])
print(f"📦  Raw jobs after URL dedupe: {len(df)}")

# ═══════════════════════════════════════════════════════════
#  FILTER + SCORE
# ═══════════════════════════════════════════════════════════

candidates   = []
drop_reasons = {"already_sent": 0, "too_old": 0, "bad_title": 0,
                "bad_desc": 0, "low_score": 0}

for _, row in df.iterrows():
    link    = str(row.get("job_url",     "")).strip()
    title   = str(row.get("title",       "")).strip()
    company = str(row.get("company",     "N/A")).strip()
    desc    = str(row.get("description", "")).strip()
    posted  = row.get("date_posted", None)
    site    = str(row.get("site",        "")).upper()
    loc     = str(row.get("location",    "India")).strip()

    fp = make_fingerprint(title, company)

    if link in sent_jobs or fp in sent_jobs:
        drop_reasons["already_sent"] += 1
        continue
    if not is_recent(posted):
        drop_reasons["too_old"] += 1
        continue
    if not title_is_relevant(title):
        drop_reasons["bad_title"] += 1
        continue
    if not desc_is_ok(desc):
        drop_reasons["bad_desc"] += 1
        continue

    combined = f"{title} {company} {desc}"
    score    = resume_match_score(combined)
    if score < MIN_MATCH_SCORE:
        drop_reasons["low_score"] += 1
        continue

    matched = [s for s in YOUR_SKILLS if s in combined.lower()]

    candidates.append({
        "link": link, "fp": fp, "title": title, "company": company,
        "posted": posted, "score": score, "site": site,
        "location": loc, "matched_skills": matched[:6],
    })

print(f"\n🔎  Filter breakdown:")
for reason, count in drop_reasons.items():
    print(f"    {reason:<15}: {count}")
print(f"✅  Passing all filters: {len(candidates)}")

# ═══════════════════════════════════════════════════════════
#  SORT: newest first, then highest score
# ═══════════════════════════════════════════════════════════

candidates.sort(
    key=lambda x: (get_timestamp(x["posted"]), x["score"]),
    reverse=True,
)

# ═══════════════════════════════════════════════════════════
#  SEND
# ═══════════════════════════════════════════════════════════

new_count = 0

for job in candidates[:MAX_JOBS_PER_RUN]:
    pct        = int(job["score"] * 100)
    bar        = "█" * min(pct // 10, 10) + "░" * max(10 - pct // 10, 0)
    skills_str = ", ".join(job["matched_skills"]) if job["matched_skills"] else "—"

    msg = (
        f"🚀 <b>{job['site']}</b>  ·  <i>{format_age(job['posted'])}</i>\n"
        f"━━━━━━━━━━━━━━━━━━━━━\n"
        f"💼 <b>{job['title']}</b>\n"
        f"🏢 {job['company']}\n"
        f"📍 {job['location']}\n\n"
        f"🎯 Match: {pct}%  <code>[{bar}]</code>\n"
        f"🛠 Skills: <i>{skills_str}</i>\n\n"
        f"🔗 <a href='{job['link']}'>Apply Now →</a>"
    )

    send_telegram(msg)
    sent_jobs.add(job["link"])
    sent_jobs.add(job["fp"])
    new_count += 1
    print(f"  📨  {job['title']} @ {job['company']}  [{pct}%]  {format_age(job['posted'])}")

# ═══════════════════════════════════════════════════════════
#  SAVE STATE
# ═══════════════════════════════════════════════════════════

with open(SENT_FILE, "w") as f:
    json.dump(list(sent_jobs), f, indent=2)

print(f"\n🏁  Done. Sent {new_count} new job(s).")

if new_count == 0 and not candidates:
    print("\n💡  Still getting 0? Try this debug sequence:")
    print("    1. Set MAX_HOURS_OLD = 168  (7 days) — rules out the date filter")
    print("    2. Set MIN_MATCH_SCORE = 0.03 — rules out the score filter")
    print("    3. Check site_counts above — if LinkedIn=0, you need a proxy")
    print("    4. Add print(df[['title','site','date_posted']].head(20)) after the concat")
    print("       to see what's actually being fetched before filtering")
