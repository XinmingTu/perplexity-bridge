"""Fetch transcripts of speech: UK Commons Hansard and the Canadian House of Commons (English-original only).

Spoken, minuted text is the most reliably human-written fresh text we can get, though speeches read from a
prepared script can still have been drafted with a model. Each document keeps a `kind` so results can be split
by how spontaneous the speech is (oral questions vs. set-piece debate).
"""

import datetime as dt
import hashlib
import json
import re
import time
from pathlib import Path

import httpx
from bs4 import BeautifulSoup

from .text import Document

UA = {"User-Agent": "compression-bridge research (non-commercial)"}
CACHE = Path("cache/http")


def _get_json(url: str, pause: float = 0.2):
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / (hashlib.sha256(url.encode()).hexdigest()[:24] + ".json")
    if path.exists():
        return json.loads(path.read_text())
    for attempt in range(4):
        try:
            r = httpx.get(url, headers=UA, timeout=60, follow_redirects=True)
            if 400 <= r.status_code < 500 and r.status_code != 429:
                print(f"  skipped (HTTP {r.status_code}): {url}")
                path.write_text("null")  # a permanent client error: remember it, don't refetch
                return None
            r.raise_for_status()
            break
        except httpx.HTTPError:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)
    path.write_text(r.text)
    time.sleep(pause)
    return r.json()


def _clean(html: str) -> str:
    text = BeautifulSoup(html, "html.parser").get_text("")
    text = re.sub(r"\[\s*\d{5,}\s*\]", "", text)  # written-question numbers like [912345]
    return re.sub(r"\s+", " ", text).strip()


def _name(attribution: str) -> str:
    """'Jane Smith (Somewhere) (Lab)' -> 'Jane Smith'; 'Madam Deputy Speaker (Ms X)' keeps the office."""
    return re.sub(r"\s*\(.*$", "", attribution or "").strip()


def _days(start: str, end: str):
    d, e = dt.date.fromisoformat(start), dt.date.fromisoformat(end)
    while d <= e:
        if d.weekday() < 5:
            yield d.isoformat()
        d += dt.timedelta(days=1)


# --- UK House of Commons -------------------------------------------------------------------------------------------

UK = "https://hansard-api.parliament.uk"
UK_KIND = {"hs_8Question": "oral_questions", "hs_8GenericHdg": "oral_questions", "hs_2DebBill": "debate",
           "hs_2DebatedMotion": "debate", "hs_2cGenericHdg": "business", "hs_2cUrgentQuestion": "urgent_question",
           "hs_2cStatement": "statement", "hs_8Petition": "petition"}


def uk_commons(start: str, end: str, min_words: int = 400) -> list[Document]:
    docs = []
    for day in _days(start, end):
        trees = _get_json(f"{UK}/overview/sectiontrees.json?date={day}&house=Commons&section=Debate")
        if not trees:
            continue
        for item in trees[0]["SectionTreeItems"]:
            kind = UK_KIND.get(item["HRSTag"])
            if not kind or not item["ExternalId"]:
                continue
            debate = _get_json(f"{UK}/debates/debate/{item['ExternalId']}.json")
            if debate is None:
                continue
            turns = []
            for it in debate["Items"]:
                if it["ItemType"] != "Contribution" or not it.get("AttributedTo"):
                    continue
                said = _clean(it["Value"])
                if kind == "oral_questions":
                    said = re.sub(r"^\d+\.\s*", "", said)  # order-paper number before a question
                if said:
                    turns.append((_name(it["AttributedTo"]), said))
            doc = Document(f"uk-{item['ExternalId'].lower()}", "uk_commons", day, item["Title"].strip(), turns,
                           dict(kind=kind, url=f"https://hansard.parliament.uk/Commons/{day}/debates/{item['ExternalId']}"))
            if sum(len(t[1].split()) for t in turns) >= min_words:
                docs.append(doc)
    return docs


# --- Canadian House of Commons (openparliament.ca) -----------------------------------------------------------------

CA = "https://api.openparliament.ca"
CA_KIND = {"Oral Questions": "oral_questions", "Government Orders": "debate", "Private Members' Business": "debate",
           "Statements by Members": "statement", "Routine Proceedings": "business", "Adjournment Proceedings": "debate"}


def _english_original(html: str) -> str:
    """Only paragraphs first spoken in English: the rest of the English record is a translation."""
    soup = BeautifulSoup(html, "html.parser")
    keep = [p for p in soup.find_all("p") if p.get("data-originallang") == "en" and "procedural" not in (p.get("class") or [])]
    return re.sub(r"\s+", " ", " ".join(p.get_text("") for p in keep)).strip()


def canada_commons(start: str, end: str, min_words: int = 400) -> list[Document]:
    docs = []
    for day in _days(start, end):
        y, m, d = day.split("-")
        url = f"{CA}/speeches/?document=/debates/{y}/{m}/{d}/&format=json&limit=500"
        speeches = []
        while url:
            page = _get_json(url)
            speeches += page["objects"]
            nxt = page["pagination"]["next_url"]
            url = CA + nxt if nxt else None
        groups: dict[tuple[str, str], list] = {}
        for s in speeches:
            if s.get("procedural"):
                continue
            h1, h2 = (s.get("h1") or {}).get("en", ""), (s.get("h2") or {}).get("en", "")
            groups.setdefault((h1, h2), []).append(s)
        for (h1, h2), group in groups.items():
            kind = CA_KIND.get(h1)
            if not kind:
                continue
            turns = []
            for s in sorted(group, key=lambda s: s["time"]):
                said = _english_original(s["content"]["en"])
                if said:
                    turns.append((_name((s.get("attribution") or {}).get("en", "")), said))
            key = hashlib.sha256(f"{day}|{h1}|{h2}".encode()).hexdigest()[:16]
            doc = Document(f"ca-{key}", "canada_commons", day, f"{h1}: {h2}".strip(": "), turns,
                           dict(kind=kind, url=f"https://openparliament.ca/debates/{y}/{m}/{d}/"))
            if sum(len(t[1].split()) for t in turns) >= min_words:
                docs.append(doc)
    return docs


SOURCES = {"uk_commons": uk_commons, "canada_commons": canada_commons}


def write_jsonl(docs: list[Document], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for d in docs:
            f.write(json.dumps(d.to_json(), ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[Document]:
    return [Document.from_json(json.loads(line)) for line in path.open()]
