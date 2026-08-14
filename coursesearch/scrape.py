"""Scrape UBC course descriptions into a dataframe."""
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup

ROOT = "https://vancouver.calendar.ubc.ca/course-descriptions/institution/120"
HEADERS = {"User-Agent": "Mozilla/5.0"}


def soup(url):
    return BeautifulSoup(requests.get(url, headers=HEADERS, timeout=30).text, "html.parser")


def subject_links():
    return [urljoin(ROOT, a["href"]) for a in soup(ROOT).find("ol", class_="list-buttons").find_all("a")]


def scrape_subject(url):
    listing = soup(url).find("ol", class_="list-none")
    rows = []
    for li in listing.find_all("li") if listing else []:
        name, desc = li.find("strong"), li.find("p", class_="mt-0")
        if not (name and desc):
            continue
        code = li.find("h3", class_="text-lg").get_text().replace(name.get_text(), "").split("(")[0].strip()
        rows.append({"code": code, "name": name.get_text(strip=True), "description": desc.get_text(strip=True)})
    return rows


def scrape_all():
    rows = []
    for link in subject_links():
        rows += scrape_subject(link)
        print(f"  {len(rows)} courses…", end="\r", flush=True)
    df = pd.DataFrame(rows).drop_duplicates("code")
    df["description"] = df["description"].str.replace(r"\s+", " ", regex=True).str.strip()
    return df[df["description"].str.len() > 25].reset_index(drop=True)
