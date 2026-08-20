"""Download EPDK electricity-market legislation (Kanun, Yonetmelik, Teblig) to data/mevzuat/.

Source: https://www.epdk.gov.tr/Detay/Icerik/23-2-3/mevzuat
The page lists documents under collapsible <h4> sections. Some leaf items link
directly to a file; others only carry a `data-id` and are populated client-side
via a POST to /Detay/GetFastAccessList, which is replicated here.
"""
import json
import re
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import requests
from bs4 import BeautifulSoup

BASE_URL = "https://www.epdk.gov.tr"
MEVZUAT_PAGE = f"{BASE_URL}/Detay/Icerik/23-2-3/mevzuat"
FAST_ACCESS_URL = f"{BASE_URL}/Detay/GetFastAccessList"

# Maps the trailing URL slug of each <h4> category link to a human-readable name.
TARGET_CATEGORIES = {
    "kanunlar": "Kanunlar",
    "yonetmelikler": "Yonetmelikler",
    "tebligler": "Tebligler",
}

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "mevzuat"
RAW_DIR = OUTPUT_DIR / "raw"
MANIFEST_PATH = OUTPUT_DIR / "manifest.json"

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; epdk-rag-mevzuat-indirici/1.0)"}
REQUEST_DELAY = 0.4  # seconds between requests, be polite to the server


def slugify(text: str, max_len: int = 100) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^\w\s-]", "", text).strip().lower()
    text = re.sub(r"[-\s]+", "-", text)
    return text[:max_len].strip("-") or "belge"


def category_slug_from_href(href: str) -> str | None:
    slug = href.rstrip("/").rsplit("/", 1)[-1]
    return slug if slug in TARGET_CATEGORIES else None


def clean_text(tag) -> str:
    return re.sub(r"\s+", " ", tag.get_text(" ", strip=True)).strip()


def fetch_category_sections(session: requests.Session) -> dict[str, list]:
    """Return {category_slug: [nested <ul> elements belonging to that section]}."""
    resp = session.get(MEVZUAT_PAGE, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    sections: dict[str, list] = {}
    for h4 in soup.find_all("h4"):
        link = h4.find("a", href=True)
        if link is None:
            continue
        slug = category_slug_from_href(link["href"])
        if slug is None or slug in sections:
            continue
        nodes = []
        for sib in h4.next_siblings:
            if getattr(sib, "name", None) == "h4":
                break
            nodes.append(sib)
        sections[slug] = nodes
    return sections


def iter_leaf_items(nodes):
    """Yield every li.accordion-pop with a data-id, across all nesting levels."""
    for node in nodes:
        if not hasattr(node, "select"):
            continue
        for li in node.select("li.accordion-pop"):
            a = li.find("a", attrs={"data-id": True}, recursive=False)
            if a is not None:
                yield li, a


def classify_version(label: str | None) -> str:
    """Normalize EPDK's free-text version labels (Son Versiyon / son versiyon / SON VERSİYON, ...)."""
    if not label:
        return "diger"
    normalized = unicodedata.normalize("NFKD", label).encode("ascii", "ignore").decode("ascii").lower()
    if "son" in normalized and ("versiyon" in normalized or "hali" in normalized):
        return "son_versiyon"
    if "degisiklik" in normalized:
        return "degisiklik"
    if "eski" in normalized:
        return "eski_versiyon"
    return "diger"


def parse_rg_date(value: str | None) -> str | None:
    """Convert a dd.mm.yyyy Resmi Gazete date into sortable ISO format."""
    if not value:
        return None
    match = re.match(r"(\d{1,2})\.(\d{1,2})\.(\d{4})", value.strip())
    if not match:
        return None
    day, month, year = match.groups()
    return f"{year}-{int(month):02d}-{int(day):02d}"


def direct_download_links(li) -> list[dict]:
    """Return document dicts for links present directly in the static HTML.

    These items don't carry EPDK's structured version metadata (no RgDate/MulgaTitle),
    so version fields are left empty - there's usually only one such link per item anyway.
    """
    links = []
    for a in li.find_all("a", href=True, recursive=False):
        if "DownloadDocument" in a["href"]:
            title = a.get("title") or clean_text(a)
            links.append(
                {
                    "title": title,
                    "url": BASE_URL + a["href"],
                    "version_label": None,
                    "version_kind": "diger",
                    "rg_date": None,
                    "rg_number": None,
                    "is_mulga": False,
                }
            )
    return links


def fetch_ajax_links(session: requests.Session, fast_access_id: str, fallback_title: str) -> list[dict]:
    """Call GetFastAccessList and flatten every FastAccessDetail into document dicts."""
    resp = session.post(
        FAST_ACCESS_URL,
        headers={**HEADERS, "Content-Type": "application/json; charset=utf-8"},
        data=json.dumps({"fId": fast_access_id}),
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    if data.get("State") != 1:
        return []

    links = []
    for entry in data.get("model") or []:
        entry_title = entry.get("Title") or fallback_title
        mulga_title = entry.get("MulgaTitle") or ""
        rg_date = parse_rg_date(entry.get("RgDate"))
        rg_number = entry.get("RgNumber") or None
        for detail in entry.get("FastAccessDetail") or []:
            content_id = detail.get("ContentId")
            if not content_id:
                continue
            title = detail.get("Title") or entry_title
            # Mirrors the site's own JS string concatenation (Icerik.js) -
            # ContentId is used raw, not percent-encoded.
            url = f"{BASE_URL}/Detay/DownloadDocument?id={content_id}"
            links.append(
                {
                    "title": f"{fallback_title} - {title}" if title != fallback_title else title,
                    "url": url,
                    "version_label": entry_title,
                    "version_kind": classify_version(entry_title),
                    "rg_date": rg_date,
                    "rg_number": rg_number,
                    "is_mulga": bool(mulga_title.strip()),
                }
            )
    return links


def extension_from_response(resp: requests.Response) -> str:
    disposition = resp.headers.get("content-disposition", "")
    match = re.search(r'filename="?([^";]+)"?', disposition)
    if match:
        suffix = Path(match.group(1)).suffix
        if suffix:
            return suffix
    content_type = resp.headers.get("content-type", "")
    return {
        "application/pdf": ".pdf",
        "application/msword": ".doc",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
        "application/vnd.ms-excel": ".xls",
        "application/zip": ".zip",
    }.get(content_type.split(";")[0].strip(), ".pdf")


def download_document(session: requests.Session, url: str, title: str, category_dir: Path) -> Path | None:
    resp = session.get(url, headers=HEADERS, timeout=60)
    if resp.status_code != 200 or not resp.content:
        return None
    ext = extension_from_response(resp)
    filename = f"{slugify(title)}{ext}"
    path = category_dir / filename
    counter = 2
    while path.exists() and path.stat().st_size != len(resp.content):
        path = category_dir / f"{slugify(title)}-{counter}{ext}"
        counter += 1
    path.write_bytes(resp.content)
    return path


def assign_is_current(manifest: list[dict]) -> None:
    """Within each document_group, mark the newest non-repealed 'son_versiyon' entry as current.

    Falls back to leaving is_current False for every entry in a group when none of them
    is clearly the "son versiyon" (e.g. only "Degisiklik"/"Diger" entries exist) - safer
    for a downstream ingest step to skip an ambiguous group than to index the wrong file.
    """
    groups: dict[str, list[dict]] = {}
    for entry in manifest:
        groups.setdefault(entry["document_group"], []).append(entry)

    for entries in groups.values():
        for entry in entries:
            entry["is_current"] = False

        candidates = [e for e in entries if e["version_kind"] == "son_versiyon" and not e["is_mulga"]]
        if not candidates:
            if len(entries) == 1:
                entries[0]["is_current"] = True
            continue

        dated = [e for e in candidates if e["rg_date"]]
        best = max(dated, key=lambda e: e["rg_date"]) if dated else candidates[0]
        best["is_current"] = True


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8")) if MANIFEST_PATH.exists() else []
    seen_urls = {entry["source_url"] for entry in manifest}

    session = requests.Session()
    downloaded = skipped = errors = 0

    print(f"Fetching {MEVZUAT_PAGE} ...")
    sections = fetch_category_sections(session)

    for slug, category_name in TARGET_CATEGORIES.items():
        nodes = sections.get(slug)
        if nodes is None:
            print(f"[!] '{category_name}' section not found on the page, skipping.")
            continue

        category_dir = RAW_DIR / category_name
        category_dir.mkdir(parents=True, exist_ok=True)

        for li, a in iter_leaf_items(nodes):
            title = clean_text(a)
            fast_access_id = a["data-id"]

            links = direct_download_links(li)
            if not links:
                try:
                    time.sleep(REQUEST_DELAY)
                    links = fetch_ajax_links(session, fast_access_id, title)
                except (requests.RequestException, ValueError) as exc:
                    print(f"[x] {category_name} / {title}: fast-access lookup failed ({exc})")
                    errors += 1
                    continue

            for link in links:
                doc_title, doc_url = link["title"], link["url"]
                if doc_url in seen_urls:
                    skipped += 1
                    continue
                try:
                    time.sleep(REQUEST_DELAY)
                    local_path = download_document(session, doc_url, doc_title, category_dir)
                except requests.RequestException as exc:
                    print(f"[x] {category_name} / {doc_title}: download failed ({exc})")
                    errors += 1
                    continue
                if local_path is None:
                    print(f"[x] {category_name} / {doc_title}: empty/failed response from {doc_url}")
                    errors += 1
                    continue

                manifest.append(
                    {
                        "title": doc_title,
                        "document_group": title,
                        "category": category_name,
                        "version_label": link["version_label"],
                        "version_kind": link["version_kind"],
                        "rg_date": link["rg_date"],
                        "rg_number": link["rg_number"],
                        "is_mulga": link["is_mulga"],
                        "is_current": None,  # filled in by assign_is_current() below
                        "source_url": doc_url,
                        "local_path": str(local_path.relative_to(OUTPUT_DIR)),
                        "downloaded_at": datetime.now(timezone.utc).isoformat(),
                    }
                )
                seen_urls.add(doc_url)
                downloaded += 1
                print(f"[+] {category_name} / {doc_title} -> {local_path.name}")

    assign_is_current(manifest)
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nDone. Downloaded: {downloaded}, skipped (already present): {skipped}, errors: {errors}")
    print(f"Manifest: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
