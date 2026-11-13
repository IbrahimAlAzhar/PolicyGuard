"""Collect submissions and official reviews from OpenReview venues.

Example:
python -m collectors.openreview_collect \
  --venue-id ML4H.cc/2024/Conference \
  --out data/metadata/openreview_ml4h_2024.jsonl \
  --download-pdfs

Notes:
- Venue IDs differ by year. Check the OpenReview venue page.
- Many venues use invitation names ending in /-/Submission and /-/Official_Review.
"""
import argparse
import os
from typing import Any, Dict, List, Optional

import openreview
from tqdm import tqdm

from utils.io import append_jsonl, download_file, get_session, load_config, slugify


def content_value(content: Dict[str, Any], key: str):
    val = content.get(key)
    if isinstance(val, dict) and "value" in val:
        return val.get("value")
    return val


def flatten_content(content: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for k, v in (content or {}).items():
        if isinstance(v, dict) and "value" in v:
            out[k] = v["value"]
        else:
            out[k] = v
    return out


def is_official_review(reply: Dict[str, Any]) -> bool:
    invitations = reply.get("invitations", []) or []
    signatures = reply.get("signatures", []) or []
    text = " ".join(invitations + signatures)
    return "Official_Review" in text or "Review" in text and "Comment" not in text


def get_direct_replies(note) -> List[Dict[str, Any]]:
    details = getattr(note, "details", None) or {}
    replies = details.get("directReplies", []) if isinstance(details, dict) else []
    return replies or []


def get_pdf_url(pdf_field: Optional[str]) -> Optional[str]:
    if not pdf_field:
        return None
    if pdf_field.startswith("http"):
        return pdf_field
    if pdf_field.startswith("/pdf"):
        return "https://openreview.net" + pdf_field
    if pdf_field.startswith("/attachment"):
        return "https://openreview.net" + pdf_field
    return None


def collect_openreview(venue_id: str, out_path: str, download_pdfs: bool, user_agent: str, sleep: float, pdf_dir: str):
    client = openreview.api.OpenReviewClient(baseurl="https://api2.openreview.net")
    session = get_session(user_agent)

    submission_invitation = f"{venue_id}/-/Submission"
    submissions = client.get_all_notes(invitation=submission_invitation, details="directReplies")

    os.makedirs(pdf_dir, exist_ok=True)
    open(out_path, "w", encoding="utf-8").close()

    for note in tqdm(submissions, desc=f"OpenReview {venue_id}"):
        content = flatten_content(getattr(note, "content", {}) or {})
        title = content.get("title")
        abstract = content.get("abstract")
        pdf_url = get_pdf_url(content.get("pdf"))

        reviews = []
        for reply in get_direct_replies(note):
            if not is_official_review(reply):
                continue
            reviews.append({
                "reply_id": reply.get("id"),
                "invitations": reply.get("invitations"),
                "signatures": reply.get("signatures"),
                "tcdate": reply.get("tcdate"),
                "tmdate": reply.get("tmdate"),
                "content": flatten_content(reply.get("content", {}) or {}),
            })

        pdf_local_path = None
        if download_pdfs and pdf_url:
            fname = slugify(f"openreview_{venue_id}_{getattr(note, 'id', '')}_{title}") + ".pdf"
            pdf_local_path = os.path.join(pdf_dir, fname)
            ok = download_file(session, pdf_url, pdf_local_path, sleep=sleep)
            if not ok:
                pdf_local_path = None

        rec = {
            "source": "openreview",
            "venue_id": venue_id,
            "paper_id": getattr(note, "id", None),
            "forum": getattr(note, "forum", None),
            "number": getattr(note, "number", None),
            "title": title,
            "abstract": abstract,
            "authors": content.get("authors"),
            "keywords": content.get("keywords"),
            "pdf_url": pdf_url,
            "pdf_local_path": pdf_local_path,
            "reviews": reviews,
            "raw_content": content,
        }
        append_jsonl(out_path, rec)

    print(f"Saved {len(submissions)} submissions to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--venue-id", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--download-pdfs", action="store_true")
    parser.add_argument("--pdf-dir", default="data/pdfs/openreview")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    collect_openreview(
        venue_id=args.venue_id,
        out_path=args.out,
        download_pdfs=args.download_pdfs,
        user_agent=cfg.get("user_agent"),
        sleep=float(cfg.get("request_sleep_seconds", 1.0)),
        pdf_dir=args.pdf_dir,
    )


if __name__ == "__main__":
    main()
