import os
import json
import requests
import xml.etree.ElementTree as ET
import pandas as pd
from pathlib import Path
from typing import Any, Dict, Optional, List

# ============================================================
# DIRECTORY CONFIGURATION
# ============================================================
PDF_DIR = Path("/lstr/sahara/datalab-ml/ibrahim/limagents_update/socio_limit_shield/data/health/nature_bmj/nature_communications/peer_review_pdfs")
JSON_DIR = Path("/lstr/sahara/datalab-ml/ibrahim/limagents_update/socio_limit_shield/data/health/nature_bmj/nature_communications/peer_review_jsons")
CSV_OUT = Path("/lstr/sahara/datalab-ml/ibrahim/limagents_update/socio_limit_shield/data/health/csv_files/nature_comms_peer_review.csv")

GROBID_BASE = "http://127.0.0.1:8070"
TEI_NS = {"tei": "http://www.tei-c.org/ns/1.0"}

# Ensure output directories exist
JSON_DIR.mkdir(parents=True, exist_ok=True)
CSV_OUT.parent.mkdir(parents=True, exist_ok=True)

# ============================================================
# GROBID & XML HELPERS
# ============================================================
def grobid_process_fulltext(pdf_path: Path, timeout: int = 600) -> str:
    """POST local PDF to GROBID server."""
    url = GROBID_BASE.rstrip("/") + "/api/processFulltextDocument"
    with pdf_path.open("rb") as fp:
        files = {"input": fp}
        r = requests.post(url, files=files, timeout=timeout)
        r.raise_for_status()
        return r.text

def text_of(elem: Optional[ET.Element]) -> str:
    if elem is None:
        return ""
    return "".join(elem.itertext()).strip()

def extract_title(root: ET.Element) -> str:
    t = root.find(".//tei:teiHeader//tei:fileDesc//tei:titleStmt//tei:title", TEI_NS)
    return text_of(t)

def extract_abstract(root: ET.Element) -> str:
    a = root.find(".//tei:teiHeader//tei:profileDesc//tei:abstract", TEI_NS)
    if a is not None:
        return text_of(a)
    for div in root.findall(".//tei:text//tei:body//tei:div", TEI_NS):
        head = text_of(div.find("./tei:head", TEI_NS)).lower()
        if head == "abstract":
            paras = [text_of(p) for p in div.findall(".//tei:p", TEI_NS)]
            return "\n".join([p for p in paras if p]).strip()
    return ""

def extract_authors(root: ET.Element) -> List[Dict[str, str]]:
    authors = []
    for a in root.findall(".//tei:teiHeader//tei:fileDesc//tei:titleStmt//tei:author", TEI_NS):
        pers = a.find(".//tei:persName", TEI_NS)
        if pers is not None:
            forename = text_of(pers.find(".//tei:forename", TEI_NS))
            surname = text_of(pers.find(".//tei:surname", TEI_NS))
            name = (forename + " " + surname).strip() or text_of(pers)
        else:
            name = text_of(a)
        if name:
            authors.append({"name": name})
    return authors

def extract_sections(root: ET.Element) -> List[Dict[str, Any]]:
    sections = []
    body = root.find(".//tei:text/tei:body", TEI_NS)
    if body is None:
        return sections
    for div in body.findall(".//tei:div", TEI_NS):
        head = text_of(div.find("./tei:head", TEI_NS))
        paras = [text_of(p) for p in div.findall(".//tei:p", TEI_NS) if text_of(p)]
        if head or paras:
            sections.append({"heading": head, "paragraphs": paras})
    return sections

def tei_xml_to_json_dict(tei_xml: str, source_file: str) -> Dict[str, Any]:
    root = ET.fromstring(tei_xml.encode("utf-8", errors="ignore"))
    return {
        "source_pdf_file": source_file,
        "title": extract_title(root),
        "authors": extract_authors(root),
        "abstract": extract_abstract(root),
        "sections": extract_sections(root),
        "source": "grobid-local"
    }

# ============================================================
# MAIN PIPELINE
# ============================================================
def main():
    print(f"Scanning directory: {PDF_DIR}")
    pdf_files = list(PDF_DIR.glob("*.pdf"))
    
    if not pdf_files:
        print("No PDF files found in the target directory.")
        return

    data_list = []
    processed = 0

    for pdf_path in pdf_files:
        try:
            print(f"Processing: {pdf_path.name}")
            
            # 1. Process via GROBID
            tei_xml = grobid_process_fulltext(pdf_path)
            
            # 2. Convert XML to Dictionary
            data = tei_xml_to_json_dict(tei_xml, source_file=pdf_path.name)
            
            # 3. Save as individual JSON file
            json_file_path = JSON_DIR / f"{pdf_path.stem}.json"
            with open(json_file_path, 'w', encoding='utf-8') as jf:
                json.dump(data, jf, indent=4)
                
            # 4. Append to global list
            data_list.append(data)
            processed += 1
            
        except Exception as e:
            print(f"Error processing {pdf_path.name}: {e}")

    # 5. Convert to flattened CSV
    if data_list:
        print("\nFlattening structures and saving to CSV...")
        df = pd.json_normalize(data_list)
        df.to_csv(CSV_OUT, index=False, encoding='utf-8')
        
        print(f"\n[DONE] Successfully processed {processed}/{len(pdf_files)} PDFs.")
        print(f"JSON files saved to: {JSON_DIR}")
        print(f"CSV file saved to: {CSV_OUT}")
    else:
        print("\n[WARNING] No data was extracted. CSV not created.")

if __name__ == "__main__":
    main()