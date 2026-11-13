import pandas as pd
import requests
import xml.etree.ElementTree as ET
import time

# 1. Load your dataset and filter it FIRST
df = pd.read_csv("/lstr/sahara/datalab-ml/ibrahim/limagents_update/socio_limit_shield/data/data_medlim/pubmed_post_processed/filtered_pubmed.csv")

open_review_journals = ['elife', 'f1000research', 'bmj', 'bmj open', 'plos one', 'nature communications']
sample_df = df[df['journal'].str.lower().isin(open_review_journals)].copy()

print(f"Starting extraction for {len(sample_df)} potentially open-review papers...")

def fetch_review_text_from_xml(pmcid):
    """
    Downloads the Full Text XML from Europe PMC and extracts the peer review text.
    """
    if pd.isna(pmcid):
        return None
        
    pmcid_str = str(pmcid).strip()
    if not pmcid_str.startswith('PMC'):
        pmcid_str = f"PMC{pmcid_str}"

    # Use the fullTextXML endpoint instead of the JSON search endpoint
    url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid_str}/fullTextXML"
    
    try:
        response = requests.get(url, timeout=10)
        
        # If the XML doesn't exist or is locked, skip it
        if response.status_code != 200:
            return "XML not available via API"
            
        # Parse the XML
        root = ET.fromstring(response.content)
        extracted_reviews = []
        
        # Open-access publishers use the JATS XML standard. 
        # Peer reviews are typically stored inside <sub-article> tags.
        for sub_article in root.findall('.//sub-article'):
            article_type = sub_article.get('article-type', '').lower()
            
            # Look for reviewer reports, editor decisions, or author responses
            if 'review' in article_type or 'report' in article_type or 'reply' in article_type:
                
                # Extract all text nodes within this specific sub-article
                texts = [elem.text for elem in sub_article.iter() if elem.text and elem.text.strip()]
                
                # Join the extracted sentences together
                full_review_text = " ".join(texts).strip()
                if full_review_text:
                    extracted_reviews.append(f"[{article_type.upper()}]\n{full_review_text}")
                    
        if extracted_reviews:
            # Separate multiple reviews with a clear divider
            return "\n\n--- NEXT REVIEW REPORT ---\n\n".join(extracted_reviews)
            
        return "No peer review text found in XML"

    except Exception as e:
        print(f"Error parsing XML for {pmcid_str}: {e}")
        return None

# 2. Run the updated function on your filtered DataFrame
# We add a sleep timer so Europe PMC doesn't block your IP address for spamming requests!
def apply_with_delay(pmcid):
    time.sleep(3) # Wait 3 seconds between each call
    return fetch_review_text_from_xml(pmcid)

sample_df['api_peer_reviews_text'] = sample_df['pmc'].apply(apply_with_delay)

# 3. Save the results
out_path = "/lstr/sahara/datalab-ml/ibrahim/limagents_update/socio_limit_shield/data/data_medlim/pubmed_post_processed/filtered_pubmed_with_peer_reviews.csv"
sample_df.to_csv(out_path, index=False)
print(f"Done! Saved to {out_path}")
