import os
import json
import time
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
from openai import OpenAI

from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from bertopic import BERTopic

# --- Configuration & Paths ---
INPUT_CSV = "/lstr/sahara/datalab-ml/ibrahim/limagents_update/med_limagnets/data/pubmed_post_processed/rag_corpus.csv"
OUTPUT_DIR = Path("/lstr/sahara/datalab-ml/ibrahim/limagents_update/med_limagnets/data/pubmed_post_processed/for_agents_role")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FINAL_CSV_PATH = OUTPUT_DIR / "rag_corpus_topics_500.csv"
TOPIC_CATEGORIES_PATH = OUTPUT_DIR / "bertopic_gpt_topic_categories.csv"

GPT_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "thenlper/gte-small"
# NOTE: Since you are only using 500 documents, the min_topic_size must be 
# smaller than the default 50, otherwise everything becomes outliers (-1).
MIN_TOPIC_SIZE = 15 


def label_topic_prompt(topic_id, keywords, representative_docs):
    """Generates the prompt for GPT to label the BERTopic cluster."""
    keyword_text = ", ".join([word for word, _ in keywords[:20]])
    docs_text = "\n".join(f"- {doc[:400]}..." for doc in representative_docs[:5])
    
    return f"""You are labeling BERTopic clusters of limitations extracted from scientific papers.

Topic id: {topic_id}

Representative limitation extractions:
{docs_text}

BERTopic keywords:
{keyword_text}

Return JSON only in this format:
{{
  "cluster_title": "short technical topic label (e.g. 'Small Sample Size' or 'Selection Bias')",
  "cluster_description": "one sentence describing the shared limitation topic"
}}"""

def label_topic(client, topic_id, keywords, representative_docs):
    """Calls GPT-4o-mini to label a specific topic."""
    response = client.chat.completions.create(
        model=GPT_MODEL,
        temperature=0.1,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "Create concise, accurate labels for scientific limitation topics."},
            {"role": "user", "content": label_topic_prompt(topic_id, keywords, representative_docs)},
        ],
    )
    return json.loads(response.choices[0].message.content)

def fallback_title(topic_id, keywords):
    """Provides a basic title if GPT fails or for the outlier topic."""
    if topic_id == -1:
        return "Unclustered / Mixed Limitations"
    words = [word for word, _ in keywords[:5]]
    return " / ".join(words[:4]) if words else f"Topic {topic_id}"


def main():
    # Initialize OpenAI client (assumes OPENAI_API_KEY is in environment variables)
    client = OpenAI()

    # ---------------------------------------------------------
    # STEP 1: Load, Filter, and Sample the Data
    # ---------------------------------------------------------
    print(f"Loading data from {INPUT_CSV}...")
    df_rag = pd.read_csv(INPUT_CSV)

    # Filter for word count strictly above 50 and below 300
    print("Filtering rows where 50 < limitations_word_count < 300...")
    df_filtered = df_rag[(df_rag['limitations_word_count'] > 50) & (df_rag['limitations_word_count'] < 300)]
    
    # Sample 500 rows if there are more than 500
    if len(df_filtered) > 500:
        print(f"Found {len(df_filtered)} rows. Sampling 500 random rows...")
        df_sampled = df_filtered.sample(n=500, random_state=42).reset_index(drop=True)
    else:
        print(f"Found {len(df_filtered)} rows. Using all available rows.")
        df_sampled = df_filtered.reset_index(drop=True)

    # ---------------------------------------------------------
    # STEP 2: Extract Limitations using GPT-4o-mini
    # ---------------------------------------------------------
    print("Extracting limitations via GPT-4o-mini...")
    extracted_lims = []
    
    for text in tqdm(df_sampled['limitations_section'], desc="LLM Extraction"):
        if pd.isna(text) or not str(text).strip():
            extracted_lims.append("")
            continue
            
        try:
            response = client.chat.completions.create(
                model=GPT_MODEL,
                temperature=0.0,
                messages=[
                    {"role": "system", "content": "Extract limitations from the snippet, don't generate anything."},
                    {"role": "user", "content": str(text)}
                ]
            )
            extracted_lims.append(response.choices[0].message.content.strip())
        except Exception as e:
            print(f"\nError during extraction: {e}")
            extracted_lims.append("")
            
    df_sampled['gpt_extracted_lim'] = extracted_lims

    # Save intermediate dataframe just in case the topic modeling crashes
    intermediate_path = OUTPUT_DIR / "rag_corpus_extracted_500.csv"
    df_sampled.to_csv(intermediate_path, index=False)
    print(f"Saved intermediate extraction to {intermediate_path}")

    # ---------------------------------------------------------
    # STEP 3: BERTopic Modeling
    # ---------------------------------------------------------
    print("\nStarting BERTopic modeling...")
    docs = df_sampled['gpt_extracted_lim'].fillna("").astype(str).tolist()
    
    print(f"Encoding {len(docs)} documents with {EMBEDDING_MODEL}...")
    sentence_model = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = sentence_model.encode(docs, show_progress_bar=True, normalize_embeddings=True, batch_size=64)

    # Use a custom Vectorizer to remove common stop words for better keywords
    vectorizer_model = CountVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2)
    
    topic_model = BERTopic(
        embedding_model=sentence_model,
        vectorizer_model=vectorizer_model,
        min_topic_size=MIN_TOPIC_SIZE,
        calculate_probabilities=False,
        verbose=True,
    )
    
    topics, _ = topic_model.fit_transform(docs, embeddings)
    df_sampled["bertopic_topic_id"] = [int(t) for t in topics]

    # ---------------------------------------------------------
    # STEP 4: Label Topics with GPT-4o-mini
    # ---------------------------------------------------------
    print("\nLabeling topics with GPT-4o-mini...")
    representative_docs_by_topic = topic_model.get_representative_docs()
    category_rows = []
    
    for topic_id in sorted(df_sampled["bertopic_topic_id"].unique().tolist()):
        group = df_sampled[df_sampled["bertopic_topic_id"] == topic_id]
        keywords = topic_model.get_topic(topic_id) or []
        representative_docs = representative_docs_by_topic.get(topic_id, [])
        
        label = {
            "cluster_title": fallback_title(topic_id, keywords),
            "cluster_description": ""
        }
        
        # We don't need GPT to label the -1 (outliers) cluster
        if topic_id != -1:
            try:
                label = label_topic(client, topic_id, keywords, representative_docs)
                time.sleep(0.5) # Slight delay to avoid rate limits
            except Exception as exc:
                print(f"GPT label failed for topic {topic_id}: {exc}")

        category_rows.append({
            "topic_id": int(topic_id),
            "cluster_title": label.get("cluster_title", fallback_title(topic_id, keywords)),
            "cluster_description": label.get("cluster_description", ""),
            "n_documents": int(len(group)),
            "keywords": [{"word": word, "score": float(score)} for word, score in keywords[:10]],
            "representative_docs": representative_docs[:3],
        })

    # Create category lookup dataframe
    categories_df = pd.DataFrame(category_rows).sort_values(["topic_id"]).reset_index(drop=True)
    title_lookup = dict(zip(categories_df["topic_id"], categories_df["cluster_title"]))
    
    # Map the titles back to the main dataframe
    df_sampled["bertopic_topic_title"] = df_sampled["bertopic_topic_id"].map(title_lookup)

    # ---------------------------------------------------------
    # STEP 5: Save Final Outputs
    # ---------------------------------------------------------
    df_sampled.to_csv(FINAL_CSV_PATH, index=False)
    categories_df.to_csv(TOPIC_CATEGORIES_PATH, index=False)
    
    print(f"\n✅ Processing Complete!")
    print(f"Saved final dataframe to: {FINAL_CSV_PATH}")
    print(f"Saved topic categories to: {TOPIC_CATEGORIES_PATH}")
    print(f"Generated {len(categories_df)} distinct topics.")

if __name__ == "__main__":
    main() 
    