import os
import glob
import pandas as pd
import re
try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, desc=None: x

# [Insert the exact extract_limitations_from_end function here]
def extract_limitations_from_end(text):
    if not isinstance(text, str):
        return None
    matches = list(re.finditer(r'limitation', text, flags=re.IGNORECASE))
    if not matches: return None
    last_match = matches[-1]
    
    text_before = text[:last_match.start()]
    start_matches = list(re.finditer(r'[.!?]\s+|\n\n', text_before))
    start_idx = start_matches[-1].end() if start_matches else 0
        
    text_after = text[last_match.end():]
    sentence_end_match = re.search(r'[.!?](?:\s+)|\n\n', text_after)
    
    if sentence_end_match:
        end_of_sentence_idx = last_match.end() + sentence_end_match.start() + (1 if text_after[sentence_end_match.start()] in '.!?' else 0)
    else:
        end_of_sentence_idx = len(text)
        
    remaining_text = text[end_of_sentence_idx:]
    stop_pattern = re.compile(r'\n\n|\b(acknowledgment|acknowledgement|ethics|ethical considerations|future work|grant|appendix|funding)\b', flags=re.IGNORECASE)
    stop_match = stop_pattern.search(remaining_text)
    
    final_text = text[start_idx:end_of_sentence_idx] + remaining_text[:stop_match.start()] if stop_match else text[start_idx:]
    return final_text.strip()

# Directories
input_dir = "/lstr/sahara/datalab-ml/ibrahim/limagents_update/sci_lit_synthesis/data_bio_phy_chem/pubmed_oa_full_paper/csv"
output_dir = "/lstr/sahara/datalab-ml/ibrahim/limagents_update/lim_in_science/data/pubmed"
os.makedirs(output_dir, exist_ok=True)

# Find all files
all_files = glob.glob(os.path.join(input_dir, "*.csv.gz"))
print(f"Found {len(all_files)} files. Starting distributed processing...")

# Process and save ONE BY ONE
for file_path in tqdm(all_files, desc="Processing files"):
    # Create the new output filename (e.g., 'processed_oa_comm_xml.csv.gz')
    base_name = os.path.basename(file_path)
    output_path = os.path.join(output_dir, f"processed_{base_name}")
    
    # Skip if already processed (great if the script gets interrupted and you need to restart)
    if os.path.exists(output_path):
        continue
        
    # 1. Read
    df = pd.read_csv(file_path, compression='gzip')
    
    # 2. Process
    if 'full_text' in df.columns:
        df['limitations_section'] = df['full_text'].apply(extract_limitations_from_end)
    
    # 3. Save immediately
    df.to_csv(output_path, compression='gzip', index=False)
    
    # 4. Clear memory explicitly (optional but good practice)
    del df

print("All files processed and saved individually!") 
