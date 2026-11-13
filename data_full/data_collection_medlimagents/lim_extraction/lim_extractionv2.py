import os
import pandas as pd
import re

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

# List of specific files you want to update
files_to_process = [
    "/lstr/sahara/datalab-ml/ibrahim/limagents_update/med_limagnets/data/pubmed_post_processed/inference_set_500.csv",
    "/lstr/sahara/datalab-ml/ibrahim/limagents_update/med_limagnets/data/pubmed_post_processed/train_350.csv",
    "/lstr/sahara/datalab-ml/ibrahim/limagents_update/med_limagnets/data/pubmed_post_processed/test_150.csv"
]

print(f"Starting processing of {len(files_to_process)} specific files...\n")

for file_path in files_to_process:
    file_name = os.path.basename(file_path)
    print(f"Processing: {file_name}")
    
    # 1. Read the specific CSV file
    df = pd.read_csv(file_path)
    
    # 2. Check if 'full_text' exists before applying logic
    if 'full_text' in df.columns:
        
        # 3. Extract limitations and create 'limitations_section_upd'
        df['limitations_section_upd'] = df['full_text'].apply(extract_limitations_from_end)
        
        # 4. Function to remove the limitation from the full text
        def remove_limitation(row):
            full = row['full_text']
            limitation = row['limitations_section_upd']
            
            # If a limitation was successfully extracted, remove it from the full text
            if isinstance(limitation, str) and isinstance(full, str):
                # .replace() removes the exact matching substring
                return full.replace(limitation, '')
            
            # If no limitation was found, return the original text unchanged
            return full
            
        # Apply the removal function to create 'full_text_excl_lim'
        df['full_text_excl_lim'] = df.apply(remove_limitation, axis=1)
    
    # 5. Save the updated DataFrame back to the exact same file path (overwriting it)
    df.to_csv(file_path, index=False)
    print(f"  -> Successfully updated and saved to: {file_path}\n")

print("All specified files have been updated!")