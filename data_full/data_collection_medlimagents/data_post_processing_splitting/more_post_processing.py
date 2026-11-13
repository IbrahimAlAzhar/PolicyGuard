import pandas as pd 

df = pd.read_csv("/lstr/sahara/datalab-ml/ibrahim/limagents_update/med_limagnets/data/pubmed_post_processed/filtered_pubmed.csv") 

# Filters for rows strictly greater than 3
filtered_df = df[df['relevance_score'] > 3]

filtered_df = filtered_df[filtered_df['limitations_word_count'] > 49] 

filtered_df.to_csv("/lstr/sahara/datalab-ml/ibrahim/limagents_update/med_limagnets/data/pubmed_post_processed/filtered_pubmed_rel_above_3.csv",index=False) 

