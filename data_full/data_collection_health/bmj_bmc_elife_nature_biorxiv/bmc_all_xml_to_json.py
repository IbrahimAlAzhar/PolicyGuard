import pandas as pd
import xmltodict
import json
from pathlib import Path

# 1. Define your paths
xml_dir = Path("/lstr/sahara/datalab-ml/ibrahim/limagents_update/socio_limit_shield/data/health/europepmc/bmc_all/fulltext_xml")
json_out_dir = Path("/lstr/sahara/datalab-ml/ibrahim/limagents_update/socio_limit_shield/data/health/europepmc/bmc_all/fulltext_json")
output_csv_path = "/lstr/sahara/datalab-ml/ibrahim/limagents_update/socio_limit_shield/data/health/csv_files/bmc_fulltext_aggregated.csv" 

# 2. Ensure the output directory for JSON files exists
# parents=True creates any missing parent folders, exist_ok=True prevents errors if it's already there
json_out_dir.mkdir(parents=True, exist_ok=True)

data_list = []

# 3. Iterate through all .xml files
for xml_file in xml_dir.glob("*.xml"):
    try:
        with open(xml_file, 'r', encoding='utf-8') as f:
            # Read and parse XML
            xml_content = f.read()
            data_dict = xmltodict.parse(xml_content)
            data_dict['source_xml_file'] = xml_file.name
            
            # --- NEW: Save the dictionary as an individual JSON file ---
            # xml_file.stem gives us just the filename without the ".xml" extension
            json_filename = f"{xml_file.stem}.json" 
            json_file_path = json_out_dir / json_filename
            
            with open(json_file_path, 'w', encoding='utf-8') as jf:
                # indent=4 makes the JSON clean and readable
                json.dump(data_dict, jf, indent=4)
            # ----------------------------------------------------------
            
            # Append to list for the CSV aggregation
            data_list.append(data_dict)
            
    except Exception as e:
        print(f"Error parsing {xml_file.name}: {e}")

# 4. Flatten the nested XML dictionaries and convert to DataFrame
print("Flattening structure and converting to DataFrame...")
df = pd.json_normalize(data_list)

# 5. Save the aggregated DataFrame to CSV
df.to_csv(output_csv_path, index=False, encoding='utf-8')

print(f"Successfully saved individual JSON files to: {json_out_dir}")
print(f"Successfully saved {len(df)} rows to {output_csv_path}.")
print(f"Total dynamic columns created in CSV: {len(df.columns)}")