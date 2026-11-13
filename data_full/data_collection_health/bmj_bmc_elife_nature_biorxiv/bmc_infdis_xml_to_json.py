import pandas as pd
import xmltodict
import json
from pathlib import Path

# 1. Define your exact paths
xml_dir = Path("/lstr/sahara/datalab-ml/ibrahim/limagents_update/socio_limit_shield/data/health/europepmc/bmc_infdis/fulltext_xml")
json_out_dir = Path("/lstr/sahara/datalab-ml/ibrahim/limagents_update/socio_limit_shield/data/health/europepmc/bmc_infdis/fullext_json")
output_csv_path = Path("/lstr/sahara/datalab-ml/ibrahim/limagents_update/socio_limit_shield/data/health/csv_files/bmc_infdis.csv")

# 2. Ensure both the JSON output directory and the CSV parent directory exist
json_out_dir.mkdir(parents=True, exist_ok=True)
output_csv_path.parent.mkdir(parents=True, exist_ok=True)

data_list = []

# 3. Iterate through all .xml files in the input directory
for xml_file in xml_dir.glob("*.xml"):
    try:
        with open(xml_file, 'r', encoding='utf-8') as f:
            # Read and parse XML
            xml_content = f.read()
            data_dict = xmltodict.parse(xml_content)
            data_dict['source_xml_file'] = xml_file.name
            
            # --- Save the dictionary as an individual JSON file ---
            json_filename = f"{xml_file.stem}.json" 
            json_file_path = json_out_dir / json_filename
            
            with open(json_file_path, 'w', encoding='utf-8') as jf:
                json.dump(data_dict, jf, indent=4)
            
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
print(f"Successfully saved {len(df)} rows to: {output_csv_path}")
print(f"Total dynamic columns created in CSV: {len(df.columns)}")