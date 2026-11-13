import json
from pathlib import Path
import pandas as pd
import xmltodict

# 1. Define configurations for all three datasets
datasets = [
    {
        "name": "BMC Medicine",
        "xml_dir": "/lstr/sahara/datalab-ml/ibrahim/limagents_update/socio_limit_shield/data/health/europepmc/bmc_medicine/fulltext_xml",
        "json_dir": "/lstr/sahara/datalab-ml/ibrahim/limagents_update/socio_limit_shield/data/health/europepmc/bmc_medicine/fulltext_json",
        "csv_path": "/lstr/sahara/datalab-ml/ibrahim/limagents_update/socio_limit_shield/data/health/csv_files/bmc_medicine.csv",
    },
    {
        "name": "BMC Public Health",
        "xml_dir": "/lstr/sahara/datalab-ml/ibrahim/limagents_update/socio_limit_shield/data/health/europepmc/bmc_public_health/fulltext_xml",
        "json_dir": "/lstr/sahara/datalab-ml/ibrahim/limagents_update/socio_limit_shield/data/health/europepmc/bmc_public_health/fulltext_json",
        "csv_path": "/lstr/sahara/datalab-ml/ibrahim/limagents_update/socio_limit_shield/data/health/csv_files/bmc_public_health.csv",
    },
    {
        "name": "eLife VOR",
        "xml_dir": "/lstr/sahara/datalab-ml/ibrahim/limagents_update/socio_limit_shield/data/health/europepmc/elife_vor/fulltext_xml",
        "json_dir": "/lstr/sahara/datalab-ml/ibrahim/limagents_update/socio_limit_shield/data/health/europepmc/elife_vor/fulltext_json",
        "csv_path": "/lstr/sahara/datalab-ml/ibrahim/limagents_update/socio_limit_shield/data/health/csv_files/elife_vor.csv",
    },
]

# 2. Loop through each dataset configuration
for target in datasets:
    print(f"\n--- Processing Dataset: {target['name']} ---")

    xml_dir = Path(target["xml_dir"])
    json_out_dir = Path(target["json_dir"])
    output_csv_path = Path(target["csv_path"])

    # Ensure output directories exist
    json_out_dir.mkdir(parents=True, exist_ok=True)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    data_list = []

    # Iterate through XML files
    xml_files = list(xml_dir.glob("*.xml"))
    if not xml_files:
        print(f"Warning: No XML files found in {xml_dir}")
        continue

    for xml_file in xml_files:
        try:
            with open(xml_file, "r", encoding="utf-8") as f:
                xml_content = f.read()
                data_dict = xmltodict.parse(xml_content)
                data_dict["source_xml_file"] = xml_file.name

                # Save individual JSON file
                json_filename = f"{xml_file.stem}.json"
                json_file_path = json_out_dir / json_filename

                with open(json_file_path, "w", encoding="utf-8") as jf:
                    json.dump(data_dict, jf, indent=4)

                # Append for global DataFrame accumulation
                data_list.append(data_dict)

        except Exception as e:
            print(f"Error parsing {xml_file.name}: {e}")

    # Convert to flat DataFrame and save to CSV
    if data_list:
        print(f"Flattening and converting {target['name']} data to DataFrame...")
        df = pd.json_normalize(data_list)
        df.to_csv(output_csv_path, index=False, encoding="utf-8")

        print(f"Saved JSON files to: {json_out_dir}")
        print(f"Saved CSV file to: {output_csv_path}")
        print(f"Total rows: {len(df)} | Total dynamic columns: {len(df.columns)}")
    else:
        print(f"No valid data parsed for {target['name']}. Skipping CSV creation.")

print("\nAll pipeline tasks completed successfully.")