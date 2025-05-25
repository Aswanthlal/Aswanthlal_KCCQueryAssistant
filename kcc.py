import pandas as pd
import json

input_file = 'kcc_data_1_8th.csv'
output_jsonl = 'preprocessed_kcc_data_8.jsonl'
output_csv = 'preprocessed_kcc_data_8.csv'
chunk_size = 100_000

csv_rows = []

with open(output_jsonl, 'w', encoding='utf-8') as jsonl_out:
    for chunk in pd.read_csv(input_file, chunksize=chunk_size):
        print(f"Processing chunk with {len(chunk)} rows")

        chunk = chunk.dropna(subset=['KccAns'])
        chunk['QueryText'] = chunk['QueryText'].str.strip().str.lower()
        chunk['KccAns'] = chunk['KccAns'].str.strip().str.lower()
        chunk['Date'] = pd.to_datetime(chunk[["Year", "Month", "Day"]], errors='coerce')

        for _, row in chunk.iterrows():
            try:
                record = {
                    'question': row['QueryText'],
                    'answer': row['KccAns'],
                    'metadata': {
                        'corp': str(row['Crop']),
                        'district': row['DistrictName'],
                        'state': row['StateName'],
                        'season': row['Season'],
                        'sector': row['Sector'],
                        'date': row['Date'].strftime('%Y-%m-%d') if pd.notnull(row['Date']) else None
                    }
                }
                jsonl_out.write(json.dumps(record) + '\n')
                csv_rows.append({
                    'question': record['question'],
                    'answer': record['answer'],
                    **record['metadata']
                })
            except Exception as e:
                print(f"Error processing row: {e}")
                continue

# Save CSV
csv_df = pd.DataFrame(csv_rows)
csv_df.to_csv(output_csv, index=False)
print(f"Total rows written: {len(csv_df)}")
