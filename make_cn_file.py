import pandas as pd

input_path = "dataset_files/Steel Beads/data/all_particle_stats.csv"
output_path = "dataset_files/Steel Beads/data/cn_file.txt"
df = pd.read_csv(input_path)
max_cn, min_cn = -9999, 9999
max_cn_id, min_cn_id = -9999, -9999
with open(output_path, "w", encoding="utf-8") as f:
    for _, row in df.iterrows():
        if max_cn < row['cn']:
            max_cn = row['cn']
            max_cn_id = int(row['cp_id'])
        if min_cn > row['cn']:
            min_cn = row['cn']
            min_cn_id = int(row['cp_id']) 
        text = f"{int(row['cp_id'])} {row['cn']}\n"
        f.write(text)

print("Max: ", max_cn, max_cn_id)
print("Min: ", min_cn, min_cn_id)