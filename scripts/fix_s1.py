"""Fix Sentinel-1 scene catalog — extract dates from scene IDs."""
import pandas as pd
import re

s1_path = r"D:\Best_Shot_ROBOVANTA_Hackathon\DATASETS\DATASETS\ROBOVANTA_PROJECT\Satellite\Sentinel1\sentinel1_available_scenes.csv"
df = pd.read_csv(s1_path)
print("Original columns:", df.columns.tolist())
print("Shape:", df.shape)
print(df.head(3))

# Extract dates from scene_id using regex
def extract_date(scene_id):
    m = re.search(r'(\d{8})T\d{6}', str(scene_id))
    return pd.to_datetime(m.group(1), format='%Y%m%d') if m else None

df['date'] = df['scene_id'].apply(extract_date)
df = df.sort_values('date')

out_path = r"D:\Best_Shot_ROBOVANTA_Hackathon\data\cleaned\sentinel1_scenes.csv"
df.to_csv(out_path, index=False)
print(f"\nFixed: {len(df)} scenes")
print(f"Date range: {df['date'].min()} to {df['date'].max()}")
print("Saved to:", out_path)
