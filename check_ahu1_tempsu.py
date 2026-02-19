import pandas as pd

df = pd.read_csv('artifacts/bacnet_values.csv')

# Filter strictly for Ahu1 (not Ahu10, Ahu11, etc.)
ahu1 = df[df['object_name'].str.contains("'Ahu1'", na=False, regex=False)]

print("=== All Ahu1 datapoints with Supply/TempSu/TSu in name or description ===")
mask = (
    ahu1['object_name'].str.contains('Su|Supply|Temp', case=False, na=False) |
    ahu1['description'].str.contains('Supply', case=False, na=False)
)
print(ahu1[mask][['object_name', 'description', 'present_value']].to_string())

print("\n=== Full list of ALL Ahu1 sensor datapoints ===")
for _, r in ahu1.sort_values('object_name').iterrows():
    parts = r['object_name'].split("Ahu1'")
    dp = parts[-1].lstrip("'") if len(parts) > 1 else r['object_name']
    print(f"  {dp:25s} | {str(r['description']):50s} | {r['present_value']}")

print(f"\nTotal Ahu1 datapoints: {len(ahu1)}")

# Compare with Ahu2 and Ahu3 which DO have TempSu
print("\n=== Ahu2 TempSu ===")
ahu2_su = df[df['object_name'].str.contains("'Ahu2'", na=False, regex=False) &
             df['object_name'].str.contains('TempSu', na=False)]
print(ahu2_su[['object_name', 'description', 'present_value']].to_string())

print("\n=== Ahu3 TempSu ===")
ahu3_su = df[df['object_name'].str.contains("'Ahu3'", na=False, regex=False) &
             df['object_name'].str.contains('TempSu', na=False)]
print(ahu3_su[['object_name', 'description', 'present_value']].to_string())
