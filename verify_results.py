import pandas as pd
import numpy as np

df_old = pd.read_csv('artifacts/ahu13_mpc/static_optimization_energy_aware_25.csv')
df_new = pd.read_csv('artifacts/ahu13_mpc/static_optimization_proportional.csv')

for label, df in [("PREVIOUS (binary cutoff)", df_old), ("NEW (proportional)", df_new)]:
    occ = df[df['occupied'] == 1]
    sp = occ['optimized_SpTREff']
    
    print("=" * 60)
    print(f"  {label}")
    print("=" * 60)
    print(f"Count: {len(occ)}, mean: {sp.mean():.2f}, std: {sp.std():.2f}, min: {sp.min()}, max: {sp.max()}")
    
    print("Distribution:")
    for lo, hi in [(21, 21.5), (21.5, 22), (22, 22.5), (22.5, 23), (23, 23.5)]:
        c = len(occ[(sp >= lo) & (sp < hi)])
        print(f"  [{lo}, {hi}): {c} ({c/len(occ)*100:.1f}%)")
    print(f"  exactly 21.0: {len(occ[sp == 21.0])}")
    print(f"  below 21.0: {len(occ[sp < 21.0])}")
    
    # Warm room analysis (TempSp1 > 22 — the rows that were stuck at 21)
    warm = occ[occ['actual_TempSp1'] > 22]
    if len(warm) > 0:
        print(f"\n  WARM ROOM (TempSp1 > 22): {len(warm)} rows")
        print(f"    SpTREff: mean={warm['optimized_SpTREff'].mean():.2f}, min={warm['optimized_SpTREff'].min():.2f}, max={warm['optimized_SpTREff'].max():.2f}")
    
    slight_warm = occ[(occ['actual_TempSp1'] > 22) & (occ['actual_TempSp1'] < 23)]
    if len(slight_warm) > 0:
        print(f"  SLIGHTLY WARM (22 < TempSp1 < 23): {len(slight_warm)} rows")
        print(f"    SpTREff: mean={slight_warm['optimized_SpTREff'].mean():.2f}, min={slight_warm['optimized_SpTREff'].min():.2f}, max={slight_warm['optimized_SpTREff'].max():.2f}")
    
    very_warm = occ[occ['actual_TempSp1'] >= 23]
    if len(very_warm) > 0:
        print(f"  VERY WARM (TempSp1 >= 23): {len(very_warm)} rows")
        print(f"    SpTREff: mean={very_warm['optimized_SpTREff'].mean():.2f}, min={very_warm['optimized_SpTREff'].min():.2f}, max={very_warm['optimized_SpTREff'].max():.2f}")
    
    cool = occ[occ['actual_TempSp1'] < 21]
    if len(cool) > 0:
        print(f"  COOL ROOM (TempSp1 < 21): {len(cool)} rows")
        print(f"    SpTREff: mean={cool['optimized_SpTREff'].mean():.2f}, min={cool['optimized_SpTREff'].min():.2f}, max={cool['optimized_SpTREff'].max():.2f}")
    
    print()
