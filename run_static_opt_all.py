"""
Run static optimization for all newly trained AHUs.
Ahu1 and Ahu6 already have static_optimization_jan23_31.csv.
Ahu13 excluded per user request.
"""
import traceback
import time
import sys
import os

os.environ["PYTHONIOENCODING"] = "utf-8"

from src.bms_ai.mpc.generate_static_optimization import generate_static_optimization_test

AHUS = [
    "Ahu2", "Ahu3", "Ahu4", "Ahu5",
    "Ahu7", "Ahu8", "Ahu9", "Ahu10",
    "Ahu11", "Ahu12", "Ahu15", "Ahu16",
]

STATIC_OPT_START = "2026-01-23 00:00:00"
STATIC_OPT_END = "2026-01-31 00:00:00"
SCHEDULE_TICKET = "ac40ff65-0950-4f2b-9dec-15ec2fe7018c"

results = {}

for ahu_id in AHUS:
    print(f"\n{'='*70}")
    print(f"  STATIC OPTIMIZATION: {ahu_id}")
    print(f"{'='*70}")
    start_time = time.time()
    try:
        opt_result = generate_static_optimization_test(
            equipment_id=ahu_id,
            start_date_utc=STATIC_OPT_START,
            end_date_utc=STATIC_OPT_END,
            schedule_ticket=SCHEDULE_TICKET,
            output_filename="static_optimization_jan23_31.csv",
        )
        elapsed = time.time() - start_time
        results[ahu_id] = {"status": "OK", "time_sec": round(elapsed, 1)}
        print(f">>> {ahu_id} DONE in {elapsed:.0f}s")
    except Exception as e:
        elapsed = time.time() - start_time
        results[ahu_id] = {"status": "FAILED", "error": str(e), "time_sec": round(elapsed, 1)}
        print(f">>> {ahu_id} FAILED: {e}")
        traceback.print_exc()

print(f"\n{'='*70}")
print("  FINAL SUMMARY")
print(f"{'='*70}")
print(f"{'AHU':<8} {'Status':<10} {'Time (s)':<10}")
print("-" * 28)
for ahu_id in AHUS:
    r = results.get(ahu_id, {})
    print(f"{ahu_id:<8} {r.get('status','?'):<10} {r.get('time_sec','?'):<10}")

ok_count = sum(1 for r in results.values() if r.get("status") == "OK")
print(f"\nTotal: {ok_count}/{len(AHUS)} succeeded")
