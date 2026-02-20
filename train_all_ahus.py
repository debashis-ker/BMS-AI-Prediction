"""
Train all remaining AHU models and generate static optimization CSVs.
Already trained: Ahu1, Ahu6, Ahu13
Excluded: Ahu14 (disabled)
"""
import traceback
import time
import sys

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

from src.bms_ai.mpc.mpc_training_pipeline import train_mpc_from_scratch
from src.bms_ai.mpc.generate_static_optimization import generate_static_optimization_test

AHUS_TO_TRAIN = [
    "Ahu2", "Ahu3", "Ahu4", "Ahu5",
    "Ahu7", "Ahu8", "Ahu9", "Ahu10",
    "Ahu11", "Ahu12", "Ahu15", "Ahu16",
]

TRAINING_FROM = "2025-11-22"
TRAINING_TO = "2026-02-20"

STATIC_OPT_START = "2026-01-23 00:00:00"
STATIC_OPT_END = "2026-01-31 00:00:00"
SCHEDULE_TICKET = "ac40ff65-0950-4f2b-9dec-15ec2fe7018c"

results = {}

for ahu_id in AHUS_TO_TRAIN:
    print("\n" + "=" * 80)
    print(f"  STARTING {ahu_id}")
    print("=" * 80)
    start_time = time.time()

    try:
        print(f"\n>>> TRAINING {ahu_id} ({TRAINING_FROM} to {TRAINING_TO})")
        model = train_mpc_from_scratch(
            equipment_id=ahu_id,
            from_date=TRAINING_FROM,
            to_date=TRAINING_TO,
            save_model=True,
        )
        train_ok = True
        print(f">>> {ahu_id} TRAINING COMPLETE")
    except Exception as e:
        train_ok = False
        print(f">>> {ahu_id} TRAINING FAILED: {e}")
        traceback.print_exc()
        results[ahu_id] = {"train": "FAILED", "error": str(e)}
        continue  

    try:
        print(f"\n>>> STATIC OPTIMIZATION {ahu_id} ({STATIC_OPT_START} to {STATIC_OPT_END})")
        opt_result = generate_static_optimization_test(
            equipment_id=ahu_id,
            start_date_utc=STATIC_OPT_START,
            end_date_utc=STATIC_OPT_END,
            schedule_ticket=SCHEDULE_TICKET,
            output_filename="static_optimization_jan23_31.csv",
        )
        opt_ok = True
        print(f">>> {ahu_id} STATIC OPTIMIZATION COMPLETE")
    except Exception as e:
        opt_ok = False
        print(f">>> {ahu_id} STATIC OPTIMIZATION FAILED: {e}")
        traceback.print_exc()

    elapsed = time.time() - start_time
    results[ahu_id] = {
        "train": "OK" if train_ok else "FAILED",
        "static_opt": "OK" if opt_ok else "FAILED",
        "time_sec": round(elapsed, 1),
    }

print("\n" + "=" * 80)
print("  FINAL SUMMARY")
print("=" * 80)
print(f"{'AHU':<8} {'Training':<12} {'Static Opt':<12} {'Time (s)':<10}")
print("-" * 42)
for ahu_id in AHUS_TO_TRAIN:
    r = results.get(ahu_id, {})
    print(f"{ahu_id:<8} {r.get('train','?'):<12} {r.get('static_opt','?'):<12} {r.get('time_sec','?'):<10}")
