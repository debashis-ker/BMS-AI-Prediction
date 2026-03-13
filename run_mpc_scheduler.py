"""
MPC Scheduler  –  Standalone Runner
====================================

Calls the /mpc/optimize endpoint for every AHU on a fixed interval.

  - Reads all config from .env  (no changes to existing project code)
  - Skips AHUs whose model is not loaded (checked once at startup via /mpc/status)
  - Logs success / failure per AHU per cycle to console and log_info/mpc_scheduler.log
  - Runs until Ctrl-C

Run:
    python run_mpc_scheduler.py

.env keys used:
    PORT                            – API port            (default: 8000)
    MPC_SCHEDULER_INTERVAL_SECONDS  – seconds per cycle   (default: 10)
    MPC_TICKET                      – BMS / schedule API ticket
    MPC_TICKET_TYPE                 – optional ticket type (e.g. jobUser)
    MPC_BUILDING_ID                 – building UUID
    MPC_OCCUPIED_SETPOINT           – occupied °C target  (default: 21.0)
    MPC_UNOCCUPIED_SETPOINT         – unoccupied °C       (default: 24.0)
"""

import os
import sys
import time
import logging
import signal
from datetime import datetime
from pathlib import Path

import requests
from dotenv import load_dotenv

# ─────────────────────────────────────────────────────────────────────────────
# Bootstrap
# ─────────────────────────────────────────────────────────────────────────────

load_dotenv()

# ─── Logging ─────────────────────────────────────────────────────────────────
Path("log_info").mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("log_info/mpc_scheduler.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("mpc_scheduler")

# ─────────────────────────────────────────────────────────────────────────────
# Config from .env
# ─────────────────────────────────────────────────────────────────────────────

API_BASE_URL  = f"http://localhost:{os.getenv('PORT', '8000')}"
INTERVAL      = int(os.getenv("MPC_SCHEDULER_INTERVAL_SECONDS", "1"))
TICKET        = os.getenv("MPC_TICKET", "").strip("'\"")
TICKET_TYPE   = os.getenv("MPC_TICKET_TYPE", None)
BUILDING_ID   = os.getenv("MPC_BUILDING_ID", "36c27828-d0b4-4f1e-8a94-d962d342e7c2")
OCC_SP        = float(os.getenv("MPC_OCCUPIED_SETPOINT", "21.0"))
UNOCC_SP      = float(os.getenv("MPC_UNOCCUPIED_SETPOINT", "24.0"))

# ─────────────────────────────────────────────────────────────────────────────
# Full AHU map  –  equipment_id -> screen_id
# (mirrors ahu_configs.py without importing it)
# ─────────────────────────────────────────────────────────────────────────────

ALL_AHUS: dict[str, str] = {
    # "Ahu1":  "Screen 1",
    # "Ahu2":  "Screen 2",
    # "Ahu3":  "Screen 3",
    # "Ahu4":  "Screen 4",
    # "Ahu5":  "Screen 5",
    # "Ahu6":  "Screen 6",
    # "Ahu7":  "Screen 7",
    # "Ahu8":  "Screen 8",
    # "Ahu9":  "Screen 9",
    # "Ahu10": "Screen 10",
    # "Ahu11": "Screen 11",
    # "Ahu12": "Screen 12",
    "Ahu13": "Screen 13"
    # "Ahu14": "Screen 14",
    # "Ahu15": "Screen 15",
    # "Ahu16": "Screen 16",
}

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def check_loaded_models() -> list[str]:
    """
    Ask the API which AHU models are loaded.
    Returns a list of equipment_ids whose model is ready.
    Falls back to ALL_AHUS if the status endpoint is unreachable.
    """
    try:
        resp = requests.get(f"{API_BASE_URL}/mpc/status", timeout=10)
        resp.raise_for_status()
        data = resp.json()
        models: dict = data.get("models") or {}
        loaded = [ahu for ahu, ok in models.items() if ok]
        log.info(f"[Status] Loaded models: {loaded}")
        return loaded
    except Exception as exc:
        log.warning(f"[Status] Could not reach /mpc/status ({exc}). Will try all AHUs.")
        return list(ALL_AHUS.keys())


def optimize_ahu(equipment_id: "Ahu13", screen_id: str) -> dict:
    """POST /mpc/optimize for one AHU. Returns the response JSON."""
    payload = {
        "ticket":               TICKET,
        "ticket_type":          TICKET_TYPE,
        "account_id":           "scheduler",
        "software_id":          os.getenv("SOFTWARE_ID", "scheduler"),
        "building_id":          BUILDING_ID,
        "equipment_id":         equipment_id,
        "screen_id":            screen_id,
        "occupied_setpoint":    OCC_SP,
        "unoccupied_setpoint":  UNOCC_SP,
    }
    # Remove None values so the API doesn't choke on them
    payload = {k: v for k, v in payload.items() if v is not None}

    resp = requests.post(
        f"{API_BASE_URL}/mpc/optimize",
        json=payload,
        timeout=60,      # inference can take a few seconds
    )
    return resp.json()


# ─────────────────────────────────────────────────────────────────────────────
# Graceful shutdown
# ─────────────────────────────────────────────────────────────────────────────

_running = True

def _handle_signal(sig, frame):
    global _running
    log.info("Shutdown signal received. Stopping scheduler…")
    _running = False

signal.signal(signal.SIGINT,  _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)

# ─────────────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────────────

def main():
    log.info("=" * 65)
    log.info("  MPC Scheduler  –  Standalone Runner")
    log.info(f"  API           : {API_BASE_URL}")
    log.info(f"  Interval      : {INTERVAL}s")
    log.info(f"  Ticket        : {TICKET[:8]}…" if len(TICKET) > 8 else f"  Ticket        : {TICKET or '(empty)'}")
    log.info(f"  Occ setpoint  : {OCC_SP} °C")
    log.info(f"  Unocc setpoint: {UNOCC_SP} °C")
    log.info("=" * 65)

    if not TICKET:
        log.error("MPC_TICKET is empty. Please set it in your .env file.")
        sys.exit(1)

    # Check once at startup which models are loaded, then restrict to ALL_AHUS
    loaded_ahus = [a for a in check_loaded_models() if a in ALL_AHUS]
    if not loaded_ahus:
        log.warning("Requested AHUs not loaded via API — running against ALL_AHUS directly.")
        loaded_ahus = list(ALL_AHUS.keys())
    if not loaded_ahus:
        log.error("No AHU models are loaded. Is the API running?")
        sys.exit(1)

    cycle = 0

    while _running:
        cycle += 1
        cycle_start = time.monotonic()
        now_str = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

        log.info(f"── Cycle {cycle:>4d}  {now_str} UTC  ({len(loaded_ahus)} AHUs) ──")

        success_count = 0
        fail_count    = 0

        for ahu_id in loaded_ahus:
            if not _running:
                break

            screen_id = ALL_AHUS.get(ahu_id, f"Screen {ahu_id.replace('Ahu','')}")

            try:
                result = optimize_ahu(ahu_id, screen_id)

                if result.get("success"):
                    status   = result.get("optimization_status", "ok")
                    setpoint = result.get("optimized_setpoint")
                    mode     = result.get("mode", "")
                    log.info(
                        f"  ✓ {ahu_id:<6s}  status={status:<10s}  "
                        f"setpoint={setpoint}  mode={mode}"
                    )
                    success_count += 1
                else:
                    err = result.get("error", "unknown error")
                    log.warning(f"  ✗ {ahu_id:<6s}  FAILED  – {err}")
                    fail_count += 1

            except requests.exceptions.Timeout:
                log.warning(f"  ✗ {ahu_id:<6s}  TIMEOUT  (>60s)")
                fail_count += 1
            except requests.exceptions.ConnectionError:
                log.error(f"  ✗ {ahu_id:<6s}  CONNECTION ERROR  – Is the API up?")
                fail_count += 1
            except Exception as exc:
                log.error(f"  ✗ {ahu_id:<6s}  UNEXPECTED ERROR  – {exc}")
                fail_count += 1

        elapsed   = time.monotonic() - cycle_start
        sleep_for = max(0, INTERVAL - elapsed)

        log.info(
            f"── Cycle {cycle:>4d} done  "
            f"✓{success_count} ✗{fail_count}  "
            f"elapsed={elapsed:.1f}s  sleeping={sleep_for:.1f}s ──\n"
        )

        # Sleep in small chunks so Ctrl-C is responsive
        deadline = time.monotonic() + sleep_for
        while _running and time.monotonic() < deadline:
            time.sleep(0.5)

    log.info("MPC Scheduler stopped.")


if __name__ == "__main__":
    main()
