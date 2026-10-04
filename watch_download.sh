#!/usr/bin/env bash
# watch_download.sh — watchdog for BTC L2 orderbook download.
#
# Idempotent sandbox-reset recovery:
#   1. Repair pip deps + API key if missing
#   2. Ensure download_orderbook.py is present (git restore)
#   3. If no download_orderbook.py is running, launch it in BACKGROUND
#      using setsid + disown (survives the tool runner)
#   4. Print a structured status snapshot
set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
API_KEY_FILE="${WORKSPACE}/.cryptohft_key"
SCRIPT="${WORKSPACE}/download_orderbook.py"
OUTPUT_DIR="${WORKSPACE}/data"
STATE_FILE="${OUTPUT_DIR}/.watch_state.json"
LOG_FILE="${WORKSPACE}/download.log"
PID_FILE="${WORKSPACE}/.download.pid"

repair_env() {
    # pip deps
    if ! python3 -c "import requests, pandas, pyarrow, cryptohftdata" 2>/dev/null; then
        echo "[watch] installing pip deps..."
        pip install -q requests pandas pyarrow cryptohftdata 2>&1 | tail -3
    fi
    # API key
    if [[ ! -f "${API_KEY_FILE}" ]]; then
        echo "[watch] restoring API key..."
        echo '7b01d0d6c45dbf65f025ad40ec8aafb1f0540d64c9892c1f386cf3ebca47c9f8' \
            > "${API_KEY_FILE}"
        chmod 600 "${API_KEY_FILE}"
    fi
    # download_orderbook.py — recover from git if missing
    if [[ ! -f "${SCRIPT}" ]]; then
        echo "[watch] restoring ${SCRIPT} from git..."
        git -C "${WORKSPACE}" checkout download_orderbook.py 2>&1 || {
            echo "[watch] FATAL: ${SCRIPT} missing AND not in git."
            exit 2
        }
        chmod +x "${SCRIPT}"
    fi
    mkdir -p "${OUTPUT_DIR}"
}

has_running_download() {
    ps aux | grep '[d]ownload_orderbook.py' >/dev/null 2>&1
}

launch_download() {
    local key
    key=$(cat "${API_KEY_FILE}")
    echo "[watch] launching download_orderbook.py (setsid + disown)..."
    (
        setsid python3 "${SCRIPT}" \
            --start 2026-09-04 \
            --end 2026-10-04 \
            --assets BTC \
            --market both \
            --exchanges binance \
            --output "${OUTPUT_DIR}" \
            --api-key "${key}" \
            >> "${LOG_FILE}" 2>&1 < /dev/null &
        disown
        echo $! > "${PID_FILE}"
    )
    sleep 3
    if has_running_download; then
        echo "[watch] download pid=$(cat "${PID_FILE}") running OK"
    else
        echo "[watch] download process exited — check ${LOG_FILE}"
        tail -10 "${LOG_FILE}" 2>/dev/null || true
    fi
}

# ---- main ----
repair_env

if has_running_download; then
    echo "[watch] download_orderbook.py already running — skip launch"
else
    launch_download
fi

echo "============================================"
echo "[watch] $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "============================================"
echo "process running: $(has_running_download && echo YES || echo NO)"
echo "disk:"
df -h "${WORKSPACE}" | tail -1
echo "state:"
if [[ -f "${STATE_FILE}" ]]; then
    cat "${STATE_FILE}"
else
    echo "(no .watch_state.json yet — download in progress)"
fi
echo "log tail:"
tail -3 "${LOG_FILE}" 2>/dev/null || echo "(no log yet)"
