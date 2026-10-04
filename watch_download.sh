#!/usr/bin/env bash
# watch_download.sh — watchdog for BTC L2 orderbook download.
#
# Responsibilities:
#   1. Verify/download dependencies and API key (sandbox-reset safe).
#   2. Ensure download_orderbook.py is NOT already running (single-instance lock).
#   3. Launch/resume download_orderbook.py if needed.
#   4. Emit a structured status snapshot (reads .watch_state.json).
#
# Idempotent: safe to re-run on every sandbox reset.
set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
API_KEY_FILE="${WORKSPACE}/.cryptohft_key"
PY="${PYTHON:-python3}"
SCRIPT="${WORKSPACE}/download_orderbook.py"
OUTPUT_DIR="${WORKSPACE}/data"
STATE_FILE="${OUTPUT_DIR}/.watch_state.json"
LOG_FILE="${WORKSPACE}/download.log"

# ---- step 1: env repair (sandbox reset recovery) ----
repair_env() {
    # pip deps
    if ! ${PY} -c "import requests, pandas, pyarrow, cryptohftdata" 2>/dev/null; then
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
    # download_orderbook.py — if missing, flag and exit (should be git-tracked)
    if [[ ! -f "${SCRIPT}" ]]; then
        echo "[watch] FATAL: ${SCRIPT} missing! Check git status."
        git -C "${WORKSPACE}" ls-files 2>&1 | head
        exit 2
    fi
    mkdir -p "${OUTPUT_DIR}"
}

# ---- step 2: single-instance check ----
has_running_download() {
    # grep [d]ownload_orderbook excludes the grep itself
    ps aux | grep '[d]ownload_orderbook.py' | grep -v grep >/dev/null 2>&1
}

# ---- step 3: launch download ----
launch_download() {
    local key
    key=$(cat "${API_KEY_FILE}")
    echo "[watch] launching download_orderbook.py..."
    nohup ${PY} "${SCRIPT}" \
        --start 2026-09-04 \
        --end 2026-10-04 \
        --assets BTC \
        --market both \
        --exchanges binance \
        --output "${OUTPUT_DIR}" \
        --api-key "${key}" \
        >> "${LOG_FILE}" 2>&1 &
    echo $! > "${WORKSPACE}/.download.pid"
    sleep 2
    if has_running_download; then
        echo "[watch] download pid=$(cat "${WORKSPACE}/.download.pid") started OK"
    else
        echo "[watch] download failed to start — tail of log:"
        tail -20 "${LOG_FILE}" 2>/dev/null || true
    fi
}

# ---- step 4: print status snapshot ----
print_status() {
    echo "============================================"
    echo "[watch] $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    echo "============================================"
    echo "pid running: $(has_running_download && echo YES || echo NO)"
    echo "log tail:"
    tail -5 "${LOG_FILE}" 2>/dev/null || echo "(no log yet)"
    echo "disk:"
    df -h "${WORKSPACE}" | tail -1
    echo "state:"
    if [[ -f "${STATE_FILE}" ]]; then
        cat "${STATE_FILE}"
    else
        echo "(no .watch_state.json yet)"
    fi
}

# ---- main ----
repair_env

if has_running_download; then
    echo "[watch] download_orderbook.py already running — skip launch"
else
    launch_download
fi

print_status
