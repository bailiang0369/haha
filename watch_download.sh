#!/usr/bin/env bash
# BTC L2 Orderbook 下载看门狗
# 每次触发都做全链路恢复 / 并发守护 / 状态汇报
# Usage: bash watch_download.sh

set -u

WORKSPACE="/workspace"
DOWNLOAD_PY="${WORKSPACE}/download_orderbook.py"
STATE_JSON="${WORKSPACE}/.watch_state.json"
LOG_FILE="${WORKSPACE}/.watch_download.log"
PID_FILE="${WORKSPACE}/.download.pid"
API_KEY_FILE="${WORKSPACE}/.cryptohft_key"
DATA_DIR="${WORKSPACE}/data"

# ========== 下载参数 (与 download_orderbook.py 中保持一致) ==========
START_DATE="2026-09-04"
END_DATE="2026-10-04"
ASSETS="BTC"
MARKET="both"
EXCHANGES="binance"

log() {
    local ts
    ts="$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
    echo "[${ts}] $*" | tee -a "${LOG_FILE}"
}

# -------- 0. 基础环境恢复 --------
ensure_pip() {
    local need_pip=0
    python3 -c "import requests, pandas, pyarrow, cryptohftdata" 2>/dev/null || need_pip=1
    if [[ ${need_pip} -eq 1 ]]; then
        log "pip installing requests pandas pyarrow cryptohftdata"
        pip install requests pandas pyarrow cryptohftdata -q 2>>"${LOG_FILE}" || {
            log "ERROR: pip install failed"
            return 1
        }
    fi
    return 0
}

ensure_api_key() {
    if [[ ! -s "${API_KEY_FILE}" ]]; then
        log "API key file missing, restoring..."
        echo '7b01d0d6c45dbf65f025ad40ec8aafb1f0540d64c9892c1f386cf3ebca47c9f8' \
            > "${API_KEY_FILE}"
        chmod 600 "${API_KEY_FILE}"
    fi
    return 0
}

ensure_scripts() {
    # download_orderbook.py - 必须是 git tracked 的
    if [[ ! -f "${DOWNLOAD_PY}" ]]; then
        log "download_orderbook.py missing, trying git checkout..."
        git -C "${WORKSPACE}" checkout download_orderbook.py 2>>"${LOG_FILE}" || {
            log "ERROR: git checkout failed - script must be committed first!"
            return 1
        }
    fi
    # watch_download.sh 自身
    if [[ ! -x "${BASH_SOURCE[0]}" ]]; then
        chmod +x "${BASH_SOURCE[0]}"
    fi
    return 0
}

# -------- 1. 并发守护 --------
is_download_running() {
    # 用 ps grep 匹配 download_orderbook.py, 排除自己的 grep 进程
    if pgrep -f "[d]ownload_orderbook.py" >/dev/null 2>&1; then
        return 0
    fi
    return 1
}

kill_stale_pid_file() {
    if [[ -f "${PID_FILE}" ]]; then
        local pid
        pid=$(cat "${PID_FILE}")
        if ! kill -0 "${pid}" 2>/dev/null; then
            log "stale pid file ${PID_FILE} (pid=${pid} not alive), removing"
            rm -f "${PID_FILE}"
        fi
    fi
}

# -------- 2. 启动下载 --------
start_download() {
    local api_key
    api_key=$(cat "${API_KEY_FILE}")

    mkdir -p "${DATA_DIR}"
    export CRYPTOHFTDATA_API_KEY="${api_key}"

    log "starting download_orderbook.py in background"
    nohup python3 "${DOWNLOAD_PY}" \
        --start "${START_DATE}" \
        --end "${END_DATE}" \
        --assets "${ASSETS}" \
        --market "${MARKET}" \
        --exchanges "${EXCHANGES}" \
        --output "${DATA_DIR}" \
        --api-key "${api_key}" \
        >> "${LOG_FILE}" 2>&1 &
    local pid=$!
    echo "${pid}" > "${PID_FILE}"
    log "download pid=${pid}"
    sleep 2
    if is_download_running; then
        log "download process confirmed alive"
    else
        log "ERROR: download process died immediately, check ${LOG_FILE}"
        rm -f "${PID_FILE}"
        return 1
    fi
    return 0
}

# -------- 3. 状态采集 --------
collect_expected() {
    # 31 days * 24h * 1 symbol * 2 exchanges (binance_spot + binance_futures)
    python3 -c "
from datetime import date
s='${START_DATE}'.split('-'); e='${END_DATE}'.split('-')
d0=date(int(s[0]),int(s[1]),int(s[2])); d1=date(int(e[0]),int(e[1]),int(e[2]))
days=(d1-d0).days+1
exchanges=2; symbols=1
print(days*24*exchanges*symbols)
"
}

collect_totals() {
    local total=0 spot=0 futures=0
    if [[ -d "${DATA_DIR}" ]]; then
        total=$(find "${DATA_DIR}" -name '*.parquet' 2>/dev/null | wc -l)
        spot=$(find "${DATA_DIR}" -path '*exchange=*_spot*' -name '*.parquet' 2>/dev/null | wc -l)
        futures=$(find "${DATA_DIR}" -path '*exchange=*_futures*' -name '*.parquet' 2>/dev/null | wc -l)
    fi
    echo "${total} ${spot} ${futures}"
}

collect_size() {
    if [[ -d "${DATA_DIR}" ]]; then
        du -sb "${DATA_DIR}" 2>/dev/null | awk '{print $1}'
    else
        echo 0
    fi
}

collect_disk() {
    df -B1 "${WORKSPACE}" | tail -1 | awk '{print $4}'
}

write_state() {
    local now total spot futures expected size disk_avail progress pct running status

    now="$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
    read -r total spot futures <<< "$(collect_totals)"
    expected=$(collect_expected)
    size=$(collect_size)
    disk_avail=$(collect_disk)

    progress=0
    pct=0
    if [[ "${expected}" -gt 0 ]]; then
        progress=$(echo "${total} ${expected}" | awk '{printf "%.0f", $1/$2*100}')
        pct=$(echo "${total} ${expected}" | awk '{printf "%.2f", $1/$2*100}')
    fi

    if is_download_running; then
        running="true"
        status="downloading"
    else
        running="false"
        if [[ "${total}" -ge "${expected}" && "${expected}" -gt 0 ]]; then
            status="complete"
        elif [[ "${total}" -gt 0 ]]; then
            status="idle_partial"
        else
            status="idle_empty"
        fi
    fi

    cat > "${STATE_JSON}" <<EOF
{
  "timestamp":   "${now}",
  "total":       ${total},
  "spot":        ${spot},
  "futures":     ${futures},
  "expected":    ${expected},
  "progress":    ${progress},
  "progress_pct": ${pct},
  "size_bytes":  ${size},
  "disk_avail_bytes": ${disk_avail},
  "download_running": ${running},
  "status":      "${status}",
  "params": {
    "start":     "${START_DATE}",
    "end":       "${END_DATE}",
    "assets":    "${ASSETS}",
    "market":    "${MARKET}",
    "exchanges": "${EXCHANGES}",
    "output":    "${DATA_DIR}"
  }
}
EOF
    log "state: total=${total}/${expected} spot=${spot} futures=${futures} progress=${progress}% status=${status} running=${running}"
}

# -------- main --------
main() {
    log "===== watch_download.sh triggered ====="

    ensure_pip          || { log "ABORT: pip";   exit 1; }
    ensure_api_key      || { log "ABORT: key";   exit 1; }
    ensure_scripts      || { log "ABORT: scripts"; exit 1; }
    kill_stale_pid_file

    if is_download_running; then
        log "download_orderbook.py already running - skip start"
    else
        log "download not running - starting"
        start_download || log "start_download failed, will retry next trigger"
    fi

    write_state
    log "===== watch done ====="
}

main "$@"
