#!/usr/bin/env bash
# (Re)start the audio-extract web GUI, detached so it survives this shell.
#
#   web/run.sh [start|stop|restart|status]   (default: restart)
#
# It serves on 127.0.0.1:$PORT and is exposed to the tailnet via `tailscale
# serve` (see web/README.md). Logs go to web/server.log.

set -euo pipefail

PORT="${PORT:-8730}"
HOST="${HOST:-127.0.0.1}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"      # .../audio-extract/web
ROOT="$(cd "$HERE/.." && pwd)"                            # .../audio-extract
LOG="$HERE/server.log"
PIDFILE="$HERE/server.pid"

find_pids() {
  # match our specific server process (avoids killing unrelated python)
  pgrep -f "web/server.py" 2>/dev/null || true
}

stop() {
  local pids; pids="$(find_pids)"
  if [ -n "$pids" ]; then
    echo "stopping web server (pids: $pids)"
    # shellcheck disable=SC2086
    kill $pids 2>/dev/null || true
    sleep 1
    pids="$(find_pids)"
    # shellcheck disable=SC2086
    [ -n "$pids" ] && kill -9 $pids 2>/dev/null || true
  else
    echo "no running web server"
  fi
  rm -f "$PIDFILE"
}

start() {
  if [ -n "$(find_pids)" ]; then
    echo "already running (pids: $(find_pids)); use '$0 restart'"
    status
    return 0
  fi
  echo "starting web server on http://$HOST:$PORT  (root: $ROOT)"
  cd "$ROOT"
  PORT="$PORT" HOST="$HOST" nohup uv run python web/server.py --port "$PORT" --host "$HOST" \
    >>"$LOG" 2>&1 &
  echo $! >"$PIDFILE"
  sleep 2
  status
}

status() {
  local pids; pids="$(find_pids)"
  if [ -n "$pids" ]; then
    echo "RUNNING (pids: $pids) on http://$HOST:$PORT"
    echo "  local check: curl -s -o /dev/null -w '%{http_code}\\n' http://$HOST:$PORT/"
    echo "  logs:        $LOG"
  else
    echo "NOT running"
  fi
}

case "${1:-restart}" in
  start)   start ;;
  stop)    stop ;;
  status)  status ;;
  restart) stop; start ;;
  *) echo "usage: $0 [start|stop|restart|status]"; exit 1 ;;
esac
