#!/bin/bash
# Boot wrapper: pull latest from origin/main if reachable, then run dissman.py.
# Network failures are non-fatal — we still launch with whatever's on disk.

cd /home/dissman/Documents/app || exit 1

LOG=/home/dissman/Documents/app/boot-update.log
{
  echo "===== $(date -Is) boot ====="
  if timeout 20 git fetch --quiet origin main; then
    LOCAL=$(git rev-parse HEAD)
    REMOTE=$(git rev-parse origin/main)
    if [ "$LOCAL" != "$REMOTE" ]; then
      echo "updating: $LOCAL -> $REMOTE"
      git reset --hard origin/main
    else
      echo "already up to date at $LOCAL"
    fi
  else
    echo "git fetch failed or timed out; launching offline"
  fi
} >>"$LOG" 2>&1

exec /usr/bin/python3 /home/dissman/Documents/app/dissman.py
