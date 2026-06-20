#!/bin/bash
# Boot wrapper: pull latest from origin/main if reachable, then run dissman.py.
# Network failures are non-fatal — we still launch with whatever's on disk.

APP=/home/dissman/Documents/app
LOGDIR=/home/dissman/Documents/app-logs   # git worktree checked out on 'pi-logs'

cd "$APP" || exit 1

LOG="$APP/boot-update.log"
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

# --- Publish the previous session's logs to the 'pi-logs' branch -----------
# The kiosk is power-cycled, not shut down, so there's no exit hook. Instead,
# every boot we push whatever the last session left on disk, then start a fresh
# log below. All of this is best-effort and must NEVER block the launch.
{
  echo "===== $(date -Is) log-sync ====="

  # First boot (or after a re-clone): create the logs worktree from origin.
  if [ ! -e "$LOGDIR/.git" ]; then
    if timeout 20 git fetch --quiet origin pi-logs; then
      git worktree prune
      git worktree add --track -B pi-logs "$LOGDIR" origin/pi-logs \
        && echo "created logs worktree at $LOGDIR"
    else
      echo "could not fetch pi-logs; skipping log sync this boot"
    fi
  fi

  if [ -e "$LOGDIR/.git" ]; then
    # Carry the boot/update log over too, so it's visible remotely.
    cp -f "$LOG" "$LOGDIR/boot-update.log" 2>/dev/null

    git -C "$LOGDIR" add -A
    if ! git -C "$LOGDIR" diff --cached --quiet; then
      git -C "$LOGDIR" commit --quiet -m "pi logs $(hostname) $(date -Is)"
    fi

    # This Pi is the only writer and filenames are unique per boot, so a
    # fast-forward push is the norm; rebase just in case the branch moved.
    if timeout 20 git -C "$LOGDIR" fetch --quiet origin pi-logs; then
      git -C "$LOGDIR" rebase --quiet origin/pi-logs || git -C "$LOGDIR" rebase --abort
    fi
    timeout 30 git -C "$LOGDIR" push --quiet origin pi-logs \
      || echo "push failed; will retry next boot"
  fi
} >>"$LOG" 2>&1

# --- Launch, capturing this session's output to disk ----------------------
# python3 -u + line-buffered tee so each line is flushed to disk immediately
# and survives a hard power-off. tee also forwards stdout to journald as before.
if [ -d "$LOGDIR" ]; then
  SESSION_LOG="$LOGDIR/$(hostname)-$(date -u +%Y%m%dT%H%M%SZ).log"
  exec /usr/bin/python3 -u "$APP/dissman.py" > >(stdbuf -oL tee -a "$SESSION_LOG") 2>&1
else
  # No logs worktree (e.g. first boot was offline) — launch without on-disk capture.
  exec /usr/bin/python3 -u "$APP/dissman.py"
fi
