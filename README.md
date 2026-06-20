# pi-logs

Runtime logs pushed from the Raspberry Pi kiosk. **Do not commit application
code or merge this branch into `main`.**

This is an orphan branch with no shared history. On each boot, the Pi's
`start.sh` publishes the previous session's log file here (a git worktree on
this branch), then starts a fresh log. Because the kiosk is power-cycled rather
than shut down cleanly, each session's log lands here on the *next* boot.

Logs are written line-buffered to disk during the run (`python3 -u ... | tee`),
so they survive a hard power-off without depending on journald persistence.
