---
name: dissman-pi-ssh
description: Use when running commands on the Dissman Raspberry Pi over SSH — deploying a merged push, restarting or inspecting dissman.service, reading journal/boot logs, or opening a printer/prompt tuning window with prompt_testing/lab.py. Covers the claude-ssh-dissman MCP exec tool, reachability failures, and the printer-ownership and deploy-on-reboot constraints.
---

# Operating the Dissman Pi over SSH

## Overview

The Pi (hostname `dissman`, mDNS `dissman.local`, user `dissman`) is reachable through the
`claude-ssh-dissman` MCP server, which exposes an `exec` tool that runs shell commands on it.
Registration, credentials, and per-device setup live in `docs/ssh-mcp-setup.md`. Reachability
needs the same LAN (mDNS `.local`) or a routable host/VPN.

## When to use

- Deploying a change already merged to `main` (trigger the pull + restart)
- Restarting or checking the status of `dissman.service`
- Reading app or boot logs
- A printer or prompt-tuning session driving `prompt_testing/lab.py`

**When NOT to assume it works:** the maintainer usually has no live Pi access. If `exec` fails
with `getaddrinfo ENOTFOUND dissman.local` or a connection refused/timeout, you are simply not
on the Pi's LAN — **say so and stop; do not loop retrying the same call.** Nothing is broken;
the box just isn't reachable from here. Any change committed to `main` still deploys on the
next reboot regardless (see below).

## Quick reference

Run each via the `claude-ssh-dissman` `exec` tool.

| Task | Command |
|------|---------|
| Deploy latest `main` | `sudo systemctl restart dissman.service` (start.sh pulls origin/main first) |
| Follow app logs | `sudo journalctl -u dissman.service -b --no-pager -f` |
| Boot-pull log | `tail /home/dissman/Documents/app/boot-update.log` |
| Stop app (frees printer USB) | `sudo systemctl stop dissman` |
| Start app | `sudo systemctl start dissman` |
| App directory | `/home/dissman/Documents/app/` |

## Constraints (do not violate)

- **The printer USB is owned by `dissman.service`.** Any direct printing — `print.py`, or
  `lab.py --test print` — needs `sudo systemctl stop dissman` first and `start` after.
- **Deploy is pull-on-reboot.** `start.sh` runs `git fetch && git reset --hard origin/main`
  on every service start. Tracked repo files deploy on the next restart; steps that require
  SSH-only actions (apt/pip installs, systemd unit edits, file copies outside the repo) may
  never happen — flag them. See CLAUDE.md "Pi Access Constraint".
- **The systemd unit is NOT tracked** (`/etc/systemd/system/dissman.service`); editing it
  needs SSH. `start.sh` IS tracked — edit it in the repo and it deploys on reboot (keep LF
  line endings or the shebang breaks).

## Reference

- Setup, credentials, OS-specific paths, reachability checks: `docs/ssh-mcp-setup.md`
- Deploy flow and architecture: CLAUDE.md
