# SSH MCP Setup (`claude-ssh-dissman`)

How to give a Claude Code instance SSH access to the Dissman Raspberry Pi via the
`claude-ssh-dissman` MCP server. This is the tool referenced throughout `CLAUDE.md`
("SSH accessible via MCP tool `claude-ssh-dissman`").

The server is [jasondsmith72/claude-ssh-server](https://github.com/jasondsmith72/claude-ssh-server),
a stdio MCP server that opens an SSH session to a host and exposes it as MCP tools.

## What it connects to

| Setting   | Value                                                         |
|-----------|--------------------------------------------------------------|
| host      | `dissman.local` (mDNS — requires same LAN as the Pi)         |
| user      | `dissman`                                                     |
| password  | `<PI_PASSWORD>` (real value kept out of this repo — see note) |

## Adding it to a new device

The build output (`node_modules/`, `build/`) is platform-specific — **do not copy it
between machines.** Clone and build fresh on each device.

### 1. Install Node.js

Windows (winget):

```powershell
winget install OpenJS.NodeJS.LTS
```

macOS (`brew install node`) or Linux (distro package / nvm) work the same. Any current
LTS is fine; this was verified on Node `v24.x` / npm `11.x`.

### 2. Clone and build the server

Pick a **stable location outside the app repo** so it isn't disturbed by app updates.
This setup used `C:\Users\rmich\mcp-servers\claude-ssh-server`.

```bash
git clone https://github.com/jasondsmith72/claude-ssh-server.git
cd claude-ssh-server
npm install
npm run build          # produces build/index.js
```

### 3. Register the MCP in Claude Code

There are two ways to register the server. **They register the same server under the
same name (`claude-ssh-dissman`)** — pick one; don't run both on the same machine or
they collide. See "Which registration to use" below.

#### Option A — Project scope via committed `.mcp.json` (recommended)

The repo ships a project-scoped [`.mcp.json`](../.mcp.json) at its root that declares this
server. Claude Code offers it to anyone who opens the repo (after a one-time trust prompt).
The committed file holds **no secret and no OS-specific path** — it references environment
variables you set per device:

```json
{
  "mcpServers": {
    "claude-ssh-dissman": {
      "command": "node",
      "args": [
        "${DISSMAN_SSH_MCP_PATH}",
        "--host=${DISSMAN_HOST:-dissman.local}",
        "--user=${DISSMAN_USER:-dissman}",
        "--password=${DISSMAN_PI_PASSWORD}"
      ]
    }
  }
}
```

Set these environment variables on the device (values persist across shells/reboots):

| Variable               | Required | Value                                                        |
|------------------------|----------|--------------------------------------------------------------|
| `DISSMAN_SSH_MCP_PATH` | yes      | Absolute path to this device's built `build/index.js`        |
| `DISSMAN_PI_PASSWORD`  | yes      | The Pi's password (**never** commit this)                    |
| `DISSMAN_HOST`         | no       | Defaults to `dissman.local`; override for VPN/real IP        |
| `DISSMAN_USER`         | no       | Defaults to `dissman`                                        |

Windows (persistent, user scope) — set once, then restart Claude Code:

```powershell
setx DISSMAN_SSH_MCP_PATH "C:/Users/rmich/mcp-servers/claude-ssh-server/build/index.js"
setx DISSMAN_PI_PASSWORD  "<PI_PASSWORD>"
```

macOS/Linux: add `export DISSMAN_SSH_MCP_PATH=...` and `export DISSMAN_PI_PASSWORD=...`
to your shell profile (`~/.zshrc`, `~/.bashrc`).

#### Option B — User scope via CLI (per-device, not shared)

Use the **absolute path** to the built `build/index.js` for this device's OS. This bakes
the password into `~/.claude.json` on that one machine and does not travel with the repo.

```bash
claude mcp add claude-ssh-dissman --scope user -- \
  node /absolute/path/to/claude-ssh-server/build/index.js \
  --host=dissman.local --user=dissman --password=<PI_PASSWORD>
```

Windows example:

```powershell
claude mcp add claude-ssh-dissman --scope user -- node "C:/Users/rmich/mcp-servers/claude-ssh-server/build/index.js" --host=dissman.local --user=dissman --password=<PI_PASSWORD>
```

Alternatively, hand-edit the device's `~/.claude.json` `mcpServers` block with the same
shape — just fix the absolute path and `command`/`args` for the new OS.

#### Which registration to use

- **A machine cloning the repo fresh:** use Option A — set the two env vars and go; the
  `.mcp.json` is already there.
- **A machine that already has the user-scope server (Option B):** the project `.mcp.json`
  and the user-scope entry share the name `claude-ssh-dissman` and will conflict. Remove
  the user-scope one to switch to the shared file: `claude mcp remove claude-ssh-dissman --scope user`.

> **Secret hygiene:** `.mcp.json` is committed, so it must never contain the password —
> only the `${DISSMAN_PI_PASSWORD}` placeholder. The real password lives in a per-device
> env var and in your own records, never in git. Prefer key-based auth over a password
> (see Gotchas).

### 4. Verify

```bash
claude mcp list        # claude-ssh-dissman should report: ✔ Connected
```

The MCP tools load at **session start**, so after registering you must restart Claude
Code before the `claude-ssh-dissman` tools appear in-session. `✔ Connected` confirms the
server launches and completes the MCP handshake; it does not by itself prove the SSH
login or LAN path. Sanity-check Pi reachability:

```bash
ping dissman.local                                   # mDNS resolves?
# Windows: Test-NetConnection dissman.local -Port 22 # SSH port open?
```

## Gotchas

- **Absolute, OS-specific path.** On macOS/Linux it's `/Users/...` or `/home/...`, and
  `command`/`args` change accordingly. The Windows path above won't work elsewhere.
- **`dissman.local` is mDNS.** The device must be on the **same LAN** as the Pi for
  `.local` to resolve. Remote use needs the Pi reachable another way (VPN, real
  hostname/IP) — change `--host` accordingly.
- **The password sits in plaintext** in `~/.claude.json` args. Replicating it onto
  another machine spreads that credential. Prefer key-based auth, or at minimum treat
  `~/.claude.json` as a secret on each device. The real password is intentionally **not**
  committed to this repo — substitute it for `<PI_PASSWORD>` from your own records.
