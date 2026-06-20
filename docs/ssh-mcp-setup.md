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

### 3. Register the MCP in Claude Code (user scope)

Use the **absolute path** to the built `build/index.js` for this device's OS.

```bash
claude mcp add claude-ssh-dissman --scope user -- \
  node /absolute/path/to/claude-ssh-server/build/index.js \
  --host=dissman.local --user=dissman --password=<PI_PASSWORD>
```

Windows example (this device):

```powershell
claude mcp add claude-ssh-dissman --scope user -- node "C:/Users/rmich/mcp-servers/claude-ssh-server/build/index.js" --host=dissman.local --user=dissman --password=<PI_PASSWORD>
```

Alternatively, hand-edit the device's `~/.claude.json` `mcpServers` block with the same
shape — just fix the absolute path and `command`/`args` for the new OS.

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
