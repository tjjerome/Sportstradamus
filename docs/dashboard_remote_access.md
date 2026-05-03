# Always-on Dashboard with Tailscale

How to keep the Sportstradamus Streamlit dashboard running on the UP Board
(Ubuntu 24.04) and reach it from anywhere via Tailscale.

## Architecture

```
phone / laptop  --(WireGuard over internet)-->  upboard.<tailnet>.ts.net  -->  127.0.0.1:8501  (Streamlit)
                       Tailscale mesh                    Tailscale Serve              systemd unit
```

Two layers:

1. **systemd** keeps the Streamlit process alive across reboots and crashes.
2. **Tailscale** gives the UP Board a stable hostname reachable from any
   device signed into your tailnet — no port forwarding, no DNS, no certs
   to manage.

## 1. Keep the dashboard alive — systemd unit

Drop this in `/etc/systemd/system/sportstradamus-dashboard.service`:

```ini
[Unit]
Description=Sportstradamus dashboard
After=network.target

[Service]
WorkingDirectory=/home/trevor/Sportstradamus
ExecStart=/home/trevor/.local/bin/poetry run streamlit run src/sportstradamus/dashboard_app.py --server.port 8501 --server.address 127.0.0.1 --server.headless true
Restart=always
RestartSec=5
User=trevor

[Install]
WantedBy=multi-user.target
```

Bind to `127.0.0.1` so Streamlit is only reachable through the local
loopback. Tailscale Serve (step 7) proxies the tailnet hostname into it.

Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now sportstradamus-dashboard
sudo systemctl status sportstradamus-dashboard
```

## 2. Tailscale account

Sign up at https://tailscale.com (Google / GitHub / email login). The
free **Personal** tier covers up to 100 devices.

## 3. Install Tailscale on the UP Board

```bash
curl -fsSL https://tailscale.com/install.sh | sh
```

Installs the apt repo, the `tailscaled` daemon, and the `tailscale` CLI.
The daemon auto-starts and is enabled in systemd by the installer.

## 4. Bring the UP Board onto your tailnet

```bash
sudo tailscale up --ssh --hostname=upboard
```

- Prints a URL — open it on a logged-in device, click **Connect** to
  authorize the machine.
- `--ssh` lets you SSH in from any tailnet device using Tailscale identity
  (kills the need to expose port 22 anywhere).
- `--hostname` sets the MagicDNS name.

Verify:

```bash
tailscale status
tailscale ip -4    # prints the 100.x.y.z tailnet IP
```

## 5. Enable MagicDNS (one-time)

Open https://login.tailscale.com/admin/dns → toggle **MagicDNS: on**.
The UP Board is now reachable as `upboard` from any tailnet device.

## 6. Install Tailscale on your phone / laptop

| Platform | Install |
|---|---|
| iOS / Android | App stores |
| macOS | App store or `brew install tailscale` |
| Windows | https://tailscale.com/download |
| Linux | Same `curl … \| sh` as step 3, then `sudo tailscale up` |

Sign in with the same account. Each device gets its own tailnet IP.

## 7. HTTPS via Tailscale Serve

Streamlit on `:8501` is plain HTTP. To upgrade to a real HTTPS URL with
an automatic Let's Encrypt cert from Tailscale:

```bash
sudo tailscale cert upboard.<your-tailnet>.ts.net
sudo tailscale serve --bg --https=443 http://127.0.0.1:8501
```

Get your exact tailnet name from `tailscale status` or the admin console.

Check it:

```bash
tailscale serve status
```

Now `https://upboard.<tailnet>.ts.net` proxies to Streamlit. The
`--bg` flag persists across `tailscaled` restarts.

## 8. Lock the OS firewall

Belt and suspenders — ensure 8501 is only reachable through the tailnet:

```bash
sudo ufw allow in on tailscale0
sudo ufw allow OpenSSH        # skip if you rely on `tailscale ssh`
sudo ufw default deny incoming
sudo ufw enable
```

## 9. (Optional) Public URL — Tailscale Funnel

Only enable if you want **non-tailnet** visitors. **No auth wall** — anyone
with the URL can hit Streamlit, which has no login.

In admin console → DNS → enable Funnel for the device. Then:

```bash
sudo tailscale funnel --bg 443
```

For a public link with an auth wall instead, prefer Cloudflare Tunnel +
Cloudflare Access and keep Tailscale private.

## Smoke test

From your phone with wifi off (LTE only):

```
https://upboard.<tailnet>.ts.net
```

If the dashboard loads, you're done. Reboot the UP Board and repeat —
both `tailscaled` and `sportstradamus-dashboard` should auto-start, and
`tailscale serve` state persists.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `tailscale status` shows the device offline | `sudo systemctl restart tailscaled` |
| HTTPS URL returns 502 | Check the dashboard is listening: `ss -tlnp \| grep 8501` |
| Dashboard reachable on LAN but not via tailnet | Streamlit bound to `127.0.0.1` is correct *only* if `tailscale serve` proxies it; otherwise bind to `0.0.0.0` |
| Cert error in browser | Re-run `sudo tailscale cert <hostname>`; check it matches the URL exactly |
| Dashboard not restarting after crash | `journalctl -u sportstradamus-dashboard -n 100` — check the `ExecStart` path matches your `poetry` install |

## Day-to-day commands

```bash
# Restart dashboard after a code change
sudo systemctl restart sportstradamus-dashboard

# Tail dashboard logs
journalctl -u sportstradamus-dashboard -f

# See who's on the tailnet
tailscale status

# Stop serving (e.g. for maintenance)
sudo tailscale serve reset
```
