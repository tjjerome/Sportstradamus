# Always-on Dashboard with Tailscale

How to keep the Streamlit dashboard running on a home server and reach it from
your own devices, without exposing it to the internet.

> **Warning — the dashboard has no authentication.** Anyone who can reach the
> port can read and click everything. Keep it tailnet-only: bind Streamlit to
> localhost and front it with `tailscale serve`, which only devices signed into
> your tailnet can reach. Never expose it to the public internet — no port
> forwarding, no public reverse proxy, and never `tailscale funnel`.

## Architecture

```
phone / laptop --(WireGuard)--> <host>.<tailnet>.ts.net --> 127.0.0.1:8501 (Streamlit)
                Tailscale mesh       Tailscale Serve             systemd unit
```

1. **systemd** keeps the Streamlit process alive across reboots and crashes.
2. **Tailscale Serve** gives the box a stable HTTPS hostname reachable only
   from your tailnet — no port forwarding, no DNS, no certs to manage.

## 1. systemd unit

Drop this in `/etc/systemd/system/<unit>.service`:

```ini
[Unit]
Description=Sportstradamus dashboard
After=network.target

[Service]
WorkingDirectory=<repo-checkout>
Environment=STREAMLIT_SERVER_PORT=8501
Environment=STREAMLIT_SERVER_ADDRESS=127.0.0.1
Environment=STREAMLIT_SERVER_HEADLESS=true
ExecStart=<path-to-poetry> run dashboard
Restart=always
RestartSec=5
User=<user>

[Install]
WantedBy=multi-user.target
```

`poetry run sportstradamus dashboard` is the canonical entry point (`sportstradamus.dashboard:run`);
it launches `streamlit run` on the package's `dashboard/app.py` with the file
watcher off, so an unattended server never shows the "Source file changed,
rerun?" popup. Don't point `ExecStart` at a `.py` directly — the launcher
centralizes those production flags.

The bind comes from the `Environment=` lines (the `dashboard` command takes no
flags). `STREAMLIT_SERVER_ADDRESS=127.0.0.1` keeps Streamlit on loopback so it
is only reachable through Tailscale Serve.

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now <unit>
```

## 2. Join the tailnet

Install Tailscale (https://tailscale.com/download) on the server and on each
device you browse from, sign both into the same tailnet, then on the server:

```bash
sudo tailscale up --ssh --hostname=<host>
```

`--ssh` allows SSH from tailnet devices using Tailscale identity, so port 22
never needs exposing. Enable MagicDNS in the admin console so `<host>`
resolves by name.

## 3. HTTPS via Tailscale Serve

```bash
sudo tailscale cert <host>.<tailnet>.ts.net
sudo tailscale serve --bg --https=443 http://127.0.0.1:8501
```

Now `https://<host>.<tailnet>.ts.net` proxies to Streamlit with an automatic
Let's Encrypt cert. `--bg` persists across `tailscaled` restarts; check with
`tailscale serve status`.

If the box runs an OS firewall, allow inbound only on the `tailscale0`
interface (plus SSH if you don't rely on `tailscale ssh`) and default-deny
the rest.

## Smoke test

From your phone with wifi off (LTE only), open
`https://<host>.<tailnet>.ts.net`. Then reboot the server and repeat — both
`tailscaled` and `<unit>` should auto-start, and `tailscale serve` state
persists.

## Troubleshooting

| Symptom | Fix |
|---|---|
| Device offline in `tailscale status` | `sudo systemctl restart tailscaled` |
| HTTPS URL returns 502 | Check Streamlit is listening: `ss -tlnp \| grep 8501` |
| Reachable on LAN but not via tailnet | Loopback bind is correct *only* with `tailscale serve` proxying it |
| Dashboard not restarting after crash | `journalctl -u <unit> -n 100`; confirm the poetry path in `ExecStart` and that `poetry run sportstradamus dashboard` starts by hand |

## Day-to-day

```bash
sudo systemctl restart <unit>    # after a code change
journalctl -u <unit> -f          # tail logs
sudo tailscale serve reset       # stop serving for maintenance
```
