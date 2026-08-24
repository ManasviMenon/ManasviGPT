# Deployment Notes

Reference for the 2026-08-24 outage/migration. Read this first if the site goes down again.

## Current setup (as of this migration)

- **Backend host**: Google Cloud Compute Engine, `e2-micro` VM, zone `us-west1-b` (Always Free tier — must stay this machine type + one of us-west1/us-central1/us-east1 to remain free).
- **Static IP**: `35.203.162.235` (reserved as static, won't change on restart).
- **SSH access**: GCP Console → Compute Engine → VM instances → `manasvigpt-backend` → **SSH** button (browser terminal, uses GCP OS Login). This is the only session with passwordless `sudo` — a plain SSH key added manually to `~/.ssh/authorized_keys` can log in but cannot `sudo` without a password.
- **App location on VM**: `/home/manasvimenon2003/ManasviGPT/` (venv at `venv/`, FAISS index at `faiss_index/`, secrets in `.env` — not in git).
- **Process manager**: systemd service `manasvigpt` (`/etc/systemd/system/manasvigpt.service`), runs `gunicorn -b 127.0.0.1:5000 -w 1 --timeout 120 app:app`. `Restart=always`.
- **Reverse proxy**: Nginx, config at `/etc/nginx/sites-available/manasvigpt`, proxies `api.manasvigpt.online` → `127.0.0.1:5000`.
- **SSL**: Let's Encrypt via Certbot, auto-renews via a systemd timer certbot installed. Cert expires 2026-11-22 — should auto-renew before then; if not, run `sudo certbot renew` manually.
- **DNS**: Cloudflare, `api.manasvigpt.online` A record → `35.203.162.235`, proxy status set to **"DNS only" (grey cloud)** — deliberately not proxied through Cloudflare, to keep Certbot's domain verification and TLS simple.
- **Swap**: 2GB swapfile at `/swapfile` — added because this VM only has 1GB RAM and the embedding model needs headroom.
- **Billing safety net**: GCP budget alert set at $2 on the `manasvigpt` billing account (alerts only, no spend cap).

## Common troubleshooting

**Site down / fetch fails on manasvigpt.online:**
1. Check the backend directly: `curl https://api.manasvigpt.online/health`
2. If that times out/fails to connect: check the VM is actually running (GCP Console → VM instances) and that DNS still points to `35.203.162.235` (`nslookup api.manasvigpt.online`).
3. If the VM's up but health check fails: SSH in (via the Console SSH button) and check `sudo systemctl status manasvigpt` and `sudo systemctl status nginx`.
4. Check app logs: `sudo journalctl -u manasvigpt -n 100 --no-pager`
5. Restart if needed: `sudo systemctl restart manasvigpt` (first request after restart takes ~20s — cold model load, this is normal, not a bug).

**Chat returns "Groq error: model not found" or similar:**
Groq periodically deprecates models. Check current available models with your key:
```
curl -s https://api.groq.com/openai/v1/models -H "Authorization: Bearer $GROQ_API_KEY"
```
Update the `"model"` field in `query_chatbot.py`'s `groq_answer()` function, then `git pull`/copy the file to the VM and `sudo systemctl restart manasvigpt`.

**Chat returns "I don't have enough information to answer that" a lot:**
This is the app's own relevance-threshold logic (`context_relevance_score < 0.52` in `query_chatbot.py`), not necessarily a bug — it's intentionally conservative to avoid hallucinated answers. Confirmed working correctly for direct questions (e.g. "who is manasvi") during this migration; some phrasings for project-specific questions returned this fallback even with exact section-name matches, which may be worth revisiting/tuning later, but wasn't something this migration broke (same retrieval logic, same threshold, carried over as-is from before).

## What happened (root cause history)

1. Backend was originally on AWS EC2 (`t3.micro`, `ap-southeast-2`), behind Nginx, with a Route through Cloudflare.
2. AWS's account-level "Free Plan" (a 6-month account-wide trial, distinct from the classic per-resource Free Tier) expired and closed the account. The EC2 instance became unreachable — this is what caused the original `manasvigpt.online` fetch failures (Cloudflare 522 — origin unreachable).
3. Account was reactivated by upgrading to a paid AWS plan, but ongoing EC2 costs (~$15-17 AUD/month) weren't wanted, so we migrated instead.
4. Recovered `faiss_index/` (the built FAISS index) and `.env` (holds `GROQ_API_KEY`) from the old EC2 instance before stopping it — these were never committed to this git repo (only `data/`, the original knowledge-base source text, is genuinely lost — the built index survived, the raw source `.txt` files used to build it did not).
5. Migrated to a GCP `e2-micro` VM (Always Free tier, no ongoing cost as long as it stays within Always Free limits) — see "Current setup" above.
6. Along the way, fixed a corrupted `.gitignore` (had UTF-16-encoded garbage lines from some past encoding mishap) and a hardcoded Groq model (`llama-3.1-8b-instant`) that Groq had deprecated, replaced with `openai/gpt-oss-20b`.
7. The original repo's `README.md` still describes the old Render-based plan and is now stale/aspirational — the actual deployment target has changed twice since (Render → AWS EC2 → GCP), and was never Render in practice.

## Files worth knowing about

- `faiss_index/` — the built FAISS index + `texts.pkl` (175 chunks, sections: faq, background, experience, taxi_project, airbnb_project, skills, aiesec, coindcx). Committed to git since there's no `data/` source to rebuild it from.
- `.env` — **not in git**. Lives locally and on the VM only. Contains `GROQ_API_KEY`.
- `build_faiss.py` — would rebuild `faiss_index/` from a `data/` folder of `.txt` files, but that `data/` folder doesn't exist anywhere anymore (lost, not just gitignored). If the knowledge base ever needs to change, `data/` has to be recreated from scratch first.
