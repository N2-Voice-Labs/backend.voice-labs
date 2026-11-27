# AWS deployment (UNAI)

One EC2 server runs everything: project **Golden Hour Code**, account `509614632997`,
Region `ap-southeast-2`, instance `i-080bf3ac86470e88e` (`unai-server`, Elastic IP `13.55.24.212`).

| URL | What | Source |
|---|---|---|
| https://api.13-55-24-212.sslip.io | Platform API (`/health`) | this repo, `main` |
| https://site.13-55-24-212.sslip.io | marketing site, `/inha` portal, workspace | `frontend`, `main` |
| https://portal.13-55-24-212.sslip.io | legacy portal | archived `frontend.voice-labs`, frozen |

nginx on the host terminates HTTPS (Let's Encrypt via certbot, auto-renewing) and proxies to
`127.0.0.1:8000` (Platform), `:3001` (site) and `:3000` (legacy portal).

## How code gets there

Merging to `main` deploys automatically:

- **backend.voice-labs**: after the `test` workflow passes on `main`, `deploy.yml` runs.
- **frontend**: every push to `main` runs `deploy.yml`.

The workflow assumes the IAM role `unai-github-deploy` through GitHub OIDC (no AWS keys in
GitHub) and runs the SSM document `UNAI-Deploy`, which can only execute
`deploy/aws/deploy.sh backend|site` on the server. The script pulls `main`, rebuilds with
Docker Compose, waits for health and prunes old images. A failed build keeps the old
containers running; the workflow log shows the server output.

Manual deploy (Actions tab > deploy > Run workflow), or from a laptop with the `unai` profile:

```sh
aws ssm send-command --profile unai --region ap-southeast-2 \
  --instance-ids i-080bf3ac86470e88e --document-name UNAI-Deploy --parameters App=site
```

## Debugging on the server

No SSH key is needed; use Session Manager:

```sh
aws ssm start-session --profile unai --region ap-southeast-2 --target i-080bf3ac86470e88e
sudo -i
cd /opt/unai/backend && docker compose ps && docker compose logs -f --tail=100 platform runtime
docker logs -f unai-frontends-site-1
```

Server layout (`/opt/unai`): `backend/` and `frontend/` are git checkouts of `main` (do not
edit them on the server; the next deploy resets them). Secrets stay on the server only:
`backend/.env` and `site.env` (`PLATFORM_URL`, `PLATFORM_API_KEY`, `DEMO_AGENTS`,
`WORKSPACE_PASSWORD`). After editing `.env`, recreate: `docker compose up -d --force-recreate platform runtime`.

`watchdog.sh` runs every minute (`/etc/cron.d/unai-watchdog`) and restarts Platform when
Docker marks it unhealthy, saving its logs to `/opt/unai/logs/`. Check
`/opt/unai/logs/watchdog.log` to see whether hangs recur.

Cost: the server bills while running (~$80/month against the $100 credits). Stop it when
not demoing: `aws ec2 stop-instances --profile unai --region ap-southeast-2 --instance-ids i-080bf3ac86470e88e`.
