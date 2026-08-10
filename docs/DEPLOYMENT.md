# SHAMSU Deployment Guide

This deployment profile gives SHAMSU a backend service and MySQL database while keeping the local desktop workflow unchanged.

## Files
- `backend/Dockerfile`: containerizes the FastAPI backend.
- `docker-compose.yml`: runs backend + private MySQL network.
- `.env.deploy.example`: template for deployment secrets.

## Start
```powershell
copy .env.deploy.example .env.deploy
notepad .env.deploy
```

Replace every `replace_with_...` value with a private password, then run:

```powershell
docker compose up --build
```

## Check
```powershell
curl.exe http://127.0.0.1:8080/api/health
```

Expected result: JSON with `"status":"ok"`. When MySQL is healthy, `history_store` should show MySQL.

## Security Notes
- `.env.deploy` must stay local and must not be committed.
- MySQL is not published to the host by this compose file.
- Put HTTPS/TLS in front of production deployments with NGINX, Caddy, Cloudflare, or your hosting provider.
- Keep Ollama private to the host or trusted network.
