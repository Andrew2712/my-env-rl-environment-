---
title: My Env
emoji: 🔐
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---

# Password Policy Environment

A black-box password generation environment built with FastAPI.

## API Endpoints
- POST /reset — Start new episode
- POST /step — Submit password
- GET /state — Full internal state
- GET /health — Liveness check
