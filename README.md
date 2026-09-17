# NEXUS

**NEXUS Microfinance Analytics Platform** — Milestone 2C Baseline

## Overview

NEXUS is a microfinance analytics engine with:
- **Backend**: FastAPI + Python analytical models (M0 → M2C)
- **Frontend**: React + Vite dashboard
- **Analytical Core**: LightGBM models with temporal cross-validation, propagation features, and attribution

## Quick Start

### Backend
```bash
pip install -r requirements.txt
uvicorn src.api.main:app --reload
# API: http://localhost:8000
# Health: GET http://localhost:8000/health
```

### Frontend
```bash
cd frontend
npm install
npm run dev
# UI: http://localhost:5173
```

### Tests
```bash
pytest tests/ -q
```

## Version

See [`VERSION`](./VERSION) — current baseline is **M2C-FROZEN**.

## Development

See [`DEVELOPMENT.md`](./DEVELOPMENT.md) for parallel development rules (Antigravity / Claude Code worktree setup).
