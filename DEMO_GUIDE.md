# NEXUS Hackathon Demo Guide

This guide provides the exact steps to launch the NEXUS Product Layer for the hackathon demo. It explicitly accounts for known local constraints, including a Vite path-resolution bug caused by the `#` character in the project folder name.

## 1. Prerequisites
- Python 3.10+
- Node.js 18+
- The repository must be on the latest `product-layer` commit.
- Ensure port `8000` and port `4173` are free.

## 2. Launching the Demo

### Automated Launch (Windows)
We have provided a reliable PowerShell helper script that opens both services in separate windows:
```powershell
.\scripts\start_demo.ps1
```

### Manual Launch

**A. Start the FastAPI Backend**
From the project root:
```bash
python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```
*Expected URL: http://127.0.0.1:8000*

**B. Start the Frontend (Preview Mode)**
Due to the `#` in the directory path, Vite's dev server (`npm run dev`) will fail to resolve dependencies. You MUST use the production build preview. From the `frontend/` directory:
```bash
cd frontend
npm run build
npm run preview
```
*Expected URL: http://localhost:4173*

## 3. Demo Sequence

To demonstrate the full capability of NEXUS to the judges, follow this sequence:

1. **Command Center**: Start at `http://localhost:4173/`. Explain the overall portfolio risk distribution and the distinction between diagnostic exposure (network evidence) and predictive operational risk. Point out that NEXUS relies on the Seed909/Week127 analytical baseline.
2. **Borrower Intelligence**: Click a borrower from the Command Center risk table. **B342** is in the Elevated tier (99.6th percentile) and makes the strongest example; **B10** sits in the Standard tier (65.6th percentile) and is a useful contrast. Demonstrate the individual risk metrics and the specific network exposure values. Emphasize that the risk score is a calibrated probability (Model C) whereas the percentile is a cohort rank.
3. **Network Intelligence**: Navigate to the Network view. Show the JLG star topology and how groups aggregate exposure.
4. **Ripple Simulator**: Open the Simulator (the "Simulate shock" action carries the selected borrower across).
   - The target is a **borrower**, not a group — enter e.g. **B10**, not G2.
   - Explain that both arms run on independent deep copies, so the frozen baseline is never mutated.
   - Recommended demo input: **B10, income reduction, 500, 12 weeks.** This produces a visible ripple — the target's buffer falls to zero with ~Rs 1,829 cumulative shortfall, and average peer cash drops from ~Rs 8,749 to ~Rs 8,338 as the group absorbs part of the gap.
5. **Intervention Studio**: Navigate to the Studio (the "Explore an intervention" action carries the borrower across).
   - Run a supported intervention on the same borrower and read the baseline / modeled scenario / delta table.
   - Quote the product's own wording: **"Modeled counterfactual under stated assumptions."** The UI makes no claim about preventing default or guaranteeing recovery, and there is no historical-efficacy data behind these numbers — they are the deterministic model's arithmetic.
   - If a borrower carries no outstanding due, the amount-due panel says so explicitly instead of drawing a flat zero line; use the cash-buffer panels in that case.
6. **Model & Impact Lab**: Conclude in the Lab. 
   - Show the transparency report for the frozen M2C evaluation.
   - Honestly disclose the inconclusiveness of the current evidence (due to low target incidence) and how NEXUS avoids fabricating performance claims.

## 4. How to Reset / Stop

**Stopping the Application:**
- If you used `start_demo.ps1`, simply close the two newly opened console windows.
- If launched manually, press `Ctrl+C` in each respective terminal.

**Resetting the State:**
The application relies on frozen data artifacts (`baseline_state.pkl` and `evaluation.json`). It does not mutate the disk during the demo. 
To reset the frontend state (e.g., clearing the Simulator or Intervention selections), simply **refresh the browser page (F5)**.

## 5. Troubleshooting

- **Frontend shows a blank screen or throws console errors:** Make sure you ran `npm run build` followed by `npm run preview`. Do NOT use `npm run dev` — it fails to resolve `/src/main.jsx` because of the `#` in the path, while still printing a "ready" banner.
- **Port 4173 already taken:** `npm run preview -- --port 4180` and open that port instead.
- **A different API port:** set `VITE_API_BASE` in `frontend/.env` (see `.env.example`) and rebuild; the base URL is defined once in `src/lib/api.js`.
- **Backend fails with "address already in use":** Find and kill the process holding port 8000:
  ```powershell
  netstat -ano | findstr :8000
  taskkill /PID <ProcessId> /F
  ```
- **"Network Error" when connecting frontend to backend:** Ensure the backend is explicitly running on `127.0.0.1:8000` (not just `localhost:8000`), as the centralized `VITE_API_BASE` points there.
