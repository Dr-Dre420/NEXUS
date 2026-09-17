# NEXUS Hackathon Demo Guide

This guide provides the exact steps to launch the NEXUS Product Layer for the hackathon demo. It explicitly accounts for known local constraints, including a Vite path-resolution bug caused by the `#` character in the project folder name.

## 1. Prerequisites
- Python 3.10+
- Node.js 18+
- The repository must be exactly on commit `2ea41fcc` (the frozen Product Layer state).
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
2. **Borrower Intelligence**: Click on a high-risk borrower (e.g., **B10** or **B342**). Demonstrate the individual risk metrics and the specific network exposure values. Emphasize that the risk score is a calibrated probability (Model C) whereas the percentile is a cohort rank.
3. **Network Intelligence**: Navigate to the Network view. Show the JLG star topology and how groups aggregate exposure.
4. **Ripple Simulator**: Open the Simulator. 
   - Explain that the baseline is immutable.
   - Run a counterfactual shock to a high-exposure group (e.g., G69 or G2). 
   - Show how the shock propagates through the network, updating the counterfactual state without altering the historical baseline.
5. **Intervention Studio**: Navigate to the Studio.
   - Run a targeted intervention on the same high-risk entities.
   - Note the careful phrasing: interventions are "assumptions based on historical efficacy," not causal guarantees.
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

- **Frontend shows a blank screen or throws console errors:** Make sure you ran `npm run build` followed by `npm run preview`. Do NOT use `npm run dev`.
- **Backend fails with "address already in use":** Find and kill the process holding port 8000:
  ```powershell
  netstat -ano | findstr :8000
  taskkill /PID <ProcessId> /F
  ```
- **"Network Error" when connecting frontend to backend:** Ensure the backend is explicitly running on `127.0.0.1:8000` (not just `localhost:8000`), as the centralized `VITE_API_BASE` points there.
