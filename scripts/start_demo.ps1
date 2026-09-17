param(
    [switch]$BuildFrontend = $false
)

Write-Host "======================================================"
Write-Host " STARTING NEXUS HACKATHON DEMO ENVIRONMENT"
Write-Host "======================================================"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$rootDir = (Resolve-Path "$scriptDir\..").Path

Set-Location -Path $rootDir

if ($BuildFrontend) {
    Write-Host "`n[1/3] Building Frontend (Required for initial setup)..."
    Set-Location -Path "$rootDir\frontend"
    npm run build
    Set-Location -Path $rootDir
}

Write-Host "`n[2/3] Starting FastAPI Backend on 127.0.0.1:8000..."
Start-Process -FilePath "python" -ArgumentList "-m uvicorn src.api.main:app --host 127.0.0.1 --port 8000" -WindowStyle Normal

Write-Host "`n[3/3] Starting React/Vite Frontend (Preview Mode) on localhost:4173..."
Start-Process -FilePath "npm" -ArgumentList "run preview" -WorkingDirectory "$rootDir\frontend" -WindowStyle Normal

Write-Host "`n======================================================"
Write-Host " NEXUS IS LIVE"
Write-Host "======================================================"
Write-Host " Frontend URL: http://localhost:4173"
Write-Host " Backend URL:  http://127.0.0.1:8000"
Write-Host ""
Write-Host " Note: Due to local folder constraints, the dev server is bypassed."
Write-Host "       The frontend is being served in production preview mode."
Write-Host ""
Write-Host " To stop the demo, close the two newly opened console windows."
Write-Host "======================================================"
