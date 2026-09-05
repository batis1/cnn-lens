# CNN Visualizer Backend

This is the single backend API that will replace the separate AI_Test_UI Flask
apps during integration.

## Setup

From the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r backend\requirements.txt
```

If `AI_Test_UI` is not beside this project, point the backend to it:

```powershell
$env:AI_TEST_UI_ROOT="C:\Users\Mohammed Batis\Documents\Interpretability_projects\AI_Test_UI"
```

## Run

```powershell
python backend\app.py
```

Default URL:

```text
http://127.0.0.1:8000
```

Optional debug mode:

```powershell
$env:CNN_EXPLAINER_BACKEND_DEBUG="1"
python backend\app.py
```

Useful checks:

```text
GET http://127.0.0.1:8000/api/health
GET http://127.0.0.1:8000/api/models
```

## Current Scope

Implemented now:

- Health endpoint
- Model registry endpoint for the AI_Test_UI `.pt` files
- PyTorch `/api/explain` endpoint for registered CNN models and MNIST-style images
- Placeholder endpoints for test, data generation, and result saving

Next integration task:

- Add the frontend adapter that consumes `/api/explain`.

Example explain request:

```powershell
Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/api/explain" `
  -Method Post `
  -ContentType "application/json" `
  -Body '{"modelId":"cnn-net-28-ori","imagePath":"test_original/1_7.jpg"}'
```
