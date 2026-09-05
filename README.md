# CNN Visualizer

Local setup and run commands for this project on Windows:

```powershell
npm install
```

Run `npm install` only the first time, or after dependency changes.

For local development with Vite hot reload:

```powershell
npm run dev
```

Open the local URL shown by Vite in your browser.

For a production build:

```powershell
npm run build
```

## Backend

The PyTorch integration backend now lives in `backend/`.

Install and run it from the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r backend\requirements.txt
python backend\app.py
```

The backend defaults to:

```text
http://127.0.0.1:8000
```

Initial checks:

```text
GET http://127.0.0.1:8000/api/health
GET http://127.0.0.1:8000/api/models
```

Reference project:
https://github.com/poloclub/cnn-explainer
