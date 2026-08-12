# fastapi-hello-world

A minimal [FastAPI](https://fastapi.tiangolo.com/) app.

## Endpoints

- `GET /` — status message
- `GET /hello/{name}` — returns `{"message": "Hello, World {name}"}`
- `GET /docs` — interactive API docs

## Run locally

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

Then open http://127.0.0.1:8000

## Deploy on Render

This repo includes `render.yaml`. Create a new Web Service on
[Render](https://render.com) from this repo (free plan) and it deploys
automatically.
