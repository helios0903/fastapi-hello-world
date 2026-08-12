from fastapi import FastAPI

app = FastAPI()


@app.get("/")
def root():
    return {
        "message": "FastAPI is running!",
        "try": "/hello/YourName",
        "docs": "/docs",
    }


@app.get("/hello/{name}")
def hello(name: str):
    return {"message": f"Hello, World {name}"}
