from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI(
    title="Interactive Model Studio API",
    description="Backend API for our visual machine learning platform",
    version="1.0.0"
)


# Allow the React frontend to communicate with the backend.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home():
    return {
        "message": "Interactive Model Studio backend is running!"
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy"
    }