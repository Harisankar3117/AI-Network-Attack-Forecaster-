from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import inference
from api.services.inference_service import inference_service

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        inference_service.load_artifacts()
    except Exception as e:
        print(f"Warning during startup: {e}")
    yield

app = FastAPI(
    title="ByteStorm- Forecast API",
    description="Backend API for AI Network Attack Forecasting (SIH 26153)",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS for local React development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health_check():
    if not inference_service.is_loaded:
        try:
            inference_service.load_artifacts()
        except Exception:
            pass
    return {
        "status": "online",
        "model_loaded": inference_service.is_loaded,
        "scaler_loaded": inference_service.scaler is not None,
        "version": "1.0.0"
    }

from fastapi.staticfiles import StaticFiles
from starlette.responses import FileResponse
import os

# Register routes
app.include_router(inference.router, prefix="/api/v1")

# Serve React App (Frontend)
if os.path.exists("dashboard/dist"):
    app.mount("/assets", StaticFiles(directory="dashboard/dist/assets"), name="assets")
    
    @app.get("/{full_path:path}")
    async def serve_frontend(full_path: str):
        # Allow API routes to pass through
        if full_path.startswith("api/"):
            return None
            
        file_path = os.path.join("dashboard/dist", full_path)
        if os.path.isfile(file_path):
            return FileResponse(file_path)
        return FileResponse("dashboard/dist/index.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
