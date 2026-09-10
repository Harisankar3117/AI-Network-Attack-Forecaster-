# Stage 1: Build the React Application
FROM node:20-alpine as frontend-builder
WORKDIR /app/dashboard
COPY dashboard/package*.json ./
RUN npm install
COPY dashboard/ .
RUN npm run build

# Stage 2: Build the Python Backend
FROM python:3.10-slim
WORKDIR /app

# Install CPU-only PyTorch first. 
# We use PyTorch index as primary to get the CPU version, and PyPI as extra to get its dependencies (typing-extensions).
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple

# Copy requirements and install the rest of dependencies
COPY requirements.txt .
# Remove torch from requirements.txt temporarily during install to avoid re-downloading the heavy CUDA version
RUN sed -i '/torch==/d' requirements.txt && pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY . .

# Copy built frontend from Stage 1 into the backend directory
COPY --from=frontend-builder /app/dashboard/dist /app/dashboard/dist

# Expose port (Render sets $PORT dynamically)
EXPOSE 8000

# Start the FastAPI server using the dynamically assigned PORT (default 8000)
CMD uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}
