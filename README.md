# ByteStorm - AI Network Attack Forecaster (SIH 26153)

An advanced Predictive Defense Console for analyzing, forecasting, and mitigating network attacks. Built using an LSTM-based world model trained on the CIC-IDS2018 dataset, ByteStorm provides proactive security operations through an interactive Command Center.

## Key Features
- **K-Step Probabilistic Forecasting**: Uses LSTM to forecast the progression of network state parameters up to $K$ steps into the future.
- **Threat Trajectory Analysis**: Visualizes how an attack develops over time, providing proactive early warnings before the peak probability of an attack hits.
- **Real-Time Command Center**: A sleek React dashboard for tracking active threats, evaluating network health, and assessing model benchmarking.
- **Custom CSV Offline Inference Mode**: Allows security operators to upload custom packet capture summaries (CSV) directly to the console for offline inference and prediction without modifying backend code.
- **Explainable AI (XAI)**: Identifies driving factors of the forecasted risks to assist security analysts in root-cause identification.

## Tech Stack
- **Backend Model**: PyTorch, Scikit-learn, Pandas, NumPy
- **Backend API**: FastAPI, Uvicorn
- **Frontend Dashboard**: React, Vite, Tailwind CSS, Recharts, Lucide React

## Setup Instructions

### 1. Backend Setup
Make sure you have Python 3 installed. Install the backend dependencies:
```powershell
pip install -r requirements.txt
```

Run the backend FastAPI server:
```powershell
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```
The API will be available at `http://localhost:8000`.

### 2. Frontend Setup
Make sure you have Node.js installed. Navigate to the `dashboard` directory and install the React dependencies:
```powershell
cd dashboard
npm install
```

Run the frontend Vite server:
```powershell
npm run dev
```
The Command Center will open in your browser at `http://localhost:5173`.

## Custom CSV Inference (Demo Mode)
To run custom offline inference:
1. Open the application dashboard and navigate to **Demo Mode**.
2. Click the file upload button and select a pre-processed network traffic CSV file. (Example files can be found in `data/sample/`).
3. Click **Run Offline Forecast**. The React app will stream the CSV to the FastAPI backend.
4. The LSTM model will predict the $K$-step probability of attack and automatically redirect you to the **Attack Forecast** page to visualize the results!

## License
MIT License