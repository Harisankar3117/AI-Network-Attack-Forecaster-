from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class ScenarioRequest(BaseModel):
    scenario: str = Field(..., description="Name of the scenario to load (e.g., 'wednesday', 'friday', 'march1')")
    k_steps: int = Field(5, description="Number of steps to forecast")

class RiskSummary(BaseModel):
    peak_forecast_risk_probability: float
    peak_forecast_risk_level: str
    average_forecast_risk_probability: float
    average_forecast_risk_level: str
    risk_trend: str
    earliest_high_risk_window: Optional[int]
    predicted_attack_count: int
    warning_message: str

class ForecastResponse(BaseModel):
    scenario: str
    current_probability: float
    current_risk_level: str
    forecast_windows: List[Dict[str, Any]]
    summary: RiskSummary
    mitre_context: List[Dict[str, Any]]
    explainability: List[Dict[str, Any]]
    raw_features: Dict[str, Any]

class BenchmarkResponse(BaseModel):
    lstm_results: Dict[str, Dict[str, Any]]
    logistic_regression_baseline: Dict[str, Any]
    limitations: str
