const API_BASE = import.meta.env.VITE_API_BASE || '/api/v1';

export const fetchSystemHealth = async () => {
    const res = await fetch(`/health`);
    if (!res.ok) throw new Error('Failed to fetch health');
    return res.json();
};

export const fetchModelStatus = async () => {
    const res = await fetch(`${API_BASE}/model/status`);
    if (!res.ok) throw new Error('Failed to fetch model status');
    return res.json();
};

export const fetchForecast = async (scenario = 'wednesday', k_steps = 5) => {
    const res = await fetch(`${API_BASE}/forecast/${scenario}?k_steps=${k_steps}`);
    if (!res.ok) throw new Error('Failed to fetch forecast');
    return res.json();
};

export const uploadCsv = async (file, k_steps = 5) => {
    const formData = new FormData();
    formData.append('file', file);
    
    const res = await fetch(`${API_BASE}/upload?k_steps=${k_steps}`, {
        method: 'POST',
        body: formData
    });
    if (!res.ok) throw new Error('Failed to upload CSV');
    return res.json();
};

export const fetchBenchmark = async () => {
    const res = await fetch(`${API_BASE}/benchmark`);
    if (!res.ok) throw new Error('Failed to fetch benchmark');
    return res.json();
};
