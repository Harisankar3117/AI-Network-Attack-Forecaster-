import React, { createContext, useState } from 'react';

export const ForecastContext = createContext(null);

export const ForecastProvider = ({ children }) => {
  const [forecastData, setForecastData] = useState(null);

  const clearForecastData = () => {
    setForecastData(null);
  };

  return (
    <ForecastContext.Provider value={{ forecastData, setForecastData, clearForecastData }}>
      {children}
    </ForecastContext.Provider>
  );
};
