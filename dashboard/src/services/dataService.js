import Papa from 'papaparse';

// Local CSV Paths inside public/data
const DATA_FILES = {
  wednesday: '/data/lstm_attack_predictions.csv',
  friday: '/data/lstm_unseen_friday_predictions.csv',
  march1: '/data/lstm_unseen_march1_predictions.csv',
  summary: '/data/cross_day_lstm_summary.csv'
};

/**
 * Downsamples data to a maximum number of points for efficient chart rendering.
 */
function downsample(data, maxPoints = 1500) {
  if (data.length <= maxPoints) return data;
  const step = Math.ceil(data.length / maxPoints);
  const sampled = [];
  for (let i = 0; i < data.length; i += step) {
    sampled.push(data[i]);
  }
  // Ensure the last point is included for completion
  if (sampled[sampled.length - 1] !== data[data.length - 1]) {
    sampled.push(data[data.length - 1]);
  }
  return sampled;
}

export const loadPredictions = (scenarioKey) => {
  return new Promise((resolve, reject) => {
    const fileUrl = DATA_FILES[scenarioKey];
    if (!fileUrl) {
      reject(new Error(`Invalid scenario key: ${scenarioKey}`));
      return;
    }

    Papa.parse(fileUrl, {
      download: true,
      header: true,
      dynamicTyping: true,
      skipEmptyLines: true,
      complete: (results) => {
        const rawData = results.data;
        const sampledData = downsample(rawData, 1500);
        
        // Compute basic stats
        let total = rawData.length;
        let attackCount = 0;
        let maxProb = 0;
        let sumProb = 0;

        rawData.forEach(row => {
          if (row.Actual_Attack === 1) attackCount++;
          if (row.Attack_Probability > maxProb) maxProb = row.Attack_Probability;
          sumProb += (row.Attack_Probability || 0);
        });

        const stats = {
          totalFlows: total,
          attackFlows: attackCount,
          benignFlows: total - attackCount,
          attackRate: total > 0 ? (attackCount / total) * 100 : 0,
          avgProb: total > 0 ? sumProb / total : 0,
          maxProb: maxProb
        };

        resolve({
          rawCount: total,
          data: sampledData,
          stats
        });
      },
      error: (err) => {
        reject(err);
      }
    });
  });
};
