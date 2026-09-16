import { 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  Legend, 
  ResponsiveContainer 
} from "recharts";

const MODEL_COLORS = ["#4f46e5", "#06b6d4", "#10b981", "#f59e0b", "#ec4899"];

export default function FeatureImportanceChart({ experiments = [] }) {
  if (!experiments || experiments.length === 0) {
    return <div className="text-muted text-sm text-center py-4">No experiments selected for feature importance comparison.</div>;
  }

  // Aggregate all unique features from the experiments
  const featureMap = {};
  experiments.forEach((exp) => {
    const importances = exp.feature_importances || [];
    importances.forEach((item) => {
      if (!featureMap[item.feature]) {
        featureMap[item.feature] = { feature: item.feature };
      }
      featureMap[item.feature][exp.model_name] = item.importance;
    });
  });

  const chartData = Object.values(featureMap).slice(0, 7); // Top 7 features

  return (
    <div className="feature-importance-container">
      <div style={{ width: "100%", height: 340 }}>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            data={chartData}
            layout="vertical"
            margin={{ top: 20, right: 30, left: 40, bottom: 20 }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" horizontal={false} />
            <XAxis 
              type="number" 
              stroke="#64748b" 
              domain={[0, 0.5]} 
              tickFormatter={(v) => `${(v * 100).toFixed(0)}%`} 
            />
            <YAxis 
              type="category" 
              dataKey="feature" 
              stroke="#64748b" 
              tick={{ fontSize: 12, fill: "#334155" }} 
              width={110}
            />
            <Tooltip
              formatter={(val, name) => [`${(val * 100).toFixed(1)}%`, name]}
              contentStyle={{ backgroundColor: "#0f172a", borderRadius: "8px", color: "#fff", border: "none" }}
            />
            <Legend />
            {experiments.map((exp, index) => (
              <Bar
                key={exp.id}
                dataKey={exp.model_name}
                fill={MODEL_COLORS[index % MODEL_COLORS.length]}
                radius={[0, 4, 4, 0]}
              />
            ))}
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
