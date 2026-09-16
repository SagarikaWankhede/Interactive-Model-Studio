import { Award, Zap, Target, Sparkles } from "lucide-react";

export default function BestModelBadge({ type = "accuracy", label }) {
  const configs = {
    accuracy: {
      defaultLabel: "Top Accuracy",
      icon: Award,
      className: "badge-winner-emerald",
    },
    f1_score: {
      defaultLabel: "Best F1 Score",
      icon: Target,
      className: "badge-winner-indigo",
    },
    train_time: {
      defaultLabel: "Fastest Training",
      icon: Zap,
      className: "badge-winner-cyan",
    },
    overall: {
      defaultLabel: "Best Overall",
      icon: Sparkles,
      className: "badge-winner-gold",
    }
  };

  const config = configs[type] || configs.accuracy;
  const Icon = config.icon;

  return (
    <span className={`best-model-badge ${config.className}`}>
      <Icon size={12} className="badge-icon" />
      <span>{label || config.defaultLabel}</span>
    </span>
  );
}
