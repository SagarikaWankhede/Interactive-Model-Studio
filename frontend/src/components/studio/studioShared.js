import { createContext } from "react";
import { Database, Droplets, Scale, Tags, Trash2, Scissors, Brain, BarChart3 } from "lucide-react";

/** node_id -> { errors: string[], warnings: string[] } — provided by the Studio page. */
export const NodeIssuesContext = createContext({});

export const NODE_ICONS = {
  dataset: Database,
  impute: Droplets,
  scale: Scale,
  encode: Tags,
  drop_columns: Trash2,
  split: Scissors,
  model: Brain,
  evaluate: BarChart3,
};
