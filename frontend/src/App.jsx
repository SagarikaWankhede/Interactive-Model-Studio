import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import { AuthProvider } from "./context/AuthContext";
import Navbar from "./components/Navbar";
import Upload from "./pages/Upload";
import DatasetView from "./pages/DatasetView";
import Preprocessing from "./pages/Preprocessing";
import ModelConfig from "./pages/ModelConfig";
import RunExperiment from "./pages/RunExperiment";
import ExperimentList from "./pages/ExperimentList";
import RestoreExperiment from "./pages/RestoreExperiment";
import Pipeline from "./pages/Pipeline";
import Dashboard from "./pages/Dashboard";
import EdgeLab from "./pages/EdgeLab";
import Login from "./pages/Login";
import "./App.css";

export default function App() {
  return (
    <BrowserRouter>
      <AuthProvider>
        <div className="app-shell">
          <Navbar />
          <main className="main-content">
            <Routes>
              <Route path="/" element={<Navigate to="/upload" replace />} />
              <Route path="/upload" element={<Upload />} />
              <Route path="/dataset/:id" element={<DatasetView />} />
              <Route path="/preprocessing" element={<Preprocessing />} />
              <Route path="/models" element={<ModelConfig />} />
              <Route path="/experiments/run" element={<RunExperiment />} />
              <Route path="/experiments" element={<ExperimentList />} />
              <Route path="/restore/:id" element={<RestoreExperiment />} />
              <Route path="/pipeline" element={<Pipeline />} />
              <Route path="/dashboard" element={<Dashboard />} />
              <Route path="/edge-lab" element={<EdgeLab />} />
              <Route path="/login" element={<Login />} />
              <Route path="/register" element={<Login />} />
              <Route path="*" element={<Navigate to="/upload" replace />} />
            </Routes>
          </main>
        </div>
      </AuthProvider>
    </BrowserRouter>
  );
}
