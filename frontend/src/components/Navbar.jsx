import { useEffect, useState } from "react";
import { NavLink, Link } from "react-router-dom";
import { 
  Database, 
  Sliders,
  Cpu,
  History,
  Workflow, 
  BarChart3, 
  CheckCircle2, 
  AlertCircle, 
  Loader2,
  Sparkles,
  Zap,
  LogIn,
  LogOut,
  User
} from "lucide-react";
import api from "../api/client";
import { useAuth } from "../context/AuthContext";

export default function Navbar() {
  const { user, isAuthenticated, logout } = useAuth();
  const [status, setStatus] = useState("checking"); // 'checking' | 'connected' | 'error'
  const [backendMessage, setBackendMessage] = useState("");

  const checkConnection = () => {
    setStatus("checking");
    api.healthCheck()
      .then((data) => {
        setStatus("connected");
        setBackendMessage(data?.message || "Connected");
      })
      .catch(() => {
        setStatus("error");
        setBackendMessage("Offline");
      });
  };

  useEffect(() => {
    checkConnection();
    const interval = setInterval(checkConnection, 15000);
    return () => clearInterval(interval);
  }, []);

  return (
    <header className="navbar">
      <div className="nav-container">
        <Link to="/upload" className="brand-section" style={{ textDecoration: "none" }}>
          <div className="brand-icon">
            <Sparkles className="icon-pulse" size={20} />
          </div>
          <div className="brand-text">
            <span className="brand-title">Interactive Model Studio</span>
            <span className="brand-tagline">Visual ML & Codegen Platform</span>
          </div>
        </Link>

        <nav className="nav-links">
          <NavLink 
            to="/upload" 
            className={({ isActive }) => `nav-link ${isActive ? "active" : ""}`}
          >
            <Database size={15} />
            <span>Datasets</span>
          </NavLink>
          <NavLink 
            to="/preprocessing" 
            className={({ isActive }) => `nav-link ${isActive ? "active" : ""}`}
          >
            <Sliders size={15} />
            <span>Recipe</span>
          </NavLink>
          <NavLink 
            to="/models" 
            className={({ isActive }) => `nav-link ${isActive ? "active" : ""}`}
          >
            <Cpu size={15} />
            <span>Models</span>
          </NavLink>
          <NavLink 
            to="/pipeline" 
            className={({ isActive }) => `nav-link ${isActive ? "active" : ""}`}
          >
            <Workflow size={15} />
            <span>Pipeline</span>
          </NavLink>
          <NavLink 
            to="/experiments" 
            className={({ isActive }) => `nav-link ${isActive ? "active" : ""}`}
          >
            <History size={15} />
            <span>Experiments</span>
          </NavLink>
          <NavLink 
            to="/dashboard" 
            className={({ isActive }) => `nav-link ${isActive ? "active" : ""}`}
          >
            <BarChart3 size={15} />
            <span>Dashboard</span>
          </NavLink>
          <NavLink 
            to="/edge-lab" 
            className={({ isActive }) => `nav-link ${isActive ? "active" : ""}`}
            style={{ position: "relative" }}
          >
            <Zap size={15} color="#10b981" />
            <span style={{ fontWeight: 600 }}>Edge Lab</span>
            <span style={{ 
              position: "absolute", 
              top: "-4px", 
              right: "-6px", 
              background: "#10b981", 
              color: "#fff", 
              fontSize: "0.6rem", 
              padding: "1px 4px", 
              borderRadius: "4px",
              fontWeight: 700 
            }}>
              ONNX
            </span>
          </NavLink>
        </nav>

        <div className="status-section" style={{ display: "flex", alignItems: "center", gap: "10px" }}>
          {/* Health status pill */}
          <div 
            className={`status-pill ${status}`} 
            title={`Backend: ${backendMessage}`}
            onClick={checkConnection}
            style={{ cursor: "pointer" }}
          >
            {status === "connected" && <CheckCircle2 size={13} className="text-success" />}
            {status === "checking" && <Loader2 size={13} className="animate-spin text-warning" />}
            {status === "error" && <AlertCircle size={13} className="text-danger" />}
            <span style={{ fontSize: "0.75rem" }}>
              {status === "connected" ? "Live" : status === "checking" ? "Checking..." : "Offline"}
            </span>
          </div>

          {/* User Auth Section */}
          {isAuthenticated ? (
            <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
              <div 
                style={{ 
                  display: "flex", 
                  alignItems: "center", 
                  gap: "6px", 
                  background: "rgba(99, 102, 241, 0.15)", 
                  padding: "5px 10px", 
                  borderRadius: "20px",
                  fontSize: "0.8rem",
                  color: "var(--accent-primary)",
                  fontWeight: 500
                }}
              >
                <User size={13} />
                <span>{user?.username}</span>
              </div>
              <button 
                onClick={logout} 
                className="btn btn-secondary btn-sm" 
                title="Sign Out"
                style={{ padding: "5px 8px" }}
              >
                <LogOut size={13} />
              </button>
            </div>
          ) : (
            <Link 
              to="/login" 
              className="btn btn-primary btn-sm" 
              style={{ fontSize: "0.8rem", padding: "6px 12px", textDecoration: "none", gap: "5px" }}
            >
              <LogIn size={13} />
              <span>Sign In</span>
            </Link>
          )}
        </div>
      </div>
    </header>
  );
}
