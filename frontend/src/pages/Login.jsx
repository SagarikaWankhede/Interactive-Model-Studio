import { useState } from "react";
import { useNavigate, Link } from "react-router-dom";
import { useAuth } from "../context/AuthContext";
import { 
  LogIn, 
  UserPlus, 
  ShieldCheck, 
  Database, 
  Sparkles, 
  AlertCircle, 
  CheckCircle2, 
  Loader2,
  Lock,
  Mail,
  User
} from "lucide-react";

export default function Login() {
  const navigate = useNavigate();
  const { login, register, isAuthenticated, user, logout } = useAuth();

  const [mode, setMode] = useState("login"); // 'login' | 'register'
  const [username, setUsername] = useState("");
  const [email, setEmail] = useState("");
  const [fullName, setFullName] = useState("");
  const [password, setPassword] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [successMsg, setSuccessMsg] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(null);
    setSuccessMsg(null);
    setLoading(true);

    try {
      if (mode === "login") {
        await login(username, password);
        setSuccessMsg("Signed in successfully! Redirecting...");
        setTimeout(() => navigate("/upload"), 800);
      } else {
        await register(username, email, password, fullName);
        setSuccessMsg("Account created! Redirecting to studio...");
        setTimeout(() => navigate("/upload"), 800);
      }
    } catch (err) {
      setError(err.message || "Authentication failed. Please check your credentials.");
    } finally {
      setLoading(false);
    }
  };

  const handleQuickDemo = async () => {
    setError(null);
    setLoading(true);
    try {
      // First try to log in demo account
      try {
        await login("demo_scientist", "Password123!");
      } catch {
        // If not created yet, register it!
        await register("demo_scientist", "scientist@modelstudio.ai", "Password123!", "Lead ML Researcher");
      }
      setSuccessMsg("Logged in as Demo ML Researcher! Redirecting...");
      setTimeout(() => navigate("/upload"), 800);
    } catch (err) {
      setError(err.message || "Failed to initialize demo session.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="page-container" style={{ maxWidth: "600px", margin: "40px auto" }}>
      {/* Brand Header */}
      <div className="text-center mb-8">
        <div style={{ display: "inline-flex", padding: "12px", background: "rgba(99, 102, 241, 0.1)", borderRadius: "16px", marginBottom: "12px", color: "var(--accent-primary)" }}>
          <Sparkles size={32} />
        </div>
        <h1 className="page-title" style={{ fontSize: "2rem", marginBottom: "8px" }}>
          Interactive Model Studio
        </h1>
        <p className="page-subtitle" style={{ margin: "0 auto" }}>
          Create an account to save your private datasets, visual pipelines, and trained ONNX models.
        </p>
      </div>

      {isAuthenticated ? (
        <div className="card text-center p-8">
          <div style={{ display: "inline-flex", padding: "12px", background: "rgba(16, 185, 129, 0.15)", borderRadius: "50%", marginBottom: "16px", color: "#10b981" }}>
            <CheckCircle2 size={36} />
          </div>
          <h2 style={{ fontSize: "1.3rem", fontWeight: 600, marginBottom: "8px" }}>
            You are currently signed in
          </h2>
          <p style={{ color: "var(--text-muted)", marginBottom: "20px" }}>
            Signed in as <strong>{user?.username}</strong> ({user?.email})
          </p>
          <div className="flex justify-center gap-4">
            <Link to="/upload" className="btn btn-primary">
              <Database size={16} /> Go to Datasets
            </Link>
            <button onClick={logout} className="btn btn-secondary">
              Sign Out
            </button>
          </div>
        </div>
      ) : (
        <div className="card shadow-lg" style={{ border: "1px solid rgba(255, 255, 255, 0.1)" }}>
          {/* Tab Selector */}
          <div style={{ display: "flex", borderBottom: "1px solid var(--border-color)", marginBottom: "24px" }}>
            <button
              type="button"
              onClick={() => { setMode("login"); setError(null); }}
              style={{
                flex: 1,
                padding: "14px",
                background: "transparent",
                border: "none",
                borderBottom: mode === "login" ? "2px solid var(--accent-primary)" : "none",
                color: mode === "login" ? "var(--text-primary)" : "var(--text-muted)",
                fontWeight: mode === "login" ? 600 : 400,
                cursor: "pointer",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                gap: "8px"
              }}
            >
              <LogIn size={16} />
              <span>Sign In</span>
            </button>
            <button
              type="button"
              onClick={() => { setMode("register"); setError(null); }}
              style={{
                flex: 1,
                padding: "14px",
                background: "transparent",
                border: "none",
                borderBottom: mode === "register" ? "2px solid var(--accent-primary)" : "none",
                color: mode === "register" ? "var(--text-primary)" : "var(--text-muted)",
                fontWeight: mode === "register" ? 600 : 400,
                cursor: "pointer",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                gap: "8px"
              }}
            >
              <UserPlus size={16} />
              <span>Create Account</span>
            </button>
          </div>

          {/* Quick Demo Button */}
          <div style={{ marginBottom: "20px", background: "rgba(99, 102, 241, 0.08)", padding: "12px 16px", borderRadius: "10px", border: "1px dashed rgba(99, 102, 241, 0.3)" }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <div>
                <span style={{ fontSize: "0.85rem", fontWeight: 600, color: "var(--accent-primary)" }}>Testing or Evaluating?</span>
                <p style={{ fontSize: "0.75rem", color: "var(--text-muted)", margin: 0 }}>Jump in instantly with a pre-configured demo account.</p>
              </div>
              <button
                type="button"
                onClick={handleQuickDemo}
                disabled={loading}
                className="btn btn-primary btn-sm"
                style={{ fontSize: "0.8rem", whiteSpace: "nowrap" }}
              >
                {loading ? <Loader2 size={13} className="animate-spin" /> : <Sparkles size={13} />}
                <span>One-Click Demo</span>
              </button>
            </div>
          </div>

          {/* Error & Success Alerts */}
          {error && (
            <div style={{ background: "rgba(239, 68, 68, 0.12)", border: "1px solid rgba(239, 68, 68, 0.3)", borderRadius: "8px", padding: "12px", color: "#f87171", fontSize: "0.85rem", marginBottom: "16px", display: "flex", alignItems: "center", gap: "8px" }}>
              <AlertCircle size={16} />
              <span>{error}</span>
            </div>
          )}

          {successMsg && (
            <div style={{ background: "rgba(16, 185, 129, 0.12)", border: "1px solid rgba(16, 185, 129, 0.3)", borderRadius: "8px", padding: "12px", color: "#34d399", fontSize: "0.85rem", marginBottom: "16px", display: "flex", alignItems: "center", gap: "8px" }}>
              <CheckCircle2 size={16} />
              <span>{successMsg}</span>
            </div>
          )}

          <form onSubmit={handleSubmit} style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
            {mode === "register" && (
              <div>
                <label style={{ display: "block", fontSize: "0.85rem", fontWeight: 500, marginBottom: "6px" }}>
                  Full Name (Optional)
                </label>
                <div style={{ position: "relative" }}>
                  <User size={15} style={{ position: "absolute", left: "12px", top: "12px", color: "var(--text-muted)" }} />
                  <input
                    type="text"
                    value={fullName}
                    onChange={(e) => setFullName(e.target.value)}
                    placeholder="e.g. Dr. Jane Doe"
                    className="input-field"
                    style={{ paddingLeft: "36px", width: "100%" }}
                  />
                </div>
              </div>
            )}

            <div>
              <label style={{ display: "block", fontSize: "0.85rem", fontWeight: 500, marginBottom: "6px" }}>
                {mode === "login" ? "Username or Email" : "Username"} *
              </label>
              <div style={{ position: "relative" }}>
                <User size={15} style={{ position: "absolute", left: "12px", top: "12px", color: "var(--text-muted)" }} />
                <input
                  type="text"
                  required
                  value={username}
                  onChange={(e) => setUsername(e.target.value)}
                  placeholder={mode === "login" ? "e.g. scientist or user@example.com" : "e.g. scientist"}
                  className="input-field"
                  style={{ paddingLeft: "36px", width: "100%" }}
                />
              </div>
            </div>

            {mode === "register" && (
              <div>
                <label style={{ display: "block", fontSize: "0.85rem", fontWeight: 500, marginBottom: "6px" }}>
                  Email Address *
                </label>
                <div style={{ position: "relative" }}>
                  <Mail size={15} style={{ position: "absolute", left: "12px", top: "12px", color: "var(--text-muted)" }} />
                  <input
                    type="email"
                    required
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    placeholder="e.g. scientist@company.com"
                    className="input-field"
                    style={{ paddingLeft: "36px", width: "100%" }}
                  />
                </div>
              </div>
            )}

            <div>
              <label style={{ display: "block", fontSize: "0.85rem", fontWeight: 500, marginBottom: "6px" }}>
                Password *
              </label>
              <div style={{ position: "relative" }}>
                <Lock size={15} style={{ position: "absolute", left: "12px", top: "12px", color: "var(--text-muted)" }} />
                <input
                  type="password"
                  required
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  placeholder="••••••••"
                  className="input-field"
                  style={{ paddingLeft: "36px", width: "100%" }}
                />
              </div>
            </div>

            <button
              type="submit"
              disabled={loading}
              className="btn btn-primary"
              style={{ marginTop: "8px", width: "100%", padding: "12px", justifyContent: "center" }}
            >
              {loading ? (
                <>
                  <Loader2 size={16} className="animate-spin" />
                  <span>Processing...</span>
                </>
              ) : mode === "login" ? (
                <>
                  <LogIn size={16} />
                  <span>Sign In</span>
                </>
              ) : (
                <>
                  <UserPlus size={16} />
                  <span>Create Account</span>
                </>
              )}
            </button>
          </form>

          {/* Privacy Footnote */}
          <div style={{ marginTop: "24px", paddingTop: "16px", borderTop: "1px solid var(--border-color)", textAlign: "center", fontSize: "0.78rem", color: "var(--text-muted)", display: "flex", alignItems: "center", justifyContent: "center", gap: "6px" }}>
            <ShieldCheck size={14} color="#10b981" />
            <span>Passswords securely hashed with PBKDF2-HMAC-SHA256. Datasets isolated per user.</span>
          </div>
        </div>
      )}
    </div>
  );
}
