import { useEffect, useState } from "react";
import "./App.css";

function App() {
  const [message, setMessage] = useState("Connecting to backend...");
  const [status, setStatus] = useState("checking");

  useEffect(() => {
    fetch("http://127.0.0.1:8000/")
      .then((response) => {
        if (!response.ok) {
          throw new Error("Backend returned an error");
        }

        return response.json();
      })
      .then((data) => {
        setMessage(data.message);
        setStatus("connected");
      })
      .catch((error) => {
        console.error("Backend connection error:", error);
        setMessage("Could not connect to backend");
        setStatus("error");
      });
  }, []);

  return (
    <div className="app">
      <header className="header">
        <h1>Interactive Model Studio</h1>
        <p>Visual machine learning platform</p>
      </header>

      <main className="main">
        <div className="welcome-card">
          <h2>Frontend ↔ Backend Test</h2>

          <div className={`status ${status}`}>
            {status === "connected"
              ? "✓ Connected"
              : status === "error"
              ? "✗ Error"
              : "⟳ Checking..."}
          </div>

          <p>{message}</p>
        </div>
      </main>
    </div>
  );
}

export default App;