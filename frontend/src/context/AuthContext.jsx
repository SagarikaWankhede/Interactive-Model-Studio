import { createContext, useContext, useState, useEffect } from "react";
import api from "../api/client";

const AuthContext = createContext(null);

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null);
  const [token, setToken] = useState(() => {
    return typeof localStorage !== "undefined" ? localStorage.getItem("ims_token") : null;
  });
  const [loading, setLoading] = useState(true);

  // Validate session on mount if token exists
  useEffect(() => {
    if (token) {
      api.getMe()
        .then((userData) => {
          setUser(userData);
        })
        .catch(() => {
          // Token invalid or expired
          localStorage.removeItem("ims_token");
          setToken(null);
          setUser(null);
        })
        .finally(() => {
          setLoading(false);
        });
    } else {
      setLoading(false);
    }
  }, [token]);

  const login = async (username, password) => {
    const res = await api.login({ username, password });
    if (res.token) {
      localStorage.setItem("ims_token", res.token);
      setToken(res.token);
      setUser(res.user);
      return res.user;
    }
    throw new Error("Login failed. No token received.");
  };

  const register = async (username, email, password, full_name) => {
    const res = await api.register({ username, email, password, full_name });
    if (res.token) {
      localStorage.setItem("ims_token", res.token);
      setToken(res.token);
      setUser(res.user);
      return res.user;
    }
    throw new Error("Registration failed. No token received.");
  };

  const logout = () => {
    localStorage.removeItem("ims_token");
    setToken(null);
    setUser(null);
  };

  return (
    <AuthContext.Provider
      value={{
        user,
        token,
        isAuthenticated: !!user,
        loading,
        login,
        register,
        logout,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error("useAuth must be used within an AuthProvider");
  }
  return context;
}

export default AuthContext;
