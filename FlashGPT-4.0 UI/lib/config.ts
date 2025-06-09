export const API_CONFIG = {
  BASE_URL: process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8080/api",
  ENDPOINTS: {
    AUTH: {
      LOGIN: "/auth/login",
      REGISTER: "/auth/register",
      LOGOUT: "/auth/logout",
      REFRESH: "/auth/refresh",
      PROFILE: "/auth/profile",
      OAUTH_URL: (provider: string) => `/auth/oauth/${provider}/url`,
      OAUTH_CALLBACK: "/auth/oauth/callback",
    },
    COMPARISONS: {
      LIST: "/comparisons",
      CREATE: "/comparisons",
      DELETE: (id: string) => `/comparisons/${id}`,
      GET: (id: string) => `/comparisons/${id}`,
    },
    LLM: {
      GENERATE: "/llm/generate",
      MODELS: "/llm/models",
    },
    USER: {
      PREFERENCES: "/user/preferences",
    },
  },
}

export const OAUTH_CONFIG = {
  REDIRECT_URI: process.env.NEXT_PUBLIC_OAUTH_REDIRECT_URI || `${window?.location?.origin}/auth/callback`,
  PROVIDERS: {
    GOOGLE: {
      name: "Google",
      id: "google",
    },
    GITHUB: {
      name: "GitHub",
      id: "github",
    },
  },
}

export const getAuthHeaders = () => {
  const token = localStorage.getItem("auth_token")
  return token ? { Authorization: `Bearer ${token}` } : {}
}
