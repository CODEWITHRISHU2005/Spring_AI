import { API_CONFIG, getAuthHeaders } from "./config"

class ApiClient {
  private baseURL: string

  constructor() {
    this.baseURL = API_CONFIG.BASE_URL
  }

  private async request<T>(endpoint: string, options: RequestInit = {}): Promise<{ data?: T; error?: string }> {
    try {
      const url = `${this.baseURL}${endpoint}`
      const response = await fetch(url, {
        headers: {
          "Content-Type": "application/json",
          ...getAuthHeaders(),
          ...options.headers,
        },
        ...options,
      })

      const data = await response.json()

      if (!response.ok) {
        return { error: data.message || "An error occurred" }
      }

      return { data }
    } catch (error) {
      console.error("API request failed:", error)
      // Return mock data for demo purposes
      return this.getMockData<T>(endpoint)
    }
  }

  private getMockData<T>(endpoint: string): { data?: T; error?: string } {
    // Mock data for demo purposes
    if (endpoint === "/llm/models") {
      const mockModels = [
        {
          id: "gpt-4",
          name: "GPT-4.0",
          description: "Most capable GPT model, great for complex reasoning tasks",
          provider: "OpenAI",
          color: "bg-gradient-to-br from-green-500 to-emerald-600",
          textColor: "text-white",
          isAvailable: true,
        },
        {
          id: "ollama-mistral",
          name: "Ollama Mistral",
          description: "Open-source Mistral model running locally via Ollama",
          provider: "Ollama",
          color: "bg-gradient-to-br from-orange-500 to-amber-600",
          textColor: "text-white",
          isAvailable: true,
        },
        {
          id: "claude-3-opus",
          name: "Claude 3 Opus",
          description: "Anthropic's most powerful model for complex analysis",
          provider: "Anthropic",
          color: "bg-gradient-to-br from-purple-500 to-violet-600",
          textColor: "text-white",
          isAvailable: true,
        },
        {
          id: "claude-3-sonnet",
          name: "Claude 3 Sonnet",
          description: "Balanced performance and speed for most tasks",
          provider: "Anthropic",
          color: "bg-gradient-to-br from-indigo-500 to-purple-600",
          textColor: "text-white",
          isAvailable: true,
        },
        {
          id: "gemini-pro",
          name: "Gemini Pro",
          description: "Google's advanced multimodal AI model",
          provider: "Google",
          color: "bg-gradient-to-br from-orange-500 to-red-600",
          textColor: "text-white",
          isAvailable: true,
        },
        {
          id: "llama-2-70b",
          name: "Llama 2 70B",
          description: "Meta's open-source large language model",
          provider: "Meta",
          color: "bg-gradient-to-br from-gray-600 to-gray-700",
          textColor: "text-white",
          isAvailable: false,
        },
      ] as Model[]
      return { data: mockModels as T }
    }

    if (endpoint === "/llm/generate") {
      const mockResponses = {
        "gpt-4": {
          content:
            "This is a sample response from GPT-4.0. It demonstrates the model's capability to provide detailed and thoughtful answers to complex questions.",
          responseTime: 2.3,
          tokenCount: 45,
          model: "gpt-4",
        },
        "ollama-mistral": {
          content:
            "Ollama Mistral provides this response, showcasing the power of running open-source models locally with privacy and control.",
          responseTime: 2.1,
          tokenCount: 42,
          model: "ollama-mistral",
        },
        "gemini-pro": {
          content:
            "Gemini Pro offers this response, showcasing its multimodal capabilities and comprehensive understanding of the query.",
          responseTime: 1.5,
          tokenCount: 32,
          model: "gemini-pro",
        },
      }
      return { data: { responses: mockResponses, requestId: "demo-123" } as T }
    }

    if (endpoint === "/auth/profile") {
      const mockUser = {
        id: "demo-user-123",
        name: "Demo User",
        email: "demo@example.com",
        avatar: "",
        provider: "email",
        bio: "AI enthusiast and researcher exploring the capabilities of different language models.",
        createdAt: "2024-01-15T10:30:00Z",
        updatedAt: "2024-01-20T14:45:00Z",
      }
      return { data: mockUser as T }
    }

    return { error: "Service temporarily unavailable - using demo mode" }
  }

  // Authentication methods
  async login(email: string, password: string) {
    return this.request<{ token: string; user: User }>("/auth/login", {
      method: "POST",
      body: JSON.stringify({ email, password }),
    })
  }

  async register(userData: RegisterData) {
    return this.request<{ token: string; user: User }>("/auth/register", {
      method: "POST",
      body: JSON.stringify(userData),
    })
  }

  async oauthLogin(provider: string, code: string, state?: string) {
    return this.request<{ token: string; user: User }>("/auth/oauth/callback", {
      method: "POST",
      body: JSON.stringify({ provider, code, state }),
    })
  }

  async getOAuthUrl(provider: string, redirectUri: string) {
    return this.request<{ authUrl: string; state: string }>(`/auth/oauth/${provider}/url`, {
      method: "POST",
      body: JSON.stringify({ redirectUri }),
    })
  }

  async logout() {
    return this.request("/auth/logout", {
      method: "POST",
    })
  }

  async getProfile() {
    return this.request<User>("/auth/profile")
  }

  // Profile methods
  async updateProfile(profileData: Partial<User>) {
    return this.request<User>("/auth/profile", {
      method: "PUT",
      body: JSON.stringify(profileData),
    })
  }

  async uploadAvatar(formData: FormData) {
    return this.request<{ avatarUrl: string }>("/auth/avatar", {
      method: "POST",
      body: formData,
      headers: {
        // Don't set Content-Type for FormData, let browser set it
        ...getAuthHeaders(),
      },
    })
  }

  // LLM methods
  async generateResponses(data: GenerateRequest) {
    return this.request<GenerateResponse>("/llm/generate", {
      method: "POST",
      body: JSON.stringify(data),
    })
  }

  async getAvailableModels() {
    return this.request<Model[]>("/llm/models")
  }

  // Comparison methods
  async getComparisons() {
    return this.request<Comparison[]>("/comparisons")
  }

  async saveComparison(data: SaveComparisonRequest) {
    return this.request<Comparison>("/comparisons", {
      method: "POST",
      body: JSON.stringify(data),
    })
  }

  async deleteComparison(id: string) {
    return this.request(`/comparisons/${id}`, {
      method: "DELETE",
    })
  }

  async getComparison(id: string) {
    return this.request<Comparison>(`/comparisons/${id}`)
  }

  // User preferences
  async getUserPreferences() {
    const mockPreferences = {
      theme: "dark" as const,
      fontSize: "medium" as const,
      defaultModels: ["gpt-4", "claude-3-opus", "gemini-pro"],
      autoSave: false,
    }
    return { data: mockPreferences }
  }

  async updateUserPreferences(preferences: Partial<UserPreferences>) {
    return this.request<UserPreferences>("/user/preferences", {
      method: "PUT",
      body: JSON.stringify(preferences),
    })
  }
}

export const apiClient = new ApiClient()

// Type definitions
export interface User {
  id: string
  name: string
  email: string
  avatar?: string
  provider?: string
  bio?: string
  createdAt: string
  updatedAt: string
}

export interface RegisterData {
  name: string
  email: string
  password: string
}

export interface Model {
  id: string
  name: string
  description: string
  provider: string
  color: string
  textColor: string
  isAvailable: boolean
}

export interface GenerateRequest {
  prompt: string
  models: string[]
  temperature?: number
  maxTokens?: number
}

export interface GenerateResponse {
  responses: Record<string, ModelResponse>
  requestId: string
}

export interface ModelResponse {
  content: string
  responseTime: number
  tokenCount: number
  model: string
}

export interface Comparison {
  id: string
  name: string
  prompt: string
  models: string[]
  responses: Record<string, ModelResponse>
  isPublic: boolean
  createdAt: string
  updatedAt: string
  userId: string
}

export interface SaveComparisonRequest {
  name: string
  prompt: string
  models: string[]
  responses: Record<string, ModelResponse>
  isPublic: boolean
}

export interface UserPreferences {
  theme: "light" | "dark" | "system"
  fontSize: "small" | "medium" | "large"
  defaultModels: string[]
  autoSave: boolean
}
