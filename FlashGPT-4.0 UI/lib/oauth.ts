import { apiClient } from "./api-client"

export const OAUTH_PROVIDERS = {
  GOOGLE: "google",
  GITHUB: "github",
} as const

export type OAuthProvider = (typeof OAUTH_PROVIDERS)[keyof typeof OAUTH_PROVIDERS]

export class OAuthManager {
  private static instance: OAuthManager
  private pendingStates = new Map<string, { provider: OAuthProvider; timestamp: number }>()

  static getInstance(): OAuthManager {
    if (!OAuthManager.instance) {
      OAuthManager.instance = new OAuthManager()
    }
    return OAuthManager.instance
  }

  async initiateOAuth(provider: OAuthProvider): Promise<void> {
    try {
      const redirectUri = `${window.location.origin}/auth/callback`
      const { data, error } = await apiClient.getOAuthUrl(provider, redirectUri)

      if (error || !data) {
        throw new Error(error || "Failed to get OAuth URL")
      }

      // Store the state for verification
      this.pendingStates.set(data.state, {
        provider,
        timestamp: Date.now(),
      })

      // Clean up old states (older than 10 minutes)
      this.cleanupOldStates()

      // Redirect to OAuth provider
      window.location.href = data.authUrl
    } catch (error) {
      console.error(`OAuth initiation failed for ${provider}:`, error)
      throw error
    }
  }

  async handleCallback(code: string, state: string): Promise<{ token: string; user: any }> {
    try {
      // Verify state
      const pendingState = this.pendingStates.get(state)
      if (!pendingState) {
        throw new Error("Invalid or expired OAuth state")
      }

      // Remove used state
      this.pendingStates.delete(state)

      const { data, error } = await apiClient.oauthLogin(pendingState.provider, code, state)

      if (error || !data) {
        throw new Error(error || "OAuth login failed")
      }

      return data
    } catch (error) {
      console.error("OAuth callback handling failed:", error)
      throw error
    }
  }

  private cleanupOldStates(): void {
    const tenMinutesAgo = Date.now() - 10 * 60 * 1000
    for (const [state, data] of this.pendingStates.entries()) {
      if (data.timestamp < tenMinutesAgo) {
        this.pendingStates.delete(state)
      }
    }
  }
}

export const oauthManager = OAuthManager.getInstance()
