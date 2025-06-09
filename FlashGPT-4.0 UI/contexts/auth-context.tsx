"use client"

import type React from "react"

import { createContext, useContext, useEffect, useState } from "react"
import { apiClient, type User } from "@/lib/api-client"
import { oauthManager } from "@/lib/oauth"

interface AuthContextType {
  user: User | null
  isLoading: boolean
  login: (email: string, password: string) => Promise<{ success: boolean; error?: string }>
  register: (userData: { name: string; email: string; password: string }) => Promise<{
    success: boolean
    error?: string
  }>
  handleOAuthCallback: (code: string, state: string) => Promise<{ success: boolean; error?: string }>
  logout: () => Promise<void>
  refreshUser: () => Promise<void>
  updateProfile: (profileData: Partial<User>) => Promise<{ success: boolean; error?: string }>
}

const AuthContext = createContext<AuthContextType | undefined>(undefined)

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<User | null>(null)
  const [isLoading, setIsLoading] = useState(true)

  useEffect(() => {
    checkAuthStatus()
  }, [])

  const checkAuthStatus = async () => {
    const token = localStorage.getItem("auth_token")
    if (!token) {
      setIsLoading(false)
      return
    }

    const { data, error } = await apiClient.getProfile()
    if (data && !error) {
      setUser(data)
    } else {
      localStorage.removeItem("auth_token")
    }
    setIsLoading(false)
  }

  const login = async (email: string, password: string) => {
    const { data, error } = await apiClient.login(email, password)

    if (data && !error) {
      localStorage.setItem("auth_token", data.token)
      setUser(data.user)
      return { success: true }
    }

    return { success: false, error: error || "Login failed" }
  }

  const register = async (userData: { name: string; email: string; password: string }) => {
    const { data, error } = await apiClient.register(userData)

    if (data && !error) {
      localStorage.setItem("auth_token", data.token)
      setUser(data.user)
      return { success: true }
    }

    return { success: false, error: error || "Registration failed" }
  }

  const handleOAuthCallback = async (code: string, state: string) => {
    try {
      const result = await oauthManager.handleCallback(code, state)
      localStorage.setItem("auth_token", result.token)
      setUser(result.user)
      return { success: true }
    } catch (error) {
      console.error("OAuth callback failed:", error)
      return { success: false, error: error instanceof Error ? error.message : "OAuth authentication failed" }
    }
  }

  const logout = async () => {
    await apiClient.logout()
    localStorage.removeItem("auth_token")
    setUser(null)
  }

  const refreshUser = async () => {
    const { data } = await apiClient.getProfile()
    if (data) {
      setUser(data)
    }
  }

  const updateProfile = async (profileData: Partial<User>) => {
    const { data, error } = await apiClient.updateProfile(profileData)

    if (data && !error) {
      setUser(data)
      return { success: true }
    }

    return { success: false, error: error || "Profile update failed" }
  }

  return (
    <AuthContext.Provider
      value={{ user, isLoading, login, register, handleOAuthCallback, logout, refreshUser, updateProfile }}
    >
      {children}
    </AuthContext.Provider>
  )
}

export function useAuth() {
  const context = useContext(AuthContext)
  if (context === undefined) {
    throw new Error("useAuth must be used within an AuthProvider")
  }
  return context
}
