"use client"

import { useEffect, useState } from "react"
import { useRouter, useSearchParams } from "next/navigation"
import { useAuth } from "@/contexts/auth-context"
import { Loader2, AlertCircle, CheckCircle } from "lucide-react"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { Button } from "@/components/ui/button"
import Link from "next/link"

export default function OAuthCallbackPage() {
  const router = useRouter()
  const searchParams = useSearchParams()
  const { handleOAuthCallback } = useAuth()
  const [status, setStatus] = useState<"loading" | "success" | "error">("loading")
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const handleCallback = async () => {
      const code = searchParams.get("code")
      const state = searchParams.get("state")
      const errorParam = searchParams.get("error")

      if (errorParam) {
        setStatus("error")
        setError("OAuth authentication was cancelled or failed")
        return
      }

      if (!code || !state) {
        setStatus("error")
        setError("Invalid OAuth callback parameters")
        return
      }

      try {
        const result = await handleOAuthCallback(code, state)

        if (result.success) {
          setStatus("success")
          // Redirect after a short delay to show success message
          setTimeout(() => {
            router.push("/")
          }, 2000)
        } else {
          setStatus("error")
          setError(result.error || "OAuth authentication failed")
        }
      } catch (error) {
        setStatus("error")
        setError("An unexpected error occurred during authentication")
      }
    }

    handleCallback()
  }, [searchParams, handleOAuthCallback, router])

  return (
    <div className="min-h-screen bg-background text-foreground flex items-center justify-center">
      <div className="max-w-md w-full p-8">
        <div className="text-center">
          {status === "loading" && (
            <div className="space-y-4">
              <Loader2 className="h-12 w-12 animate-spin mx-auto text-primary" />
              <h1 className="text-2xl font-bold">Completing Authentication</h1>
              <p className="text-muted-foreground">Please wait while we sign you in...</p>
            </div>
          )}

          {status === "success" && (
            <div className="space-y-4">
              <CheckCircle className="h-12 w-12 mx-auto text-green-500" />
              <h1 className="text-2xl font-bold text-green-600">Authentication Successful!</h1>
              <p className="text-muted-foreground">You will be redirected to the dashboard shortly...</p>
            </div>
          )}

          {status === "error" && (
            <div className="space-y-6">
              <AlertCircle className="h-12 w-12 mx-auto text-destructive" />
              <h1 className="text-2xl font-bold text-destructive">Authentication Failed</h1>

              <Alert variant="destructive">
                <AlertCircle className="h-4 w-4" />
                <AlertDescription>{error}</AlertDescription>
              </Alert>

              <div className="space-y-3">
                <Button asChild className="w-full">
                  <Link href="/login">Try Again</Link>
                </Button>
                <Button asChild variant="outline" className="w-full">
                  <Link href="/">Go Home</Link>
                </Button>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
