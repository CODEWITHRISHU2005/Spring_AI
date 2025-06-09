import type { Metadata } from "next"
import Link from "next/link"
import { LoginForm } from "@/components/auth/login-form"

export const metadata: Metadata = {
  title: "Login - LLM Comparison Tool",
  description: "Login to your account to save comparisons and preferences",
}

export default function LoginPage() {
  return (
    <div className="min-h-screen flex items-center justify-center bg-background">
      <div className="w-full max-w-md p-8">
        <div className="text-center mb-8">
          <h1 className="text-3xl font-bold bg-gradient-to-r from-purple-400 via-blue-400 to-green-400 bg-clip-text text-transparent">
            Sign In
          </h1>
          <p className="text-muted-foreground mt-2">Sign in to your account to save comparisons and preferences</p>
        </div>
        <LoginForm />
        <div className="text-center mt-6 text-muted-foreground">
          <p>
            Don't have an account?{" "}
            <Link href="/register" className="text-primary hover:underline">
              Sign up
            </Link>
          </p>
        </div>
      </div>
    </div>
  )
}
