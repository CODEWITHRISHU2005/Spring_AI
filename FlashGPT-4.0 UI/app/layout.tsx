import type React from "react"
import "./globals.css"
import type { Metadata } from "next"
import { Inter } from "next/font/google"
import { ThemeProvider } from "@/components/theme-provider"
import { NavBar } from "@/components/layout/nav-bar"
import { AuthProvider } from "@/contexts/auth-context"
import { Toaster } from "@/components/ui/toaster"
import { AppWrapper } from "@/components/app-wrapper"

const inter = Inter({ subsets: ["latin"] })

export const metadata: Metadata = {
  title: "LLM Compare - AI Model Comparison Platform",
  description: "Compare responses from various large language models side by side",
    generator: 'v0.dev'
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className={inter.className}>
        <AuthProvider>
          <ThemeProvider attribute="class" defaultTheme="dark" enableSystem disableTransitionOnChange>
            <AppWrapper>
              <NavBar />
              {children}
              <Toaster />
            </AppWrapper>
          </ThemeProvider>
        </AuthProvider>
      </body>
    </html>
  )
}
