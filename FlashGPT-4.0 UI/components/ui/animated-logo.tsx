"use client"

import { motion } from "framer-motion"
import { Brain, Sparkles } from "lucide-react"
import Link from "next/link"

interface AnimatedLogoProps {
  size?: "sm" | "md" | "lg"
  showIcon?: boolean
}

export function AnimatedLogo({ size = "md", showIcon = true }: AnimatedLogoProps) {
  const sizeClasses = {
    sm: "text-lg",
    md: "text-xl",
    lg: "text-4xl",
  }

  const iconSizes = {
    sm: "w-5 h-5",
    md: "w-6 h-6",
    lg: "w-10 h-10",
  }

  return (
    <Link href="/" className="flex items-center space-x-3 group">
      {showIcon && (
        <motion.div whileHover={{ rotate: 360, scale: 1.1 }} transition={{ duration: 0.5 }} className="relative">
          <motion.div
            animate={{
              boxShadow: [
                "0 0 20px rgba(168, 85, 247, 0.4)",
                "0 0 30px rgba(59, 130, 246, 0.4)",
                "0 0 20px rgba(16, 185, 129, 0.4)",
                "0 0 30px rgba(168, 85, 247, 0.4)",
              ],
            }}
            transition={{ duration: 3, repeat: Number.POSITIVE_INFINITY }}
            className="bg-gradient-to-r from-purple-500 via-blue-500 to-emerald-500 rounded-lg p-2"
          >
            <Brain className={`${iconSizes[size]} text-white`} />
          </motion.div>

          <motion.div
            animate={{ rotate: 360 }}
            transition={{ duration: 8, repeat: Number.POSITIVE_INFINITY, ease: "linear" }}
            className="absolute -top-1 -right-1"
          >
            <Sparkles className="w-3 h-3 text-yellow-400" />
          </motion.div>
        </motion.div>
      )}

      <motion.div whileHover={{ scale: 1.05 }} className="flex items-center space-x-1">
        <motion.span
          className={`${sizeClasses[size]} font-bold bg-gradient-to-r from-purple-600 via-blue-600 to-emerald-600 bg-clip-text text-transparent`}
          whileHover={{
            backgroundImage: "linear-gradient(45deg, #8b5cf6, #3b82f6, #10b981, #f59e0b)",
          }}
          transition={{ duration: 0.3 }}
        >
          LLM
        </motion.span>
        <motion.span
          className={`${sizeClasses[size]} font-bold bg-gradient-to-r from-emerald-600 via-blue-600 to-purple-600 bg-clip-text text-transparent`}
          whileHover={{
            backgroundImage: "linear-gradient(45deg, #10b981, #3b82f6, #8b5cf6, #f59e0b)",
          }}
          transition={{ duration: 0.3 }}
        >
          Compare
        </motion.span>
      </motion.div>
    </Link>
  )
}
