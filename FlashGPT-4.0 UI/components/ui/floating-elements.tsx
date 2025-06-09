"use client"

import { motion } from "framer-motion"
import { Brain, Zap, Sparkles, Cpu, Network, Code, Database, Globe } from "lucide-react"

export function FloatingElements() {
  const icons = [Brain, Zap, Sparkles, Cpu, Network, Code, Database, Globe]

  return (
    <div className="fixed inset-0 pointer-events-none overflow-hidden z-0">
      {icons.map((Icon, index) => (
        <motion.div
          key={index}
          className="absolute text-primary/10"
          initial={{
            x: Math.random() * window.innerWidth,
            y: Math.random() * window.innerHeight,
            rotate: 0,
            scale: 0.5 + Math.random() * 0.5,
          }}
          animate={{
            x: Math.random() * window.innerWidth,
            y: Math.random() * window.innerHeight,
            rotate: 360,
          }}
          transition={{
            duration: 20 + Math.random() * 20,
            repeat: Number.POSITIVE_INFINITY,
            ease: "linear",
          }}
        >
          <Icon className="w-6 h-6" />
        </motion.div>
      ))}
    </div>
  )
}
