"use client"

import { useState, useEffect } from "react"
import { motion, AnimatePresence } from "framer-motion"
import { Sparkles, Zap, Brain, Cpu, Network, Code } from "lucide-react"

interface SplashScreenProps {
  onComplete: () => void
}

export function SplashScreen({ onComplete }: SplashScreenProps) {
  const [currentStep, setCurrentStep] = useState(0)
  const [isComplete, setIsComplete] = useState(false)

  const steps = [
    { icon: Brain, text: "Initializing AI Models", color: "text-purple-500" },
    { icon: Network, text: "Connecting Neural Networks", color: "text-blue-500" },
    { icon: Cpu, text: "Loading Comparison Engine", color: "text-green-500" },
    { icon: Code, text: "Preparing Interface", color: "text-orange-500" },
  ]

  useEffect(() => {
    const timer = setInterval(() => {
      setCurrentStep((prev) => {
        if (prev < steps.length - 1) {
          return prev + 1
        } else {
          setIsComplete(true)
          setTimeout(() => onComplete(), 1000)
          return prev
        }
      })
    }, 800)

    return () => clearInterval(timer)
  }, [onComplete, steps.length])

  const CurrentIcon = steps[currentStep]?.icon || Brain

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="fixed inset-0 z-50 flex items-center justify-center bg-gradient-to-br from-purple-900 via-blue-900 to-indigo-900"
      >
        {/* Animated Background Particles */}
        <div className="absolute inset-0 overflow-hidden">
          {[...Array(50)].map((_, i) => (
            <motion.div
              key={i}
              className="absolute w-1 h-1 bg-white rounded-full opacity-30"
              initial={{
                x: Math.random() * 1000,
                y: Math.random() * 800,
              }}
              animate={{
                x: Math.random() * 1000,
                y: Math.random() * 800,
              }}
              transition={{
                duration: Math.random() * 10 + 10,
                repeat: Number.POSITIVE_INFINITY,
                ease: "linear",
              }}
            />
          ))}
        </div>

        {/* Main Content */}
        <div className="relative z-10 text-center space-y-8">
          {/* App Logo/Icon */}
          <motion.div
            initial={{ scale: 0, rotate: -180 }}
            animate={{ scale: 1, rotate: 0 }}
            transition={{ duration: 1, ease: "easeOut" }}
            className="relative mx-auto w-24 h-24 mb-8"
          >
            <div className="absolute inset-0 bg-gradient-to-r from-purple-500 via-blue-500 to-emerald-500 rounded-2xl blur-lg opacity-75 animate-pulse" />
            <div className="relative bg-gradient-to-r from-purple-500 via-blue-500 to-emerald-500 rounded-2xl p-4 shadow-2xl">
              <Brain className="w-16 h-16 text-white" />
            </div>
          </motion.div>

          {/* App Name with Animated Text */}
          <motion.div
            initial={{ opacity: 0, y: 50 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 1, delay: 0.5 }}
            className="space-y-4"
          >
            <h1 className="text-6xl font-bold text-white">
              <motion.span
                className="inline-block bg-gradient-to-r from-purple-400 via-blue-400 to-emerald-400 bg-clip-text text-transparent"
                initial={{ opacity: 0, x: -50 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.8, delay: 0.7 }}
              >
                LLM
              </motion.span>
              <motion.span
                className="inline-block ml-4 bg-gradient-to-r from-emerald-400 via-blue-400 to-purple-400 bg-clip-text text-transparent"
                initial={{ opacity: 0, x: 50 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.8, delay: 0.9 }}
              >
                Compare
              </motion.span>
            </h1>

            <motion.p
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ duration: 1, delay: 1.2 }}
              className="text-xl text-blue-200 font-light tracking-wide"
            >
              AI Model Comparison Platform
            </motion.p>
          </motion.div>

          {/* Loading Steps */}
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 1.5 }}
            className="space-y-6"
          >
            {/* Progress Bar */}
            <div className="w-80 mx-auto">
              <div className="h-2 bg-white/20 rounded-full overflow-hidden">
                <motion.div
                  className="h-full bg-gradient-to-r from-purple-500 to-blue-500"
                  initial={{ width: "0%" }}
                  animate={{ width: `${((currentStep + 1) / steps.length) * 100}%` }}
                  transition={{ duration: 0.5 }}
                />
              </div>
            </div>

            {/* Current Step */}
            <AnimatePresence mode="wait">
              <motion.div
                key={currentStep}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                transition={{ duration: 0.5 }}
                className="flex items-center justify-center space-x-3"
              >
                <motion.div
                  animate={{ rotate: 360 }}
                  transition={{ duration: 2, repeat: Number.POSITIVE_INFINITY, ease: "linear" }}
                  className={`${steps[currentStep]?.color || "text-white"}`}
                >
                  <CurrentIcon className="w-6 h-6" />
                </motion.div>
                <span className="text-white font-medium">{steps[currentStep]?.text || "Loading..."}</span>
              </motion.div>
            </AnimatePresence>
          </motion.div>

          {/* Completion Animation */}
          <AnimatePresence>
            {isComplete && (
              <motion.div
                initial={{ opacity: 0, scale: 0 }}
                animate={{ opacity: 1, scale: 1 }}
                className="absolute inset-0 flex items-center justify-center"
              >
                <motion.div animate={{ rotate: 360 }} transition={{ duration: 1 }} className="text-emerald-400">
                  <Sparkles className="w-16 h-16" />
                </motion.div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {/* Floating Elements */}
        <div className="absolute inset-0 pointer-events-none">
          {[Zap, Sparkles, Brain, Cpu].map((Icon, index) => (
            <motion.div
              key={index}
              className="absolute text-white/20"
              initial={{
                x: Math.random() * 1000,
                y: Math.random() * 800,
                rotate: 0,
              }}
              animate={{
                x: Math.random() * 1000,
                y: Math.random() * 800,
                rotate: 360,
              }}
              transition={{
                duration: Math.random() * 20 + 20,
                repeat: Number.POSITIVE_INFINITY,
                ease: "linear",
              }}
            >
              <Icon className="w-8 h-8" />
            </motion.div>
          ))}
        </div>
      </motion.div>
    </AnimatePresence>
  )
}

export default SplashScreen
