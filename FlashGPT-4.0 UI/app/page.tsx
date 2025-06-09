"use client"

import type React from "react"

import { useState, useEffect } from "react"
import { useRouter } from "next/navigation"
import { useTheme } from "next-themes"
import { motion } from "framer-motion"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Checkbox } from "@/components/ui/checkbox"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Textarea } from "@/components/ui/textarea"
import { Badge } from "@/components/ui/badge"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import {
  Copy,
  ThumbsUp,
  ThumbsDown,
  Download,
  Share,
  Info,
  Save,
  Loader2,
  Sparkles,
  Zap,
  Clock,
  Users,
  CheckCircle2,
} from "lucide-react"
import { Separator } from "@/components/ui/separator"
import { useAuth } from "@/contexts/auth-context"
import { apiClient, type Model, type ModelResponse, type UserPreferences } from "@/lib/api-client"
import { toast } from "@/components/ui/use-toast"
import { cn } from "@/lib/utils"
import { PageTransition } from "@/components/ui/page-transition"
import { FloatingElements } from "@/components/ui/floating-elements"
import { ExportDialog } from "@/components/comparison/export-dialog"
import { ShareDialog } from "@/components/comparison/share-dialog"

export default function LLMComparisonTool() {
  const { user } = useAuth()
  const router = useRouter()
  const { theme, setTheme } = useTheme()
  const [activeTab, setActiveTab] = useState("compare")
  const [availableModels, setAvailableModels] = useState<Model[]>([])
  const [selectedModels, setSelectedModels] = useState<string[]>([])
  const [prompt, setPrompt] = useState("")
  const [preferences, setPreferences] = useState<UserPreferences>({
    theme: "dark",
    fontSize: "medium",
    defaultModels: [],
    autoSave: false,
  })
  const [responses, setResponses] = useState<Record<string, ModelResponse>>({})
  const [isGenerating, setIsGenerating] = useState(false)
  const [isSaving, setIsSaving] = useState(false)
  const [comparisonName, setComparisonName] = useState("")
  const [isPublic, setIsPublic] = useState(false)
  const [saveDialogOpen, setSaveDialogOpen] = useState(false)
  const [isLoadingModels, setIsLoadingModels] = useState(true)
  const [isLoadingPreferences, setIsLoadingPreferences] = useState(false)
  const [exportDialogOpen, setExportDialogOpen] = useState(false)
  const [shareDialogOpen, setShareDialogOpen] = useState(false)

  useEffect(() => {
    loadAvailableModels()
    if (user) {
      loadUserPreferences()
    }
  }, [user])

  // Sync theme with preferences
  useEffect(() => {
    if (preferences.theme !== theme) {
      setTheme(preferences.theme)
    }
  }, [preferences.theme, theme, setTheme])

  const loadAvailableModels = async () => {
    setIsLoadingModels(true)
    const { data, error } = await apiClient.getAvailableModels()

    if (data && !error) {
      setAvailableModels(data)
      // Set default selected models if none selected
      if (selectedModels.length === 0) {
        const defaultModels = data
          .filter((model) => model.isAvailable)
          .slice(0, 3)
          .map((model) => model.id)
        setSelectedModels(defaultModels)
      }
    } else {
      toast({
        title: "Demo Mode",
        description: "Using demo data - connect to backend for live models",
        variant: "default",
      })
    }
    setIsLoadingModels(false)
  }

  const loadUserPreferences = async () => {
    setIsLoadingPreferences(true)
    const { data, error } = await apiClient.getUserPreferences()

    if (data && !error) {
      setPreferences(data)
      if (data.defaultModels.length > 0) {
        setSelectedModels(data.defaultModels)
      }
      setTheme(data.theme)
    }
    setIsLoadingPreferences(false)
  }

  const handleModelToggle = (modelId: string) => {
    setSelectedModels((prev) =>
      prev.includes(modelId) ? prev.filter((id) => id !== modelId) : prev.length < 5 ? [...prev, modelId] : prev,
    )
  }

  const generateResponses = async () => {
    if (!prompt.trim()) {
      toast({
        title: "Missing Prompt",
        description: "Please enter a prompt to compare responses",
        variant: "destructive",
      })
      return
    }

    if (selectedModels.length === 0) {
      toast({
        title: "No Models Selected",
        description: "Please select at least one model to compare",
        variant: "destructive",
      })
      return
    }

    setIsGenerating(true)
    setResponses({})

    const { data, error } = await apiClient.generateResponses({
      prompt: prompt.trim(),
      models: selectedModels,
    })

    if (data && !error) {
      setResponses(data.responses)
      toast({
        title: "Responses Generated! ✨",
        description: `Generated ${Object.keys(data.responses).length} responses successfully`,
      })
    } else {
      toast({
        title: "Generation Failed",
        description: error || "Failed to generate responses",
        variant: "destructive",
      })
    }

    setIsGenerating(false)
  }

  const saveComparison = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault()
    if (!user) {
      router.push("/login")
      return
    }

    if (!comparisonName.trim()) {
      toast({
        title: "Missing Name",
        description: "Please enter a name for this comparison",
        variant: "destructive",
      })
      return
    }

    setIsSaving(true)

    const { data, error } = await apiClient.saveComparison({
      name: comparisonName.trim(),
      prompt: prompt.trim(),
      models: selectedModels,
      responses,
      isPublic,
    })

    if (data && !error) {
      toast({
        title: "Comparison Saved! 🎉",
        description: "Your comparison has been saved successfully",
      })
      setSaveDialogOpen(false)
      setComparisonName("")
    } else {
      toast({
        title: "Save Failed",
        description: error || "Failed to save comparison",
        variant: "destructive",
      })
    }

    setIsSaving(false)
  }

  const savePreferences = async () => {
    if (!user) return

    const { data, error } = await apiClient.updateUserPreferences({
      ...preferences,
      defaultModels: selectedModels,
    })

    if (data && !error) {
      toast({
        title: "Preferences Saved! ⚙️",
        description: "Your settings have been updated",
      })
    } else {
      toast({
        title: "Save Failed",
        description: error || "Failed to save preferences",
        variant: "destructive",
      })
    }
  }

  const handleThemeChange = (newTheme: "light" | "dark" | "system") => {
    setPreferences((prev) => ({ ...prev, theme: newTheme }))
    setTheme(newTheme)
  }

  const copyToClipboard = async (text: string) => {
    try {
      await navigator.clipboard.writeText(text)
      toast({
        title: "Copied! 📋",
        description: "Response copied to clipboard",
      })
    } catch (error) {
      toast({
        title: "Copy Failed",
        description: "Failed to copy to clipboard",
        variant: "destructive",
      })
    }
  }

  if (isLoadingModels) {
    return (
      <div className="min-h-screen bg-background text-foreground flex items-center justify-center">
        <motion.div
          initial={{ opacity: 0, scale: 0.8 }}
          animate={{ opacity: 1, scale: 1 }}
          className="text-center space-y-4"
        >
          <div className="relative">
            <motion.div
              animate={{ rotate: 360 }}
              transition={{ duration: 2, repeat: Number.POSITIVE_INFINITY, ease: "linear" }}
            >
              <Loader2 className="h-12 w-12 mx-auto text-primary" />
            </motion.div>
            <motion.div
              animate={{ scale: [1, 1.2, 1] }}
              transition={{ duration: 1.5, repeat: Number.POSITIVE_INFINITY }}
              className="absolute top-0 left-1/2 transform -translate-x-1/2"
            >
              <Sparkles className="h-6 w-6 text-yellow-500" />
            </motion.div>
          </div>
          <div>
            <h2 className="text-xl font-semibold mb-2">Loading AI Models</h2>
            <p className="text-muted-foreground">Preparing the comparison environment...</p>
          </div>
        </motion.div>
      </div>
    )
  }

  return (
    <PageTransition>
      <div className="min-h-screen bg-background relative">
        <FloatingElements />

        <div className="container mx-auto px-4 py-8 max-w-7xl relative z-10">
          {/* Hero Section */}
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            className="text-center mb-12 space-y-4"
          >
            <motion.div
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.6, delay: 0.2 }}
              className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-primary/10 text-primary text-sm font-medium mb-4"
            >
              <motion.div
                animate={{ rotate: 360 }}
                transition={{ duration: 3, repeat: Number.POSITIVE_INFINITY, ease: "linear" }}
              >
                <Sparkles className="w-4 h-4" />
              </motion.div>
              AI Model Comparison Platform
            </motion.div>

            <motion.h1
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8, delay: 0.3 }}
              className="text-5xl font-bold mb-6 leading-tight"
            >
              <motion.span
                className="bg-gradient-to-r from-purple-600 via-blue-600 to-emerald-600 bg-clip-text text-transparent"
                animate={{
                  backgroundPosition: ["0% 50%", "100% 50%", "0% 50%"],
                }}
                transition={{ duration: 5, repeat: Number.POSITIVE_INFINITY }}
              >
                Compare AI Models
              </motion.span>
              <br />
              <motion.span
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.8 }}
                className="text-3xl text-foreground"
              >
                Side by Side
              </motion.span>
            </motion.h1>

            <motion.p
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8, delay: 0.5 }}
              className="text-xl text-muted-foreground max-w-3xl mx-auto leading-relaxed"
            >
              Discover the unique strengths of different AI models by comparing their responses to your prompts. Make
              informed decisions about which AI works best for your specific needs.
            </motion.p>

            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8, delay: 0.7 }}
              className="flex items-center justify-center gap-6 mt-8 text-sm text-muted-foreground"
            >
              {[
                { icon: Zap, text: "Real-time comparison", color: "text-yellow-500" },
                { icon: Users, text: "Multiple AI models", color: "text-blue-500" },
                { icon: Save, text: "Save & share results", color: "text-green-500" },
              ].map((item, index) => (
                <motion.div
                  key={index}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ duration: 0.5, delay: 0.8 + index * 0.1 }}
                  className="flex items-center gap-2"
                >
                  <motion.div
                    animate={{ scale: [1, 1.1, 1] }}
                    transition={{ duration: 2, repeat: Number.POSITIVE_INFINITY, delay: index * 0.5 }}
                  >
                    <item.icon className={`w-4 h-4 ${item.color}`} />
                  </motion.div>
                  <span>{item.text}</span>
                </motion.div>
              ))}
            </motion.div>
          </motion.div>

          {/* Rest of the component remains the same but wrapped in motion.div for animations */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.9 }}
          >
            {/* Navigation Tabs */}
            <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
              <TabsList className="grid w-full grid-cols-2 mb-8 h-12">
                <TabsTrigger value="compare" className="text-base font-medium">
                  <Zap className="w-4 h-4 mr-2" />
                  Compare Models
                </TabsTrigger>
                <TabsTrigger value="settings" className="text-base font-medium">
                  <Info className="w-4 h-4 mr-2" />
                  Settings
                </TabsTrigger>
              </TabsList>

              <TabsContent value="compare" className="space-y-8">
                {/* Model Selection */}
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.5, delay: 0.1 }}
                >
                  <Card className="border-2 border-dashed border-muted-foreground/20 hover:border-primary/50 transition-colors duration-200">
                    <CardHeader className="pb-4">
                      <div className="flex items-center gap-3">
                        <motion.div
                          animate={{ rotate: [0, 10, -10, 0] }}
                          transition={{ duration: 2, repeat: Number.POSITIVE_INFINITY }}
                          className="p-2 rounded-lg bg-primary/10"
                        >
                          <Users className="w-5 h-5 text-primary" />
                        </motion.div>
                        <div>
                          <CardTitle className="text-xl">Select AI Models</CardTitle>
                          <CardDescription className="text-base">
                            Choose up to 5 models to compare. Each model has unique strengths and capabilities.
                          </CardDescription>
                        </div>
                      </div>
                    </CardHeader>
                    <CardContent className="space-y-4">
                      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
                        {availableModels.map((model, index) => (
                          <motion.div
                            key={model.id}
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ duration: 0.3, delay: index * 0.1 }}
                            whileHover={{ scale: 1.02, y: -2 }}
                            className={cn(
                              "relative p-4 rounded-xl border-2 transition-all duration-200 cursor-pointer group",
                              selectedModels.includes(model.id)
                                ? "border-primary bg-primary/5 shadow-lg shadow-primary/20"
                                : "border-muted hover:border-muted-foreground/50",
                              !model.isAvailable && "opacity-50 cursor-not-allowed",
                            )}
                            onClick={() => model.isAvailable && handleModelToggle(model.id)}
                          >
                            <div className="flex items-start space-x-3">
                              <Checkbox
                                id={model.id}
                                checked={selectedModels.includes(model.id)}
                                onCheckedChange={() => model.isAvailable && handleModelToggle(model.id)}
                                disabled={!model.isAvailable}
                                className="mt-1"
                              />
                              <div className="flex-1 min-w-0">
                                <div className="flex items-center space-x-2 mb-2">
                                  <motion.div
                                    animate={{ scale: [1, 1.1, 1] }}
                                    transition={{ duration: 2, repeat: Number.POSITIVE_INFINITY, delay: index * 0.2 }}
                                    className={cn(
                                      "w-3 h-3 rounded-full",
                                      model.color.replace("bg-gradient-to-br", "bg-gradient-to-r"),
                                    )}
                                  />
                                  <h3
                                    className={cn(
                                      "font-semibold text-sm",
                                      model.isAvailable ? "text-foreground" : "text-muted-foreground",
                                    )}
                                  >
                                    {model.name}
                                  </h3>
                                  {!model.isAvailable && (
                                    <Badge variant="secondary" className="text-xs">
                                      Coming Soon
                                    </Badge>
                                  )}
                                  {selectedModels.includes(model.id) && (
                                    <motion.div
                                      initial={{ scale: 0 }}
                                      animate={{ scale: 1 }}
                                      transition={{ duration: 0.2 }}
                                    >
                                      <CheckCircle2 className="w-4 h-4 text-primary" />
                                    </motion.div>
                                  )}
                                </div>
                                <p className="text-xs text-muted-foreground leading-relaxed">{model.description}</p>
                                <div className="mt-2">
                                  <Badge variant="outline" className="text-xs">
                                    {model.provider}
                                  </Badge>
                                </div>
                              </div>
                            </div>
                          </motion.div>
                        ))}
                      </div>

                      {selectedModels.length > 0 && (
                        <motion.div
                          initial={{ opacity: 0, height: 0 }}
                          animate={{ opacity: 1, height: "auto" }}
                          transition={{ duration: 0.3 }}
                          className="mt-6 p-4 bg-primary/5 rounded-lg border border-primary/20"
                        >
                          <div className="flex items-center gap-2 mb-2">
                            <motion.div
                              animate={{ scale: [1, 1.1, 1] }}
                              transition={{ duration: 1.5, repeat: Number.POSITIVE_INFINITY }}
                            >
                              <CheckCircle2 className="w-4 h-4 text-primary" />
                            </motion.div>
                            <span className="font-medium text-sm">
                              {selectedModels.length} model{selectedModels.length !== 1 ? "s" : ""} selected
                            </span>
                          </div>
                          <div className="flex flex-wrap gap-2">
                            {selectedModels.map((modelId, index) => {
                              const model = availableModels.find((m) => m.id === modelId)
                              return model ? (
                                <motion.div
                                  key={modelId}
                                  initial={{ opacity: 0, scale: 0 }}
                                  animate={{ opacity: 1, scale: 1 }}
                                  transition={{ duration: 0.2, delay: index * 0.05 }}
                                >
                                  <Badge variant="secondary" className="text-xs">
                                    {model.name}
                                  </Badge>
                                </motion.div>
                              ) : null
                            })}
                          </div>
                        </motion.div>
                      )}
                    </CardContent>
                  </Card>
                </motion.div>

                {/* Prompt Input */}
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.5, delay: 0.2 }}
                >
                  <Card className="border-2 border-dashed border-muted-foreground/20 hover:border-primary/50 transition-colors duration-200">
                    <CardHeader className="pb-4">
                      <div className="flex items-center gap-3">
                        <motion.div
                          animate={{ rotate: [0, 360] }}
                          transition={{ duration: 4, repeat: Number.POSITIVE_INFINITY, ease: "linear" }}
                          className="p-2 rounded-lg bg-primary/10"
                        >
                          <Sparkles className="w-5 h-5 text-primary" />
                        </motion.div>
                        <div>
                          <CardTitle className="text-xl">Your Prompt</CardTitle>
                          <CardDescription className="text-base">
                            Enter your question or prompt to see how different AI models respond.
                          </CardDescription>
                        </div>
                      </div>
                    </CardHeader>
                    <CardContent className="space-y-6">
                      <Textarea
                        value={prompt}
                        onChange={(e) => setPrompt(e.target.value)}
                        placeholder="Ask anything... For example: 'Explain quantum computing in simple terms' or 'Write a creative story about time travel'"
                        className="min-h-[120px] text-base resize-none border-2 focus:border-primary transition-colors"
                      />

                      <div className="flex items-center justify-between">
                        <div className="text-sm text-muted-foreground">
                          {prompt.length > 0 && <span>{prompt.length} characters</span>}
                        </div>

                        <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
                          <Button
                            onClick={generateResponses}
                            disabled={selectedModels.length === 0 || isGenerating || !prompt.trim()}
                            size="lg"
                            className="px-8 py-3 text-base font-medium bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-700 hover:to-blue-700 transition-all duration-200 shadow-lg hover:shadow-xl"
                          >
                            {isGenerating ? (
                              <>
                                <motion.div
                                  animate={{ rotate: 360 }}
                                  transition={{ duration: 1, repeat: Number.POSITIVE_INFINITY, ease: "linear" }}
                                >
                                  <Loader2 className="w-5 h-5 mr-2" />
                                </motion.div>
                                Generating Responses...
                              </>
                            ) : (
                              <>
                                <Zap className="w-5 h-5 mr-2" />
                                Generate Responses
                              </>
                            )}
                          </Button>
                        </motion.div>
                      </div>
                    </CardContent>
                  </Card>
                </motion.div>

                {/* Responses */}
                {Object.keys(responses).length > 0 && (
                  <motion.div
                    initial={{ opacity: 0, y: 30 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.6 }}
                    className="space-y-8"
                  >
                    <div className="flex flex-col sm:flex-row sm:items-end sm:justify-between gap-4">
                      <div className="space-y-2">
                        <motion.h2
                          initial={{ opacity: 0, x: -20 }}
                          animate={{ opacity: 1, x: 0 }}
                          className="text-3xl font-bold flex items-center gap-3"
                        >
                          <motion.div
                            animate={{ rotate: [0, 360] }}
                            transition={{ duration: 3, repeat: Number.POSITIVE_INFINITY, ease: "linear" }}
                          >
                            <Sparkles className="w-8 h-8 text-yellow-500" />
                          </motion.div>
                          AI Responses
                        </motion.h2>
                        <motion.p
                          initial={{ opacity: 0 }}
                          animate={{ opacity: 1 }}
                          transition={{ delay: 0.2 }}
                          className="text-muted-foreground text-base"
                        >
                          Compare how different models approach your prompt
                        </motion.p>
                      </div>

                      <motion.div
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: 0.3 }}
                        className="flex flex-wrap items-center gap-3"
                      >
                        <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
                          <Button
                            variant="outline"
                            size="sm"
                            className="h-9 px-4 border-2 border-dashed border-muted-foreground/30 hover:border-primary/50 hover:bg-primary/5 transition-all duration-200 font-medium"
                            onClick={() => setExportDialogOpen(true)}
                          >
                            <Download className="w-4 h-4 mr-2" />
                            Export
                          </Button>
                        </motion.div>

                        <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
                          <Button
                            variant="outline"
                            size="sm"
                            className="h-9 px-4 border-2 border-dashed border-muted-foreground/30 hover:border-primary/50 hover:bg-primary/5 transition-all duration-200 font-medium"
                            onClick={() => setShareDialogOpen(true)}
                          >
                            <Share className="w-4 h-4 mr-2" />
                            Share
                          </Button>
                        </motion.div>

                        {/* Stats Badge */}
                        <motion.div
                          initial={{ opacity: 0, scale: 0.8 }}
                          animate={{ opacity: 1, scale: 1 }}
                          transition={{ delay: 0.4 }}
                          className="hidden sm:flex items-center gap-2 px-3 py-1.5 rounded-full bg-primary/10 border border-primary/20"
                        >
                          <div className="flex items-center gap-1 text-xs font-medium text-primary">
                            <Users className="w-3 h-3" />
                            {selectedModels.length} models
                          </div>
                          <div className="w-1 h-1 rounded-full bg-primary/50" />
                          <div className="flex items-center gap-1 text-xs font-medium text-primary">
                            <Zap className="w-3 h-3" />
                            {Object.keys(responses).length} responses
                          </div>
                        </motion.div>
                      </motion.div>
                    </div>

                    {/* Mobile Stats */}
                    <motion.div
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      transition={{ delay: 0.5 }}
                      className="sm:hidden flex items-center justify-center gap-4 p-3 rounded-lg bg-muted/30 border"
                    >
                      <div className="flex items-center gap-2 text-sm font-medium">
                        <Users className="w-4 h-4 text-primary" />
                        {selectedModels.length} models selected
                      </div>
                      <div className="w-1 h-1 rounded-full bg-muted-foreground" />
                      <div className="flex items-center gap-2 text-sm font-medium">
                        <Zap className="w-4 h-4 text-primary" />
                        {Object.keys(responses).length} responses generated
                      </div>
                    </motion.div>

                    <div className="grid gap-6 lg:grid-cols-2 xl:grid-cols-3">
                      {selectedModels.map((modelId, index) => {
                        const model = availableModels.find((m) => m.id === modelId)
                        const response = responses[modelId]

                        if (!model || !response) return null

                        return (
                          <motion.div
                            key={modelId}
                            initial={{ opacity: 0, y: 30, scale: 0.9 }}
                            animate={{ opacity: 1, y: 0, scale: 1 }}
                            transition={{ duration: 0.5, delay: index * 0.1 }}
                            whileHover={{ y: -5, scale: 1.02 }}
                          >
                            <Card
                              className={cn(
                                "group hover:shadow-xl transition-all duration-300 border-2 overflow-hidden",
                                model.color,
                                model.textColor,
                              )}
                            >
                              <CardHeader className="pb-4">
                                <div className="flex items-center justify-between">
                                  <div className="flex items-center space-x-3">
                                    <motion.div
                                      whileHover={{ rotate: 360 }}
                                      transition={{ duration: 0.5 }}
                                      className="p-2 rounded-lg bg-white/20 backdrop-blur-sm"
                                    >
                                      <Info className="w-4 h-4" />
                                    </motion.div>
                                    <div>
                                      <CardTitle className="text-lg font-semibold">{model.name}</CardTitle>
                                      <p className="text-white/80 text-sm">{model.provider}</p>
                                    </div>
                                  </div>
                                  <Button
                                    variant="ghost"
                                    size="sm"
                                    className="text-white/80 hover:text-white hover:bg-white/20"
                                  >
                                    •••
                                  </Button>
                                </div>
                              </CardHeader>
                              <CardContent className="space-y-4">
                                <motion.div
                                  initial={{ opacity: 0 }}
                                  animate={{ opacity: 1 }}
                                  transition={{ delay: 0.3 + index * 0.1 }}
                                  className="bg-white/10 backdrop-blur-sm rounded-lg p-4"
                                >
                                  <div className="text-sm leading-relaxed whitespace-pre-wrap text-white">
                                    {response.content}
                                  </div>
                                </motion.div>

                                <Separator className="bg-white/20" />

                                <div className="flex items-center justify-between text-sm">
                                  <div className="flex space-x-3">
                                    <div className="flex items-center gap-1">
                                      <Clock className="w-3 h-3" />
                                      <Badge variant="secondary" className="bg-white/20 text-white border-white/30">
                                        {response.responseTime}s
                                      </Badge>
                                    </div>
                                    {response.tokenCount && (
                                      <Badge variant="secondary" className="bg-white/20 text-white border-white/30">
                                        {response.tokenCount} tokens
                                      </Badge>
                                    )}
                                  </div>
                                  <div className="flex space-x-1">
                                    <motion.div whileHover={{ scale: 1.1 }} whileTap={{ scale: 0.9 }}>
                                      <Button
                                        variant="ghost"
                                        size="sm"
                                        className="text-white/80 hover:text-white hover:bg-white/20 p-2"
                                        onClick={() => copyToClipboard(response.content)}
                                      >
                                        <Copy className="w-4 h-4" />
                                      </Button>
                                    </motion.div>
                                    <motion.div whileHover={{ scale: 1.1 }} whileTap={{ scale: 0.9 }}>
                                      <Button
                                        variant="ghost"
                                        size="sm"
                                        className="text-white/80 hover:text-white hover:bg-white/20 p-2"
                                      >
                                        <ThumbsUp className="w-4 h-4" />
                                      </Button>
                                    </motion.div>
                                    <motion.div whileHover={{ scale: 1.1 }} whileTap={{ scale: 0.9 }}>
                                      <Button
                                        variant="ghost"
                                        size="sm"
                                        className="text-white/80 hover:text-white hover:bg-white/20 p-2"
                                      >
                                        <ThumbsDown className="w-4 h-4" />
                                      </Button>
                                    </motion.div>
                                  </div>
                                </div>
                              </CardContent>
                            </Card>
                          </motion.div>
                        )
                      })}
                    </div>
                  </motion.div>
                )}
              </TabsContent>

              <TabsContent value="settings" className="space-y-8">
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.5 }}
                >
                  <Card>
                    <CardHeader>
                      <div className="flex items-center gap-3">
                        <motion.div
                          animate={{ rotate: [0, 360] }}
                          transition={{ duration: 4, repeat: Number.POSITIVE_INFINITY, ease: "linear" }}
                          className="p-2 rounded-lg bg-primary/10"
                        >
                          <Info className="w-5 h-5 text-primary" />
                        </motion.div>
                        <div>
                          <CardTitle className="text-xl">Display Settings</CardTitle>
                          <CardDescription className="text-base">
                            Customize your experience and preferences
                          </CardDescription>
                        </div>
                      </div>
                    </CardHeader>
                    <CardContent className="space-y-6">
                      <div className="grid gap-6 md:grid-cols-2">
                        <div className="space-y-3">
                          <label className="font-medium text-sm">Theme Preference</label>
                          <Select value={preferences.theme} onValueChange={handleThemeChange}>
                            <SelectTrigger className="h-11">
                              <SelectValue />
                            </SelectTrigger>
                            <SelectContent>
                              <SelectItem value="light">☀️ Light Mode</SelectItem>
                              <SelectItem value="dark">🌙 Dark Mode</SelectItem>
                              <SelectItem value="system">💻 System Default</SelectItem>
                            </SelectContent>
                          </Select>
                        </div>

                        <div className="space-y-3">
                          <label className="font-medium text-sm">Font Size</label>
                          <Select
                            value={preferences.fontSize}
                            onValueChange={(value: "small" | "medium" | "large") =>
                              setPreferences((prev) => ({ ...prev, fontSize: value }))
                            }
                          >
                            <SelectTrigger className="h-11">
                              <SelectValue />
                            </SelectTrigger>
                            <SelectContent>
                              <SelectItem value="small">Small</SelectItem>
                              <SelectItem value="medium">Medium</SelectItem>
                              <SelectItem value="large">Large</SelectItem>
                            </SelectContent>
                          </Select>
                        </div>
                      </div>

                      <div className="flex items-center space-x-3 p-4 rounded-lg border bg-muted/50">
                        <Checkbox
                          id="autoSave"
                          checked={preferences.autoSave}
                          onCheckedChange={(checked) =>
                            setPreferences((prev) => ({ ...prev, autoSave: checked === true }))
                          }
                        />
                        <div className="flex-1">
                          <label htmlFor="autoSave" className="font-medium cursor-pointer text-sm">
                            Auto-save comparisons
                          </label>
                          <p className="text-xs text-muted-foreground mt-1">
                            Automatically save your comparisons when you generate responses
                          </p>
                        </div>
                      </div>

                      {user && (
                        <motion.div whileHover={{ scale: 1.02 }} whileTap={{ scale: 0.98 }}>
                          <Button
                            onClick={savePreferences}
                            disabled={isLoadingPreferences}
                            size="lg"
                            className="w-full md:w-auto"
                          >
                            {isLoadingPreferences ? (
                              <>
                                <motion.div
                                  animate={{ rotate: 360 }}
                                  transition={{ duration: 1, repeat: Number.POSITIVE_INFINITY, ease: "linear" }}
                                >
                                  <Loader2 className="w-4 h-4 mr-2" />
                                </motion.div>
                                Saving Preferences...
                              </>
                            ) : (
                              <>
                                <Save className="w-4 h-4 mr-2" />
                                Save Preferences
                              </>
                            )}
                          </Button>
                        </motion.div>
                      )}
                    </CardContent>
                  </Card>
                </motion.div>

                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.5, delay: 0.1 }}
                >
                  <Card>
                    <CardHeader>
                      <div className="flex items-center gap-3">
                        <motion.div
                          animate={{ scale: [1, 1.1, 1] }}
                          transition={{ duration: 2, repeat: Number.POSITIVE_INFINITY }}
                          className="p-2 rounded-lg bg-primary/10"
                        >
                          <Sparkles className="w-5 h-5 text-primary" />
                        </motion.div>
                        <div>
                          <CardTitle className="text-xl">About This Tool</CardTitle>
                          <CardDescription className="text-base">Learn more about AI model comparison</CardDescription>
                        </div>
                      </div>
                    </CardHeader>
                    <CardContent className="space-y-4">
                      <p className="text-muted-foreground leading-relaxed">
                        This tool allows you to compare responses from different large language models side by side.
                        It's designed to help you understand the unique strengths, weaknesses, and characteristics of
                        each AI model, enabling you to make informed decisions about which AI works best for your
                        specific use cases.
                      </p>

                      <div className="grid gap-4 md:grid-cols-2">
                        <motion.div whileHover={{ scale: 1.02, y: -2 }} className="p-4 rounded-lg border bg-muted/50">
                          <h4 className="font-medium mb-2 flex items-center gap-2">
                            <motion.div
                              animate={{ rotate: [0, 360] }}
                              transition={{ duration: 2, repeat: Number.POSITIVE_INFINITY, ease: "linear" }}
                            >
                              <Zap className="w-4 h-4 text-yellow-500" />
                            </motion.div>
                            Real-time Comparison
                          </h4>
                          <p className="text-sm text-muted-foreground">
                            Generate and compare responses from multiple AI models simultaneously
                          </p>
                        </motion.div>

                        <motion.div whileHover={{ scale: 1.02, y: -2 }} className="p-4 rounded-lg border bg-muted/50">
                          <h4 className="font-medium mb-2 flex items-center gap-2">
                            <motion.div
                              animate={{ scale: [1, 1.1, 1] }}
                              transition={{ duration: 1.5, repeat: Number.POSITIVE_INFINITY }}
                            >
                              <Save className="w-4 h-4 text-green-500" />
                            </motion.div>
                            Save & Share
                          </h4>
                          <p className="text-sm text-muted-foreground">
                            Save your comparisons and share insights with your team
                          </p>
                        </motion.div>
                      </div>

                      <motion.div whileHover={{ scale: 1.02 }} whileTap={{ scale: 0.98 }}>
                        <Button variant="outline" className="mt-4">
                          <Info className="w-4 h-4 mr-2" />
                          Learn More About AI Models
                        </Button>
                      </motion.div>
                    </CardContent>
                  </Card>
                </motion.div>
              </TabsContent>
            </Tabs>
          </motion.div>
        </div>

        {/* Export Dialog */}
        <ExportDialog
          open={exportDialogOpen}
          onOpenChange={setExportDialogOpen}
          prompt={prompt}
          responses={responses}
          selectedModels={selectedModels}
          availableModels={availableModels}
        />

        {/* Share Dialog */}
        <ShareDialog
          open={shareDialogOpen}
          onOpenChange={setShareDialogOpen}
          prompt={prompt}
          responses={responses}
          selectedModels={selectedModels}
          availableModels={availableModels}
        />
      </div>
    </PageTransition>
  )
}
