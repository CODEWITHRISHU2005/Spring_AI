"use client"

import { useState, useEffect } from "react"
import { Textarea } from "@/components/ui/textarea"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Skeleton } from "@/components/ui/skeleton"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import {
  AlertDialog,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from "@/components/ui/alert-dialog"
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from "@/components/ui/dropdown-menu"
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip"
import {
  Brain,
  Clock,
  Copy,
  Download,
  History,
  Info,
  Loader2,
  MoreHorizontal,
  Save,
  Share2,
  Sparkles,
  ThumbsDown,
  ThumbsUp,
} from "lucide-react"
import { useToast } from "@/hooks/use-toast"
import { useTheme } from "next-themes"

// Define model types
type Model = {
  id: string
  name: string
  provider: string
  color: string
  description: string
}

type ModelResponse = {
  modelId: string
  content: string
  error: boolean
  loading: boolean
  responseTime?: number
}

export default function LLMComparisonTool() {
  const [prompt, setPrompt] = useState("")
  const [responses, setResponses] = useState<ModelResponse[]>([])
  const [selectedModels, setSelectedModels] = useState<string[]>(["claude", "gpt4o", "gemma"])
  const [isComparing, setIsComparing] = useState(false)
  const [promptHistory, setPromptHistory] = useState<string[]>([])
  const [activeTab, setActiveTab] = useState("compare")
  const { toast } = useToast()
  const { theme, setTheme } = useTheme()
  const [fontSize, setFontSize] = useState("medium")
  const [mounted, setMounted] = useState(false)

  // Prevent hydration mismatch
  useEffect(() => {
    setMounted(true)
  }, [])

  // Apply font size to body element
  useEffect(() => {
    if (mounted) {
      document.body.style.fontSize = {
        small: '0.875rem',
        medium: '1rem',
        large: '1.125rem'
      }[fontSize] || '1rem';
    }
  }, [fontSize, mounted]);

  if (!mounted) {
    return null;
  }

  // Available models
  const models: Model[] = [
    {
      id: "claude",
      name: "Anthropic (Claude)",
      provider: "Anthropic",
      color: "bg-purple-500",
      description: "Claude excels at thoughtful, nuanced conversations and complex reasoning tasks.",
    },
    {
      id: "gpt4o",
      name: "OpenAI (GPT-4o)",
      provider: "OpenAI",
      color: "bg-green-500",
      description: "GPT-4o is OpenAI's most advanced model with multimodal capabilities and fast responses.",
    },
    {
      id: "gemma",
      name: "Ollama (Gemma 2)",
      provider: "Ollama",
      color: "bg-orange-500",
      description: "Gemma is an open-source, lightweight model that can run locally through Ollama.",
    },
    {
      id: "mistral",
      name: "Mistral Large",
      provider: "Mistral AI",
      color: "bg-blue-500",
      description: "Mistral Large is a powerful model with strong reasoning capabilities and efficiency.",
    },
    {
      id: "llama3",
      name: "Meta (Llama 3)",
      provider: "Meta",
      color: "bg-pink-500",
      description: "Llama 3 is Meta's latest open model with strong performance across various tasks.",
    },
  ]

  // Mock responses for demonstration
  const mockResponses = {
    claude: `Java is a high-level, class-based, object-oriented programming language designed to have as few implementation dependencies as possible. It was developed by James Gosling at Sun Microsystems (which has since been acquired by Oracle) and released in 1995.

Key features of Java include:
- Platform independence ("Write Once, Run Anywhere")
- Object-oriented design
- Automatic memory management through garbage collection
- Strong type checking
- Extensive standard library

Java is widely used for developing enterprise applications, web applications, mobile apps (Android), and more.`,

    gpt4o: `Java is a versatile, object-oriented programming language first released by Sun Microsystems in 1995 and now owned by Oracle Corporation. It's designed to be platform-independent through its "write once, run anywhere" approach, where code is compiled to bytecode that runs on the Java Virtual Machine (JVM).

Key characteristics of Java include:
• Strong typing and compile-time error checking
• Automatic memory management (garbage collection)
• Multi-threading capabilities
• Rich standard library (Java API)
• Large ecosystem of frameworks and tools

Java is widely used for enterprise applications, Android mobile development, web services, and large-scale systems. Its syntax is influenced by C and C++, but with fewer low-level facilities and a stronger emphasis on object-oriented design principles.`,

    gemma: `Java is a programming language and software development platform developed by Java Inc. Inc. in 1996. It has become the most popular programming language for Android, Windows, Linux, Mac, Web, and other platforms worldwide.

Java is known for its "write once, run anywhere" capability, meaning that compiled Java code can run on all platforms that support Java without recompilation. It is an object-oriented language with syntax similar to C and C++.

Key features of Java include:
- Platform independence
- Object-oriented programming
- Automatic memory management (garbage collection)
- Strong type checking
- Rich standard library
- Multithreading support`,

    mistral: `Java is a high-level, class-based, object-oriented programming language that was designed to have as few implementation dependencies as possible. Created by James Gosling at Sun Microsystems (now owned by Oracle), Java was first released in 1995.

The language follows the WORA (Write Once, Run Anywhere) principle, meaning that compiled Java code can run on all platforms that support Java without recompilation. This is achieved through the Java Virtual Machine (JVM), which executes the Java bytecode.

Key features of Java include:
• Object-oriented design
• Platform independence
• Automatic memory management
• Strong type checking
• Extensive standard libraries
• Multithreading capabilities

Java is widely used for developing enterprise applications, web applications, Android mobile applications, and embedded systems. Its syntax is influenced by C and C++, but with fewer low-level facilities.`,

    llama3: `Java is a high-level, class-based, object-oriented programming language that was first released by Sun Microsystems in 1995 (now owned by Oracle Corporation). It was designed with the principle of "Write Once, Run Anywhere" (WORA), meaning that compiled Java code can run on any device that has a Java Virtual Machine (JVM) without needing to be recompiled.

Key characteristics of Java include:

1. Platform independence through the JVM
2. Strong typing and compile-time error checking
3. Automatic memory management (garbage collection)
4. Object-oriented programming paradigm
5. Rich standard library and ecosystem
6. Multithreading support
7. Security features

Java is widely used for developing:
- Enterprise applications
- Android mobile applications
- Web applications and services
- Scientific applications
- Financial services applications
- Big data technologies

Its syntax is similar to C and C++ but with a simpler object model and fewer low-level facilities.`,
  }

  // Function to simulate fetching responses
  const compareModels = async () => {
    if (!prompt.trim()) {
      toast({
        title: "Empty prompt",
        description: "Please enter a prompt to compare models.",
        variant: "destructive",
      })
      return
    }

    setIsComparing(true)

    // Initialize responses with loading state
    const initialResponses = selectedModels.map((modelId) => ({
      modelId,
      content: "",
      error: false,
      loading: true,
    }))

    setResponses(initialResponses)

    // Add to history if not already there
    if (!promptHistory.includes(prompt)) {
      setPromptHistory((prev) => [prompt, ...prev].slice(0, 10))
    }

    // Simulate different response times
    const delays = {
      claude: 1500 + Math.random() * 1000,
      gpt4o: 800 + Math.random() * 700,
      gemma: 2000 + Math.random() * 1500,
      mistral: 1200 + Math.random() * 900,
      llama3: 1800 + Math.random() * 1200,
    }

    // Simulate fetching responses with different timings
    selectedModels.forEach(async (modelId, index) => {
      const delay = delays[modelId as keyof typeof delays] || 1500

      setTimeout(() => {
        setResponses((prev) => {
          const newResponses = [...prev]
          const responseIndex = newResponses.findIndex((r) => r.modelId === modelId)

          if (responseIndex !== -1) {
            // 10% chance of error for demo purposes
            const hasError = Math.random() < 0.1

            newResponses[responseIndex] = {
              modelId,
              content: hasError ? "" : mockResponses[modelId as keyof typeof mockResponses],
              error: hasError,
              loading: false,
              responseTime: delay,
            }
          }

          return newResponses
        })

        // Check if all responses are done
        setTimeout(() => {
          setIsComparing(false)
        }, 100)
      }, delay)
    })
  }

  // Function to copy response to clipboard
  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text)
    toast({
      title: "Copied to clipboard",
      description: "Response has been copied to your clipboard.",
    })
  }

  // Function to save comparison
  const saveComparison = () => {
    const comparisonData = {
      prompt,
      responses,
      timestamp: new Date().toISOString(),
    }

    localStorage.setItem(`comparison-${Date.now()}`, JSON.stringify(comparisonData))

    toast({
      title: "Comparison saved",
      description: "This comparison has been saved to your browser.",
    })
  }

  // Function to share comparison
  const shareComparison = () => {
    // In a real app, this would generate a shareable link
    toast({
      title: "Share feature",
      description: "In a real app, this would generate a shareable link to this comparison.",
    })
  }

  // Function to export comparison as JSON
  const exportComparison = () => {
    const comparisonData = {
      prompt,
      responses: responses.map((r) => ({
        model: models.find((m) => m.id === r.modelId)?.name || r.modelId,
        content: r.content,
        error: r.error,
        responseTime: r.responseTime,
      })),
      timestamp: new Date().toISOString(),
    }

    const dataStr = JSON.stringify(comparisonData, null, 2)
    const dataUri = `data:application/json;charset=utf-8,${encodeURIComponent(dataStr)}`

    const exportFileDefaultName = `llm-comparison-${Date.now()}.json`

    const linkElement = document.createElement("a")
    linkElement.setAttribute("href", dataUri)
    linkElement.setAttribute("download", exportFileDefaultName)
    linkElement.click()
  }

  // Get model details by ID
  const getModelById = (id: string) => {
    return models.find((model) => model.id === id)
  }

  // Sort responses by response time (fastest first)
  const sortedResponses = [...responses].sort((a, b) => {
    if (!a.responseTime) return 1
    if (!b.responseTime) return -1
    return a.responseTime - b.responseTime
  })

  return (
    <div className="container mx-auto py-8 px-4 max-w-6xl">
      <header className="mb-8 text-center">
        <h1 className="text-4xl font-bold mb-2 bg-gradient-to-r from-purple-600 via-blue-500 to-green-500 bg-clip-text text-transparent">
          Exploring Different LLM Models
        </h1>
        <p className="text-muted-foreground max-w-2xl mx-auto">
          Compare responses from various large language models to see how they differ in style, knowledge, and
          reasoning.
        </p>
      </header>

      <Tabs value={activeTab} onValueChange={setActiveTab} className="mb-8">
        <TabsList className="grid w-full max-w-md mx-auto grid-cols-2">
          <TabsTrigger value="compare">Compare Models</TabsTrigger>
          <TabsTrigger value="settings">Settings</TabsTrigger>
        </TabsList>

        <TabsContent value="compare" className="mt-6">
          <div className="grid gap-6">
            <div className="flex flex-col gap-4">
              <div className="flex flex-wrap gap-2 mb-2">
                {selectedModels.map((modelId) => {
                  const model = getModelById(modelId)
                  return (
                    <Badge key={modelId} variant="outline" className="flex items-center gap-1">
                      <div className={`w-2 h-2 rounded-full ${model?.color}`} />
                      {model?.name}
                    </Badge>
                  )
                })}
              </div>

              <div className="relative">
                <Textarea
                  placeholder="Enter your prompt here... (e.g., 'What is Java?', 'Explain quantum computing', etc.)"
                  value={prompt}
                  onChange={(e) => setPrompt(e.target.value)}
                  className="min-h-[120px] resize-none pr-10"
                />

                <TooltipProvider>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <Button
                        variant="ghost"
                        size="icon"
                        className="absolute right-2 top-2 text-muted-foreground"
                        onClick={() => setActiveTab("history")}
                      >
                        <History className="h-4 w-4" />
                      </Button>
                    </TooltipTrigger>
                    <TooltipContent>
                      <p>View prompt history</p>
                    </TooltipContent>
                  </Tooltip>
                </TooltipProvider>
              </div>

              <div className="flex justify-between items-center">
                <DropdownMenu>
                  <DropdownMenuTrigger asChild>
                    <Button variant="outline" size="sm" className="text-muted-foreground">
                      <History className="h-4 w-4 mr-2" />
                      History
                    </Button>
                  </DropdownMenuTrigger>
                  <DropdownMenuContent align="start" className="w-[300px]">
                    {promptHistory.length > 0 ? (
                      promptHistory.map((historyItem, index) => (
                        <DropdownMenuItem
                          key={index}
                          onClick={() => setPrompt(historyItem)}
                          className="cursor-pointer truncate"
                        >
                          {historyItem}
                        </DropdownMenuItem>
                      ))
                    ) : (
                      <DropdownMenuItem disabled>No history yet</DropdownMenuItem>
                    )}
                  </DropdownMenuContent>
                </DropdownMenu>

                <Button onClick={compareModels} disabled={isComparing || !prompt.trim()} className="ml-auto">
                  {isComparing ? (
                    <>
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                      Comparing...
                    </>
                  ) : (
                    <>
                      <Sparkles className="mr-2 h-4 w-4" />
                      Compare All Models
                    </>
                  )}
                </Button>
              </div>
            </div>

            {responses.length > 0 && (
              <div className="mt-4">
                <div className="mb-4">
                  <h2 className="text-xl font-semibold mb-2">Response Order:</h2>
                  <ol className="list-decimal list-inside space-y-1 pl-2">
                    {sortedResponses.map((response, index) => {
                      const model = getModelById(response.modelId)
                      return (
                        <li key={index} className="flex items-center gap-2">
                          <span className={`text-${model?.color.replace("bg-", "")}`}>{model?.name}</span>
                          {!response.loading && !response.error && response.responseTime && (
                            <span className="text-xs text-muted-foreground">
                              ({(response.responseTime / 1000).toFixed(1)}s)
                            </span>
                          )}
                          {index === 0 && (
                            <Badge variant="outline" className="text-xs ml-2">
                              fastest
                            </Badge>
                          )}
                        </li>
                      )
                    })}
                  </ol>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  {responses.map((response, index) => {
                    const model = getModelById(response.modelId)
                    return (
                      <Card key={index} className={`overflow-hidden border-t-4 ${model?.color}`}>
                        <CardHeader className="pb-2">
                          <CardTitle className="flex justify-between items-center">
                            <div className="flex items-center gap-2">
                              <Brain className="h-5 w-5" />
                              <span>{model?.name}</span>
                            </div>
                            <DropdownMenu>
                              <DropdownMenuTrigger asChild>
                                <Button variant="ghost" size="icon" className="h-8 w-8">
                                  <MoreHorizontal className="h-4 w-4" />
                                </Button>
                              </DropdownMenuTrigger>
                              <DropdownMenuContent align="end">
                                <DropdownMenuItem onClick={() => copyToClipboard(response.content)}>
                                  <Copy className="h-4 w-4 mr-2" />
                                  Copy response
                                </DropdownMenuItem>
                                <DropdownMenuItem
                                  onClick={() =>
                                    toast({ title: "Feedback recorded", description: "Thank you for your feedback!" })
                                  }
                                >
                                  <ThumbsUp className="h-4 w-4 mr-2" />
                                  Mark as helpful
                                </DropdownMenuItem>
                                <DropdownMenuItem
                                  onClick={() =>
                                    toast({ title: "Feedback recorded", description: "Thank you for your feedback!" })
                                  }
                                >
                                  <ThumbsDown className="h-4 w-4 mr-2" />
                                  Mark as unhelpful
                                </DropdownMenuItem>
                              </DropdownMenuContent>
                            </DropdownMenu>
                          </CardTitle>
                        </CardHeader>
                        <CardContent>
                          <div className="h-[300px] overflow-y-auto">
                            {response.loading ? (
                              <div className="space-y-2">
                                <Skeleton className="h-4 w-full" />
                                <Skeleton className="h-4 w-[90%]" />
                                <Skeleton className="h-4 w-[95%]" />
                                <Skeleton className="h-4 w-[85%]" />
                                <Skeleton className="h-4 w-[90%]" />
                              </div>
                            ) : response.error ? (
                              <div className="flex flex-col items-center justify-center h-full text-center text-muted-foreground">
                                <p className="text-red-500 mb-2">Error: Failed to fetch</p>
                                <Button
                                  variant="outline"
                                  size="sm"
                                  onClick={() => {
                                    setResponses((prev) => {
                                      const newResponses = [...prev]
                                      const responseIndex = newResponses.findIndex(
                                        (r) => r.modelId === response.modelId,
                                      )
                                      if (responseIndex !== -1) {
                                        newResponses[responseIndex].loading = true
                                        newResponses[responseIndex].error = false

                                        // Simulate retry
                                        setTimeout(() => {
                                          setResponses((prev) => {
                                            const retryResponses = [...prev]
                                            const retryIndex = retryResponses.findIndex(
                                              (r) => r.modelId === response.modelId,
                                            )
                                            if (retryIndex !== -1) {
                                              retryResponses[retryIndex] = {
                                                modelId: response.modelId,
                                                content: mockResponses[response.modelId as keyof typeof mockResponses],
                                                error: false,
                                                loading: false,
                                                responseTime: 1500,
                                              }
                                            }
                                            return retryResponses
                                          })
                                        }, 1500)
                                      }
                                      return newResponses
                                    })
                                  }}
                                >
                                  Retry
                                </Button>
                              </div>
                            ) : (
                              <div className="prose prose-sm max-w-none dark:prose-invert">
                                {response.content.split("\n").map((paragraph, i) => (
                                  <p key={i}>{paragraph}</p>
                                ))}
                              </div>
                            )}
                          </div>

                          {!response.loading && !response.error && (
                            <div className="flex items-center justify-between mt-4 pt-2 border-t">
                              <div className="flex items-center text-xs text-muted-foreground">
                                <Clock className="h-3 w-3 mr-1" />
                                {response.responseTime ? `${(response.responseTime / 1000).toFixed(1)}s` : "N/A"}
                              </div>
                              <div className="flex gap-1">
                                <Button
                                  variant="ghost"
                                  size="icon"
                                  className="h-8 w-8"
                                  onClick={() => copyToClipboard(response.content)}
                                >
                                  <Copy className="h-4 w-4" />
                                </Button>
                                <Button
                                  variant="ghost"
                                  size="icon"
                                  className="h-8 w-8"
                                  onClick={() =>
                                    toast({ title: "Feedback recorded", description: "Thank you for your feedback!" })
                                  }
                                >
                                  <ThumbsUp className="h-4 w-4" />
                                </Button>
                                <Button
                                  variant="ghost"
                                  size="icon"
                                  className="h-8 w-8"
                                  onClick={() =>
                                    toast({ title: "Feedback recorded", description: "Thank you for your feedback!" })
                                  }
                                >
                                  <ThumbsDown className="h-4 w-4" />
                                </Button>
                              </div>
                            </div>
                          )}
                        </CardContent>
                      </Card>
                    )
                  })}
                </div>

                {responses.length > 0 && !isComparing && (
                  <div className="flex justify-end gap-2 mt-6">
                    <Button variant="outline" size="sm" onClick={saveComparison}>
                      <Save className="h-4 w-4 mr-2" />
                      Save
                    </Button>
                    <Button variant="outline" size="sm" onClick={shareComparison}>
                      <Share2 className="h-4 w-4 mr-2" />
                      Share
                    </Button>
                    <Button variant="outline" size="sm" onClick={exportComparison}>
                      <Download className="h-4 w-4 mr-2" />
                      Export
                    </Button>
                  </div>
                )}
              </div>
            )}
          </div>
        </TabsContent>

        <TabsContent value="settings" className="mt-6">
          <Card>
            <CardHeader>
              <CardTitle>Model Selection</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid gap-4">
                <p className="text-sm text-muted-foreground">
                  Select which models you want to compare. You can select up to 5 models.
                </p>

                <div className="grid gap-4">
                  {models.map((model) => (
                    <div key={model.id} className="flex items-center space-x-2">
                      <input
                        type="checkbox"
                        id={model.id}
                        checked={selectedModels.includes(model.id)}
                        onChange={(e) => {
                          if (e.target.checked) {
                            if (selectedModels.length < 5) {
                              setSelectedModels([...selectedModels, model.id])
                            } else {
                              toast({
                                title: "Maximum models reached",
                                description: "You can select up to 5 models at a time.",
                                variant: "destructive",
                              })
                            }
                          } else {
                            if (selectedModels.length > 1) {
                              setSelectedModels(selectedModels.filter((id) => id !== model.id))
                            } else {
                              toast({
                                title: "Minimum models required",
                                description: "You must select at least one model.",
                                variant: "destructive",
                              })
                            }
                          }
                        }}
                        className="rounded border-gray-300 text-primary focus:ring-primary"
                      />
                      <div className="grid gap-1.5">
                        <label
                          htmlFor={model.id}
                          className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70 flex items-center gap-2"
                        >
                          <div className={`w-3 h-3 rounded-full ${model.color}`} />
                          {model.name}
                        </label>
                        <p className="text-sm text-muted-foreground">{model.description}</p>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className="mt-4">
            <CardHeader>
              <CardTitle>Display Settings</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid gap-4">
                <div className="grid gap-2">
                  <label htmlFor="theme" className="text-sm font-medium">
                    Theme
                  </label>
                  <Select value={theme} onValueChange={setTheme}>
                    <SelectTrigger id="theme">
                      <SelectValue placeholder="Select theme" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="light">Light</SelectItem>
                      <SelectItem value="dark">Dark</SelectItem>
                      <SelectItem value="system">System</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div className="grid gap-2">
                  <label htmlFor="font-size" className="text-sm font-medium">
                    Font Size
                  </label>
                  <Select value={fontSize} onValueChange={setFontSize}>
                    <SelectTrigger id="font-size">
                      <SelectValue placeholder="Select font size" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="small">Small</SelectItem>
                      <SelectItem value="medium">Medium</SelectItem>
                      <SelectItem value="large">Large</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className="mt-4">
            <CardHeader>
              <CardTitle>About</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid gap-4">
                <p className="text-sm text-muted-foreground">
                  This tool allows you to compare responses from different large language models side by side. It's
                  useful for understanding the strengths and weaknesses of each model, as well as their different styles
                  and approaches to answering questions.
                </p>

                <AlertDialog>
                  <AlertDialogTrigger asChild>
                    <Button variant="outline" className="w-full">
                      <Info className="h-4 w-4 mr-2" />
                      Learn More
                    </Button>
                  </AlertDialogTrigger>
                  <AlertDialogContent>
                    <AlertDialogHeader>
                      <AlertDialogTitle>About LLM Comparison Tool</AlertDialogTitle>
                      <AlertDialogDescription>
                        <p className="mb-4">
                          This tool is designed to help researchers, developers, and AI enthusiasts compare the outputs
                          of different large language models.
                        </p>
                        <p className="mb-4">By seeing how different models respond to the same prompt, you can:</p>
                        <ul className="list-disc pl-6 mb-4 space-y-1">
                          <li>Evaluate factual accuracy across models</li>
                          <li>Compare reasoning capabilities</li>
                          <li>Assess writing styles and tones</li>
                          <li>Identify biases and limitations</li>
                          <li>Determine which model is best for specific use cases</li>
                        </ul>
                        <p>
                          Note: In a production environment, this tool would connect to actual API endpoints for each
                          model. The current implementation uses mock data for demonstration purposes.
                        </p>
                      </AlertDialogDescription>
                    </AlertDialogHeader>
                    <AlertDialogFooter>
                      <AlertDialogCancel>Close</AlertDialogCancel>
                    </AlertDialogFooter>
                  </AlertDialogContent>
                </AlertDialog>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )
}
