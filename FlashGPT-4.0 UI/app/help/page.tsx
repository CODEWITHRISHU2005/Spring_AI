"use client"

import { useState } from "react"
import { motion } from "framer-motion"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Badge } from "@/components/ui/badge"
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion"
import {
  HelpCircle,
  Search,
  Zap,
  Users,
  Save,
  Settings,
  MessageCircle,
  BookOpen,
  Video,
  FileText,
  ExternalLink,
  ChevronRight,
  Sparkles,
  Brain,
  Shield,
  ArrowRight,
} from "lucide-react"
import { PageTransition } from "@/components/ui/page-transition"
import Link from "next/link"

export default function HelpPage() {
  const [searchQuery, setSearchQuery] = useState("")
  const [activeCategory, setActiveCategory] = useState("getting-started")

  const helpCategories = [
    {
      id: "getting-started",
      title: "Getting Started",
      icon: Sparkles,
      color: "text-green-500",
      description: "Learn the basics of using LLM Compare",
    },
    {
      id: "models",
      title: "AI Models",
      icon: Brain,
      color: "text-blue-500",
      description: "Understanding different AI models and their capabilities",
    },
    {
      id: "features",
      title: "Features",
      icon: Zap,
      color: "text-yellow-500",
      description: "Explore all the features and tools available",
    },
    {
      id: "account",
      title: "Account & Settings",
      icon: Settings,
      color: "text-purple-500",
      description: "Manage your account, preferences, and data",
    },
    {
      id: "troubleshooting",
      title: "Troubleshooting",
      icon: Shield,
      color: "text-red-500",
      description: "Common issues and how to resolve them",
    },
  ]

  const faqData = [
    {
      category: "getting-started",
      question: "How do I start comparing AI models?",
      answer:
        "Simply select the models you want to compare from the model selection panel, enter your prompt in the text area, and click 'Generate Responses'. The tool will query all selected models and display their responses side by side.",
    },
    {
      category: "getting-started",
      question: "Do I need an account to use LLM Compare?",
      answer:
        "You can use the basic comparison features without an account. However, creating an account allows you to save comparisons, customize preferences, and access additional features like sharing and exporting.",
    },
    {
      category: "getting-started",
      question: "How many models can I compare at once?",
      answer:
        "You can compare up to 5 AI models simultaneously. This limit ensures optimal performance and readability of the comparison results.",
    },
    {
      category: "models",
      question: "What's the difference between GPT-4.0 and Claude 3 Opus?",
      answer:
        "GPT-4.0 excels at complex reasoning and creative tasks, while Claude 3 Opus is known for its analytical capabilities and safety features. Both are top-tier models but have different strengths depending on your use case.",
    },
    {
      category: "models",
      question: "Why is Ollama Mistral different from other models?",
      answer:
        "Ollama Mistral runs locally on your machine, providing privacy and control over your data. It's an open-source model that doesn't send your prompts to external servers, making it ideal for sensitive or private content.",
    },
    {
      category: "models",
      question: "Which model should I choose for my task?",
      answer:
        "It depends on your specific needs: GPT-4.0 for creative writing and complex reasoning, Claude for analysis and safety-critical tasks, Gemini Pro for multimodal content, and Ollama Mistral for privacy-focused applications.",
    },
    {
      category: "features",
      question: "How do I save a comparison?",
      answer:
        "After generating responses, click the 'Save Comparison' button. You'll need to be logged in to save comparisons. You can give it a name and choose whether to make it public or private.",
    },
    {
      category: "features",
      question: "Can I export my comparison results?",
      answer:
        "Yes! Click the 'Export' button after generating responses. You can export in various formats including PDF for reports and JSON for data analysis.",
    },
    {
      category: "features",
      question: "How does the sharing feature work?",
      answer:
        "When you save a comparison as 'public', it gets a unique URL that you can share with others. They'll be able to view the comparison results without needing an account.",
    },
    {
      category: "account",
      question: "How do I change my theme preferences?",
      answer:
        "Go to Settings (gear icon in the navigation) or use the Settings tab on the main page. You can choose between Light, Dark, or System theme, and adjust font size and other preferences.",
    },
    {
      category: "account",
      question: "Can I set default models for new comparisons?",
      answer:
        "Yes! In your Settings, you can select which models should be pre-selected when you start a new comparison. This saves time if you frequently use the same set of models.",
    },
    {
      category: "account",
      question: "How do I delete my account and data?",
      answer:
        "In Settings under 'Data & Privacy', you'll find options to export your data or permanently delete your account. Account deletion is irreversible and removes all your saved comparisons.",
    },
    {
      category: "troubleshooting",
      question: "Why are some models showing as 'Coming Soon'?",
      answer:
        "Some models may be temporarily unavailable due to maintenance, API limits, or they're still being integrated. Check back later or contact support if a model remains unavailable for extended periods.",
    },
    {
      category: "troubleshooting",
      question: "My responses are taking too long to generate. What should I do?",
      answer:
        "Response times vary by model and server load. If responses consistently take too long, try using fewer models simultaneously or check your internet connection. Some models are naturally slower but provide higher quality responses.",
    },
    {
      category: "troubleshooting",
      question: "I'm getting error messages when generating responses. How do I fix this?",
      answer:
        "Common causes include network issues, API rate limits, or server maintenance. Try refreshing the page, checking your internet connection, or waiting a few minutes before trying again. If problems persist, contact support.",
    },
  ]

  const quickStartSteps = [
    {
      step: 1,
      title: "Select AI Models",
      description: "Choose up to 5 models from our available selection",
      icon: Users,
    },
    {
      step: 2,
      title: "Enter Your Prompt",
      description: "Type your question or request in the prompt field",
      icon: MessageCircle,
    },
    {
      step: 3,
      title: "Generate Responses",
      description: "Click the generate button to get responses from all selected models",
      icon: Zap,
    },
    {
      step: 4,
      title: "Compare & Save",
      description: "Review the responses side by side and save your comparison",
      icon: Save,
    },
  ]

  const resources = [
    {
      title: "Video Tutorials",
      description: "Watch step-by-step guides on using LLM Compare",
      icon: Video,
      link: "#",
      badge: "New",
    },
    {
      title: "Best Practices Guide",
      description: "Learn how to write effective prompts for better results",
      icon: BookOpen,
      link: "#",
      badge: "Popular",
    },
    {
      title: "API Documentation",
      description: "Integrate LLM Compare into your own applications",
      icon: FileText,
      link: "#",
      badge: "Developer",
    },
    {
      title: "Community Forum",
      description: "Connect with other users and share insights",
      icon: MessageCircle,
      link: "#",
      badge: "Community",
    },
  ]

  const filteredFAQs = faqData.filter(
    (faq) =>
      faq.category === activeCategory &&
      (faq.question.toLowerCase().includes(searchQuery.toLowerCase()) ||
        faq.answer.toLowerCase().includes(searchQuery.toLowerCase())),
  )

  // Find the active category once
  const activeCategory_data = helpCategories.find((cat) => cat.id === activeCategory)
  // Get the icon component
  const CategoryIcon = activeCategory_data?.icon || HelpCircle

  return (
    <PageTransition>
      <div className="min-h-screen bg-background">
        <div className="container mx-auto px-4 py-8 max-w-6xl">
          {/* Header */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            className="text-center mb-12"
          >
            <motion.div
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.5, delay: 0.2 }}
              className="inline-flex items-center gap-3 mb-6"
            >
              <div className="p-3 rounded-xl bg-primary/10">
                <HelpCircle className="w-8 h-8 text-primary" />
              </div>
              <h1 className="text-4xl font-bold">Help Center</h1>
            </motion.div>
            <motion.p
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.4 }}
              className="text-xl text-muted-foreground max-w-2xl mx-auto"
            >
              Everything you need to know about using LLM Compare effectively
            </motion.p>
          </motion.div>

          {/* Search */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.3 }}
            className="mb-8"
          >
            <div className="relative max-w-md mx-auto">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-muted-foreground w-4 h-4" />
              <Input
                placeholder="Search help articles..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="pl-10 h-12 text-base"
              />
            </div>
          </motion.div>

          {/* Quick Start Guide */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.4 }}
            className="mb-12"
          >
            <Card className="border-2 border-dashed border-primary/20 bg-primary/5">
              <CardHeader>
                <div className="flex items-center gap-3">
                  <motion.div
                    animate={{ rotate: [0, 360] }}
                    transition={{ duration: 4, repeat: Number.POSITIVE_INFINITY, ease: "linear" }}
                    className="p-2 rounded-lg bg-primary/10"
                  >
                    <Sparkles className="w-5 h-5 text-primary" />
                  </motion.div>
                  <div>
                    <CardTitle className="text-xl">Quick Start Guide</CardTitle>
                    <CardDescription>Get up and running in 4 simple steps</CardDescription>
                  </div>
                </div>
              </CardHeader>
              <CardContent>
                <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
                  {quickStartSteps.map((step, index) => (
                    <motion.div
                      key={step.step}
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ duration: 0.3, delay: 0.5 + index * 0.1 }}
                      className="relative"
                    >
                      <div className="flex flex-col items-center text-center p-4 rounded-lg border bg-background/50 hover:bg-background/80 transition-colors">
                        <div className="flex items-center justify-center w-12 h-12 rounded-full bg-primary/10 mb-3">
                          <step.icon className="w-6 h-6 text-primary" />
                        </div>
                        <div className="absolute -top-2 -left-2 w-6 h-6 rounded-full bg-primary text-primary-foreground text-sm font-bold flex items-center justify-center">
                          {step.step}
                        </div>
                        <h3 className="font-semibold mb-2">{step.title}</h3>
                        <p className="text-sm text-muted-foreground">{step.description}</p>
                      </div>
                      {index < quickStartSteps.length - 1 && (
                        <div className="hidden lg:block absolute top-1/2 -right-2 transform -translate-y-1/2">
                          <ArrowRight className="w-4 h-4 text-muted-foreground" />
                        </div>
                      )}
                    </motion.div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </motion.div>

          {/* Main Content */}
          <div className="grid gap-8 lg:grid-cols-4">
            {/* Categories Sidebar */}
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.5, delay: 0.5 }}
              className="lg:col-span-1"
            >
              <Card>
                <CardHeader>
                  <CardTitle className="text-lg">Categories</CardTitle>
                </CardHeader>
                <CardContent className="space-y-2">
                  {helpCategories.map((category, index) => (
                    <motion.button
                      key={category.id}
                      initial={{ opacity: 0, x: -10 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ duration: 0.3, delay: 0.6 + index * 0.1 }}
                      onClick={() => setActiveCategory(category.id)}
                      className={`w-full flex items-center gap-3 p-3 rounded-lg text-left transition-colors ${
                        activeCategory === category.id
                          ? "bg-primary/10 text-primary border border-primary/20"
                          : "hover:bg-muted/50"
                      }`}
                    >
                      <category.icon className={`w-5 h-5 ${category.color}`} />
                      <div className="flex-1 min-w-0">
                        <div className="font-medium text-sm">{category.title}</div>
                        <div className="text-xs text-muted-foreground truncate">{category.description}</div>
                      </div>
                      <ChevronRight className="w-4 h-4 text-muted-foreground" />
                    </motion.button>
                  ))}
                </CardContent>
              </Card>
            </motion.div>

            {/* FAQ Content */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.6 }}
              className="lg:col-span-3"
            >
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <CategoryIcon className={activeCategory_data?.color || "text-primary"} />
                    {activeCategory_data?.title || "Help"}
                  </CardTitle>
                  <CardDescription>
                    {activeCategory_data?.description || "Find answers to your questions"}
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  {filteredFAQs.length > 0 ? (
                    <Accordion type="single" collapsible className="space-y-2">
                      {filteredFAQs.map((faq, index) => (
                        <motion.div
                          key={index}
                          initial={{ opacity: 0, y: 10 }}
                          animate={{ opacity: 1, y: 0 }}
                          transition={{ duration: 0.3, delay: index * 0.1 }}
                        >
                          <AccordionItem value={`item-${index}`} className="border rounded-lg px-4">
                            <AccordionTrigger className="text-left hover:no-underline">{faq.question}</AccordionTrigger>
                            <AccordionContent className="text-muted-foreground leading-relaxed">
                              {faq.answer}
                            </AccordionContent>
                          </AccordionItem>
                        </motion.div>
                      ))}
                    </Accordion>
                  ) : (
                    <div className="text-center py-8">
                      <Search className="w-12 h-12 text-muted-foreground mx-auto mb-4" />
                      <h3 className="font-medium mb-2">No results found</h3>
                      <p className="text-muted-foreground">Try adjusting your search or browse a different category</p>
                    </div>
                  )}
                </CardContent>
              </Card>
            </motion.div>
          </div>

          {/* Resources Section */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.7 }}
            className="mt-12"
          >
            <div className="text-center mb-8">
              <h2 className="text-2xl font-bold mb-2">Additional Resources</h2>
              <p className="text-muted-foreground">Explore more ways to learn and get help</p>
            </div>

            <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-4">
              {resources.map((resource, index) => (
                <motion.div
                  key={index}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.3, delay: 0.8 + index * 0.1 }}
                  whileHover={{ y: -5, scale: 1.02 }}
                >
                  <Card className="h-full hover:shadow-lg transition-all duration-200 cursor-pointer group">
                    <CardHeader>
                      <div className="flex items-start justify-between">
                        <div className="flex items-center gap-3">
                          <div className="p-2 rounded-lg bg-primary/10 group-hover:bg-primary/20 transition-colors">
                            <resource.icon className="w-5 h-5 text-primary" />
                          </div>
                          <div>
                            <CardTitle className="text-base">{resource.title}</CardTitle>
                          </div>
                        </div>
                        {resource.badge && (
                          <Badge variant="secondary" className="text-xs">
                            {resource.badge}
                          </Badge>
                        )}
                      </div>
                      <CardDescription className="text-sm leading-relaxed">{resource.description}</CardDescription>
                    </CardHeader>
                    <CardContent className="pt-0">
                      <Button variant="ghost" size="sm" className="w-full justify-between group-hover:bg-muted/50">
                        Learn More
                        <ExternalLink className="w-4 h-4" />
                      </Button>
                    </CardContent>
                  </Card>
                </motion.div>
              ))}
            </div>
          </motion.div>

          {/* Contact Support */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.9 }}
            className="mt-12"
          >
            <Card className="border-2 border-dashed border-muted-foreground/20 bg-muted/20">
              <CardContent className="text-center py-8">
                <MessageCircle className="w-12 h-12 text-primary mx-auto mb-4" />
                <h3 className="text-xl font-semibold mb-2">Still need help?</h3>
                <p className="text-muted-foreground mb-6 max-w-md mx-auto">
                  Can't find what you're looking for? Our support team is here to help you get the most out of LLM
                  Compare.
                </p>
                <div className="flex flex-col sm:flex-row gap-3 justify-center">
                  <Button size="lg">
                    <MessageCircle className="w-4 h-4 mr-2" />
                    Contact Support
                  </Button>
                  <Button variant="outline" size="lg" asChild>
                    <Link href="/">
                      <ArrowRight className="w-4 h-4 mr-2" />
                      Back to App
                    </Link>
                  </Button>
                </div>
              </CardContent>
            </Card>
          </motion.div>
        </div>
      </div>
    </PageTransition>
  )
}
