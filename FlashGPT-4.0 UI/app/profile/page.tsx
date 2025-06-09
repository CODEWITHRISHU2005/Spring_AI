"use client"

import type React from "react"

import { useState, useEffect } from "react"
import { motion } from "framer-motion"
import { useRouter } from "next/navigation"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Textarea } from "@/components/ui/textarea"
import { Badge } from "@/components/ui/badge"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"
import {
  User,
  Settings,
  History,
  Clock,
  MessageSquare,
  Zap,
  Eye,
  Trash2,
  Search,
  Filter,
  Download,
  Share2,
  MoreHorizontal,
  ChevronRight,
  Sparkles,
  Brain,
  Users,
} from "lucide-react"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import { useAuth } from "@/contexts/auth-context"
import { toast } from "@/components/ui/use-toast"
import { PageTransition } from "@/components/ui/page-transition"

// Mock data for prompt history - replace with real API calls later
const mockPromptHistory = [
  {
    id: "1",
    prompt: "Explain quantum computing in simple terms",
    models: ["GPT-4.0", "Claude 3 Opus", "Gemini Pro"],
    createdAt: "2024-01-20T14:30:00Z",
    responseCount: 3,
    avgResponseTime: 2.1,
    category: "Education",
    isFavorite: true,
  },
  {
    id: "2",
    prompt: "Write a creative story about time travel and its consequences on human relationships",
    models: ["GPT-4.0", "Ollama Mistral"],
    createdAt: "2024-01-19T10:15:00Z",
    responseCount: 2,
    avgResponseTime: 3.5,
    category: "Creative Writing",
    isFavorite: false,
  },
  {
    id: "3",
    prompt: "Compare the pros and cons of renewable energy sources",
    models: ["Claude 3 Opus", "Gemini Pro", "GPT-4.0"],
    createdAt: "2024-01-18T16:45:00Z",
    responseCount: 3,
    avgResponseTime: 2.8,
    category: "Analysis",
    isFavorite: true,
  },
  {
    id: "4",
    prompt: "Help me debug this Python code for data analysis",
    models: ["GPT-4.0", "Claude 3 Sonnet"],
    createdAt: "2024-01-17T09:20:00Z",
    responseCount: 2,
    avgResponseTime: 1.9,
    category: "Programming",
    isFavorite: false,
  },
  {
    id: "5",
    prompt: "What are the latest trends in artificial intelligence and machine learning?",
    models: ["Gemini Pro", "GPT-4.0", "Claude 3 Opus"],
    createdAt: "2024-01-16T13:10:00Z",
    responseCount: 3,
    avgResponseTime: 2.4,
    category: "Technology",
    isFavorite: false,
  },
  {
    id: "6",
    prompt: "Create a business plan for a sustainable food delivery service",
    models: ["GPT-4.0", "Claude 3 Opus"],
    createdAt: "2024-01-15T11:30:00Z",
    responseCount: 2,
    avgResponseTime: 4.2,
    category: "Business",
    isFavorite: true,
  },
]

interface PromptHistoryItem {
  id: string
  prompt: string
  models: string[]
  createdAt: string
  responseCount: number
  avgResponseTime: number
  category: string
  isFavorite: boolean
}

export default function ProfilePage() {
  const { user, isLoading: authLoading, updateProfile } = useAuth()
  const router = useRouter()
  const [activeTab, setActiveTab] = useState("profile")
  const [isEditing, setIsEditing] = useState(false)
  const [isSaving, setIsSaving] = useState(false)

  // Profile form state
  const [profileData, setProfileData] = useState({
    name: "",
    email: "",
    bio: "",
    avatar: "",
  })

  // History state
  const [promptHistory, setPromptHistory] = useState<PromptHistoryItem[]>(mockPromptHistory)
  const [searchQuery, setSearchQuery] = useState("")
  const [selectedCategory, setSelectedCategory] = useState("all")
  const [sortBy, setSortBy] = useState("newest")

  useEffect(() => {
    if (authLoading) return

    if (!user) {
      router.push("/login")
      return
    }

    // Initialize profile data
    setProfileData({
      name: user.name || "",
      email: user.email || "",
      bio: user.bio || "",
      avatar: user.avatar || "",
    })
  }, [user, authLoading, router])

  const handleProfileUpdate = async (e: React.FormEvent) => {
    e.preventDefault()
    setIsSaving(true)

    try {
      const result = await updateProfile(profileData)
      if (result.success) {
        toast({
          title: "Profile Updated! ✨",
          description: "Your profile has been updated successfully",
        })
        setIsEditing(false)
      } else {
        toast({
          title: "Update Failed",
          description: result.error || "Failed to update profile",
          variant: "destructive",
        })
      }
    } catch (error) {
      toast({
        title: "Update Failed",
        description: "An unexpected error occurred",
        variant: "destructive",
      })
    } finally {
      setIsSaving(false)
    }
  }

  const getUserInitials = (name?: string, email?: string) => {
    if (name) {
      return name
        .split(" ")
        .map((n) => n[0])
        .join("")
        .toUpperCase()
        .slice(0, 2)
    }
    if (email) {
      return email[0].toUpperCase()
    }
    return "U"
  }

  const formatDate = (dateString: string) => {
    const date = new Date(dateString)
    const now = new Date()
    const diffInDays = Math.floor((now.getTime() - date.getTime()) / (1000 * 60 * 60 * 24))

    if (diffInDays === 0) return "Today"
    if (diffInDays === 1) return "Yesterday"
    if (diffInDays < 7) return `${diffInDays} days ago`
    return date.toLocaleDateString()
  }

  const categories = ["all", "Education", "Creative Writing", "Analysis", "Programming", "Technology", "Business"]

  const filteredHistory = promptHistory
    .filter((item) => {
      const matchesSearch =
        item.prompt.toLowerCase().includes(searchQuery.toLowerCase()) ||
        item.category.toLowerCase().includes(searchQuery.toLowerCase())
      const matchesCategory = selectedCategory === "all" || item.category === selectedCategory
      return matchesSearch && matchesCategory
    })
    .sort((a, b) => {
      switch (sortBy) {
        case "newest":
          return new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime()
        case "oldest":
          return new Date(a.createdAt).getTime() - new Date(b.createdAt).getTime()
        case "most-models":
          return b.models.length - a.models.length
        case "favorites":
          return b.isFavorite ? 1 : -1
        default:
          return 0
      }
    })

  const historyStats = {
    totalPrompts: promptHistory.length,
    totalModels: [...new Set(promptHistory.flatMap((item) => item.models))].length,
    avgResponseTime: promptHistory.reduce((sum, item) => sum + item.avgResponseTime, 0) / promptHistory.length,
    favoritePrompts: promptHistory.filter((item) => item.isFavorite).length,
  }

  if (authLoading) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <div className="text-center space-y-4">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary mx-auto" />
          <p>Loading profile...</p>
        </div>
      </div>
    )
  }

  if (!user) {
    return null
  }

  return (
    <PageTransition>
      <div className="min-h-screen bg-background">
        <div className="container mx-auto px-4 py-8 max-w-6xl">
          {/* Header */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            className="mb-8"
          >
            <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
              <div className="flex items-center gap-4">
                <motion.div whileHover={{ scale: 1.05 }} className="relative">
                  <Avatar className="h-16 w-16 border-4 border-background shadow-lg">
                    <AvatarImage src={user.avatar || ""} alt={user.name || "User"} />
                    <AvatarFallback className="text-lg font-bold bg-gradient-to-br from-purple-500 to-blue-500 text-white">
                      {getUserInitials(user.name, user.email)}
                    </AvatarFallback>
                  </Avatar>
                  <motion.div
                    animate={{ scale: [1, 1.2, 1] }}
                    transition={{ duration: 2, repeat: Number.POSITIVE_INFINITY }}
                    className="absolute -bottom-1 -right-1 w-5 h-5 bg-green-500 rounded-full border-2 border-background"
                  />
                </motion.div>
                <div>
                  <h1 className="text-3xl font-bold">{user.name || "Anonymous User"}</h1>
                  <p className="text-muted-foreground">{user.email}</p>
                  <div className="flex items-center gap-2 mt-1">
                    <Badge variant="secondary" className="text-xs">
                      Member since {new Date(user.createdAt).toLocaleDateString()}
                    </Badge>
                  </div>
                </div>
              </div>

              <div className="flex items-center gap-3">
                <Button variant="outline" onClick={() => setIsEditing(!isEditing)} className="flex items-center gap-2">
                  <Settings className="w-4 h-4" />
                  {isEditing ? "Cancel" : "Edit Profile"}
                </Button>
              </div>
            </div>
          </motion.div>

          {/* Tabs */}
          <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
            <TabsList className="grid w-full grid-cols-2 mb-8 h-12">
              <TabsTrigger value="profile" className="text-base font-medium">
                <User className="w-4 h-4 mr-2" />
                Profile
              </TabsTrigger>
              <TabsTrigger value="history" className="text-base font-medium">
                <History className="w-4 h-4 mr-2" />
                Prompt History
              </TabsTrigger>
            </TabsList>

            {/* Profile Tab */}
            <TabsContent value="profile" className="space-y-6">
              <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5 }}>
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <User className="w-5 h-5" />
                      Profile Information
                    </CardTitle>
                    <CardDescription>Manage your personal information and preferences</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <form onSubmit={handleProfileUpdate} className="space-y-6">
                      <div className="grid gap-6 md:grid-cols-2">
                        <div className="space-y-2">
                          <Label htmlFor="name">Full Name</Label>
                          <Input
                            id="name"
                            value={profileData.name}
                            onChange={(e) => setProfileData((prev) => ({ ...prev, name: e.target.value }))}
                            disabled={!isEditing}
                            className={!isEditing ? "bg-muted" : ""}
                          />
                        </div>
                        <div className="space-y-2">
                          <Label htmlFor="email">Email Address</Label>
                          <Input id="email" type="email" value={profileData.email} disabled className="bg-muted" />
                          <p className="text-xs text-muted-foreground">Email cannot be changed</p>
                        </div>
                      </div>

                      <div className="space-y-2">
                        <Label htmlFor="bio">Bio</Label>
                        <Textarea
                          id="bio"
                          value={profileData.bio}
                          onChange={(e) => setProfileData((prev) => ({ ...prev, bio: e.target.value }))}
                          disabled={!isEditing}
                          className={!isEditing ? "bg-muted" : ""}
                          placeholder="Tell us about yourself..."
                          rows={4}
                        />
                      </div>

                      <div className="space-y-2">
                        <Label htmlFor="avatar">Avatar URL</Label>
                        <Input
                          id="avatar"
                          value={profileData.avatar}
                          onChange={(e) => setProfileData((prev) => ({ ...prev, avatar: e.target.value }))}
                          disabled={!isEditing}
                          className={!isEditing ? "bg-muted" : ""}
                          placeholder="https://example.com/avatar.jpg"
                        />
                      </div>

                      {isEditing && (
                        <motion.div
                          initial={{ opacity: 0, y: 10 }}
                          animate={{ opacity: 1, y: 0 }}
                          className="flex justify-end gap-3"
                        >
                          <Button type="button" variant="outline" onClick={() => setIsEditing(false)}>
                            Cancel
                          </Button>
                          <Button type="submit" disabled={isSaving}>
                            {isSaving ? "Saving..." : "Save Changes"}
                          </Button>
                        </motion.div>
                      )}
                    </form>
                  </CardContent>
                </Card>
              </motion.div>
            </TabsContent>

            {/* History Tab */}
            <TabsContent value="history" className="space-y-6">
              {/* History Stats */}
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5 }}
                className="grid gap-4 md:grid-cols-4"
              >
                <Card className="border-2 border-dashed border-primary/20 bg-primary/5">
                  <CardContent className="p-4">
                    <div className="flex items-center gap-3">
                      <div className="p-2 rounded-lg bg-primary/10">
                        <MessageSquare className="w-4 h-4 text-primary" />
                      </div>
                      <div>
                        <p className="text-2xl font-bold">{historyStats.totalPrompts}</p>
                        <p className="text-xs text-muted-foreground">Total Prompts</p>
                      </div>
                    </div>
                  </CardContent>
                </Card>

                <Card className="border-2 border-dashed border-blue-500/20 bg-blue-500/5">
                  <CardContent className="p-4">
                    <div className="flex items-center gap-3">
                      <div className="p-2 rounded-lg bg-blue-500/10">
                        <Brain className="w-4 h-4 text-blue-500" />
                      </div>
                      <div>
                        <p className="text-2xl font-bold">{historyStats.totalModels}</p>
                        <p className="text-xs text-muted-foreground">Models Used</p>
                      </div>
                    </div>
                  </CardContent>
                </Card>

                <Card className="border-2 border-dashed border-green-500/20 bg-green-500/5">
                  <CardContent className="p-4">
                    <div className="flex items-center gap-3">
                      <div className="p-2 rounded-lg bg-green-500/10">
                        <Clock className="w-4 h-4 text-green-500" />
                      </div>
                      <div>
                        <p className="text-2xl font-bold">{historyStats.avgResponseTime.toFixed(1)}s</p>
                        <p className="text-xs text-muted-foreground">Avg Response Time</p>
                      </div>
                    </div>
                  </CardContent>
                </Card>

                <Card className="border-2 border-dashed border-yellow-500/20 bg-yellow-500/5">
                  <CardContent className="p-4">
                    <div className="flex items-center gap-3">
                      <div className="p-2 rounded-lg bg-yellow-500/10">
                        <Sparkles className="w-4 h-4 text-yellow-500" />
                      </div>
                      <div>
                        <p className="text-2xl font-bold">{historyStats.favoritePrompts}</p>
                        <p className="text-xs text-muted-foreground">Favorites</p>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </motion.div>

              {/* History Controls */}
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: 0.1 }}
              >
                <Card>
                  <CardHeader>
                    <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
                      <div>
                        <CardTitle className="flex items-center gap-2">
                          <History className="w-5 h-5" />
                          Prompt History
                        </CardTitle>
                        <CardDescription>View and manage your previous AI model comparisons</CardDescription>
                      </div>
                      <div className="flex items-center gap-2">
                        <Button variant="outline" size="sm">
                          <Download className="w-4 h-4 mr-2" />
                          Export
                        </Button>
                      </div>
                    </div>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    {/* Search and Filters */}
                    <div className="flex flex-col sm:flex-row gap-4">
                      <div className="relative flex-1">
                        <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-muted-foreground w-4 h-4" />
                        <Input
                          placeholder="Search prompts..."
                          value={searchQuery}
                          onChange={(e) => setSearchQuery(e.target.value)}
                          className="pl-10"
                        />
                      </div>
                      <div className="flex gap-2">
                        <DropdownMenu>
                          <DropdownMenuTrigger asChild>
                            <Button variant="outline" size="sm">
                              <Filter className="w-4 h-4 mr-2" />
                              {selectedCategory === "all" ? "All Categories" : selectedCategory}
                            </Button>
                          </DropdownMenuTrigger>
                          <DropdownMenuContent>
                            {categories.map((category) => (
                              <DropdownMenuItem key={category} onClick={() => setSelectedCategory(category)}>
                                {category === "all" ? "All Categories" : category}
                              </DropdownMenuItem>
                            ))}
                          </DropdownMenuContent>
                        </DropdownMenu>

                        <DropdownMenu>
                          <DropdownMenuTrigger asChild>
                            <Button variant="outline" size="sm">
                              Sort:{" "}
                              {sortBy === "newest"
                                ? "Newest"
                                : sortBy === "oldest"
                                  ? "Oldest"
                                  : sortBy === "most-models"
                                    ? "Most Models"
                                    : "Favorites"}
                            </Button>
                          </DropdownMenuTrigger>
                          <DropdownMenuContent>
                            <DropdownMenuItem onClick={() => setSortBy("newest")}>Newest First</DropdownMenuItem>
                            <DropdownMenuItem onClick={() => setSortBy("oldest")}>Oldest First</DropdownMenuItem>
                            <DropdownMenuItem onClick={() => setSortBy("most-models")}>Most Models</DropdownMenuItem>
                            <DropdownMenuItem onClick={() => setSortBy("favorites")}>Favorites</DropdownMenuItem>
                          </DropdownMenuContent>
                        </DropdownMenu>
                      </div>
                    </div>

                    {/* History List */}
                    <div className="space-y-3">
                      {filteredHistory.length > 0 ? (
                        filteredHistory.map((item, index) => (
                          <motion.div
                            key={item.id}
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ duration: 0.3, delay: index * 0.05 }}
                            whileHover={{ scale: 1.01, y: -2 }}
                            className="p-4 rounded-lg border bg-card hover:bg-muted/50 transition-all duration-200 cursor-pointer group"
                          >
                            <div className="flex items-start justify-between gap-4">
                              <div className="flex-1 min-w-0">
                                <div className="flex items-center gap-2 mb-2">
                                  <Badge variant="outline" className="text-xs">
                                    {item.category}
                                  </Badge>
                                  {item.isFavorite && <Sparkles className="w-3 h-3 text-yellow-500" />}
                                  <span className="text-xs text-muted-foreground">{formatDate(item.createdAt)}</span>
                                </div>
                                <h4 className="font-medium text-sm mb-2 line-clamp-2 group-hover:text-primary transition-colors">
                                  {item.prompt}
                                </h4>
                                <div className="flex items-center gap-4 text-xs text-muted-foreground">
                                  <div className="flex items-center gap-1">
                                    <Users className="w-3 h-3" />
                                    {item.models.length} models
                                  </div>
                                  <div className="flex items-center gap-1">
                                    <Clock className="w-3 h-3" />
                                    {item.avgResponseTime}s avg
                                  </div>
                                  <div className="flex items-center gap-1">
                                    <Zap className="w-3 h-3" />
                                    {item.responseCount} responses
                                  </div>
                                </div>
                                <div className="flex flex-wrap gap-1 mt-2">
                                  {item.models.map((model) => (
                                    <Badge key={model} variant="secondary" className="text-xs">
                                      {model}
                                    </Badge>
                                  ))}
                                </div>
                              </div>
                              <div className="flex items-center gap-1">
                                <Button
                                  variant="ghost"
                                  size="sm"
                                  className="opacity-0 group-hover:opacity-100 transition-opacity"
                                >
                                  <Eye className="w-4 h-4" />
                                </Button>
                                <Button
                                  variant="ghost"
                                  size="sm"
                                  className="opacity-0 group-hover:opacity-100 transition-opacity"
                                >
                                  <Share2 className="w-4 h-4" />
                                </Button>
                                <DropdownMenu>
                                  <DropdownMenuTrigger asChild>
                                    <Button
                                      variant="ghost"
                                      size="sm"
                                      className="opacity-0 group-hover:opacity-100 transition-opacity"
                                    >
                                      <MoreHorizontal className="w-4 h-4" />
                                    </Button>
                                  </DropdownMenuTrigger>
                                  <DropdownMenuContent>
                                    <DropdownMenuItem>
                                      <Eye className="w-4 h-4 mr-2" />
                                      View Details
                                    </DropdownMenuItem>
                                    <DropdownMenuItem>
                                      <Share2 className="w-4 h-4 mr-2" />
                                      Share
                                    </DropdownMenuItem>
                                    <DropdownMenuItem>
                                      <Download className="w-4 h-4 mr-2" />
                                      Export
                                    </DropdownMenuItem>
                                    <DropdownMenuSeparator />
                                    <DropdownMenuItem className="text-destructive">
                                      <Trash2 className="w-4 h-4 mr-2" />
                                      Delete
                                    </DropdownMenuItem>
                                  </DropdownMenuContent>
                                </DropdownMenu>
                                <ChevronRight className="w-4 h-4 text-muted-foreground" />
                              </div>
                            </div>
                          </motion.div>
                        ))
                      ) : (
                        <div className="text-center py-12">
                          <History className="w-12 h-12 text-muted-foreground mx-auto mb-4" />
                          <h3 className="font-medium mb-2">No prompts found</h3>
                          <p className="text-muted-foreground mb-4">
                            {searchQuery || selectedCategory !== "all"
                              ? "Try adjusting your search or filters"
                              : "Start comparing AI models to see your history here"}
                          </p>
                          <Button variant="outline" onClick={() => router.push("/")}>
                            <Zap className="w-4 h-4 mr-2" />
                            Create New Comparison
                          </Button>
                        </div>
                      )}
                    </div>
                  </CardContent>
                </Card>
              </motion.div>
            </TabsContent>
          </Tabs>
        </div>
      </div>
    </PageTransition>
  )
}
