"use client"

import { useEffect, useState } from "react"
import { useRouter } from "next/navigation"
import { useTheme } from "next-themes"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Checkbox } from "@/components/ui/checkbox"
import { Badge } from "@/components/ui/badge"
import { SettingsIcon, Palette, Save, Loader2, Bell, Shield, Database, Download, Trash2 } from "lucide-react"
import { useAuth } from "@/contexts/auth-context"
import { apiClient, type UserPreferences } from "@/lib/api-client"
import { toast } from "@/components/ui/use-toast"

export default function SettingsPage() {
  const { user, isLoading: authLoading } = useAuth()
  const router = useRouter()
  const { theme, setTheme } = useTheme()
  const [preferences, setPreferences] = useState<UserPreferences>({
    theme: "dark",
    fontSize: "medium",
    defaultModels: [],
    autoSave: false,
  })
  const [isLoading, setIsLoading] = useState(true)
  const [isSaving, setIsSaving] = useState(false)

  useEffect(() => {
    if (authLoading) return

    if (!user) {
      router.push("/login")
      return
    }

    loadPreferences()
  }, [user, authLoading, router])

  const loadPreferences = async () => {
    setIsLoading(true)
    const { data, error } = await apiClient.getUserPreferences()

    if (data && !error) {
      setPreferences(data)
      setTheme(data.theme)
    }
    setIsLoading(false)
  }

  const handleThemeChange = (newTheme: "light" | "dark" | "system") => {
    setPreferences((prev) => ({ ...prev, theme: newTheme }))
    setTheme(newTheme)
  }

  const savePreferences = async () => {
    if (!user) return

    setIsSaving(true)
    const { data, error } = await apiClient.updateUserPreferences(preferences)

    if (data && !error) {
      toast({
        title: "Settings Saved! ⚙️",
        description: "Your preferences have been updated successfully",
      })
    } else {
      toast({
        title: "Save Failed",
        description: error || "Failed to save preferences",
        variant: "destructive",
      })
    }
    setIsSaving(false)
  }

  if (authLoading || isLoading) {
    return (
      <div className="min-h-screen bg-background text-foreground flex items-center justify-center">
        <div className="text-center space-y-4">
          <Loader2 className="h-8 w-8 animate-spin mx-auto" />
          <p>Loading settings...</p>
        </div>
      </div>
    )
  }

  if (!user) {
    return null
  }

  return (
    <div className="min-h-screen bg-background">
      <div className="container mx-auto px-4 py-8 max-w-4xl">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold mb-2 flex items-center gap-3">
            <SettingsIcon className="w-8 h-8" />
            Settings
          </h1>
          <p className="text-muted-foreground">Manage your application preferences and account settings</p>
        </div>

        <div className="space-y-8">
          {/* Appearance Settings */}
          <Card>
            <CardHeader>
              <div className="flex items-center gap-3">
                <div className="p-2 rounded-lg bg-primary/10">
                  <Palette className="w-5 h-5 text-primary" />
                </div>
                <div>
                  <CardTitle>Appearance</CardTitle>
                  <CardDescription>Customize how the application looks and feels</CardDescription>
                </div>
              </div>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="grid gap-6 md:grid-cols-2">
                <div className="space-y-3">
                  <Label>Theme</Label>
                  <Select value={preferences.theme} onValueChange={handleThemeChange}>
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="light">☀️ Light</SelectItem>
                      <SelectItem value="dark">🌙 Dark</SelectItem>
                      <SelectItem value="system">💻 System</SelectItem>
                    </SelectContent>
                  </Select>
                  <p className="text-xs text-muted-foreground">Choose your preferred color scheme</p>
                </div>

                <div className="space-y-3">
                  <Label>Font Size</Label>
                  <Select
                    value={preferences.fontSize}
                    onValueChange={(value: "small" | "medium" | "large") =>
                      setPreferences((prev) => ({ ...prev, fontSize: value }))
                    }
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="small">Small</SelectItem>
                      <SelectItem value="medium">Medium</SelectItem>
                      <SelectItem value="large">Large</SelectItem>
                    </SelectContent>
                  </Select>
                  <p className="text-xs text-muted-foreground">Adjust text size for better readability</p>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Comparison Settings */}
          <Card>
            <CardHeader>
              <div className="flex items-center gap-3">
                <div className="p-2 rounded-lg bg-primary/10">
                  <Database className="w-5 h-5 text-primary" />
                </div>
                <div>
                  <CardTitle>Comparison Preferences</CardTitle>
                  <CardDescription>Configure how comparisons work and are saved</CardDescription>
                </div>
              </div>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="space-y-4">
                <div className="flex items-center justify-between p-4 border rounded-lg">
                  <div className="space-y-1">
                    <div className="flex items-center gap-2">
                      <Checkbox
                        id="autoSave"
                        checked={preferences.autoSave}
                        onCheckedChange={(checked) =>
                          setPreferences((prev) => ({ ...prev, autoSave: checked === true }))
                        }
                      />
                      <Label htmlFor="autoSave" className="font-medium">
                        Auto-save comparisons
                      </Label>
                    </div>
                    <p className="text-sm text-muted-foreground">
                      Automatically save your comparisons when responses are generated
                    </p>
                  </div>
                </div>

                <div className="space-y-3">
                  <Label>Default Models</Label>
                  <div className="grid grid-cols-2 md:grid-cols-3 gap-2">
                    {["GPT-4", "Claude 3 Opus", "Gemini Pro", "GPT-3.5", "Claude 3 Sonnet", "Llama 2"].map((model) => (
                      <div key={model} className="flex items-center space-x-2 p-2 border rounded">
                        <Checkbox id={model} />
                        <Label htmlFor={model} className="text-sm">
                          {model}
                        </Label>
                      </div>
                    ))}
                  </div>
                  <p className="text-xs text-muted-foreground">
                    Select models that will be pre-selected for new comparisons
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Notifications */}
          <Card>
            <CardHeader>
              <div className="flex items-center gap-3">
                <div className="p-2 rounded-lg bg-primary/10">
                  <Bell className="w-5 h-5 text-primary" />
                </div>
                <div>
                  <CardTitle>Notifications</CardTitle>
                  <CardDescription>Control what notifications you receive</CardDescription>
                </div>
              </div>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-4">
                <div className="flex items-center justify-between p-4 border rounded-lg">
                  <div className="space-y-1">
                    <div className="flex items-center gap-2">
                      <Checkbox id="emailNotifications" defaultChecked />
                      <Label htmlFor="emailNotifications" className="font-medium">
                        Email notifications
                      </Label>
                    </div>
                    <p className="text-sm text-muted-foreground">Receive updates about your account via email</p>
                  </div>
                </div>

                <div className="flex items-center justify-between p-4 border rounded-lg">
                  <div className="space-y-1">
                    <div className="flex items-center gap-2">
                      <Checkbox id="comparisonUpdates" defaultChecked />
                      <Label htmlFor="comparisonUpdates" className="font-medium">
                        Comparison updates
                      </Label>
                    </div>
                    <p className="text-sm text-muted-foreground">
                      Get notified when your saved comparisons are viewed or shared
                    </p>
                  </div>
                </div>

                <div className="flex items-center justify-between p-4 border rounded-lg">
                  <div className="space-y-1">
                    <div className="flex items-center gap-2">
                      <Checkbox id="newFeatures" defaultChecked />
                      <Label htmlFor="newFeatures" className="font-medium">
                        New features
                      </Label>
                    </div>
                    <p className="text-sm text-muted-foreground">
                      Be the first to know about new AI models and features
                    </p>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Data & Privacy */}
          <Card>
            <CardHeader>
              <div className="flex items-center gap-3">
                <div className="p-2 rounded-lg bg-primary/10">
                  <Shield className="w-5 h-5 text-primary" />
                </div>
                <div>
                  <CardTitle>Data & Privacy</CardTitle>
                  <CardDescription>Manage your data and privacy settings</CardDescription>
                </div>
              </div>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-4">
                <div className="flex items-center justify-between p-4 border rounded-lg">
                  <div className="space-y-1">
                    <p className="font-medium text-sm">Export your data</p>
                    <p className="text-sm text-muted-foreground">Download all your comparisons and account data</p>
                  </div>
                  <Button variant="outline" size="sm">
                    <Download className="w-4 h-4 mr-2" />
                    Export
                  </Button>
                </div>

                <div className="flex items-center justify-between p-4 border rounded-lg">
                  <div className="space-y-1">
                    <p className="font-medium text-sm">Data retention</p>
                    <p className="text-sm text-muted-foreground">
                      Your comparisons are kept for 2 years after creation
                    </p>
                  </div>
                  <Badge variant="secondary">2 years</Badge>
                </div>

                <div className="flex items-center justify-between p-4 border border-destructive/20 rounded-lg bg-destructive/5">
                  <div className="space-y-1">
                    <p className="font-medium text-sm text-destructive">Delete all data</p>
                    <p className="text-sm text-muted-foreground">Permanently delete all your comparisons and data</p>
                  </div>
                  <Button variant="destructive" size="sm">
                    <Trash2 className="w-4 h-4 mr-2" />
                    Delete
                  </Button>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Save Button */}
          <div className="flex justify-end">
            <Button onClick={savePreferences} disabled={isSaving} size="lg" className="px-8">
              {isSaving ? (
                <>
                  <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                  Saving Settings...
                </>
              ) : (
                <>
                  <Save className="w-4 h-4 mr-2" />
                  Save All Settings
                </>
              )}
            </Button>
          </div>
        </div>
      </div>
    </div>
  )
}
