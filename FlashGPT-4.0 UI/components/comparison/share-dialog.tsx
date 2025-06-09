"use client"

import { useState } from "react"
import { motion } from "framer-motion"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Textarea } from "@/components/ui/textarea"
import { Badge } from "@/components/ui/badge"
import { Separator } from "@/components/ui/separator"
import {
  Share2,
  Copy,
  Mail,
  MessageSquare,
  Twitter,
  Linkedin,
  LinkIcon,
  CheckCircle,
  Loader2,
  Globe,
  Lock,
  Eye,
} from "lucide-react"
import { toast } from "@/components/ui/use-toast"
import type { Model, ModelResponse } from "@/lib/api-client"

interface ShareDialogProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  prompt: string
  responses: Record<string, ModelResponse>
  selectedModels: string[]
  availableModels: Model[]
}

export function ShareDialog({
  open,
  onOpenChange,
  prompt,
  responses,
  selectedModels,
  availableModels,
}: ShareDialogProps) {
  const [shareTitle, setShareTitle] = useState("")
  const [shareDescription, setShareDescription] = useState("")
  const [isPublic, setIsPublic] = useState(true)
  const [isGenerating, setIsGenerating] = useState(false)
  const [shareUrl, setShareUrl] = useState("")
  const [copied, setCopied] = useState(false)

  // Generate a mock share URL (in real app, this would come from your backend)
  const generateShareUrl = async () => {
    setIsGenerating(true)

    // Simulate API call
    await new Promise((resolve) => setTimeout(resolve, 1500))

    const shareId = Math.random().toString(36).substring(2, 15)
    const mockUrl = `${window.location.origin}/shared/${shareId}`
    setShareUrl(mockUrl)
    setIsGenerating(false)

    toast({
      title: "Share Link Generated! 🔗",
      description: "Your comparison is now ready to share",
    })
  }

  const copyToClipboard = async (text: string) => {
    try {
      await navigator.clipboard.writeText(text)
      setCopied(true)
      toast({
        title: "Copied! 📋",
        description: "Share link copied to clipboard",
      })
      setTimeout(() => setCopied(false), 2000)
    } catch (error) {
      toast({
        title: "Copy Failed",
        description: "Failed to copy to clipboard",
        variant: "destructive",
      })
    }
  }

  const shareViaEmail = () => {
    const subject = encodeURIComponent(shareTitle || "LLM Comparison Results")
    const body = encodeURIComponent(`Check out this AI model comparison:\n\n${shareUrl}\n\n${shareDescription}`)
    window.open(`mailto:?subject=${subject}&body=${body}`)
  }

  const shareViaTwitter = () => {
    const text = encodeURIComponent(`${shareTitle || "Check out this AI model comparison"} ${shareUrl}`)
    window.open(`https://twitter.com/intent/tweet?text=${text}`)
  }

  const shareViaLinkedIn = () => {
    const url = encodeURIComponent(shareUrl)
    window.open(`https://www.linkedin.com/sharing/share-offsite/?url=${url}`)
  }

  const shareViaNative = async () => {
    if (navigator.share) {
      try {
        await navigator.share({
          title: shareTitle || "LLM Comparison Results",
          text: shareDescription,
          url: shareUrl,
        })
      } catch (error) {
        console.log("Share cancelled")
      }
    } else {
      copyToClipboard(shareUrl)
    }
  }

  const modelNames = selectedModels
    .map((id) => availableModels.find((m) => m.id === id)?.name)
    .filter(Boolean)
    .join(", ")

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <motion.div
              animate={{ rotate: [0, 360] }}
              transition={{ duration: 2, repeat: Number.POSITIVE_INFINITY, ease: "linear" }}
            >
              <Share2 className="w-5 h-5" />
            </motion.div>
            Share Comparison
          </DialogTitle>
          <DialogDescription>Share your AI model comparison with others</DialogDescription>
        </DialogHeader>

        <div className="space-y-6 py-4">
          {!shareUrl ? (
            <>
              {/* Share Configuration */}
              <div className="space-y-4">
                <div className="space-y-2">
                  <Label htmlFor="title" className="text-sm font-medium">
                    Share Title
                  </Label>
                  <Input
                    id="title"
                    placeholder="Give your comparison a descriptive title..."
                    value={shareTitle}
                    onChange={(e) => setShareTitle(e.target.value)}
                  />
                </div>

                <div className="space-y-2">
                  <Label htmlFor="description" className="text-sm font-medium">
                    Description (optional)
                  </Label>
                  <Textarea
                    id="description"
                    placeholder="Add context about your comparison..."
                    value={shareDescription}
                    onChange={(e) => setShareDescription(e.target.value)}
                    className="min-h-[80px] resize-none"
                  />
                </div>

                {/* Privacy Setting */}
                <div className="flex items-center justify-between p-4 rounded-lg border bg-muted/20">
                  <div className="flex items-center space-x-3">
                    {isPublic ? (
                      <Globe className="w-5 h-5 text-green-500" />
                    ) : (
                      <Lock className="w-5 h-5 text-orange-500" />
                    )}
                    <div>
                      <div className="font-medium text-sm">{isPublic ? "Public" : "Private"}</div>
                      <div className="text-xs text-muted-foreground">
                        {isPublic
                          ? "Anyone with the link can view this comparison"
                          : "Only people you share the link with can view"}
                      </div>
                    </div>
                  </div>
                  <Button variant="outline" size="sm" onClick={() => setIsPublic(!isPublic)}>
                    {isPublic ? "Make Private" : "Make Public"}
                  </Button>
                </div>
              </div>

              {/* Preview */}
              <div className="space-y-3">
                <Label className="text-sm font-medium">Preview</Label>
                <div className="p-4 rounded-lg border bg-muted/10">
                  <div className="space-y-2">
                    <h4 className="font-medium">{shareTitle || "LLM Comparison Results"}</h4>
                    <p className="text-sm text-muted-foreground">
                      {shareDescription ||
                        `Comparing responses from ${modelNames} for: "${prompt.slice(0, 100)}${prompt.length > 100 ? "..." : ""}"`}
                    </p>
                    <div className="flex items-center gap-2 mt-2">
                      <Badge variant="secondary" className="text-xs">
                        {selectedModels.length} models
                      </Badge>
                      <Badge variant="secondary" className="text-xs">
                        {Object.keys(responses).length} responses
                      </Badge>
                      {isPublic ? (
                        <Badge variant="outline" className="text-xs text-green-600">
                          <Globe className="w-3 h-3 mr-1" />
                          Public
                        </Badge>
                      ) : (
                        <Badge variant="outline" className="text-xs text-orange-600">
                          <Lock className="w-3 h-3 mr-1" />
                          Private
                        </Badge>
                      )}
                    </div>
                  </div>
                </div>
              </div>
            </>
          ) : (
            <>
              {/* Share URL Generated */}
              <div className="space-y-4">
                <div className="text-center space-y-2">
                  <motion.div initial={{ scale: 0 }} animate={{ scale: 1 }} transition={{ duration: 0.5 }}>
                    <CheckCircle className="w-12 h-12 text-green-500 mx-auto" />
                  </motion.div>
                  <h3 className="font-medium">Share Link Ready!</h3>
                  <p className="text-sm text-muted-foreground">Your comparison is now available at this link</p>
                </div>

                {/* Share URL */}
                <div className="space-y-2">
                  <Label className="text-sm font-medium">Share URL</Label>
                  <div className="flex space-x-2">
                    <Input value={shareUrl} readOnly className="font-mono text-sm" />
                    <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
                      <Button variant="outline" size="sm" onClick={() => copyToClipboard(shareUrl)} className="px-3">
                        {copied ? <CheckCircle className="w-4 h-4 text-green-500" /> : <Copy className="w-4 h-4" />}
                      </Button>
                    </motion.div>
                  </div>
                </div>

                <Separator />

                {/* Share Options */}
                <div className="space-y-3">
                  <Label className="text-sm font-medium">Share via</Label>
                  <div className="grid grid-cols-2 gap-2">
                    <motion.div whileHover={{ scale: 1.02 }} whileTap={{ scale: 0.98 }}>
                      <Button variant="outline" size="sm" onClick={shareViaEmail} className="w-full justify-start">
                        <Mail className="w-4 h-4 mr-2" />
                        Email
                      </Button>
                    </motion.div>

                    <motion.div whileHover={{ scale: 1.02 }} whileTap={{ scale: 0.98 }}>
                      <Button variant="outline" size="sm" onClick={shareViaTwitter} className="w-full justify-start">
                        <Twitter className="w-4 h-4 mr-2" />
                        Twitter
                      </Button>
                    </motion.div>

                    <motion.div whileHover={{ scale: 1.02 }} whileTap={{ scale: 0.98 }}>
                      <Button variant="outline" size="sm" onClick={shareViaLinkedIn} className="w-full justify-start">
                        <Linkedin className="w-4 h-4 mr-2" />
                        LinkedIn
                      </Button>
                    </motion.div>

                    {navigator.share && (
                      <motion.div whileHover={{ scale: 1.02 }} whileTap={{ scale: 0.98 }}>
                        <Button variant="outline" size="sm" onClick={shareViaNative} className="w-full justify-start">
                          <MessageSquare className="w-4 h-4 mr-2" />
                          More
                        </Button>
                      </motion.div>
                    )}
                  </div>
                </div>

                {/* Stats */}
                <div className="p-3 rounded-lg bg-muted/20 text-center">
                  <div className="flex items-center justify-center gap-2 text-sm text-muted-foreground">
                    <Eye className="w-4 h-4" />
                    <span>0 views</span>
                    <span>•</span>
                    <span>Just created</span>
                  </div>
                </div>
              </div>
            </>
          )}
        </div>

        <DialogFooter className="gap-2">
          <Button variant="outline" onClick={() => onOpenChange(false)}>
            {shareUrl ? "Close" : "Cancel"}
          </Button>
          {!shareUrl && (
            <motion.div whileHover={{ scale: 1.02 }} whileTap={{ scale: 0.98 }}>
              <Button onClick={generateShareUrl} disabled={isGenerating}>
                {isGenerating ? (
                  <>
                    <motion.div
                      animate={{ rotate: 360 }}
                      transition={{ duration: 1, repeat: Number.POSITIVE_INFINITY, ease: "linear" }}
                    >
                      <Loader2 className="w-4 h-4 mr-2" />
                    </motion.div>
                    Generating Link...
                  </>
                ) : (
                  <>
                    <LinkIcon className="w-4 h-4 mr-2" />
                    Generate Share Link
                  </>
                )}
              </Button>
            </motion.div>
          )}
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
