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
import { Label } from "@/components/ui/label"
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group"
import { Checkbox } from "@/components/ui/checkbox"
import { Input } from "@/components/ui/input"
import { Textarea } from "@/components/ui/textarea"
import { Download, FileText, Code, Loader2, CheckCircle } from "lucide-react"
import { toast } from "@/components/ui/use-toast"
import type { Model, ModelResponse } from "@/lib/api-client"

interface ExportDialogProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  prompt: string
  responses: Record<string, ModelResponse>
  selectedModels: string[]
  availableModels: Model[]
}

export function ExportDialog({
  open,
  onOpenChange,
  prompt,
  responses,
  selectedModels,
  availableModels,
}: ExportDialogProps) {
  const [exportFormat, setExportFormat] = useState<"pdf" | "json" | "markdown" | "csv">("pdf")
  const [includeMetadata, setIncludeMetadata] = useState(true)
  const [includeTimestamps, setIncludeTimestamps] = useState(true)
  const [customTitle, setCustomTitle] = useState("")
  const [customNotes, setCustomNotes] = useState("")
  const [isExporting, setIsExporting] = useState(false)
  const [exportComplete, setExportComplete] = useState(false)

  const formatOptions = [
    {
      value: "pdf",
      label: "PDF Report",
      description: "Professional report format for sharing and printing",
      icon: FileText,
    },
    {
      value: "json",
      label: "JSON Data",
      description: "Structured data format for analysis and integration",
      icon: Code,
    },
    {
      value: "markdown",
      label: "Markdown",
      description: "Text format for documentation and wikis",
      icon: FileText,
    },
    {
      value: "csv",
      label: "CSV Spreadsheet",
      description: "Tabular format for data analysis",
      icon: FileText,
    },
  ]

  const generateExportData = () => {
    const timestamp = new Date().toISOString()
    const title = customTitle || `LLM Comparison - ${new Date().toLocaleDateString()}`

    const exportData = {
      title,
      prompt,
      timestamp,
      notes: customNotes,
      models: selectedModels.map((modelId) => {
        const model = availableModels.find((m) => m.id === modelId)
        const response = responses[modelId]
        return {
          id: modelId,
          name: model?.name || modelId,
          provider: model?.provider || "Unknown",
          response: response?.content || "",
          responseTime: response?.responseTime || 0,
          tokenCount: response?.tokenCount || 0,
        }
      }),
      metadata: includeMetadata
        ? {
            totalModels: selectedModels.length,
            totalTokens: Object.values(responses).reduce((sum, r) => sum + (r.tokenCount || 0), 0),
            averageResponseTime:
              Object.values(responses).reduce((sum, r) => sum + (r.responseTime || 0), 0) /
              Object.values(responses).length,
            exportedAt: timestamp,
          }
        : undefined,
    }

    return exportData
  }

  const downloadFile = (content: string, filename: string, mimeType: string) => {
    const blob = new Blob([content], { type: mimeType })
    const url = URL.createObjectURL(blob)
    const link = document.createElement("a")
    link.href = url
    link.download = filename
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
    URL.revokeObjectURL(url)
  }

  const exportToPDF = async (data: any) => {
    // For demo purposes, we'll create a simple HTML that could be converted to PDF
    const htmlContent = `
      <!DOCTYPE html>
      <html>
      <head>
        <title>${data.title}</title>
        <style>
          body { font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }
          .header { border-bottom: 2px solid #333; padding-bottom: 20px; margin-bottom: 30px; }
          .prompt { background: #f5f5f5; padding: 20px; border-radius: 8px; margin: 20px 0; }
          .response { margin: 20px 0; padding: 20px; border: 1px solid #ddd; border-radius: 8px; }
          .model-name { color: #333; font-weight: bold; font-size: 18px; }
          .metadata { background: #f9f9f9; padding: 15px; border-radius: 8px; margin-top: 20px; }
          .notes { background: #fff3cd; padding: 15px; border-radius: 8px; margin: 20px 0; }
        </style>
      </head>
      <body>
        <div class="header">
          <h1>${data.title}</h1>
          <p>Generated on: ${new Date(data.timestamp).toLocaleString()}</p>
        </div>
        
        <div class="prompt">
          <h2>Prompt</h2>
          <p>${data.prompt}</p>
        </div>

        ${data.notes ? `<div class="notes"><h3>Notes</h3><p>${data.notes}</p></div>` : ""}

        <h2>Model Responses</h2>
        ${data.models
          .map(
            (model: any) => `
          <div class="response">
            <div class="model-name">${model.name} (${model.provider})</div>
            <p>${model.response}</p>
            <div style="font-size: 12px; color: #666; margin-top: 10px;">
              Response Time: ${model.responseTime}s | Tokens: ${model.tokenCount}
            </div>
          </div>
        `,
          )
          .join("")}

        ${
          data.metadata
            ? `
          <div class="metadata">
            <h3>Export Metadata</h3>
            <p>Total Models: ${data.metadata.totalModels}</p>
            <p>Total Tokens: ${data.metadata.totalTokens}</p>
            <p>Average Response Time: ${data.metadata.averageResponseTime.toFixed(2)}s</p>
          </div>
        `
            : ""
        }
      </body>
      </html>
    `

    downloadFile(htmlContent, `${data.title.replace(/[^a-z0-9]/gi, "_")}.html`, "text/html")
  }

  const exportToJSON = (data: any) => {
    const jsonContent = JSON.stringify(data, null, 2)
    downloadFile(jsonContent, `llm_comparison_${Date.now()}.json`, "application/json")
  }

  const exportToMarkdown = (data: any) => {
    const markdownContent = `# ${data.title}

**Generated:** ${new Date(data.timestamp).toLocaleString()}

## Prompt
\`\`\`
${data.prompt}
\`\`\`

${data.notes ? `## Notes\n${data.notes}\n` : ""}

## Model Responses

${data.models
  .map(
    (model: any) => `### ${model.name} (${model.provider})

${model.response}

- **Response Time:** ${model.responseTime}s
- **Tokens:** ${model.tokenCount}

---
`,
  )
  .join("\n")}

${
  data.metadata
    ? `## Export Metadata

- **Total Models:** ${data.metadata.totalModels}
- **Total Tokens:** ${data.metadata.totalTokens}
- **Average Response Time:** ${data.metadata.averageResponseTime.toFixed(2)}s
- **Exported At:** ${data.metadata.exportedAt}
`
    : ""
}
`

    downloadFile(markdownContent, `llm_comparison_${Date.now()}.md`, "text/markdown")
  }

  const exportToCSV = (data: any) => {
    const headers = ["Model", "Provider", "Response", "Response Time (s)", "Token Count"]
    const rows = data.models.map((model: any) => [
      model.name,
      model.provider,
      `"${model.response.replace(/"/g, '""')}"`, // Escape quotes in CSV
      model.responseTime,
      model.tokenCount,
    ])

    const csvContent = [
      `"Prompt","${data.prompt.replace(/"/g, '""')}"`,
      `"Generated","${new Date(data.timestamp).toLocaleString()}"`,
      "",
      headers.join(","),
      ...rows.map((row) => row.join(",")),
    ].join("\n")

    downloadFile(csvContent, `llm_comparison_${Date.now()}.csv`, "text/csv")
  }

  const handleExport = async () => {
    setIsExporting(true)
    setExportComplete(false)

    try {
      const data = generateExportData()

      // Simulate processing time
      await new Promise((resolve) => setTimeout(resolve, 1500))

      switch (exportFormat) {
        case "pdf":
          await exportToPDF(data)
          break
        case "json":
          exportToJSON(data)
          break
        case "markdown":
          exportToMarkdown(data)
          break
        case "csv":
          exportToCSV(data)
          break
      }

      setExportComplete(true)

      toast({
        title: "Export Successful! 📁",
        description: `Your comparison has been exported as ${exportFormat.toUpperCase()}`,
      })

      // Auto-close after success
      setTimeout(() => {
        onOpenChange(false)
        setExportComplete(false)
      }, 2000)
    } catch (error) {
      toast({
        title: "Export Failed",
        description: "There was an error exporting your comparison. Please try again.",
        variant: "destructive",
      })
    } finally {
      setIsExporting(false)
    }
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-lg">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <motion.div
              animate={{ rotate: isExporting ? 360 : 0 }}
              transition={{ duration: 1, repeat: isExporting ? Number.POSITIVE_INFINITY : 0, ease: "linear" }}
            >
              <Download className="w-5 h-5" />
            </motion.div>
            Export Comparison
          </DialogTitle>
          <DialogDescription>Choose your preferred format and customize the export options</DialogDescription>
        </DialogHeader>

        <div className="space-y-6 py-4">
          {/* Format Selection */}
          <div className="space-y-3">
            <Label className="text-sm font-medium">Export Format</Label>
            <RadioGroup value={exportFormat} onValueChange={(value: any) => setExportFormat(value)}>
              <div className="grid gap-3">
                {formatOptions.map((option) => (
                  <motion.div
                    key={option.value}
                    whileHover={{ scale: 1.02 }}
                    className="flex items-center space-x-3 p-3 rounded-lg border hover:bg-muted/50 transition-colors"
                  >
                    <RadioGroupItem value={option.value} id={option.value} />
                    <div className="flex items-center space-x-3 flex-1">
                      <div className="p-2 rounded-lg bg-primary/10">
                        <option.icon className="w-4 h-4 text-primary" />
                      </div>
                      <div className="flex-1">
                        <Label htmlFor={option.value} className="font-medium cursor-pointer">
                          {option.label}
                        </Label>
                        <p className="text-xs text-muted-foreground">{option.description}</p>
                      </div>
                    </div>
                  </motion.div>
                ))}
              </div>
            </RadioGroup>
          </div>

          {/* Customization Options */}
          <div className="space-y-4">
            <Label className="text-sm font-medium">Customization</Label>

            <div className="space-y-2">
              <Label htmlFor="title" className="text-xs">
                Custom Title (optional)
              </Label>
              <Input
                id="title"
                placeholder="Enter a custom title for your export..."
                value={customTitle}
                onChange={(e) => setCustomTitle(e.target.value)}
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="notes" className="text-xs">
                Additional Notes (optional)
              </Label>
              <Textarea
                id="notes"
                placeholder="Add any notes or context about this comparison..."
                value={customNotes}
                onChange={(e) => setCustomNotes(e.target.value)}
                className="min-h-[80px] resize-none"
              />
            </div>

            <div className="space-y-3">
              <div className="flex items-center space-x-3 p-3 rounded-lg border bg-muted/20">
                <Checkbox
                  id="metadata"
                  checked={includeMetadata}
                  onCheckedChange={(checked) => setIncludeMetadata(checked === true)}
                />
                <div className="flex-1">
                  <Label htmlFor="metadata" className="text-sm font-medium cursor-pointer">
                    Include metadata
                  </Label>
                  <p className="text-xs text-muted-foreground">Add statistics like token counts and response times</p>
                </div>
              </div>

              <div className="flex items-center space-x-3 p-3 rounded-lg border bg-muted/20">
                <Checkbox
                  id="timestamps"
                  checked={includeTimestamps}
                  onCheckedChange={(checked) => setIncludeTimestamps(checked === true)}
                />
                <div className="flex-1">
                  <Label htmlFor="timestamps" className="text-sm font-medium cursor-pointer">
                    Include timestamps
                  </Label>
                  <p className="text-xs text-muted-foreground">Add generation and export timestamps</p>
                </div>
              </div>
            </div>
          </div>
        </div>

        <DialogFooter className="gap-2">
          <Button variant="outline" onClick={() => onOpenChange(false)} disabled={isExporting}>
            Cancel
          </Button>
          <motion.div whileHover={{ scale: 1.02 }} whileTap={{ scale: 0.98 }}>
            <Button onClick={handleExport} disabled={isExporting || exportComplete}>
              {isExporting ? (
                <>
                  <motion.div
                    animate={{ rotate: 360 }}
                    transition={{ duration: 1, repeat: Number.POSITIVE_INFINITY, ease: "linear" }}
                  >
                    <Loader2 className="w-4 h-4 mr-2" />
                  </motion.div>
                  Exporting...
                </>
              ) : exportComplete ? (
                <>
                  <CheckCircle className="w-4 h-4 mr-2 text-green-500" />
                  Exported!
                </>
              ) : (
                <>
                  <Download className="w-4 h-4 mr-2" />
                  Export {exportFormat.toUpperCase()}
                </>
              )}
            </Button>
          </motion.div>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
