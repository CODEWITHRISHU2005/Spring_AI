"use client"

import { useState } from "react"
import Link from "next/link"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from "@/components/ui/alert-dialog"
import { Eye, Trash2, Clock, Share2 } from "lucide-react"
import type { Comparison } from "@/lib/api-client"

interface ComparisonsListProps {
  comparisons: Comparison[]
  onDelete: (id: string) => Promise<void>
}

export function ComparisonsList({ comparisons, onDelete }: ComparisonsListProps) {
  const [deletingId, setDeletingId] = useState<string | null>(null)

  async function handleDelete(id: string) {
    setDeletingId(id)
    await onDelete(id)
    setDeletingId(null)
  }

  const formatDistanceToNow = (dateString: string) => {
    const date = new Date(dateString)
    const now = new Date()
    const diffInDays = Math.floor((now.getTime() - date.getTime()) / (1000 * 60 * 60 * 24))

    if (diffInDays === 0) return "Today"
    if (diffInDays === 1) return "1 day ago"
    return `${diffInDays} days ago`
  }

  return (
    <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
      {comparisons.map((comparison) => (
        <Card key={comparison.id} className="bg-gray-900 border-gray-800">
          <CardHeader>
            <div className="flex justify-between items-start">
              <CardTitle className="text-white">{comparison.name}</CardTitle>
              <Badge
                variant={comparison.isPublic ? "default" : "secondary"}
                className={comparison.isPublic ? "bg-green-700" : "bg-gray-700"}
              >
                {comparison.isPublic ? "Public" : "Private"}
              </Badge>
            </div>
            <CardDescription className="text-gray-400 line-clamp-2">{comparison.prompt}</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="flex flex-wrap gap-2">
              {comparison.models.map((model) => (
                <Badge key={model} variant="outline" className="border-gray-700 text-gray-300">
                  {model}
                </Badge>
              ))}
            </div>
          </CardContent>
          <CardFooter className="flex justify-between border-t border-gray-800 pt-4">
            <div className="flex items-center text-gray-400 text-sm">
              <Clock className="mr-1 h-3 w-3" />
              {formatDistanceToNow(comparison.createdAt)}
            </div>
            <div className="flex space-x-2">
              <Button asChild size="sm" variant="ghost" className="text-gray-400 hover:text-white">
                <Link href={`/comparison/${comparison.id}`}>
                  <Eye className="h-4 w-4" />
                </Link>
              </Button>
              {comparison.isPublic && (
                <Button size="sm" variant="ghost" className="text-gray-400 hover:text-white">
                  <Share2 className="h-4 w-4" />
                </Button>
              )}
              <AlertDialog>
                <AlertDialogTrigger asChild>
                  <Button size="sm" variant="ghost" className="text-gray-400 hover:text-red-500">
                    <Trash2 className="h-4 w-4" />
                  </Button>
                </AlertDialogTrigger>
                <AlertDialogContent className="bg-gray-900 border-gray-800 text-white">
                  <AlertDialogHeader>
                    <AlertDialogTitle>Delete Comparison</AlertDialogTitle>
                    <AlertDialogDescription className="text-gray-400">
                      Are you sure you want to delete this comparison? This action cannot be undone.
                    </AlertDialogDescription>
                  </AlertDialogHeader>
                  <AlertDialogFooter>
                    <AlertDialogCancel className="bg-gray-800 text-white border-gray-700 hover:bg-gray-700">
                      Cancel
                    </AlertDialogCancel>
                    <AlertDialogAction
                      onClick={() => handleDelete(comparison.id)}
                      className="bg-red-600 hover:bg-red-700"
                      disabled={deletingId === comparison.id}
                    >
                      {deletingId === comparison.id ? "Deleting..." : "Delete"}
                    </AlertDialogAction>
                  </AlertDialogFooter>
                </AlertDialogContent>
              </AlertDialog>
            </div>
          </CardFooter>
        </Card>
      ))}
    </div>
  )
}
