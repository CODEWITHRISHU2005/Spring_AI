"use client"

import { useEffect, useState } from "react"
import { useRouter } from "next/navigation"
import { Button } from "@/components/ui/button"
import Link from "next/link"
import { PlusCircle, Loader2 } from "lucide-react"
import { useAuth } from "@/contexts/auth-context"
import { apiClient, type Comparison } from "@/lib/api-client"
import { ComparisonsList } from "@/components/comparisons/comparisons-list"
import { toast } from "@/components/ui/use-toast"

export default function MyComparisonsPage() {
  const { user, isLoading: authLoading } = useAuth()
  const router = useRouter()
  const [comparisons, setComparisons] = useState<Comparison[]>([])
  const [isLoading, setIsLoading] = useState(true)

  useEffect(() => {
    if (authLoading) return

    if (!user) {
      router.push("/login")
      return
    }

    loadComparisons()
  }, [user, authLoading, router])

  const loadComparisons = async () => {
    setIsLoading(true)
    const { data, error } = await apiClient.getComparisons()

    if (data && !error) {
      setComparisons(data)
    } else {
      toast({
        title: "Error",
        description: error || "Failed to load comparisons",
        variant: "destructive",
      })
    }
    setIsLoading(false)
  }

  const handleDelete = async (id: string) => {
    const { error } = await apiClient.deleteComparison(id)

    if (!error) {
      setComparisons((prev) => prev.filter((c) => c.id !== id))
      toast({
        title: "Success",
        description: "Comparison deleted successfully",
      })
    } else {
      toast({
        title: "Error",
        description: error || "Failed to delete comparison",
        variant: "destructive",
      })
    }
  }

  if (authLoading || isLoading) {
    return (
      <div className="min-h-screen bg-background text-foreground flex items-center justify-center">
        <div className="text-center">
          <Loader2 className="h-8 w-8 animate-spin mx-auto mb-4" />
          <p>Loading comparisons...</p>
        </div>
      </div>
    )
  }

  if (!user) {
    return null
  }

  return (
    <div className="min-h-screen bg-background text-foreground">
      <div className="container mx-auto px-4 py-8">
        <div className="flex items-center justify-between mb-8">
          <h1 className="text-3xl font-bold">My Saved Comparisons</h1>
          <Button asChild>
            <Link href="/">
              <PlusCircle className="mr-2 h-4 w-4" />
              New Comparison
            </Link>
          </Button>
        </div>

        {comparisons.length === 0 ? (
          <div className="text-center py-12">
            <h2 className="text-xl font-medium text-muted-foreground mb-4">You don't have any saved comparisons yet</h2>
            <p className="text-muted-foreground mb-6">Create a comparison and save it to see it here</p>
            <Button asChild>
              <Link href="/">
                <PlusCircle className="mr-2 h-4 w-4" />
                Create Comparison
              </Link>
            </Button>
          </div>
        ) : (
          <ComparisonsList comparisons={comparisons} onDelete={handleDelete} />
        )}
      </div>
    </div>
  )
}
