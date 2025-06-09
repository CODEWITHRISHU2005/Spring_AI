import { db, savedComparisons } from "@/lib/db"
import { eq } from "drizzle-orm"

export async function getSavedComparisons(userId: string) {
  const comparisons = await db
    .select()
    .from(savedComparisons)
    .where(eq(savedComparisons.userId, userId))
    .orderBy(savedComparisons.createdAt)

  return comparisons
}

export async function getSavedComparisonById(id: number, userId: string) {
  const [comparison] = await db
    .select()
    .from(savedComparisons)
    .where(eq(savedComparisons.id, id))
    .where(eq(savedComparisons.userId, userId))
    .limit(1)

  return comparison
}

export async function saveComparison(data: {
  userId: string
  name: string
  prompt: string
  models: string[]
  responses: Record<string, string>
  isPublic?: boolean
}) {
  const [result] = await db
    .insert(savedComparisons)
    .values({
      userId: data.userId,
      name: data.name,
      prompt: data.prompt,
      models: data.models,
      responses: data.responses,
      isPublic: data.isPublic || false,
    })
    .returning({ id: savedComparisons.id })

  return result
}

export async function deleteComparison(id: number, userId: string) {
  await db.delete(savedComparisons).where(eq(savedComparisons.id, id)).where(eq(savedComparisons.userId, userId))
}
