"use server"

import { z } from "zod"

const saveComparisonSchema = z.object({
  name: z.string().min(1, "Name is required"),
  prompt: z.string().min(1, "Prompt is required"),
  models: z.array(z.string()).min(1, "At least one model is required"),
  responses: z.record(z.string(), z.string()),
  isPublic: z.boolean().optional(),
})

export async function saveUserComparison(formData: FormData) {
  // For demo purposes, we'll just simulate saving
  const validatedFields = saveComparisonSchema.safeParse({
    name: formData.get("name"),
    prompt: formData.get("prompt"),
    models: JSON.parse(formData.get("models") as string),
    responses: JSON.parse(formData.get("responses") as string),
    isPublic: formData.get("isPublic") === "true",
  })

  if (!validatedFields.success) {
    return { error: validatedFields.error.flatten().fieldErrors }
  }

  try {
    // Simulate saving to database
    await new Promise((resolve) => setTimeout(resolve, 1000))

    console.log("Saving comparison:", validatedFields.data)

    return { success: true, id: Math.floor(Math.random() * 1000) }
  } catch (error) {
    return { error: "Failed to save comparison" }
  }
}

export async function deleteUserComparison(id: number) {
  try {
    // Simulate deletion
    await new Promise((resolve) => setTimeout(resolve, 500))

    console.log("Deleting comparison:", id)

    return { success: true }
  } catch (error) {
    return { error: "Failed to delete comparison" }
  }
}

export async function saveUserPreferences(formData: FormData) {
  const theme = formData.get("theme") as string
  const fontSize = formData.get("fontSize") as string
  const defaultModels = JSON.parse(formData.get("defaultModels") as string)

  try {
    // Simulate saving preferences
    await new Promise((resolve) => setTimeout(resolve, 500))

    console.log("Saving preferences:", { theme, fontSize, defaultModels })

    return { success: true }
  } catch (error) {
    return { error: "Failed to save preferences" }
  }
}
