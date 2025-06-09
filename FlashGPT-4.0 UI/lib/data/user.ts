import { db, users, userPreferences } from "@/lib/db"
import { eq } from "drizzle-orm"
import { hash } from "bcrypt"
import { v4 as uuidv4 } from "uuid"

export async function getUserByEmail(email: string) {
  const [user] = await db.select().from(users).where(eq(users.email, email)).limit(1)

  return user
}

export async function getUserById(id: string) {
  const [user] = await db.select().from(users).where(eq(users.id, id)).limit(1)

  return user
}

export async function createUser(data: { name: string; email: string; password: string }) {
  const hashedPassword = await hash(data.password, 10)
  const userId = uuidv4()

  await db.insert(users).values({
    id: userId,
    name: data.name,
    email: data.email,
    password: hashedPassword,
  })

  // Create default preferences
  await db.insert(userPreferences).values({
    userId,
    theme: "dark",
    fontSize: "medium",
    defaultModels: ["anthropic", "openai", "ollama"],
  })

  return userId
}

export async function getUserPreferences(userId: string) {
  const [preferences] = await db.select().from(userPreferences).where(eq(userPreferences.userId, userId)).limit(1)

  return preferences
}

export async function updateUserPreferences(
  userId: string,
  data: {
    theme?: string
    fontSize?: string
    defaultModels?: string[]
  },
) {
  await db
    .update(userPreferences)
    .set({
      ...data,
      updatedAt: new Date(),
    })
    .where(eq(userPreferences.userId, userId))
}
