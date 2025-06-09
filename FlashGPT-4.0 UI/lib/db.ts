import { drizzle } from "drizzle-orm/vercel-postgres"
import { sql } from "@vercel/postgres"
import { pgTable, serial, text, timestamp, json, boolean, primaryKey } from "drizzle-orm/pg-core"
import { createInsertSchema, createSelectSchema } from "drizzle-zod"
import { z } from "zod"

// Create database connection
export const db = drizzle(sql)

// Define users table
export const users = pgTable("users", {
  id: text("id").primaryKey().notNull(),
  name: text("name"),
  email: text("email").notNull().unique(),
  password: text("password"),
  emailVerified: timestamp("emailVerified", { mode: "date" }),
  image: text("image"),
  createdAt: timestamp("createdAt", { mode: "date" }).defaultNow().notNull(),
  updatedAt: timestamp("updatedAt", { mode: "date" }).defaultNow().notNull(),
})

// Define user preferences table
export const userPreferences = pgTable(
  "user_preferences",
  {
    userId: text("userId")
      .notNull()
      .references(() => users.id, { onDelete: "cascade" }),
    theme: text("theme").default("dark").notNull(),
    fontSize: text("fontSize").default("medium").notNull(),
    defaultModels: json("defaultModels").$type<string[]>().default(["anthropic", "openai", "ollama"]),
    updatedAt: timestamp("updatedAt", { mode: "date" }).defaultNow().notNull(),
  },
  (table) => {
    return {
      pk: primaryKey({ columns: [table.userId] }),
    }
  },
)

// Define saved comparisons table
export const savedComparisons = pgTable("saved_comparisons", {
  id: serial("id").primaryKey(),
  userId: text("userId")
    .notNull()
    .references(() => users.id, { onDelete: "cascade" }),
  name: text("name").notNull(),
  prompt: text("prompt").notNull(),
  models: json("models").$type<string[]>().notNull(),
  responses: json("responses").$type<Record<string, string>>().notNull(),
  isPublic: boolean("isPublic").default(false).notNull(),
  createdAt: timestamp("createdAt", { mode: "date" }).defaultNow().notNull(),
  updatedAt: timestamp("updatedAt", { mode: "date" }).defaultNow().notNull(),
})

// Zod schemas for validation
export const insertUserSchema = createInsertSchema(users)
export const selectUserSchema = createSelectSchema(users)
export const insertPreferencesSchema = createInsertSchema(userPreferences)
export const selectPreferencesSchema = createSelectSchema(userPreferences)
export const insertComparisonSchema = createInsertSchema(savedComparisons)
export const selectComparisonSchema = createSelectSchema(savedComparisons)

export const newUserSchema = insertUserSchema
  .extend({
    password: z.string().min(8, "Password must be at least 8 characters"),
    confirmPassword: z.string(),
  })
  .refine((data) => data.password === data.confirmPassword, {
    message: "Passwords do not match",
    path: ["confirmPassword"],
  })
