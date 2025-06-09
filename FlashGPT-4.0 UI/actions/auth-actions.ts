"use server"

import { z } from "zod"
import { signIn } from "@/auth"
import { redirect } from "next/navigation"

const registerSchema = z
  .object({
    name: z.string().min(1, "Name is required"),
    email: z.string().email("Invalid email address"),
    password: z.string().min(8, "Password must be at least 8 characters"),
    confirmPassword: z.string(),
  })
  .refine((data) => data.password === data.confirmPassword, {
    message: "Passwords do not match",
    path: ["confirmPassword"],
  })

export async function register(formData: FormData) {
  const validatedFields = registerSchema.safeParse({
    name: formData.get("name"),
    email: formData.get("email"),
    password: formData.get("password"),
    confirmPassword: formData.get("confirmPassword"),
  })

  if (!validatedFields.success) {
    return { error: validatedFields.error.flatten().fieldErrors }
  }

  const { name, email, password } = validatedFields.data

  // For demo purposes, we'll just show a success message
  // In a real app, you'd save the user to your database
  console.log("User registration:", { name, email })

  try {
    // For demo, we'll try to sign in with demo credentials
    const result = await signIn("credentials", {
      email: "demo@example.com",
      password: "password",
      redirect: false,
    })

    if (result?.error) {
      return { error: { _form: ["Registration successful! Please sign in with demo@example.com / password"] } }
    }

    redirect("/")
  } catch (error) {
    return { error: { _form: ["Registration successful! Please sign in with demo@example.com / password"] } }
  }
}

const loginSchema = z.object({
  email: z.string().email("Invalid email address"),
  password: z.string().min(1, "Password is required"),
})

export async function login(formData: FormData) {
  const validatedFields = loginSchema.safeParse({
    email: formData.get("email"),
    password: formData.get("password"),
  })

  if (!validatedFields.success) {
    return { error: validatedFields.error.flatten().fieldErrors }
  }

  const { email, password } = validatedFields.data

  try {
    const result = await signIn("credentials", {
      email,
      password,
      redirect: false,
    })

    if (result?.error) {
      return { error: { _form: ["Invalid email or password. Try demo@example.com / password"] } }
    }

    redirect("/")
  } catch (error) {
    console.error("Login error:", error)
    return { error: { _form: ["Something went wrong. Please try again."] } }
  }
}
