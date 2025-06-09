import NextAuth from "next-auth"
import CredentialsProvider from "next-auth/providers/credentials"
import { compare } from "bcrypt"

// We'll create a simple in-memory user store for now
// In production, you'd use a real database
const users = [
  {
    id: "1",
    name: "Demo User",
    email: "demo@example.com",
    password: "$2b$10$K7L1OJ45/4Y2nIvhRVpCe.FSmhDlwjuuoD.ZMxVBvFP4jLM/qK9S6", // "password"
  },
]

async function getUserByEmail(email: string) {
  return users.find((user) => user.email === email)
}

export const { handlers, auth, signIn, signOut } = NextAuth({
  pages: {
    signIn: "/login",
  },
  callbacks: {
    jwt: async ({ token, user }) => {
      if (user) {
        token.id = user.id
        token.email = user.email
        token.name = user.name
      }
      return token
    },
    session: async ({ session, token }) => {
      if (token && session.user) {
        session.user.id = token.id as string
      }
      return session
    },
  },
  providers: [
    CredentialsProvider({
      name: "credentials",
      credentials: {
        email: { label: "Email", type: "email" },
        password: { label: "Password", type: "password" },
      },
      async authorize(credentials) {
        if (!credentials?.email || !credentials?.password) {
          return null
        }

        const user = await getUserByEmail(credentials.email)
        if (!user || !user.password) {
          return null
        }

        const passwordMatch = await compare(credentials.password, user.password)
        if (!passwordMatch) {
          return null
        }

        return {
          id: user.id,
          email: user.email,
          name: user.name,
        }
      },
    }),
  ],
})
