"use client"

import Link from "next/link"
import { usePathname } from "next/navigation"
import { Button } from "@/components/ui/button"
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import { LogOut, Settings, Save, Home, User, Bell, HelpCircle } from "lucide-react"
import { useAuth } from "@/contexts/auth-context"
import { Badge } from "@/components/ui/badge"
import { AnimatedLogo } from "@/components/ui/animated-logo"
import { motion } from "framer-motion"

export function NavBar() {
  const { user, isLoading, logout } = useAuth()
  const pathname = usePathname()

  const isActive = (path: string) => pathname === path

  const handleLogout = async () => {
    await logout()
  }

  const getUserInitials = (name?: string, email?: string) => {
    if (name) {
      return name
        .split(" ")
        .map((n) => n[0])
        .join("")
        .toUpperCase()
        .slice(0, 2)
    }
    if (email) {
      return email[0].toUpperCase()
    }
    return "U"
  }

  const getProviderBadge = (provider?: string) => {
    if (!provider || provider === "email") return null

    const badges = {
      google: "🔍",
      github: "🐙",
    }

    return badges[provider as keyof typeof badges] || null
  }

  return (
    <motion.header
      initial={{ y: -100, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{ duration: 0.5, ease: "easeOut" }}
      className="sticky top-0 z-50 w-full border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60"
    >
      <div className="container mx-auto px-4 py-3 flex items-center justify-between">
        {/* Left side - Logo and User Profile */}
        <div className="flex items-center space-x-6">
          {/* Animated Logo */}
          <AnimatedLogo size="md" />

          {/* User Profile Section - Only show when logged in */}
          {user && (
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.5, delay: 0.2 }}
              className="hidden md:flex items-center space-x-3 px-3 py-2 rounded-lg bg-muted/50 border"
            >
              <motion.div whileHover={{ scale: 1.1 }} transition={{ duration: 0.2 }}>
                <Avatar className="h-8 w-8 border-2 border-background shadow-sm">
                  <AvatarImage src={user.avatar || ""} alt={user.name || "User"} />
                  <AvatarFallback className="text-xs font-medium bg-gradient-to-br from-purple-500 to-blue-500 text-white">
                    {getUserInitials(user.name, user.email)}
                  </AvatarFallback>
                </Avatar>
              </motion.div>
              <div className="flex flex-col min-w-0">
                <div className="flex items-center space-x-2">
                  <span className="text-sm font-medium truncate max-w-[120px]">{user.name || "Anonymous User"}</span>
                  {getProviderBadge(user.provider) && (
                    <span className="text-xs">{getProviderBadge(user.provider)}</span>
                  )}
                </div>
                <span className="text-xs text-muted-foreground truncate max-w-[120px]">{user.email}</span>
              </div>
            </motion.div>
          )}
        </div>

        {/* Center - Navigation */}
        <motion.nav
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.3 }}
          className="hidden lg:flex items-center space-x-1"
        >
          <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
            <Button asChild variant={isActive("/") ? "secondary" : "ghost"} size="sm">
              <Link href="/">
                <Home className="mr-2 h-4 w-4" />
                Compare
              </Link>
            </Button>
          </motion.div>

          {user && (
            <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
              <Button asChild variant={isActive("/my-comparisons") ? "secondary" : "ghost"} size="sm">
                <Link href="/my-comparisons">
                  <Save className="mr-2 h-4 w-4" />
                  My Comparisons
                </Link>
              </Button>
            </motion.div>
          )}

          <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
            <Button asChild variant={isActive("/help") ? "secondary" : "ghost"} size="sm">
              <Link href="/help">
                <HelpCircle className="mr-2 h-4 w-4" />
                Help
              </Link>
            </Button>
          </motion.div>
        </motion.nav>

        {/* Right side - Actions and User Menu */}
        <motion.div
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.5, delay: 0.4 }}
          className="flex items-center space-x-3"
        >
          {isLoading ? (
            <motion.div
              animate={{ rotate: 360 }}
              transition={{ duration: 1, repeat: Number.POSITIVE_INFINITY, ease: "linear" }}
              className="h-8 w-8 rounded-full bg-muted"
            />
          ) : user ? (
            <>
              {/* Notifications */}
              <motion.div whileHover={{ scale: 1.1 }} whileTap={{ scale: 0.9 }}>
                <Button variant="ghost" size="sm" className="relative">
                  <Bell className="h-4 w-4" />
                  <motion.div
                    animate={{ scale: [1, 1.2, 1] }}
                    transition={{ duration: 2, repeat: Number.POSITIVE_INFINITY }}
                  >
                    <Badge className="absolute -top-1 -right-1 h-5 w-5 flex items-center justify-center p-0 text-xs bg-red-500">
                      3
                    </Badge>
                  </motion.div>
                </Button>
              </motion.div>

              {/* User Menu */}
              <DropdownMenu>
                <DropdownMenuTrigger asChild>
                  <motion.div whileHover={{ scale: 1.1 }} whileTap={{ scale: 0.9 }}>
                    <Button variant="ghost" className="relative rounded-full h-8 w-8 p-0">
                      <Avatar className="h-8 w-8 border-2 border-background shadow-sm">
                        <AvatarImage src={user.avatar || ""} alt={user.name || "User"} />
                        <AvatarFallback className="bg-gradient-to-br from-purple-500 to-blue-500 text-white">
                          {getUserInitials(user.name, user.email)}
                        </AvatarFallback>
                      </Avatar>
                      {getProviderBadge(user.provider) && (
                        <span className="absolute -bottom-1 -right-1 text-xs bg-background rounded-full p-1">
                          {getProviderBadge(user.provider)}
                        </span>
                      )}
                    </Button>
                  </motion.div>
                </DropdownMenuTrigger>
                <DropdownMenuContent align="end" className="w-64">
                  {/* User Info Header */}
                  <div className="flex items-center justify-start gap-3 p-3 border-b">
                    <Avatar className="h-10 w-10 border-2 border-background shadow-sm">
                      <AvatarImage src={user.avatar || ""} alt={user.name || "User"} />
                      <AvatarFallback className="bg-gradient-to-br from-purple-500 to-blue-500 text-white">
                        {getUserInitials(user.name, user.email)}
                      </AvatarFallback>
                    </Avatar>
                    <div className="flex flex-col space-y-1 leading-none min-w-0">
                      <div className="flex items-center space-x-2">
                        <p className="font-medium text-sm truncate">{user.name || "Anonymous User"}</p>
                        {getProviderBadge(user.provider) && (
                          <span className="text-xs">{getProviderBadge(user.provider)}</span>
                        )}
                      </div>
                      <p className="text-xs text-muted-foreground truncate">{user.email}</p>
                    </div>
                  </div>

                  {/* Menu Items */}
                  <DropdownMenuItem asChild className="cursor-pointer">
                    <Link href="/profile" className="flex items-center">
                      <User className="mr-2 h-4 w-4" />
                      <span>View Profile</span>
                    </Link>
                  </DropdownMenuItem>

                  <DropdownMenuItem asChild className="cursor-pointer">
                    <Link href="/my-comparisons" className="flex items-center">
                      <Save className="mr-2 h-4 w-4" />
                      <span>My Comparisons</span>
                    </Link>
                  </DropdownMenuItem>

                  <DropdownMenuItem asChild className="cursor-pointer">
                    <Link href="/settings" className="flex items-center">
                      <Settings className="mr-2 h-4 w-4" />
                      <span>Settings</span>
                    </Link>
                  </DropdownMenuItem>

                  <DropdownMenuSeparator />

                  <DropdownMenuItem className="cursor-pointer">
                    <Bell className="mr-2 h-4 w-4" />
                    <span>Notifications</span>
                    <Badge className="ml-auto bg-red-500 text-white">3</Badge>
                  </DropdownMenuItem>

                  <DropdownMenuItem asChild className="cursor-pointer">
                    <Link href="/help" className="flex items-center">
                      <HelpCircle className="mr-2 h-4 w-4" />
                      <span>Help & Support</span>
                    </Link>
                  </DropdownMenuItem>

                  <DropdownMenuSeparator />

                  <DropdownMenuItem onClick={handleLogout} className="cursor-pointer text-destructive">
                    <LogOut className="mr-2 h-4 w-4" />
                    <span>Sign Out</span>
                  </DropdownMenuItem>
                </DropdownMenuContent>
              </DropdownMenu>
            </>
          ) : (
            <div className="flex items-center space-x-2">
              <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
                <Button asChild variant="ghost" size="sm">
                  <Link href="/login">Sign in</Link>
                </Button>
              </motion.div>
              <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
                <Button asChild size="sm">
                  <Link href="/register">Sign up</Link>
                </Button>
              </motion.div>
            </div>
          )}
        </motion.div>
      </div>
    </motion.header>
  )
}
