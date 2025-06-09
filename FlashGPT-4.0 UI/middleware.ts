import { NextResponse } from "next/server"
import type { NextRequest } from "next/server"

export function middleware(request: NextRequest) {
  const { nextUrl } = request

  // For now, we'll handle authentication checks on the client side
  // This middleware will just handle basic routing

  // You can add additional middleware logic here if needed
  return NextResponse.next()
}

// Specify which paths should be checked by the middleware
export const config = {
  matcher: ["/my-comparisons/:path*", "/settings/:path*", "/comparison/:path*"],
}
