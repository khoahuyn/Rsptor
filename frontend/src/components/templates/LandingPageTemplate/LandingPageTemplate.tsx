import React, { type ReactNode } from 'react'
import { Header, Footer } from '../../organisms'

interface LandingPageTemplateProps {
  children: ReactNode
  className?: string
}

export const LandingPageTemplate = ({ children, className }: LandingPageTemplateProps) => {
  return (
    <div className={`min-h-screen bg-gradient-to-br from-slate-50 to-indigo-50 ${className || ''}`}>
      {/* Header + HeroSection area (100vh) */}
      <div className="h-screen flex flex-col">
        <Header />
        <main className="flex-1">
          {/* Extract first child (HeroSection) for the 100vh area */}
          {React.Children.toArray(children)[0]}
        </main>
      </div>

      {/* Remaining sections below 100vh */}
      <div className="bg-gradient-to-br from-slate-50 to-indigo-50">
        {React.Children.toArray(children).slice(1)}
        <Footer />
      </div>
    </div>
  )
}
