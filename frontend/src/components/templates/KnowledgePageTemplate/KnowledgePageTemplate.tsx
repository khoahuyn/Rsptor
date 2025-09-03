import type { ReactNode } from 'react'
import { KnowledgeHeader } from '../../organisms'

interface KnowledgePageTemplateProps {
  children: ReactNode
  className?: string
}

export const KnowledgePageTemplate = ({ children, className }: KnowledgePageTemplateProps) => {
  return (
    <div className={`min-h-screen bg-gray-50 ${className || ''}`}>
      <KnowledgeHeader />
      <main>
        {children}
      </main>
    </div>
  )
}
