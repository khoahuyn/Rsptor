import type { ReactNode } from 'react'

interface IconBoxProps {
  children: ReactNode
  size?: 'sm' | 'md' | 'lg'
  color?: 'blue' | 'green' | 'purple' | 'gray'
  className?: string
}

export const IconBox = ({ children, size = 'md', color = 'blue', className }: IconBoxProps) => {
  const sizeStyles = {
    sm: 'w-8 h-8',
    md: 'w-12 h-12',
    lg: 'w-16 h-16'
  }

  const colorStyles = {
    blue: 'bg-blue-100 text-blue-600',
    green: 'bg-green-100 text-green-600',
    purple: 'bg-purple-100 text-purple-600',
    gray: 'bg-gray-100 text-gray-600'
  }

  return (
    <div className={`${sizeStyles[size]} ${colorStyles[color]} rounded-lg flex items-center justify-center ${className || ''}`}>
      {children}
    </div>
  )
}
