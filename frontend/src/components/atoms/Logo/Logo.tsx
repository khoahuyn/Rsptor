import { Flex, Heading } from '@radix-ui/themes'
import { FileTextIcon } from '@radix-ui/react-icons'
import { Link } from '@tanstack/react-router'

interface LogoProps {
  size?: 'sm' | 'md' | 'lg'
  showText?: boolean
  className?: string
}

export const Logo = ({ size = 'md', showText = true, className }: LogoProps) => {
  const sizeStyles = {
    sm: { icon: 'w-4 h-4', container: 'w-6 h-6', heading: '3' as const },
    md: { icon: 'w-5 h-5', container: 'w-8 h-8', heading: '5' as const },
    lg: { icon: 'w-6 h-6', container: 'w-10 h-10', heading: '6' as const }
  }

  const style = sizeStyles[size]

  return (
    <Link to="/" className={`no-underline ${className || ''}`}>
      <Flex align="center" gap="3">
        <div className={`${style.container} bg-gradient-to-r from-indigo-500 to-purple-600 rounded-lg flex items-center justify-center`}>
          <FileTextIcon className={`${style.icon} text-white`} />
        </div>
        {showText && (
          <Heading size={style.heading} className="text-gray-900">
            Raptor Service
          </Heading>
        )}
      </Flex>
    </Link>
  )
}
