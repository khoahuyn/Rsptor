import { Text } from '@radix-ui/themes'

interface NavItemProps {
  label: string
  href?: string
  onClick?: () => void
  className?: string
}

export const NavItem = ({ label, href, onClick, className }: NavItemProps) => {
  const baseClasses = "text-gray-600 dark:text-gray-300 hover:text-gray-900 dark:hover:text-gray-100 cursor-pointer transition-colors"

  if (href) {
    return (
      <a href={href} className={`${baseClasses} ${className || ''}`}>
        <Text>{label}</Text>
      </a>
    )
  }

  return (
    <Text
      className={`${baseClasses} ${className || ''}`}
      onClick={onClick}
    >
      {label}
    </Text>
  )
}
