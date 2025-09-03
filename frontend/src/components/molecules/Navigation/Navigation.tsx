import { Link } from '@tanstack/react-router'
import { NavItem } from '../../atoms'

interface NavigationProps {
  className?: string
}

const navigationItems = [
  { label: 'Features', href: '/#features' },
  { label: 'Knowledge', href: '/knowledge' },
  { label: 'About', href: '/about' },
  { label: 'Docs', href: '#docs' }
]

export const Navigation = ({ className }: NavigationProps) => {
  return (
    <nav className={`hidden md:flex items-center gap-6 ${className || ''}`}>
      {navigationItems.map((item) => (
        item.href.startsWith('#') ? (
          <NavItem
            key={item.label}
            label={item.label}
            href={item.href}
          />
        ) : (
          <Link key={item.label} to={item.href} className="text-gray-600 dark:text-gray-300 hover:text-gray-900 dark:hover:text-gray-100 cursor-pointer transition-colors">
            {item.label}
          </Link>
        )
      ))}
    </nav>
  )
}
