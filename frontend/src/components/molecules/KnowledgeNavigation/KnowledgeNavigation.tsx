import { Button } from '@heroui/react'
import { Flex } from '@radix-ui/themes'
import { useNavigate, useLocation } from '@tanstack/react-router'
import {
  FileTextIcon,
  ChatBubbleIcon
} from '@radix-ui/react-icons'

interface KnowledgeNavigationProps {
  className?: string
}

const navigationItems = [
  {
    id: 'knowledge-base',
    label: 'Knowledge Base',
    icon: FileTextIcon,
    route: '/knowledge'
  },
  {
    id: 'search',
    label: 'Chat',
    icon: ChatBubbleIcon,
    route: '/chat'
  },
  {
    id: 'agent',
    label: 'Agent',
    icon: ChatBubbleIcon,
    route: '/agent'
  }
]

export const KnowledgeNavigation = ({ className }: KnowledgeNavigationProps) => {
  const navigate = useNavigate()
  const location = useLocation()

  // Determine active item based on current route
  const getActiveItemId = () => {
    const currentPath = location.pathname

    // Check if we're on a dataset route (starts with /dataset)
    if (currentPath.startsWith('/dataset')) {
      return 'knowledge-base'
    }

    const activeItem = navigationItems.find(item => item.route === currentPath)
    return activeItem?.id || 'knowledge-base' // Default to knowledge-base
  }

  const activeItemId = getActiveItemId()

  const handleItemClick = (itemId: string) => {
    const item = navigationItems.find(nav => nav.id === itemId)
    if (item && item.route !== '/agent') { // Skip navigation for agent (not implemented)
      if (item.route === '/knowledge') {
        navigate({ to: '/knowledge' })
      } else if (item.route === '/chat') {
        navigate({ to: '/chat' })
      }
    }
  }

  return (
    <div className="bg-gray-50/80 dark:bg-gray-800/80 border border-transparent hover:border hover:border-gray-100 dark:hover:border-gray-600 rounded-lg px-3 py-2 backdrop-blur-sm">
      <Flex align="center" gap="3" className={className}>
        {navigationItems.map((item) => {
          const IconComponent = item.icon
          return (
            <Button
              key={item.id}
              variant={activeItemId === item.id ? "solid" : "ghost"}
              color={activeItemId === item.id ? "primary" : "default"}
              size="md"
              startContent={<IconComponent className="w-4 h-4" />}
              className={`
                px-4 py-2 font-bold border-transparent rounded-md transition-colors
                ${activeItemId === item.id
                  ? 'bg-indigo-600 dark:bg-indigo-500 text-white shadow-md'
                  : 'text-gray-600 dark:text-gray-300 hover:text-gray-900 dark:hover:text-gray-100 hover:bg-gray-100 dark:hover:bg-gray-700'
                }
                ${item.id === 'agent' ? 'opacity-50 cursor-not-allowed' : ''}
              `}
              disabled={item.id === 'agent'}
              onClick={() => handleItemClick(item.id)}
            >
              {item.label}
            </Button>
          )
        })}
      </Flex>
    </div>
  )
}
