import { Button, Avatar } from '@heroui/react'
import { Flex } from '@radix-ui/themes'
import {
  FileTextIcon,
  MagnifyingGlassIcon,
  GearIcon
} from '@radix-ui/react-icons'
import { useState } from 'react'

interface DatasetSidebarProps {
  className?: string
}

const menuItems = [
  {
    id: 'dataset',
    label: 'Dataset',
    icon: FileTextIcon,
    active: true
  },
  {
    id: 'retrieval',
    label: 'Retrieval',
    icon: MagnifyingGlassIcon,
    active: false
  },
  {
    id: 'configuration',
    label: 'Configuration',
    icon: GearIcon,
    active: false
  }
]

export const DatasetSidebar = ({ className }: DatasetSidebarProps) => {
  const [activeItem, setActiveItem] = useState('dataset')

  const handleItemClick = (itemId: string) => {
    setActiveItem(itemId)
    // TODO: Add navigation logic when other sections are implemented
  }

  return (
    <div className={`w-64 bg-white dark:bg-gray-900 border-r border-gray-200 dark:border-gray-700 flex flex-col ${className || ''}`}>
      {/* User Section */}
      <div className="p-6">
        <div className="flex flex-col items-center">
          <Avatar
            size="lg"
            className="bg-indigo-600 text-white mb-3"
            fallback="U"
          />
          <span className="text-gray-800 dark:text-gray-200 font-medium text-base">John Doe</span>
        </div>
      </div>

      {/* Navigation Menu */}
      <nav className="flex-1 p-4">
        <Flex direction="column" gap="2">
          {menuItems.map((item) => {
            const IconComponent = item.icon
            return (
              <Button
                key={item.id}
                variant="ghost"
                size="lg"
                startContent={<IconComponent className={`w-5 h-5 ${activeItem === item.id ? 'text-blue-700 dark:text-blue-400' : 'text-gray-600 dark:text-gray-400'}`} />}
                className={`
                  !w-full !justify-start !px-4 !py-3 !font-medium !text-left !border-0 !shadow-none
                  ${activeItem === item.id
                    ? '!bg-blue-100 dark:!bg-blue-900/30 !text-blue-700 dark:!text-blue-400 hover:!bg-blue-100 dark:hover:!bg-blue-900/30'
                    : '!text-gray-700 dark:!text-gray-300 hover:!text-gray-900 dark:hover:!text-gray-100 hover:!bg-gray-100 dark:hover:!bg-gray-800'
                  }
                `}
                onClick={() => handleItemClick(item.id)}
              >
                {item.label}
              </Button>
            )
          })}
        </Flex>
      </nav>
    </div>
  )
}
