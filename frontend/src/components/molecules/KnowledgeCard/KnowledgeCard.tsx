import { Card, CardBody, Dropdown, DropdownTrigger, DropdownMenu, DropdownItem, Button } from '@heroui/react'
import { Heading, Text, Flex } from '@radix-ui/themes'
import { ClockIcon, FileTextIcon, DotsVerticalIcon, Pencil1Icon, TrashIcon } from '@radix-ui/react-icons'
import { Link } from '@tanstack/react-router'
import { useState } from 'react'

interface KnowledgeCardProps {
  id?: string
  title: string
  description: string
  documentCount: number
  createdAt: string
  className?: string
  onClick?: () => void
  onRename?: (id: string, newName: string) => void
  onDelete?: (id: string) => void
}

export const KnowledgeCard = ({
  id,
  title,
  description,
  documentCount,
  createdAt,
  className,
  onClick,
  onRename,
  onDelete
}: KnowledgeCardProps) => {
  const [isMenuOpen, setIsMenuOpen] = useState(false)
  
  const handleCardClick = (e: React.MouseEvent) => {
    // Prevent navigation if menu is open or if clicking on menu area
    if (isMenuOpen) {
      e.preventDefault()
      e.stopPropagation()
      return
    }
    
    if (onClick) {
      onClick()
    }
  }

  const handleMenuAction = (key: React.Key) => {
    if (!id) return
    
    const keyStr = String(key)
    if (keyStr === 'rename' && onRename) {
      const newName = prompt('Enter new name:', title)
      if (newName && newName.trim() && newName !== title) {
        onRename(id, newName.trim())
      }
    } else if (keyStr === 'delete' && onDelete) {
      onDelete(id)
    }
    setIsMenuOpen(false)
  }

  const formatTimestamp = (dateString: string) => {
    const date = new Date(dateString)
    return date.toLocaleString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    })
  }

  const cardContent = (
    <Card
      className={`px-6 py-6 h-[280px] hover:shadow-lg transition-all duration-200 border border-gray-200 hover:border-indigo-300 relative overflow-visible ${
        isMenuOpen ? 'shadow-lg border-indigo-400' : 'cursor-pointer'
      } ${className || ''}`}
      onClick={onClick ? handleCardClick : undefined}
    >
      <CardBody className="flex flex-col justify-between h-full">
        {/* Three-dot menu */}
        {(onRename || onDelete) && id && (
          <div 
            className="absolute top-2 right-2 z-20"
            onClick={(e) => {
              // Always prevent card click when clicking menu area
              e.stopPropagation()
              e.preventDefault()
            }}
          >
            <Dropdown 
              isOpen={isMenuOpen} 
              onOpenChange={setIsMenuOpen}
              placement="right-start"
              offset={8}
              crossOffset={0}
              shouldFlip={true}
              containerPadding={16}
              classNames={{
                content: "z-50 min-w-32 shadow-lg border border-gray-200 bg-white"
              }}
            >
              <DropdownTrigger>
                <Button
                  isIconOnly
                  size="sm"
                  variant="light"
                  className={`min-w-8 w-8 h-8 transition-colors ${
                    isMenuOpen ? 'bg-gray-200 hover:bg-gray-300' : 'hover:bg-gray-100'
                  }`}
                  onClick={(e) => {
                    e.stopPropagation()
                    e.preventDefault()
                    setIsMenuOpen(!isMenuOpen)
                  }}
                >
                  <DotsVerticalIcon className="w-4 h-4" />
                </Button>
              </DropdownTrigger>
              <DropdownMenu 
                onAction={(key) => {
                  handleMenuAction(key)
                  setIsMenuOpen(false)
                }}
                itemClasses={{
                  base: "gap-2 text-sm py-2 px-3 hover:bg-gray-50 transition-colors"
                }}
              >
                <DropdownItem 
                  key="rename" 
                  startContent={<Pencil1Icon className="w-4 h-4" />}
                  className="text-default-500"
                >
                  Rename
                </DropdownItem>
                <DropdownItem 
                  key="delete" 
                  className="text-danger" 
                  color="danger"
                  startContent={<TrashIcon className="w-4 h-4" />}
                >
                  Delete
                </DropdownItem>
              </DropdownMenu>
            </Dropdown>
          </div>
        )}

        <Heading size="4" className="text-gray-900 font-bold mb-4 pr-10" style={{ color: '#1f2937' }}>
          {title}
        </Heading>

        <Text className="text-gray-600 leading-relaxed mb-6 line-clamp-3 flex-grow">
          {description}
        </Text>

        <Flex align="start" direction={"column"} justify="between" gap={"2"} className="text-base text-gray-500">
          <Flex gap="1">
            <Flex align="center" gap="2">
              <FileTextIcon className="w-4 h-4" />
              <span>{documentCount} documents</span>
            </Flex>
          </Flex>

          <Flex gap="1">
            <Flex align="center" gap="2">
              <ClockIcon className="w-4 h-4" />
              <span>{formatTimestamp(createdAt)}</span>
            </Flex>
          </Flex>
        </Flex>
      </CardBody>
    </Card>
  )

  // If no custom onClick handler and has id, wrap with Link for navigation
  if (!onClick && id) {
    return (
      <Link 
        to="/dataset/$id" 
        params={{ id }} 
        className="block"
        onClick={(e) => {
          // Prevent navigation if menu is open
          if (isMenuOpen) {
            e.preventDefault()
            e.stopPropagation()
            return false
          }
        }}
      >
        {cardContent}
      </Link>
    )
  }

  return cardContent
}
