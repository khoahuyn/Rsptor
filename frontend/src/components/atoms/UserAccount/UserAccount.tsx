import { Avatar, Button } from '@heroui/react'
import { useState } from 'react'

interface UserAccountProps {
  userName?: string
  userAvatar?: string
  className?: string
}

export const UserAccount = ({
  userName = 'John Doe',
  userAvatar = 'https://i.pravatar.cc/40?img=5',
  className
}: UserAccountProps) => {
  const [isOpen, setIsOpen] = useState(false)

  return (
    <div className={`relative ${className || ''}`}>
      <Button
        variant="ghost"
        size="sm"
        className="w-10 h-10 min-w-0 p-0 rounded-full border-none"
        onClick={() => setIsOpen(!isOpen)}
      >
        <Avatar
          src={userAvatar}
          className="w-8 h-8 rounded-full"
          size="sm"
        />
      </Button>

      {isOpen && (
        <div className="absolute top-full right-0 mt-1 bg-white border border-gray-200 rounded-lg shadow-lg z-50 min-w-48">
          <div className="px-3 py-2 border-b border-gray-100">
            <p className="text-sm font-medium text-gray-900">{userName}</p>
            <p className="text-xs text-gray-500">john.doe@example.com</p>
          </div>
          <div className="py-1">
            <button className="w-full px-3 py-2 text-left text-sm hover:bg-gray-50">
              Profile Settings
            </button>
            <button className="w-full px-3 py-2 text-left text-sm hover:bg-gray-50">
              Preferences
            </button>
            <hr className="my-1" />
            <button className="w-full px-3 py-2 text-left text-sm text-red-600 hover:bg-gray-50">
              Sign Out
            </button>
          </div>
        </div>
      )}
    </div>
  )
}
