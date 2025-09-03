import { Button } from '@heroui/react'
import { GitHubLogoIcon } from '@radix-ui/react-icons'

interface GitButtonProps {
  className?: string
  onClick?: () => void
}

export const GitButton = ({ className, onClick }: GitButtonProps) => {
  return (
    <Button
      variant="ghost"
      size="sm"
      isIconOnly
      className={`text-gray-600 hover:text-gray-900 ${className || ''}`}
      onClick={onClick}
    >
      <GitHubLogoIcon className="w-4 h-4" />
    </Button>
  )
}
