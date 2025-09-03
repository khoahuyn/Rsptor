import { Button } from '@heroui/react'
import { QuestionMarkCircledIcon } from '@radix-ui/react-icons'

interface QuestionButtonProps {
  className?: string
  onClick?: () => void
}

export const QuestionButton = ({ className, onClick }: QuestionButtonProps) => {
  return (
    <Button
      variant="ghost"
      size="sm"
      isIconOnly
      className={`text-gray-600 hover:text-gray-900 ${className || ''}`}
      onClick={onClick}
    >
      <QuestionMarkCircledIcon className="w-4 h-4" />
    </Button>
  )
}
