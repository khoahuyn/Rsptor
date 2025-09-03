import {
  Button,
  Dropdown,
  DropdownTrigger,
  DropdownMenu,
  DropdownItem,
  Tooltip
} from '@heroui/react'
import {
  Pencil1Icon,
  ChevronDownIcon,
  DownloadIcon,
  TrashIcon
} from '@radix-ui/react-icons'

export interface ActionButtonsProps {
  documentId: string
  onRename: (documentId: string) => void
  onEditChunking: (documentId: string) => void
  onDownload: (documentId: string) => void
  onDelete: (documentId: string) => void
  className?: string
}

export const ActionButtons = ({
  documentId,
  onRename,
  onEditChunking,
  onDownload,
  onDelete,
  className
}: ActionButtonsProps) => {
  return (
    <div className={`flex items-center gap-1 ${className || ''}`}>
      <Tooltip content="Rename document" placement="top">
        <Button
          isIconOnly
          size="sm"
          variant="ghost"
          onPress={() => onRename(documentId)}
          className="text-gray-600 hover:text-blue-600"
        >
          <Pencil1Icon className="w-4 h-4" />
        </Button>
      </Tooltip>

      <Tooltip content="Edit chunking method" placement="top">
        <Dropdown>
          <DropdownTrigger>
            <Button
              isIconOnly
              size="sm"
              variant="ghost"
              className="text-gray-600 hover:text-orange-600"
            >
              <ChevronDownIcon className="w-4 h-4" />
            </Button>
          </DropdownTrigger>
          <DropdownMenu>
            <DropdownItem
              key="edit-chunking"
              onPress={() => onEditChunking(documentId)}
            >
              Edit Chunking Method
            </DropdownItem>
          </DropdownMenu>
        </Dropdown>
      </Tooltip>

      <Tooltip content="Download document" placement="top">
        <Button
          isIconOnly
          size="sm"
          variant="ghost"
          onPress={() => onDownload(documentId)}
          className="text-gray-600 hover:text-green-600"
        >
          <DownloadIcon className="w-4 h-4" />
        </Button>
      </Tooltip>

      <Tooltip content="Delete document" placement="top">
        <Button
          isIconOnly
          size="sm"
          variant="ghost"
          onPress={() => onDelete(documentId)}
          className="text-gray-600 hover:text-red-600"
        >
          <TrashIcon className="w-4 h-4" />
        </Button>
      </Tooltip>
    </div>
  )
}
