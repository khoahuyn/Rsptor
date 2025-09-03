import { Button, Card, CardBody } from '@heroui/react'
import { Text, Flex } from '@radix-ui/themes'
import { ClipboardIcon, Cross1Icon, FileIcon } from '@radix-ui/react-icons'
import { useState, useRef } from 'react'
import { validateMarkdownFile } from '../../../utils/fileValidation'

export interface FileUploadItem {
  id: string
  name: string
  size: number
  type: string
  file: File
}

interface FileUploadProps {
  onFilesSelected: (files: FileUploadItem[]) => void
  selectedFiles: FileUploadItem[]
  onRemoveFile: (fileId: string) => void
  maxFiles?: number
  maxSizeBytes?: number
  acceptedTypes?: string[]
  className?: string
}

export const FileUpload = ({
  onFilesSelected,
  selectedFiles,
  onRemoveFile,
  maxFiles = 5,
  maxSizeBytes = 10 * 1024 * 1024, // 10MB
  acceptedTypes = ['text/markdown', 'text/x-markdown', '.md', '.markdown'],
  className
}: FileUploadProps) => {
  const [isDragging, setIsDragging] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes'
    const k = 1024
    const sizes = ['Bytes', 'KB', 'MB', 'GB']
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
  }

  const validateFile = (file: File) => {
    // Use the comprehensive Markdown validation
    const validation = validateMarkdownFile(file)
    
    if (!validation.isValid) {
      alert(`File "${file.name}" is invalid: ${validation.error}`)
      return false
    }
    
    // Show warnings if any
    if (validation.warnings && validation.warnings.length > 0) {
      console.warn(`File "${file.name}" has warnings:`, validation.warnings)
    }

    return true
  }

  const handleFileSelection = (files: FileList) => {
    const validFiles: FileUploadItem[] = []
    const remainingSlots = maxFiles - selectedFiles.length

    Array.from(files).slice(0, remainingSlots).forEach(file => {
      if (validateFile(file)) {
        const fileItem: FileUploadItem = {
          id: Date.now().toString() + Math.random().toString(36).substr(2, 9),
          name: file.name,
          size: file.size,
          type: file.type,
          file
        }
        validFiles.push(fileItem)
      }
    })

    if (validFiles.length > 0) {
      onFilesSelected(validFiles)
    }

    if (files.length > remainingSlots) {
      alert(`Only ${remainingSlots} more files can be added. Maximum ${maxFiles} files allowed.`)
    }
  }

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(true)
  }

  const handleDragLeave = () => {
    setIsDragging(false)
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)

    const files = e.dataTransfer.files
    if (files.length > 0) {
      handleFileSelection(files)
    }
  }

  const handleFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files
    if (files && files.length > 0) {
      handleFileSelection(files)
    }
    // Reset input value to allow selecting the same file again
    if (fileInputRef.current) {
      fileInputRef.current.value = ''
    }
  }

  const canAddMore = selectedFiles.length < maxFiles

  return (
    <div className={className}>
      {/* File Upload Area */}
      {canAddMore && (
        <div
          className={`border-2 border-dashed rounded-lg p-4 text-center transition-colors cursor-pointer ${
            isDragging
              ? 'border-indigo-500 bg-indigo-50'
              : 'border-gray-300 hover:border-gray-400'
          }`}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
        >
          <ClipboardIcon className="w-8 h-8 text-gray-400 mx-auto mb-2" />
          <Text className="text-sm text-gray-600 mb-1">
            Drop files here or click to select
          </Text>
          <Text className="text-xs text-gray-500">
            Max {maxFiles} files, up to {formatFileSize(maxSizeBytes)} each
          </Text>

          <input
            ref={fileInputRef}
            type="file"
            multiple
            accept={acceptedTypes.join(',')}
            onChange={handleFileInputChange}
            className="hidden"
          />
        </div>
      )}

      {/* Selected Files List */}
      {selectedFiles.length > 0 && (
        <div className="mt-3 space-y-2">
          <Text className="text-sm font-medium text-gray-700">
            Selected Files ({selectedFiles.length}/{maxFiles})
          </Text>
          <div className="max-h-64 overflow-y-auto space-y-2">
            {selectedFiles.map((file) => (
              <Card key={file.id} className="bg-gray-50 border border-gray-200">
              <CardBody className="p-3">
                <Flex align="start" justify="between" gap="3">
                  <Flex align="start" gap="3" className="min-w-0 flex-1">
                    <FileIcon className="w-4 h-4 text-gray-500 flex-shrink-0 mt-0.5" />
                    <div className="min-w-0 flex-1">
                      <Text className="text-sm font-medium text-gray-900 break-words leading-tight">
                        {file.name}
                      </Text>
                      <Text className="text-xs text-gray-500 mt-1">
                        {formatFileSize(file.size)}
                      </Text>
                    </div>
                  </Flex>
                  <Button
                    size="sm"
                    variant="light"
                    color="danger"
                    isIconOnly
                    onClick={() => onRemoveFile(file.id)}
                    className="flex-shrink-0 mt-0.5"
                  >
                    <Cross1Icon className="w-3 h-3" />
                  </Button>
                </Flex>
              </CardBody>
            </Card>
          ))}
          </div>
        </div>
      )}
    </div>
  )
}
