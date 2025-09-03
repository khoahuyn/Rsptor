import {
  Modal,
  ModalContent,
  ModalHeader,
  ModalBody,
  ModalFooter,
  Button
} from '@heroui/react'
import { Text } from '@radix-ui/themes'
import { FileUpload } from '../../molecules'
import type { FileUploadItem } from '../../molecules'
import { useState, useEffect } from 'react'

export interface AddDocumentModalProps {
  isOpen: boolean
  onClose: () => void
  onSubmit: (files: FileUploadItem[]) => void
}

export const AddDocumentModal = ({ isOpen, onClose, onSubmit }: AddDocumentModalProps) => {
  const [selectedFiles, setSelectedFiles] = useState<FileUploadItem[]>([])
  const [isSubmitting, setIsSubmitting] = useState(false)

  // Reset state when modal closes
  useEffect(() => {
    if (!isOpen) {
      setSelectedFiles([])
      setIsSubmitting(false)
    }
  }, [isOpen])

  const handleFilesSelected = (files: FileUploadItem[]) => {
    setSelectedFiles(prev => [...prev, ...files])
  }

  const handleRemoveFile = (fileId: string) => {
    setSelectedFiles(prev => prev.filter(file => file.id !== fileId))
  }

  const handleSubmit = async () => {
    if (selectedFiles.length === 0) return

    setIsSubmitting(true)
    
    // Close modal immediately when upload starts
    onClose()
    
    try {
      await onSubmit(selectedFiles)
      // Success - user will see progress in main interface
      console.log('Upload initiated successfully')
    } catch (error) {
      console.error('Error uploading documents:', error)
      // Note: Error handling should be shown in main interface, not modal
    } finally {
      setIsSubmitting(false)
    }
  }

  const handleClose = () => {
    if (!isSubmitting) {
      onClose()
    }
  }

  const isValid = selectedFiles.length > 0

  return (
    <Modal
      isOpen={isOpen}
      onClose={handleClose}
      size="lg"
      placement="center"
      backdrop="blur"
      classNames={{
        closeButton: "text-xl p-2 m-2"
      }}
    >
      <ModalContent>
        <ModalHeader className="flex flex-col gap-1">
          <h2 className="text-xl font-bold text-gray-900">Add Documents</h2>
          <Text size="2" className="text-gray-600">
            Upload documents to add to your knowledge base
          </Text>
        </ModalHeader>

        <ModalBody className="py-6">
          <div className="space-y-4">
            <Text size="3" className="text-gray-700 font-medium">
              Select Files
            </Text>
            <FileUpload
              onFilesSelected={handleFilesSelected}
              selectedFiles={selectedFiles}
              onRemoveFile={handleRemoveFile}
              maxFiles={5}
            />
            {selectedFiles.length > 0 && (
              <Text size="2" className="text-gray-600">
                {selectedFiles.length} file(s) selected for upload
              </Text>
            )}
          </div>
        </ModalBody>

        <ModalFooter>
          <Button
            variant="ghost"
            size="lg"
            onPress={handleClose}
            disabled={isSubmitting}
          >
            Cancel
          </Button>
          <Button
            color="primary"
            size="lg"
            onPress={handleSubmit}
            disabled={!isValid || isSubmitting}
            isLoading={isSubmitting}
          >
            {isSubmitting ? 'Uploading...' : 'Upload Documents'}
          </Button>
        </ModalFooter>
      </ModalContent>
    </Modal>
  )
}
