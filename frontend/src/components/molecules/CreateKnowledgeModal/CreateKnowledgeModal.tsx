import {
  Modal,
  ModalContent,
  ModalHeader,
  ModalBody,
  ModalFooter
} from '@heroui/react'
import { Text } from '@radix-ui/themes'
import { useRef, useEffect } from 'react'
import { type CreateKnowledgeFormData } from '../../../schemas'
import { CreateKnowledgeForm, type CreateKnowledgeFormRef } from '../CreateKnowledgeForm'
import { ModalActions } from '../../atoms/ModalActions'

export interface CreateKnowledgeModalProps {
  isOpen: boolean
  onClose: () => void
  onSubmit: (data: CreateKnowledgeFormData) => void
}

export const CreateKnowledgeModal = ({ isOpen, onClose, onSubmit }: CreateKnowledgeModalProps) => {
  const formRef = useRef<CreateKnowledgeFormRef>(null)

  // Reset form when modal closes
  useEffect(() => {
    if (!isOpen && formRef.current) {
      formRef.current.reset()
    }
  }, [isOpen])

  const handleSubmit = async (data: CreateKnowledgeFormData) => {
    try {
      await onSubmit(data)
      onClose()
    } catch (error) {
      console.error('Error creating knowledge base:', error)
      throw error // Re-throw to let form handle the error state
    }
  }

  const handleClose = () => {
    if (!formRef.current?.isSubmitting) {
      onClose()
    }
  }

  const handleConfirm = async () => {
    if (formRef.current) {
      await formRef.current.submit()
    }
  }

  return (
    <Modal
      isOpen={isOpen}
      onClose={handleClose}
      size="md"
      placement="center"
      backdrop="blur"
      classNames={{
        closeButton: "text-xl p-2 m-2"
      }}
    >
      <ModalContent>
        <ModalHeader className="flex flex-col gap-1">
          <h2 className="text-xl font-bold text-gray-900">Create New Knowledge Base</h2>
          <Text size="2" className="text-gray-600">
            Create a new knowledge base to organize your documents
          </Text>
        </ModalHeader>

        <ModalBody className="py-6">
          <CreateKnowledgeForm
            ref={formRef}
            onSubmit={handleSubmit}
            disabled={formRef.current?.isSubmitting}
          />
        </ModalBody>

        <ModalFooter>
          <ModalActions
            onCancel={handleClose}
            onConfirm={handleConfirm}
            cancelText="Cancel"
            confirmText={formRef.current?.isSubmitting ? 'Creating...' : 'Create Knowledge Base'}
            isLoading={formRef.current?.isSubmitting}
            disabled={formRef.current?.isSubmitting}
          />
        </ModalFooter>
      </ModalContent>
    </Modal>
  )
}
