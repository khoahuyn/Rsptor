import { Button, Modal, ModalContent, ModalHeader, ModalBody, ModalFooter } from '@heroui/react'
import { Flex, Text } from '@radix-ui/themes'
import { ExclamationTriangleIcon } from '@radix-ui/react-icons'

interface DeleteAssistantModalProps {
  isOpen: boolean
  onClose: () => void
  onConfirm: () => void
  assistantName: string
  isDeleting?: boolean
}

export const DeleteAssistantModal = ({
  isOpen,
  onClose,
  onConfirm,
  assistantName,
  isDeleting = false
}: DeleteAssistantModalProps) => {
  return (
    <Modal isOpen={isOpen} onClose={onClose} size="md">
      <ModalContent>
        <ModalHeader className="flex flex-col gap-1">
          <Flex align="center" gap="2">
            <ExclamationTriangleIcon className="w-5 h-5 text-red-500 mr-2" />
            <span>Confirm Deletion</span>
          </Flex>
        </ModalHeader>
        <ModalBody>
          <Text className="text-gray-600">
            Are you sure you want to delete the assistant <span className="font-medium">"{assistantName}"</span>? 
            This action cannot be undone.
          </Text>
        </ModalBody>
        <ModalFooter>
          <Button
            color="default"
            variant="light"
            onClick={onClose}
            isDisabled={isDeleting}
          >
            Cancel
          </Button>
          <Button
            color="danger"
            onClick={onConfirm}
            isLoading={isDeleting}
          >
            {isDeleting ? 'Deleting...' : 'Delete'}
          </Button>
        </ModalFooter>
      </ModalContent>
    </Modal>
  )
}