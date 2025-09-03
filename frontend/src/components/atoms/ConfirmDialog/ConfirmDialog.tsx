import { Modal, ModalContent, ModalHeader, ModalBody, ModalFooter } from '@heroui/react'
import { Text } from '@radix-ui/themes'
import { ModalActions } from '../../atoms/ModalActions'

interface ConfirmDialogProps {
  isOpen: boolean
  onClose: () => void
  onConfirm: () => void
  title: string
  message: string
  confirmText?: string
  cancelText?: string
  isLoading?: boolean
  confirmColor?: 'primary' | 'secondary' | 'success' | 'warning' | 'danger'
}

export const ConfirmDialog = ({
  isOpen,
  onClose,
  onConfirm,
  title,
  message,
  confirmText = 'Confirm',
  cancelText = 'Cancel',
  isLoading = false,
  confirmColor = 'danger'
}: ConfirmDialogProps) => {
  const handleConfirm = () => {
    onConfirm()
  }

  return (
    <Modal
      isOpen={isOpen}
      onClose={onClose}
      size="sm"
      placement="center"
      backdrop="blur"
      classNames={{
        closeButton: "text-xl p-2 m-2"
      }}
    >
      <ModalContent>
        <ModalHeader className="flex flex-col gap-1">
          <h2 className="text-lg font-bold text-gray-900">{title}</h2>
        </ModalHeader>

        <ModalBody className="py-4">
          <Text className="text-gray-600">
            {message}
          </Text>
        </ModalBody>

        <ModalFooter>
          <ModalActions
            onCancel={onClose}
            onConfirm={handleConfirm}
            cancelText={cancelText}
            confirmText={confirmText}
            isLoading={isLoading}
            disabled={isLoading}
            confirmColor={confirmColor}
          />
        </ModalFooter>
      </ModalContent>
    </Modal>
  )
}