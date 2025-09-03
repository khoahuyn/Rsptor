import { Button } from '@heroui/react'

export interface ModalActionsProps {
  onCancel?: () => void
  onConfirm?: () => void | Promise<void>
  cancelText?: string
  confirmText?: string
  isLoading?: boolean
  disabled?: boolean
  cancelDisabled?: boolean
  confirmDisabled?: boolean
  confirmColor?: 'primary' | 'secondary' | 'success' | 'warning' | 'danger'
}

export const ModalActions = ({
  onCancel,
  onConfirm,
  cancelText = 'Cancel',
  confirmText = 'Confirm',
  isLoading = false,
  disabled = false,
  cancelDisabled = false,
  confirmDisabled = false,
  confirmColor = 'primary'
}: ModalActionsProps) => {
  return (
    <>
      {onCancel && (
        <Button
          variant="ghost"
          onPress={onCancel}
          disabled={disabled || cancelDisabled}
          size="lg"
        >
          {cancelText}
        </Button>
      )}
      {onConfirm && (
        <Button
          color={confirmColor}
          onPress={onConfirm}
          disabled={disabled || confirmDisabled}
          isLoading={isLoading}
          size="lg"
        >
          {confirmText}
        </Button>
      )}
    </>
  )
}
