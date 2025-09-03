import { Card, CardBody, Button } from '@heroui/react'
import { Heading, Text, Flex } from '@radix-ui/themes'
import { PlusIcon, PersonIcon } from '@radix-ui/react-icons'
import { useState } from 'react'
import { CreateAssistantModal } from '../../molecules'
import { useAssistants, useToast } from '../../../hooks'
import type { Assistant } from '../../../services/assistantService'
import { AssistantItem } from '../../atoms/AssistantItem/AssistantItem'
import { AssistantItemSkeleton } from '../../atoms/AssistantItemSkeleton/AssistantItemSkeleton'
import { DeleteAssistantModal } from '../../molecules/DeleteAssistantModal/DeleteAssistantModal'

interface AssistantCreationProps {
  className?: string
  onAssistantSelect?: (assistant: Assistant) => void
}

export const AssistantCreation = ({ className, onAssistantSelect }: AssistantCreationProps) => {
  const [isModalOpen, setIsModalOpen] = useState(false)
  const [selectedAssistant, setSelectedAssistant] = useState<Assistant | null>(null)
  const [assistantToDelete, setAssistantToDelete] = useState<Assistant | null>(null)
  const [isDeleteModalOpen, setIsDeleteModalOpen] = useState(false)
  const [isDeleting, setIsDeleting] = useState(false)

  const { 
    assistants, 
    loading, 
    error, 
    deleteAssistant,
    refetch
  } = useAssistants()
  const { success: showSuccessToast, error: showErrorToast } = useToast()

  const handleDeleteAssistant = async (id: string) => {
    try {
      setIsDeleting(true)
      await deleteAssistant(id)
      showSuccessToast('Assistant deleted successfully')
      await refetch()
      
      // Clear selection if the deleted assistant was selected
      if (selectedAssistant?.assistant_id === id) {
        setSelectedAssistant(null)
      }
    } catch (error) {
      console.error('Failed to delete assistant:', error)
      showErrorToast('Failed to delete assistant')
    } finally {
      setIsDeleting(false)
      setIsDeleteModalOpen(false)
      setAssistantToDelete(null)
    }
  }

  const handleAssistantSelect = (assistant: Assistant) => {
    setSelectedAssistant(assistant)
    onAssistantSelect?.(assistant)
  }

  const handleAssistantCreated = (newAssistant: Assistant) => {
    showSuccessToast('Assistant created successfully!')
    refetch()
    if (onAssistantSelect) {
      onAssistantSelect(newAssistant)
    }
  }

  const openDeleteModal = (assistant: Assistant) => {
    setAssistantToDelete(assistant)
    setIsDeleteModalOpen(true)
  }

  const closeDeleteModal = () => {
    setIsDeleteModalOpen(false)
    setAssistantToDelete(null)
  }

  return (
    <div className={`h-full ${className || ''}`}>
      <Card className="h-full border border-gray-200 shadow-sm">
        <CardBody className="p-6">
          <Flex direction="column" gap="4" className="h-full">
            {/* Header */}
            <div>
              <Flex align="center" gap="2" className="mb-2">
                <PersonIcon className="w-5 h-5 text-indigo-600" />
                <Heading size="4" className="text-gray-900 font-bold">
                  AI Assistants
                </Heading>
              </Flex>
              <Text className="text-gray-600 text-sm">
                Create and manage your AI assistants
              </Text>
            </div>

            {/* Create Assistant Button */}
            <Button
              color="primary"
              variant="solid"
              size="md"
              startContent={<PlusIcon className="w-4 h-4" />}
              className="w-full font-medium"
              onClick={() => setIsModalOpen(true)}
            >
              Create Assistant
            </Button>

            {/* Assistants List */}
            <div className="flex-1 overflow-y-auto">
              {loading ? (
                <div className="space-y-3">
                  <Text className="text-sm font-medium text-gray-700 mb-3">
                    Your Assistants
                  </Text>
                  {/* Skeleton loaders */}
                  {[...Array(3)].map((_, index) => (
                    <AssistantItemSkeleton key={index} />
                  ))}
                </div>
              ) : error ? (
                <div className="flex flex-col items-center justify-center h-32 text-center">
                  <PersonIcon className="w-12 h-12 text-red-300 mb-2" />
                  <Text className="text-red-500 text-sm">
                    {error}
                  </Text>
                </div>
              ) : assistants.length === 0 ? (
                <div className="flex flex-col items-center justify-center h-32 text-center">
                  <PersonIcon className="w-12 h-12 text-gray-300 mb-2" />
                  <Text className="text-gray-500 text-sm">
                    No assistants created yet
                  </Text>
                  <Text className="text-gray-400 text-xs">
                    Click "Create Assistant" to get started
                  </Text>
                </div>
              ) : (
                <div className="space-y-3">
                  <Text className="text-sm font-medium text-gray-700 mb-3">
                    Your Assistants ({assistants.length})
                  </Text>
                  {assistants.map((assistant) => (
                    <AssistantItem
                      key={assistant.assistant_id}
                      assistant={assistant}
                      isSelected={selectedAssistant?.assistant_id === assistant.assistant_id}
                      onSelect={handleAssistantSelect}
                      onDelete={() => openDeleteModal(assistant)}
                    />
                  ))}
                </div>
              )}
            </div>
          </Flex>
        </CardBody>
      </Card>

      {/* Create Assistant Modal */}
      <CreateAssistantModal
        isOpen={isModalOpen}
        onClose={() => setIsModalOpen(false)}
        onAssistantCreated={handleAssistantCreated}
      />
      
      {/* Delete Assistant Confirmation Modal */}
      <DeleteAssistantModal
        isOpen={isDeleteModalOpen}
        onClose={closeDeleteModal}
        onConfirm={() => assistantToDelete && handleDeleteAssistant(assistantToDelete.assistant_id)}
        assistantName={assistantToDelete?.name || ''}
        isDeleting={isDeleting}
      />
    </div>
  )
}