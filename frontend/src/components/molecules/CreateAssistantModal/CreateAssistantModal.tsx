import { Button, Modal, ModalContent, ModalHeader, ModalBody, ModalFooter, Tabs, Tab } from '@heroui/react'
import { Flex } from '@radix-ui/themes'
import { PersonIcon } from '@radix-ui/react-icons'
import { useState, useEffect } from 'react'
import { AssistantSettingsTab, ModelSettingsTab } from '../../atoms'
import { useDatasets, useToast } from '../../../hooks'
import { assistantService, type AssistantData, type Assistant } from '../../../services/assistantService'

interface ModelSettings {
  model: string
  temperature: number
  topP: number
  presencePenalty: number
  frequencyPenalty: number
}

interface CreateAssistantModalProps {
  isOpen: boolean
  onClose: () => void
  onAssistantCreated?: (assistant: Assistant) => void
  className?: string
}

export const CreateAssistantModal = ({
  isOpen,
  onClose,
  onAssistantCreated,
  className
}: CreateAssistantModalProps) => {
  const [activeTab, setActiveTab] = useState('assistant')
  const [assistantName, setAssistantName] = useState('')
  const [assistantDescription, setAssistantDescription] = useState('')
  const [selectedKnowledge, setSelectedKnowledge] = useState<string>('')
  const [isSubmitting, setIsSubmitting] = useState(false)

  const { datasets, loading: datasetsLoading } = useDatasets()
  const { success: showSuccessToast, error: showErrorToast } = useToast()

  // Model settings with defaults
  const [modelSettings, setModelSettings] = useState<ModelSettings>({
    model: 'DeepSeek-V3',
    temperature: 0.7,
    topP: 1.0,
    presencePenalty: 0.0,
    frequencyPenalty: 0.0
  })

  // Convert datasets to knowledge base format for the AssistantSettingsTab
  const knowledgeBases = datasets.map(dataset => ({
    id: dataset.id,
    name: dataset.name,
    docs: dataset.document_count || 0
  }))

  // Available models aligned with backend
  const availableModels = [
    { key: 'DeepSeek-V3', label: 'DeepSeek-V3 (Recommended)' },
    { key: 'GPT-4o-mini', label: 'GPT-4o Mini' },
    { key: 'Gemini-2.5-Flash', label: 'Gemini 2.5 Flash' },
    { key: 'Claude-3.5-Haiku', label: 'Claude 3.5 Haiku' },
    { key: 'gpt-4o', label: 'GPT-4o (Latest)' },
    { key: 'gpt-4-turbo', label: 'GPT-4 Turbo' }
  ]

  // Reset form when modal closes
  useEffect(() => {
    if (!isOpen) {
      setActiveTab('assistant')
      setAssistantName('')
      setAssistantDescription('')
      setSelectedKnowledge('')
      setModelSettings({
        model: 'DeepSeek-V3',
        temperature: 0.7,
        topP: 1.0,
        presencePenalty: 0.0,
        frequencyPenalty: 0.0
      })
    }
  }, [isOpen])

  const handleCreateAssistant = async () => {
    if (!assistantName.trim() || !selectedKnowledge || isSubmitting) return

    setIsSubmitting(true)

    try {
      const assistantData: AssistantData = {
        name: assistantName.trim(),
        description: assistantDescription.trim() || undefined,
        knowledge_bases: [selectedKnowledge],
        model_settings: {
          model: modelSettings.model,
          temperature: modelSettings.temperature,
          topP: modelSettings.topP,
          presencePenalty: modelSettings.presencePenalty,
          frequencyPenalty: modelSettings.frequencyPenalty
        }
      }

      const newAssistant = await assistantService.createAssistant(assistantData)
      
      showSuccessToast('Assistant created successfully!')
      onAssistantCreated?.(newAssistant)
      onClose()
    } catch (error) {
      console.error('Failed to create assistant:', error)
      showErrorToast(
        error instanceof Error ? error.message : 'Failed to create assistant'
      )
    } finally {
      setIsSubmitting(false)
    }
  }

  const updateModelSetting = (key: keyof ModelSettings, value: string | number) => {
    setModelSettings(prev => ({ ...prev, [key]: value }))
  }

  return (
    <Modal
      isOpen={isOpen}
      onClose={onClose}
      size="3xl"
      className={className}
    >
      <ModalContent>
        <ModalHeader className="flex flex-col gap-1">
          <Flex align="center" gap="2">
            <PersonIcon className="w-5 h-5 text-indigo-600" />
            <span>Create New Assistant</span>
          </Flex>
        </ModalHeader>
        <ModalBody>
          <Tabs
            selectedKey={activeTab}
            onSelectionChange={(key) => setActiveTab(key as string)}
            className="w-full"
          >
            <Tab key="assistant" title="Assistant Settings">
              <AssistantSettingsTab
                assistantName={assistantName}
                assistantDescription={assistantDescription}
                selectedKnowledge={selectedKnowledge}
                knowledgeBases={knowledgeBases}
                onNameChange={setAssistantName}
                onDescriptionChange={setAssistantDescription}
                onKnowledgeChange={setSelectedKnowledge}
              />
            </Tab>

            <Tab key="model" title="Model Settings">
              <ModelSettingsTab
                modelSettings={modelSettings}
                availableModels={availableModels}
                onModelSettingChange={updateModelSetting}
              />
            </Tab>
          </Tabs>
        </ModalBody>
        <ModalFooter>
          <Button
            color="danger"
            variant="light"
            onClick={onClose}
          >
            Cancel
          </Button>
          <Button
            color="primary"
            onClick={handleCreateAssistant}
            isDisabled={!assistantName.trim() || !selectedKnowledge || isSubmitting || datasetsLoading}
            isLoading={isSubmitting}
          >
            {isSubmitting ? 'Creating...' : 'Create Assistant'}
          </Button>
        </ModalFooter>
      </ModalContent>
    </Modal>
  )
}
