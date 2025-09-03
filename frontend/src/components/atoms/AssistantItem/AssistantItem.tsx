import { Button, Avatar, Chip } from '@heroui/react'
import { Text, Flex } from '@radix-ui/themes'
import { TrashIcon, GearIcon } from '@radix-ui/react-icons'
import { motion } from 'framer-motion'
import type { Assistant } from '../../../services/assistantService'

interface AssistantItemProps {
  assistant: Assistant
  isSelected: boolean
  onSelect: (assistant: Assistant) => void
  onDelete: (id: string) => void
}

const cardVariants = {
  hidden: { opacity: 0, y: 20 },
  visible: { opacity: 1, y: 0, transition: { duration: 0.3 } }
}

export const AssistantItem = ({ 
  assistant, 
  isSelected, 
  onSelect, 
  onDelete 
}: AssistantItemProps) => {
  const getKnowledgeBaseNames = (knowledgeBaseIds: string[]) => {
    return knowledgeBaseIds.length > 0 ? `${knowledgeBaseIds.length} knowledge base(s)` : 'No knowledge bases'
  }

  return (
    <motion.div
      variants={cardVariants}
      initial="hidden"
      animate="visible"
      className={`p-4 rounded-lg border cursor-pointer hover:shadow-sm transition-all ${
        isSelected
          ? 'border-indigo-500 bg-indigo-50'
          : 'bg-gray-50 border-gray-200 hover:border-gray-300'
      }`}
      onClick={() => onSelect(assistant)}
    >
      <Flex align="center" justify="between" className="mb-2">
        <Flex align="center" gap="3">
          <Avatar
            size="sm"
            className="bg-indigo-600 text-white flex-shrink-0"
            fallback={assistant.name.charAt(0).toUpperCase()}
          />
          <div className="flex-1 min-w-0">
            <Text className="text-sm font-medium text-gray-900 truncate">
              {assistant.name}
            </Text>
          </div>
        </Flex>
        <Button
          size="sm"
          variant="light"
          color="danger"
          isIconOnly
          onClick={(e) => {
            e.stopPropagation()
            onDelete(assistant.assistant_id)
          }}
        >
          <TrashIcon className="w-4 h-4" />
        </Button>
      </Flex>

      {/* Description */}
      <Text className="text-xs text-gray-500 mb-3 block">
        {assistant.description || 'No description'}
      </Text>

      {/* Model Settings Display */}
      <div className="mb-2">
        <Flex align="center" gap="2" className="mb-1">
          <GearIcon className="w-3 h-3 text-gray-400" />
          <Text className="text-xs text-gray-600">
            Model: {assistant.model_settings.model}
          </Text>
        </Flex>
        <Flex gap="1" wrap="wrap">
          <Chip size="sm" variant="flat" className="text-xs">
            T: {assistant.model_settings.temperature}
          </Chip>
          <Chip size="sm" variant="flat" className="text-xs">
            P: {assistant.model_settings.topP}
          </Chip>
        </Flex>
      </div>

      <Text className="text-xs text-gray-500 truncate mb-1 pr-3">
        Knowledge: {getKnowledgeBaseNames(assistant.knowledge_bases)}
      </Text>
      <Text className="text-xs text-gray-400">
        Created: {new Date(assistant.created_at).toLocaleDateString()}
      </Text>
    </motion.div>
  )
}