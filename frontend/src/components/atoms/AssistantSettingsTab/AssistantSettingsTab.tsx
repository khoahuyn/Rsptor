import { Input, Textarea, Select, SelectItem } from '@heroui/react'
import { Text } from '@radix-ui/themes'
import { motion } from 'framer-motion'

interface AssistantSettingsTabProps {
  assistantName: string
  assistantDescription: string
  selectedKnowledge: string
  knowledgeBases: Array<{ id: string; name: string; docs: number }>
  onNameChange: (value: string) => void
  onDescriptionChange: (value: string) => void
  onKnowledgeChange: (value: string) => void
}

const tabVariants = {
  hidden: { opacity: 0, x: 20 },
  visible: { opacity: 1, x: 0, transition: { duration: 0.3 } }
}

export const AssistantSettingsTab = ({
  assistantName,
  assistantDescription,
  selectedKnowledge,
  knowledgeBases,
  onNameChange,
  onDescriptionChange,
  onKnowledgeChange
}: AssistantSettingsTabProps) => {
  return (
    <motion.div
      variants={tabVariants}
      initial="hidden"
      animate="visible"
      className="space-y-4 py-4"
    >
      {/* Assistant Configuration */}
      <div>
        <Text className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
          Assistant Name *
        </Text>
        <Input
          placeholder="Enter assistant name"
          value={assistantName}
          onChange={(e) => onNameChange(e.target.value)}
          variant="bordered"
          size="md"
        />
      </div>

      <div>
        <Text className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
          Description
        </Text>
        <Textarea
          placeholder="Describe your assistant's purpose..."
          value={assistantDescription}
          onChange={(e) => onDescriptionChange(e.target.value)}
          variant="bordered"
          size="md"
          minRows={3}
        />
      </div>

      {/* Knowledge Base Selection */}
      <div>
        <Text className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-3">
          Select Knowledge Base *
        </Text>
        <Select
          selectedKeys={selectedKnowledge ? new Set([selectedKnowledge]) : new Set()}
          onSelectionChange={(keys) => {
            const selectedKey = Array.from(keys)[0] as string
            onKnowledgeChange(selectedKey || '')
          }}
          variant="bordered"
          placeholder="Choose a knowledge base..."
          className="w-full"
          renderValue={(items) => {
            if (items.length === 0) return "Choose a knowledge base..."
            const selectedKb = knowledgeBases.find(kb => kb.id === Array.from(items)[0]?.key)
            return selectedKb ? (
              <div className="flex justify-between items-center w-full">
                <span>{selectedKb.name}</span>
                <span className="text-xs text-gray-500 dark:text-gray-400">{selectedKb.docs} docs</span>
              </div>
            ) : "Choose a knowledge base..."
          }}
        >
          {knowledgeBases.map((kb) => (
            <SelectItem key={kb.id}>
              <div className="flex justify-between items-center w-full">
                <span>{kb.name}</span>
                <span className="text-xs text-gray-500 dark:text-gray-400">{kb.docs} docs</span>
              </div>
            </SelectItem>
          ))}
        </Select>
      </div>
    </motion.div>
  )
}
