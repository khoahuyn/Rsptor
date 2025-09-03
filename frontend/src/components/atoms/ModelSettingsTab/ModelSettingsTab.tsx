import { Select, SelectItem } from '@heroui/react'
import { Text } from '@radix-ui/themes'
import { motion } from 'framer-motion'
import { ModelParameterSlider } from '../ModelParameterSlider'

interface ModelSettings {
  model: string
  temperature: number
  topP: number
  presencePenalty: number
  frequencyPenalty: number
}

interface ModelSettingsTabProps {
  modelSettings: ModelSettings
  availableModels: Array<{ key: string; label: string }>
  onModelSettingChange: (key: keyof ModelSettings, value: string | number) => void
}

const tabVariants = {
  hidden: { opacity: 0, x: 20 },
  visible: { opacity: 1, x: 0, transition: { duration: 0.3 } }
}

export const ModelSettingsTab = ({
  modelSettings,
  availableModels,
  onModelSettingChange
}: ModelSettingsTabProps) => {
  return (
    <motion.div
      variants={tabVariants}
      initial="hidden"
      animate="visible"
      className="space-y-6 py-4"
    >
      {/* Model Selection */}
      <div>
        <Text className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
          Model
        </Text>
        <Select
          selectedKeys={[modelSettings.model]}
          onSelectionChange={(keys) => {
            const selectedKey = Array.from(keys)[0] as string
            onModelSettingChange('model', selectedKey)
          }}
          variant="bordered"
          placeholder="Select a model"
        >
          {availableModels.map((model) => (
            <SelectItem key={model.key}>
              {model.label}
            </SelectItem>
          ))}
        </Select>
      </div>

      {/* Model Parameters */}
      <ModelParameterSlider
        label="Temperature"
        value={modelSettings.temperature}
        minValue={0}
        maxValue={2}
        step={0.1}
        description="Controls randomness: lower values for more focused responses, higher values for more creative responses"
        onChange={(value) => onModelSettingChange('temperature', value)}
      />

      <ModelParameterSlider
        label="Top P"
        value={modelSettings.topP}
        minValue={0.1}
        maxValue={1}
        step={0.1}
        description="Controls diversity via nucleus sampling: lower values for more focused responses"
        onChange={(value) => onModelSettingChange('topP', value)}
      />

      <ModelParameterSlider
        label="Presence Penalty"
        value={modelSettings.presencePenalty}
        minValue={-2}
        maxValue={2}
        step={0.1}
        description="Penalizes new tokens based on their existing frequency in the text"
        onChange={(value) => onModelSettingChange('presencePenalty', value)}
      />

      <ModelParameterSlider
        label="Frequency Penalty"
        value={modelSettings.frequencyPenalty}
        minValue={-2}
        maxValue={2}
        step={0.1}
        description="Penalizes new tokens based on their frequency in the training data"
        onChange={(value) => onModelSettingChange('frequencyPenalty', value)}
      />
    </motion.div>
  )
}
