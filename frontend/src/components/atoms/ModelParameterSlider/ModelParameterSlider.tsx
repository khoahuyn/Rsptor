import { Slider } from '@heroui/react'
import { Text, Flex } from '@radix-ui/themes'

interface ModelParameterSliderProps {
  label: string
  value: number
  minValue: number
  maxValue: number
  step: number
  description: string
  onChange: (value: number) => void
}

export const ModelParameterSlider = ({
  label,
  value,
  minValue,
  maxValue,
  step,
  description,
  onChange
}: ModelParameterSliderProps) => {
  return (
    <div>
      <Flex align="center" justify="between" className="mb-3">
        <Text className="text-base font-semibold text-gray-800 dark:text-gray-200">
          {label}
        </Text>
        <Text className="text-sm text-gray-600 dark:text-gray-400 font-medium">
          {value}
        </Text>
      </Flex>
      <Slider
        size="sm"
        step={step}
        minValue={minValue}
        maxValue={maxValue}
        value={value}
        onChange={(val) => onChange(Array.isArray(val) ? val[0] : val)}
        className="max-w-full"
      />
      <Text className="text-xs text-gray-500 dark:text-gray-400 mt-2">
        {description}
      </Text>
    </div>
  )
}
