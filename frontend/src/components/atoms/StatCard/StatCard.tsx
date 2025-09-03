import { Text } from '@radix-ui/themes'

interface StatCardProps {
  value: string
  label: string
  className?: string
}

export const StatCard = ({ value, label, className }: StatCardProps) => {
  return (
    <div className={`text-center ${className || ''}`}>
      <Text size="6" weight="bold" className="text-indigo-600 block">
        {value}
      </Text>
      <Text className="text-gray-600">
        {label}
      </Text>
    </div>
  )
}
