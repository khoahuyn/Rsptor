import { Card, CardBody } from '@heroui/react'
import { Heading, Text } from '@radix-ui/themes'
import { IconBox } from '../../atoms'
import type { ReactNode } from 'react'

interface FeatureCardProps {
  icon: ReactNode
  title: string
  description: string
  color?: 'blue' | 'green' | 'purple'
  className?: string
}

export const FeatureCard = ({ icon, title, description, color = 'blue', className }: FeatureCardProps) => {
  const gradientStyles = {
    blue: 'bg-gradient-to-br from-blue-50 to-indigo-50',
    green: 'bg-gradient-to-br from-green-50 to-emerald-50',
    purple: 'bg-gradient-to-br from-purple-50 to-violet-50'
  }

  return (
    <Card className={`p-6 hover:shadow-lg transition-shadow border-0 ${gradientStyles[color]} ${className || ''}`}>
      <CardBody>
        <IconBox color={color} className="mb-4">
          {icon}
        </IconBox>
        <Heading size="4" className="text-gray-900 mb-3">
          {title}
        </Heading>
        <Text className="text-gray-600 leading-relaxed">
          {description}
        </Text>
      </CardBody>
    </Card>
  )
}
