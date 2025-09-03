import { Card, CardBody, Avatar } from '@heroui/react'
import { Flex, Text } from '@radix-ui/themes'
import { StarIcon } from '@radix-ui/react-icons'

interface TestimonialCardProps {
  quote: string
  authorName: string
  authorRole: string
  authorImage: string
  rating?: number
  className?: string
}

export const TestimonialCard = ({
  quote,
  authorName,
  authorRole,
  authorImage,
  rating = 5,
  className
}: TestimonialCardProps) => {
  return (
    <Card className={`p-6 bg-white border-0 shadow-sm ${className || ''}`}>
      <CardBody>
        <Flex align="center" gap="2" className="mb-4">
          {Array.from({ length: rating }, (_, index) => (
            <StarIcon key={index} className="w-4 h-4 text-yellow-400 fill-current" />
          ))}
        </Flex>
        <Text className="text-gray-700 mb-4 leading-relaxed">
          "{quote}"
        </Text>
        <Flex align="center" gap="3">
          <Avatar
            src={authorImage}
            className="w-10 h-10"
          />
          <div>
            <Text weight="medium" className="text-gray-900">{authorName}</Text>
            <Text size="2" className="text-gray-500">{authorRole}</Text>
          </div>
        </Flex>
      </CardBody>
    </Card>
  )
}
