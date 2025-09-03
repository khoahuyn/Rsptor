import { Card, CardBody } from '@heroui/react'
import { Flex } from '@radix-ui/themes'

interface KnowledgeCardSkeletonProps {
  className?: string
}

export const KnowledgeCardSkeleton = ({ className }: KnowledgeCardSkeletonProps) => {
  return (
    <Card
      className={`px-6 py-6 h-[280px] border border-gray-200 ${className || ''}`}
    >
      <CardBody className="flex flex-col justify-between h-full">
        {/* Title skeleton */}
        <div className="h-6 bg-gray-200 rounded-md mb-4 w-3/4 animate-pulse"></div>

        {/* Description skeleton */}
        <div className="flex-grow mb-6 space-y-2">
          <div className="h-4 bg-gray-200 rounded-md w-full animate-pulse"></div>
          <div className="h-4 bg-gray-200 rounded-md w-5/6 animate-pulse"></div>
          <div className="h-4 bg-gray-200 rounded-md w-4/6 animate-pulse"></div>
        </div>

        {/* Footer info skeleton */}
        <Flex direction="column" gap="2">
          <Flex align="center" gap="2">
            <div className="w-4 h-4 bg-gray-200 rounded animate-pulse"></div>
            <div className="h-4 bg-gray-200 rounded-md w-24 animate-pulse"></div>
          </Flex>
          <Flex align="center" gap="2">
            <div className="w-4 h-4 bg-gray-200 rounded animate-pulse"></div>
            <div className="h-4 bg-gray-200 rounded-md w-32 animate-pulse"></div>
          </Flex>
        </Flex>
      </CardBody>
    </Card>
  )
}