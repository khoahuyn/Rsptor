import { Flex } from '@radix-ui/themes'
import { motion } from 'framer-motion'

const cardVariants = {
  hidden: { opacity: 0, y: 20 },
  visible: { opacity: 1, y: 0, transition: { duration: 0.3 } }
}

export const AssistantItemSkeleton = () => {
  return (
    <motion.div
      variants={cardVariants}
      initial="hidden"
      animate="visible"
      className="p-4 rounded-lg border bg-gray-50 border-gray-200"
    >
      <Flex align="center" justify="between" className="mb-2">
        <Flex align="center" gap="3">
          <div className="w-8 h-8 rounded-full bg-gray-200 animate-pulse" />
          <div className="flex-1 min-w-0">
            <div className="h-4 bg-gray-200 rounded w-3/4 animate-pulse"></div>
          </div>
        </Flex>
        <div className="w-8 h-8 rounded bg-gray-200 animate-pulse" />
      </Flex>

      {/* Description */}
      <div className="h-3 bg-gray-200 rounded w-full animate-pulse mb-3"></div>

      <div className="mb-2">
        <div className="h-3 bg-gray-200 rounded w-1/3 animate-pulse mb-2"></div>
        <div className="flex gap-1">
          <div className="w-12 h-5 bg-gray-200 rounded animate-pulse"></div>
          <div className="w-12 h-5 bg-gray-200 rounded animate-pulse"></div>
        </div>
      </div>

      <div className="h-3 bg-gray-200 rounded w-2/3 animate-pulse mb-1"></div>
      <div className="h-3 bg-gray-200 rounded w-1/3 animate-pulse"></div>
    </motion.div>
  )
}