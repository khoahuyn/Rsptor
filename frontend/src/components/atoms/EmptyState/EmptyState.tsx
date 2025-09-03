import { Button } from '@heroui/react'
import { Text } from '@radix-ui/themes'
import { motion } from 'framer-motion'

interface EmptyStateProps {
  searchQuery: string
  onClearSearch: () => void
  className?: string
}

export const EmptyState = ({ searchQuery, onClearSearch, className }: EmptyStateProps) => {
  return (
    <motion.div
      className={`text-center py-12 ${className || ''}`}
      initial={{ opacity: 0, scale: 0.8 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.4, delay: 0.2 }}
    >
      <Text size="4" className="text-gray-500 mb-4">
        No knowledge bases found matching "{searchQuery}"
      </Text>
      <motion.div
        whileHover={{ scale: 1.05 }}
        whileTap={{ scale: 0.95 }}
      >
        <Button
          variant="ghost"
          onClick={onClearSearch}
        >
          Clear search
        </Button>
      </motion.div>
    </motion.div>
  )
}
