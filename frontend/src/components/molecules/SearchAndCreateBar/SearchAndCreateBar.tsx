import { Button, Input } from '@heroui/react'
import { MagnifyingGlassIcon, PlusIcon } from '@radix-ui/react-icons'
import { motion } from 'framer-motion'

interface SearchAndCreateBarProps {
  searchQuery: string
  onSearchChange: (value: string) => void
  onCreateClick: () => void
  className?: string
}

export const SearchAndCreateBar = ({
  searchQuery,
  onSearchChange,
  onCreateClick,
  className
}: SearchAndCreateBarProps) => {
  const itemVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        duration: 0.5
      }
    }
  }

  return (
    <motion.div
      className={`flex flex-col sm:flex-row gap-4 mb-8 ${className || ''}`}
      variants={itemVariants}
    >
      <div className="flex-1">
        <motion.div
          whileHover={{ scale: 1.02 }}
          whileFocus={{ scale: 1.02 }}
          transition={{ duration: 0.2 }}
        >
          <Input
            placeholder="Search knowledge bases..."
            value={searchQuery}
            onChange={(e) => onSearchChange(e.target.value)}
            startContent={<MagnifyingGlassIcon className="w-4 h-4 text-gray-400" />}
            className="w-full"
            size="lg"
          />
        </motion.div>
      </div>
      <motion.div
        whileHover={{ scale: 1.05 }}
        whileTap={{ scale: 0.95 }}
        transition={{ duration: 0.2 }}
      >
        <Button
          color="primary"
          size="lg"
          startContent={<PlusIcon className="w-4 h-4" />}
          className="sm:w-auto w-full"
          onPress={onCreateClick}
        >
          Create Knowledge Base
        </Button>
      </motion.div>
    </motion.div>
  )
}
