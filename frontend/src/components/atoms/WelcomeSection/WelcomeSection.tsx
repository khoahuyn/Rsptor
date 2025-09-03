import { Heading, Text } from '@radix-ui/themes'
import { motion } from 'framer-motion'

interface WelcomeSectionProps {
  userName: string
  className?: string
}

export const WelcomeSection = ({ userName, className }: WelcomeSectionProps) => {
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
    <motion.div className={`mb-8 ${className || ''}`} variants={itemVariants}>
      <Heading as="h3" size="9" className="text-gray-900 mb-2 text-2xl lg:text-5xl">
        Welcome back, <span className="username-emphasis">{userName}</span>!
      </Heading>
      <Text size="5" className="text-gray-600">
        Manage your knowledge bases and explore your documents
      </Text>
    </motion.div>
  )
}
