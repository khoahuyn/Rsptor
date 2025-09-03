import { Button } from '@heroui/react'
import { Flex } from '@radix-ui/themes'
import { Logo } from '../../atoms'
import { Navigation } from '../../molecules'

interface HeaderProps {
  className?: string
}

export const Header = ({ className }: HeaderProps) => {
  return (
    <header className={`sticky top-0 z-50 backdrop-blur-sm bg-white/80 dark:bg-gray-900/80 border-b border-gray-200 dark:border-gray-700 ${className || ''}`}>
      <div className="px-12 py-4">
        <Flex align="center" justify="between">
          <Logo />

          <Flex align="center" gap="4">
            <Navigation />
            <Button color="primary" variant="solid" className="font-medium">
              Get Started
            </Button>
          </Flex>
        </Flex>
      </div>
    </header>
  )
}
