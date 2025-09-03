import { Button } from '@heroui/react'
import { Container, Text, Heading, Flex, Section } from '@radix-ui/themes'

interface CTASectionProps {
  className?: string
}

export const CTASection = ({ className }: CTASectionProps) => {
  return (
    <Section className={`py-20 bg-gradient-to-r from-indigo-600 to-purple-700 ${className || ''}`}>
      <Container size="4">
        <div className="text-center text-white">
          <Heading size="7" className="mb-4">
            Ready to Get Started?
          </Heading>
          <Text size="4" className="mb-8 opacity-90 max-w-2xl mx-auto">
            Join thousands of teams already using Raptor Service to revolutionize their document workflows.
          </Text>
          <Flex align="center" justify="center" gap="4">
            <Button
              size="lg"
              color="default"
              variant="solid"
              className="bg-white text-indigo-600 hover:bg-gray-50 font-semibold px-8 py-3"
            >
              Start Free Trial
            </Button>
            <Button
              size="lg"
              variant="bordered"
              className="border-white text-white hover:bg-white/10 font-semibold px-8 py-3"
            >
              Contact Sales
            </Button>
          </Flex>
        </div>
      </Container>
    </Section>
  )
}
