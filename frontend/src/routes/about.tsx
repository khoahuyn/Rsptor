import { createFileRoute } from '@tanstack/react-router'
import { LandingPageTemplate } from '../components/templates'
import { Container, Section, Heading, Text } from '@radix-ui/themes'

function AboutPage() {
  return (
    <LandingPageTemplate>
      <Section className="py-20">
        <Container size="4">
          <div className="text-center max-w-4xl mx-auto">
            <Heading size="8" className="text-gray-900 mb-6">
              About Raptor Service
            </Heading>
            <Text size="5" className="text-gray-600 mb-8 leading-relaxed">
              Raptor Service is built on cutting-edge RAPTOR (Recursive Abstractive Processing
              for Tree-Organized Retrieval) technology that revolutionizes document processing
              and intelligent information retrieval.
            </Text>
            <Text size="4" className="text-gray-600 leading-relaxed">
              Our mission is to transform how organizations interact with their documents,
              making information more accessible, searchable, and actionable through advanced AI.
            </Text>
          </div>
        </Container>
      </Section>
    </LandingPageTemplate>
  )
}

export const Route = createFileRoute('/about')({
  component: AboutPage,
})
