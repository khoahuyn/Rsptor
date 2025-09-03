import { Container, Text, Heading, Section } from '@radix-ui/themes'
import { FeatureCard } from '../../molecules'
import { featuresData } from '../../../constants/featuresData'

interface FeaturesSectionProps {
  className?: string
}

export const FeaturesSection = ({ className }: FeaturesSectionProps) => {
  return (
    <Section className={`py-20 bg-white ${className || ''}`}>
      <Container size="4">
        <div className="text-center mb-16">
          <Heading size="7" className="text-gray-900 mb-4">
            {featuresData.title}
          </Heading>
          <Text size="4" className="text-gray-600 max-w-2xl mx-auto">
            {featuresData.subtitle}
          </Text>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
          {featuresData.features.map((feature, index) => (
            <FeatureCard
              key={index}
              icon={<feature.icon className="w-6 h-6" />}
              title={feature.title}
              description={feature.description}
              color={feature.color}
            />
          ))}
        </div>
      </Container>
    </Section>
  )
}
