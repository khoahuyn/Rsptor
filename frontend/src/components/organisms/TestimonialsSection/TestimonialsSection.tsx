import { Container, Text, Heading, Section } from '@radix-ui/themes'
import { TestimonialCard } from '../../molecules'
import { testimonialsData } from '../../../constants/testimonialsData'

interface TestimonialsSectionProps {
  className?: string
}

export const TestimonialsSection = ({ className }: TestimonialsSectionProps) => {
  return (
    <Section className={`py-20 bg-gray-50 ${className || ''}`}>
      <Container size="4">
        <div className="text-center mb-16">
          <Heading size="7" className="text-gray-900 mb-4">
            {testimonialsData.title}
          </Heading>
          <Text size="4" className="text-gray-600">
            {testimonialsData.subtitle}
          </Text>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
          {testimonialsData.testimonials.map((testimonial, index) => (
            <TestimonialCard
              key={index}
              quote={testimonial.quote}
              authorName={testimonial.authorName}
              authorRole={testimonial.authorRole}
              authorImage={testimonial.authorImage}
              rating={testimonial.rating}
            />
          ))}
        </div>
      </Container>
    </Section>
  )
}
