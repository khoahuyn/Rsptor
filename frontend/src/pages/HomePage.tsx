import {
  HeroSection,
  FeaturesSection,
  TestimonialsSection,
  CTASection
} from '../components/organisms'
import { LandingPageTemplate } from '../components/templates'

export const HomePage = () => {
  return (
    <LandingPageTemplate>
      <HeroSection />
      <FeaturesSection />
      <TestimonialsSection />
      <CTASection />
    </LandingPageTemplate>
  )
}
