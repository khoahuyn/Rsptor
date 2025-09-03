import { Container, Text, Flex, Separator } from '@radix-ui/themes'
import { Logo } from '../../atoms'

interface FooterProps {
  className?: string
}

const footerLinks = {
  product: [
    { label: 'Features', href: '#features' },
    { label: 'Pricing', href: '#pricing' },
    { label: 'API', href: '#api' }
  ],
  company: [
    { label: 'About', href: '#about' },
    { label: 'Blog', href: '#blog' },
    { label: 'Careers', href: '#careers' }
  ],
  support: [
    { label: 'Documentation', href: '#docs' },
    { label: 'Help Center', href: '#help' },
    { label: 'Contact', href: '#contact' }
  ]
}

export const Footer = ({ className }: FooterProps) => {
  return (
    <footer className={`bg-gray-900 text-white py-12 ${className || ''}`}>
      <Container size="4">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-8">
          <div>
            <Logo showText={true} className="mb-4" />
            <Text size="2" className="text-gray-400 leading-relaxed">
              Intelligent document processing powered by advanced AI technology.
            </Text>
          </div>

          <div>
            <Text weight="medium" className="mb-3">Product</Text>
            <div className="space-y-2">
              {footerLinks.product.map((link) => (
                <a key={link.label} href={link.href}>
                  <Text size="2" className="text-gray-400 hover:text-white cursor-pointer block">
                    {link.label}
                  </Text>
                </a>
              ))}
            </div>
          </div>

          <div>
            <Text weight="medium" className="mb-3">Company</Text>
            <div className="space-y-2">
              {footerLinks.company.map((link) => (
                <a key={link.label} href={link.href}>
                  <Text size="2" className="text-gray-400 hover:text-white cursor-pointer block">
                    {link.label}
                  </Text>
                </a>
              ))}
            </div>
          </div>

          <div>
            <Text weight="medium" className="mb-3">Support</Text>
            <div className="space-y-2">
              {footerLinks.support.map((link) => (
                <a key={link.label} href={link.href}>
                  <Text size="2" className="text-gray-400 hover:text-white cursor-pointer block">
                    {link.label}
                  </Text>
                </a>
              ))}
            </div>
          </div>
        </div>

        <Separator className="my-8 bg-gray-700" />

        <Flex align="center" justify="between" className="text-gray-400">
          <Text size="2">
            © 2024 Raptor Service. All rights reserved.
          </Text>
          <Flex align="center" gap="6">
            <Text size="2" className="hover:text-white cursor-pointer">Privacy</Text>
            <Text size="2" className="hover:text-white cursor-pointer">Terms</Text>
          </Flex>
        </Flex>
      </Container>
    </footer>
  )
}
