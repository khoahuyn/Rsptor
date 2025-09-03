import { MagnifyingGlassIcon, LightningBoltIcon, ChatBubbleIcon } from '@radix-ui/react-icons'

export const featuresData = {
  title: "Powerful Features",
  subtitle: "Everything you need to build intelligent document processing workflows",
  features: [
    {
      icon: MagnifyingGlassIcon,
      title: "Smart Search",
      description: "Advanced semantic search powered by AI embeddings. Find exactly what you're looking for.",
      color: "blue" as const
    },
    {
      icon: LightningBoltIcon,
      title: "Lightning Fast",
      description: "Optimized processing pipeline delivers results in seconds, not minutes.",
      color: "green" as const
    },
    {
      icon: ChatBubbleIcon,
      title: "AI Chat",
      description: "Ask questions about your documents in natural language and get precise answers.",
      color: "purple" as const
    }
  ]
}
