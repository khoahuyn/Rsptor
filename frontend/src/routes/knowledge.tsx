import { createFileRoute } from '@tanstack/react-router'
import { KnowledgePage } from '../pages'

export const Route = createFileRoute('/knowledge')({
  component: KnowledgePage,
})
