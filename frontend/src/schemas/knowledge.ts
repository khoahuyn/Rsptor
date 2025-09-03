import { z } from 'zod'

// Zod validation schema for creating knowledge base
export const createKnowledgeSchema = z.object({
  name: z
    .string()
    .min(1, 'Name is required')
    .min(3, 'Name must be at least 3 characters')
    .max(50, 'Name must be less than 50 characters'),
  description: z
    .string()
    .max(500, 'Description must be less than 500 characters')
})

export type CreateKnowledgeFormData = z.infer<typeof createKnowledgeSchema>
