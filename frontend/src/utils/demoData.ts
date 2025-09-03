import { KnowledgeBaseService } from '../services'

// Demo data seeder - call this once to populate localStorage with sample data
export const seedDemoKnowledgeBases = () => {
  try {
    // Clear existing data first (optional)
    KnowledgeBaseService.clear()
    
    // Create some demo knowledge bases
    const demoKBs = [
      {
        name: 'Product Documentation',
        description: 'Comprehensive documentation for all product features, API references, and user guides.'
      },
      {
        name: 'Technical Specifications', 
        description: 'Detailed technical specifications, architecture diagrams, and system requirements.'
      },
      {
        name: 'User Manuals',
        description: 'Step-by-step user manuals and tutorials for different user roles and workflows.'
      },
      {
        name: 'Research Papers',
        description: 'Collection of research papers, whitepapers, and academic publications related to AI and ML.'
      }
    ]

    const createdKBs = demoKBs.map(kb => KnowledgeBaseService.create(kb))
    
    console.log('Demo knowledge bases created:', createdKBs)
    return createdKBs
  } catch (error) {
    console.error('Failed to seed demo knowledge bases:', error)
    return []
  }
}

// Get a specific demo dataset ID for testing
export const getDemoDatasetId = (): string | null => {
  const kbs = KnowledgeBaseService.getAll()
  return kbs.length > 0 ? kbs[0].id : null
}

// Check if demo data exists
export const hasDemoData = (): boolean => {
  return KnowledgeBaseService.getAll().length > 0
}