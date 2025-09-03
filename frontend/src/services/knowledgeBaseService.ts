import { UuidUtils } from '../utils'
import { DatasetService } from './datasetService'

export interface KnowledgeBase {
  id: string
  name: string
  description: string
  documentCount: number
  createdAt: string
  updatedAt: string
}

export interface CreateKnowledgeBaseData {
  name: string
  description?: string
}

const STORAGE_KEY = 'raptor_knowledge_bases'

export class KnowledgeBaseService {
  // Get all knowledge bases from localStorage
  static getAll(): KnowledgeBase[] {
    try {
      const stored = localStorage.getItem(STORAGE_KEY)
      return stored ? JSON.parse(stored) : []
    } catch (error) {
      console.error('Error reading knowledge bases from localStorage:', error)
      return []
    }
  }

  // Get a specific knowledge base by ID
  static getById(id: string): KnowledgeBase | null {
    const knowledgeBases = this.getAll()
    return knowledgeBases.find(kb => kb.id === id) || null
  }

  // Create a new knowledge base (LOCAL ONLY - for demo/offline use)
  static create(data: CreateKnowledgeBaseData): KnowledgeBase {
    const newKnowledgeBase: KnowledgeBase = {
      id: UuidUtils.generateKnowledgeBaseId(),
      name: data.name,
      description: data.description || '',
      documentCount: 0,
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString()
    }

    const knowledgeBases = this.getAll()
    knowledgeBases.push(newKnowledgeBase)
    
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(knowledgeBases))
      return newKnowledgeBase
    } catch (error) {
      console.error('Error saving knowledge base to localStorage:', error)
      throw new Error('Failed to save knowledge base')
    }
  }

  // Create KB via Backend API + Sync to localStorage  
  static async createKB(data: CreateKnowledgeBaseData): Promise<KnowledgeBase> {
    try {
      // Call backend API to create KB in database
      const response = await fetch('http://localhost:8081/v1/kb/create', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          tenant_id: 'test_tenant',
          name: data.name,
          description: data.description || '',
          settings: null
        })
      })

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }))
        throw new Error(`Backend error: ${errorData.detail || response.statusText}`)
      }

      const backendKB = await response.json()
      
      // Transform backend response to frontend format
      const frontendKB: KnowledgeBase = {
        id: backendKB.kb_id,
        name: backendKB.name,
        description: backendKB.description || '',
        documentCount: backendKB.document_count || 0,
        createdAt: backendKB.created_at,
        updatedAt: backendKB.updated_at || backendKB.created_at
      }

      // Save to localStorage for faster access
      const knowledgeBases = this.getAll()
      knowledgeBases.push(frontendKB)
      localStorage.setItem(STORAGE_KEY, JSON.stringify(knowledgeBases))

      console.log('[KnowledgeBaseService] KB created successfully:', {
        frontend_id: frontendKB.id,
        backend_kb_id: backendKB.kb_id,
        name: frontendKB.name
      })

      return frontendKB
    } catch (error) {
      console.error('[KnowledgeBaseService] Failed to create KB via backend:', error)
      throw error
    }
  }

  // Update an existing knowledge base
  static update(id: string, updates: Partial<Omit<KnowledgeBase, 'id' | 'createdAt'>>): KnowledgeBase | null {
    const knowledgeBases = this.getAll()
    const index = knowledgeBases.findIndex(kb => kb.id === id)
    
    if (index === -1) {
      return null
    }

    knowledgeBases[index] = {
      ...knowledgeBases[index],
      ...updates,
      updatedAt: new Date().toISOString()
    }

    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(knowledgeBases))
      return knowledgeBases[index]
    } catch (error) {
      console.error('Error updating knowledge base in localStorage:', error)
      throw new Error('Failed to update knowledge base')
    }
  }

  // Delete a knowledge base
  static delete(id: string): boolean {
    const knowledgeBases = this.getAll()
    const filteredKnowledgeBases = knowledgeBases.filter(kb => kb.id !== id)
    
    if (filteredKnowledgeBases.length === knowledgeBases.length) {
      return false // Knowledge base not found
    }

    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(filteredKnowledgeBases))
      return true
    } catch (error) {
      console.error('Error deleting knowledge base from localStorage:', error)
      throw new Error('Failed to delete knowledge base')
    }
  }

  // Increment document count when a document is uploaded
  static incrementDocumentCount(id: string): KnowledgeBase | null {
    const kb = this.getById(id)
    if (!kb) {
      return null
    }

    return this.update(id, {
      documentCount: kb.documentCount + 1
    })
  }

  // Check if a knowledge base exists
  static exists(id: string): boolean {
    return this.getById(id) !== null
  }

  // Sync localStorage with backend datasets (clear old data and save new)
  static async syncWithBackend(): Promise<KnowledgeBase[]> {
    try {
      // Fetch fresh data from backend
      const response = await DatasetService.getDatasets()
      const backendDatasets = response.datasets
      
      // Clear old localStorage data
      this.clear()
      
      // Transform backend datasets to KnowledgeBase format and save to localStorage
      const syncedKnowledgeBases: KnowledgeBase[] = backendDatasets.map(dataset => ({
        id: dataset.id,
        name: dataset.name || dataset.id,
        description: dataset.description || 'No description available',
        documentCount: dataset.document_count,
        createdAt: dataset.created_at,
        updatedAt: dataset.last_updated || dataset.created_at
      }))
      
      // Save to localStorage
      localStorage.setItem(STORAGE_KEY, JSON.stringify(syncedKnowledgeBases))
      
      return syncedKnowledgeBases
    } catch (error) {
      console.error('Error syncing with backend:', error)
      // Return current localStorage data as fallback
      return this.getAll()
    }
  }

  // Add new knowledge base to current localStorage list (don't clear existing data)
  static addToCurrentList(data: CreateKnowledgeBaseData): KnowledgeBase {
    return this.create(data)
  }

  // Update an existing knowledge base (alias for update method)
  static updateKnowledgeBase(id: string, updates: Partial<Omit<KnowledgeBase, 'id' | 'createdAt'>>): KnowledgeBase | null {
    return this.update(id, updates)
  }

  // Delete a knowledge base (alias for delete method)
  static removeKnowledgeBase(id: string): boolean {
    return this.delete(id)
  }

  // Clear all knowledge bases (for testing/reset purposes)
  static clear(): void {
    try {
      localStorage.removeItem(STORAGE_KEY)
    } catch (error) {
      console.error('Error clearing knowledge bases from localStorage:', error)
    }
  }
}