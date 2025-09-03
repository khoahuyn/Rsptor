import { useState, useEffect } from 'react'
import { KnowledgeBaseService, type KnowledgeBase, type CreateKnowledgeBaseData } from '../services'

export interface UseLocalKnowledgeBasesState {
  knowledgeBases: KnowledgeBase[]
  loading: boolean
  error: string | null
}

export const useLocalKnowledgeBases = () => {
  const [state, setState] = useState<UseLocalKnowledgeBasesState>({
    knowledgeBases: [],
    loading: true,
    error: null
  })

  const loadKnowledgeBases = () => {
    try {
      setState(prev => ({ ...prev, loading: true, error: null }))
      const knowledgeBases = KnowledgeBaseService.getAll()
      setState({
        knowledgeBases,
        loading: false,
        error: null
      })
    } catch (error) {
      console.error('Failed to load knowledge bases:', error)
      setState(prev => ({
        ...prev,
        loading: false,
        error: error instanceof Error ? error.message : 'Failed to load knowledge bases'
      }))
    }
  }

  const createKnowledgeBase = async (data: CreateKnowledgeBaseData): Promise<KnowledgeBase> => {
    try {
      const newKB = await KnowledgeBaseService.createKB(data)
      loadKnowledgeBases() // Refresh the list
      return newKB
    } catch (error) {
      console.error('Failed to create knowledge base via backend:', error)
      throw error
    }
  }

  const updateKnowledgeBase = async (
    id: string, 
    updates: Partial<Omit<KnowledgeBase, 'id' | 'createdAt'>>
  ): Promise<KnowledgeBase | null> => {
    try {
      const updated = KnowledgeBaseService.update(id, updates)
      if (updated) {
        loadKnowledgeBases() // Refresh the list
      }
      return updated
    } catch (error) {
      console.error('Failed to update knowledge base:', error)
      throw error
    }
  }

  const deleteKnowledgeBase = async (id: string): Promise<boolean> => {
    try {
      const deleted = KnowledgeBaseService.delete(id)
      if (deleted) {
        loadKnowledgeBases() // Refresh the list
      }
      return deleted
    } catch (error) {
      console.error('Failed to delete knowledge base:', error)
      throw error
    }
  }

  const incrementDocumentCount = async (id: string): Promise<KnowledgeBase | null> => {
    try {
      const updated = KnowledgeBaseService.incrementDocumentCount(id)
      if (updated) {
        loadKnowledgeBases() // Refresh the list
      }
      return updated
    } catch (error) {
      console.error('Failed to increment document count:', error)
      throw error
    }
  }

  const getKnowledgeBase = (id: string): KnowledgeBase | null => {
    return state.knowledgeBases.find(kb => kb.id === id) || null
  }

  useEffect(() => {
    loadKnowledgeBases()
  }, [])

  return {
    ...state,
    createKnowledgeBase,
    updateKnowledgeBase,
    deleteKnowledgeBase,
    incrementDocumentCount,
    getKnowledgeBase,
    refetch: loadKnowledgeBases
  }
}