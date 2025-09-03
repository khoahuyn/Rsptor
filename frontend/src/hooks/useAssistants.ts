import { useState, useEffect, useCallback } from 'react'
import { assistantService, type Assistant, type AssistantData, type AssistantListResponse } from '../services/assistantService'

interface UseAssistantsOptions {
  user_id?: string
  limit?: number
  offset?: number
  autoFetch?: boolean
}

export const useAssistants = (options: UseAssistantsOptions = {}) => {
  const [assistants, setAssistants] = useState<Assistant[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [total, setTotal] = useState(0)
  const [hasMore, setHasMore] = useState(false)

  const { user_id, limit = 50, offset = 0, autoFetch = true } = options

  const fetchAssistants = useCallback(async () => {
    setLoading(true)
    setError(null)

    try {
      const response: AssistantListResponse = await assistantService.listAssistants({
        user_id,
        limit,
        offset,
      })

      setAssistants(response.assistants)
      setTotal(response.total)
      setHasMore(response.has_more)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch assistants')
      console.error('Error fetching assistants:', err)
    } finally {
      setLoading(false)
    }
  }, [user_id, limit, offset])

  const createAssistant = useCallback(async (assistantData: AssistantData): Promise<Assistant> => {
    setError(null)

    try {
      const newAssistant = await assistantService.createAssistant(assistantData)
      
      // Add to local state
      setAssistants(prev => [newAssistant, ...prev])
      setTotal(prev => prev + 1)
      
      return newAssistant
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to create assistant'
      setError(errorMessage)
      throw new Error(errorMessage)
    }
  }, [])

  const updateAssistant = useCallback(async (
    assistantId: string, 
    updates: Partial<AssistantData>
  ): Promise<Assistant> => {
    setError(null)

    try {
      const updatedAssistant = await assistantService.updateAssistant(assistantId, updates)
      
      // Update local state
      setAssistants(prev => 
        prev.map(assistant => 
          assistant.assistant_id === assistantId ? updatedAssistant : assistant
        )
      )
      
      return updatedAssistant
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to update assistant'
      setError(errorMessage)
      throw new Error(errorMessage)
    }
  }, [])

  const deleteAssistant = useCallback(async (assistantId: string) => {
    setError(null)

    try {
      await assistantService.deleteAssistant(assistantId)
      
      // Remove from local state
      setAssistants(prev => prev.filter(assistant => assistant.assistant_id !== assistantId))
      setTotal(prev => prev - 1)
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to delete assistant'
      setError(errorMessage)
      throw new Error(errorMessage)
    }
  }, [])

  const getAssistantById = useCallback((assistantId: string): Assistant | undefined => {
    return assistants.find(assistant => assistant.assistant_id === assistantId)
  }, [assistants])

  const refetch = useCallback(() => {
    fetchAssistants()
  }, [fetchAssistants])

  useEffect(() => {
    if (autoFetch) {
      fetchAssistants()
    }
  }, [fetchAssistants, autoFetch])

  return {
    assistants,
    loading,
    error,
    total,
    hasMore,
    refetch,
    createAssistant,
    updateAssistant,
    deleteAssistant,
    getAssistantById,
  }
}