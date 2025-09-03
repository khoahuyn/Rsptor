import { apiRequest } from './api'

export interface ModelSettings {
  model: string
  temperature: number
  topP: number
  presencePenalty: number
  frequencyPenalty: number
}

export interface AssistantData {
  name: string
  description?: string
  knowledge_bases: string[]
  model_settings: ModelSettings
  user_id?: string
  system_prompt?: string
}

export interface Assistant {
  assistant_id: string
  user_id?: string
  name: string
  description?: string
  knowledge_bases: string[]
  model_settings: ModelSettings
  system_prompt?: string
  status?: string
  meta?: Record<string, unknown>
  created_at: string
  updated_at?: string
  datasets?: Array<{
    id: string
    name: string
    description?: string
    doc_count: number
    created_at: string
  }>
}

export interface AssistantListResponse {
  assistants: Assistant[]
  total: number
  limit: number
  offset: number
  has_more: boolean
}

export interface ApiResponse<T> {
  code: number
  data: T
  message?: string
}

class AssistantService {
  async createAssistant(assistantData: AssistantData): Promise<Assistant> {
    const response = await apiRequest<ApiResponse<Assistant>>('/ai/assistants', {
      method: 'POST',
      body: JSON.stringify(assistantData),
    })
    
    return response.data
  }

  async getAssistant(assistantId: string): Promise<Assistant> {
    const response = await apiRequest<ApiResponse<Assistant>>(`/ai/assistants/${assistantId}`)
    return response.data
  }

  async getAssistantWithDatasets(assistantId: string): Promise<Assistant> {
    const response = await apiRequest<ApiResponse<Assistant>>(`/ai/assistants/${assistantId}/details`)
    return response.data
  }

  async listAssistants(params?: {
    user_id?: string
    limit?: number
    offset?: number
  }): Promise<AssistantListResponse> {
    const searchParams = new URLSearchParams()
    
    if (params?.user_id) searchParams.append('user_id', params.user_id)
    if (params?.limit) searchParams.append('limit', params.limit.toString())
    if (params?.offset) searchParams.append('offset', params.offset.toString())

    const queryString = searchParams.toString()
    const endpoint = `/ai/assistants${queryString ? `?${queryString}` : ''}`
    
    // Direct response without ApiResponse wrapper
    const response = await apiRequest<AssistantListResponse>(endpoint)
    return response
  }

  async updateAssistant(
    assistantId: string, 
    updates: Partial<AssistantData>
  ): Promise<Assistant> {
    const response = await apiRequest<ApiResponse<Assistant>>(`/ai/assistants/${assistantId}`, {
      method: 'PUT',
      body: JSON.stringify(updates),
    })
    
    return response.data
  }

  async deleteAssistant(assistantId: string): Promise<void> {
    await apiRequest<ApiResponse<void>>(`/ai/assistants/${assistantId}`, {
      method: 'DELETE',
    })
  }
}

export const assistantService = new AssistantService()
export default assistantService