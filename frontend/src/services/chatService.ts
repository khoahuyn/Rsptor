// Chat service for communicating with backend chat APIs
import { apiRequest } from './api'

export interface ChatRequest {
  query: string
  assistant_id: string
  session_id?: string
  include_context?: boolean
  max_context_messages?: number
}

export interface ChatResponse {
  answer: string
  session_id: string
  message_id: string
  metadata?: {
    tokens?: {
      input: number
      output: number
      total: number
    }
    retrieval?: {
      chunks_found: number
      context_included: boolean
    }
  }
}

export interface ChatSession {
  session_id: string
  name: string
  message_count: number
  created_at: string
  last_active?: string
}

export interface CreateSessionRequest {
  assistant_id: string
  name?: string
}

export interface ChatMessage {
  message_id: string
  role: 'user' | 'assistant' | 'system'
  content: string
  created_at: string
  extra_metadata?: any
}

export interface SessionDetail {
  session_id: string
  assistant_id?: string
  kb_id: string
  name: string
  messages: ChatMessage[]
  message_count: number
  created_at: string
  last_active?: string
}

class ChatService {
  /**
   * Get all chat sessions for an assistant
   */
  async getAssistantSessions(assistantId: string): Promise<ChatSession[]> {
    return apiRequest<ChatSession[]>(`/chat/assistants/${assistantId}/sessions`)
  }

  /**
   * Send a message to an AI assistant
   */
  async chatWithAssistant(request: ChatRequest): Promise<ChatResponse> {
    return apiRequest<ChatResponse>('/chat/assistant', {
      method: 'POST',
      body: JSON.stringify(request),
    })
  }

  /**
   * Create a new chat session for an assistant
   */
  async createSession(request: CreateSessionRequest): Promise<ChatSession> {
    const response = await apiRequest<{
      session_id: string
      assistant_id: string
      name: string
      created_at: string
    }>('/chat/sessions', {
      method: 'POST',
      body: JSON.stringify(request),
    })
    
    // Convert to ChatSession format
    return {
      session_id: response.session_id,
      name: response.name,
      message_count: 0,
      created_at: response.created_at,
      last_active: undefined
    }
  }

  /**
   * Get session details with message history
   */
  async getSession(sessionId: string): Promise<SessionDetail> {
    return apiRequest<SessionDetail>(`/chat/sessions/${sessionId}`)
  }

  /**
   * Get messages for a session with pagination
   */
  async getSessionMessages(
    sessionId: string,
    limit: number = 50,
    offset: number = 0
  ): Promise<ChatMessage[]> {
    return apiRequest<ChatMessage[]>(
      `/chat/sessions/${sessionId}/messages?limit=${limit}&offset=${offset}`
    )
  }

  /**
   * Delete a chat session
   */
  async deleteSession(sessionId: string): Promise<void> {
    await apiRequest<void>(`/chat/sessions/${sessionId}`, {
      method: 'DELETE',
    })
  }

  /**
   * Legacy smart chat (without session management)
   */
  async smartChat(query: string, tenantId: string, kbId: string): Promise<{ answer: string }> {
    return apiRequest<{ answer: string }>('/chat/smart', {
      method: 'POST',
      body: JSON.stringify({
        query,
        tenant_id: tenantId,
        kb_id: kbId,
      }),
    })
  }
}

export const chatService = new ChatService()
