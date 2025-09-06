import { useState, useCallback } from 'react'
import { UuidUtils } from '../utils'

export interface ModelSettings {
  model: string
  temperature: number
  topP: number
  presencePenalty: number
  frequencyPenalty: number
}

export interface Assistant {
  id: string
  name: string
  description: string
  knowledgeBases: string[]
  modelSettings: ModelSettings
  createdAt: Date
}

export interface ChatSession {
  id: string
  name: string
  assistantId: string
  createdAt: Date
  lastMessage?: string
  messageCount: number
  lastActive?: Date
}

export interface Message {
  id: string
  type: 'user' | 'assistant' | 'system'  // ✅ ADD: Include 'system' type
  content: string
  timestamp: Date
  sessionId: string
  // ✅ ADD: Context passages for citations (Raptor-service style)
  contextPassages?: Array<{
    content: string
    relevant_excerpt?: string  // ✅ ADD: Smart excerpt for display
    similarity_score?: number
    doc_id?: string
    chunk_index?: number
    owner_type?: string
  }>
}

export interface UploadedFile {
  id: string
  name: string
  size: number
  type: string
  uploadedAt: Date
  url?: string
}

export const useChatState = () => {
  const [selectedAssistant, setSelectedAssistant] = useState<Assistant | null>(null)
  const [selectedSession, setSelectedSession] = useState<ChatSession | null>(null)
  const [sessions, setSessions] = useState<ChatSession[]>([])
  const [messages, setMessages] = useState<Message[]>([])
  const [uploadedFiles, setUploadedFiles] = useState<UploadedFile[]>([])

  // Session management
  const selectSession = useCallback(async (session: ChatSession | null) => {
    setSelectedSession(session)
    if (session) {
      try {
        // Load messages from database via API
        const { chatService } = await import('../services/chatService')
        const apiMessages = await chatService.getSessionMessages(session.id, 100, 0)
        
        // Convert API messages to local format
        const localMessages: Message[] = apiMessages.map(apiMsg => ({
          id: apiMsg.message_id,
          type: apiMsg.role as 'user' | 'assistant' | 'system',
          content: apiMsg.content,
          timestamp: new Date(apiMsg.created_at),
          sessionId: session.id,
          // ✅ ADD: Include context passages from stored messages
          contextPassages: apiMsg.extra_metadata?.context_passages || []
        }))
        
        setMessages(localMessages)
      } catch (error) {
        console.error('Failed to load session messages:', error)
        // Fallback to local filtering if API fails
        const sessionMessages = messages.filter(msg => msg.sessionId === session.id)
        setMessages(sessionMessages)
      }
    } else {
      setMessages([])
    }
  }, [messages])

  const createNewSession = useCallback(async (assistantId: string, sessionName?: string) => {
    if (!assistantId) return null

    try {
      // Import chat service dynamically
      const { chatService } = await import('../services/chatService')
      
      // Create session via API
      const apiSession = await chatService.createSession({
        assistant_id: assistantId,
        name: sessionName || `New Chat ${sessions.length + 1}`
      })

      // Convert API response to local format
      const newSession: ChatSession = {
        id: apiSession.session_id,
        name: apiSession.name,
        assistantId,
        createdAt: new Date(apiSession.created_at),
        messageCount: 0
      }

      setSessions(prev => [...prev, newSession])
      setSelectedSession(newSession)
      setMessages([])
      return newSession

    } catch (error) {
      console.error('Failed to create session:', error)
      
      // Fallback to local session if API fails
      const newSession: ChatSession = {
        id: UuidUtils.generateSessionId(),
        name: sessionName || `New Chat ${sessions.length + 1}`,
        assistantId,
        createdAt: new Date(),
        messageCount: 0
      }

      setSessions(prev => [...prev, newSession])
      setSelectedSession(newSession)
      setMessages([])
      return newSession
    }
  }, [sessions.length])

  // Assistant management
  const selectAssistant = useCallback(async (assistant: Assistant | null) => {
    setSelectedAssistant(assistant)
    setSelectedSession(null) // Clear selected session when changing assistant
    setMessages([]) // Clear messages when changing assistant

    // Load sessions for the selected assistant from database
    if (assistant) {
      try {
        // Import chat service dynamically
        const { chatService } = await import('../services/chatService')
        
        // Get sessions from database
        const assistantSessions = await chatService.getAssistantSessions(assistant.id)
        
        // Convert API sessions to local format and update state
        const localSessions: ChatSession[] = assistantSessions.map(apiSession => ({
          id: apiSession.session_id,
          name: apiSession.name,
          assistantId: assistant.id,
          createdAt: new Date(apiSession.created_at),
          messageCount: apiSession.message_count,
          lastMessage: undefined,
          lastActive: apiSession.last_active ? new Date(apiSession.last_active) : undefined
        }))
        
        // Update sessions state with loaded sessions
        setSessions(localSessions)
        
        // Auto-select session with messages first, or first session if available
        const sessionWithMessages = localSessions.find(s => s.messageCount > 0)
        const sessionToSelect = sessionWithMessages || localSessions[0]
        
        if (sessionToSelect) {
          await selectSession(sessionToSelect)
        } else {
          // Only create new session if NO sessions exist
          try {
            const newSession = await createNewSession(assistant.id, `Chat with ${assistant.name}`)
            if (newSession) {
              // Session created successfully
            }
          } catch (error) {
            console.error('Failed to create initial session:', error)
          }
        }
        
      } catch (error) {
        console.error('Failed to load assistant sessions:', error)
        // Fallback to local sessions if API fails
        const assistantSessions = sessions.filter(session => session.assistantId === assistant.id)
        if (assistantSessions.length > 0) {
          selectSession(assistantSessions[0]).catch(err => 
            console.error('Failed to select session:', err)
          )
        } else {
          // Create new session as fallback
          try {
            await createNewSession(assistant.id, `Chat with ${assistant.name}`)
          } catch (error) {
            console.error('Failed to create fallback session:', error)
          }
        }
      }
    }
  }, [selectSession, createNewSession])

  // Message management
  const addMessage = useCallback((message: Omit<Message, 'id'>) => {
    const newMessage: Message = {
      ...message,
      id: UuidUtils.generateMessageId()
    }

    setMessages(prev => [...prev, newMessage])

    // Update session with last message and count
    if (selectedSession) {
      setSessions(prev => prev.map(session =>
        session.id === selectedSession.id
          ? {
              ...session,
              lastMessage: message.type === 'user' ? message.content : session.lastMessage,
              messageCount: session.messageCount + 1
            }
          : session
      ))
    }

    return newMessage
  }, [selectedSession])

  // File management
  const addUploadedFile = useCallback((file: Omit<UploadedFile, 'id' | 'uploadedAt'>) => {
    const newFile: UploadedFile = {
      ...file,
      id: UuidUtils.generateV4(),
      uploadedAt: new Date()
    }

    setUploadedFiles(prev => [...prev, newFile])
    return newFile
  }, [])

  const removeUploadedFile = useCallback((fileId: string) => {
    setUploadedFiles(prev => prev.filter(file => file.id !== fileId))
  }, [])

  // Get sessions for current assistant
  const currentAssistantSessions = sessions.filter(
    session => selectedAssistant && session.assistantId === selectedAssistant.id
  )

  return {
    // State
    selectedAssistant,
    selectedSession,
    sessions: currentAssistantSessions,
    messages,
    uploadedFiles,

    // Actions
    selectAssistant,
    createNewSession,
    selectSession,
    addMessage,
    addUploadedFile,
    removeUploadedFile
  }
}
