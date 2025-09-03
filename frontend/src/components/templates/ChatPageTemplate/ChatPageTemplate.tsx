import { Flex } from '@radix-ui/themes'
import { KnowledgeHeader } from '../../organisms'
import { AssistantCreation, ChatArea, ChartSection } from '../../organisms'
import { useChatState, type Assistant } from '../../../hooks/useChatState'
import type { FileUploadItem } from '../../molecules'
import { motion } from 'framer-motion'
import type { Assistant as ServiceAssistant } from '../../../services/assistantService'

export type { Assistant } from '../../../hooks/useChatState'

interface ChatPageTemplateProps {
  className?: string
}

// Helper function to convert ServiceAssistant to ChatState Assistant
const convertServiceAssistantToChatAssistant = (serviceAssistant: ServiceAssistant): Assistant => {
  return {
    id: serviceAssistant.assistant_id,
    name: serviceAssistant.name,
    description: serviceAssistant.description || '',
    knowledgeBases: serviceAssistant.knowledge_bases,
    modelSettings: serviceAssistant.model_settings,
    createdAt: new Date(serviceAssistant.created_at)
  }
}

export const ChatPageTemplate = ({ className }: ChatPageTemplateProps) => {
  const {
    selectedAssistant,
    selectedSession,
    sessions,
    messages,
    selectAssistant,
    createNewSession,
    selectSession,
    addMessage
  } = useChatState()

  // Animation variants
  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        duration: 0.4,
        staggerChildren: 0.2
      }
    }
  }

  const sectionVariants = {
    hidden: { opacity: 0, y: 30 },
    visible: {
      opacity: 1,
      y: 0,
      transition: {
        duration: 0.6,
        ease: "easeOut" as const
      }
    }
  }

  const headerVariants = {
    hidden: { opacity: 0, y: -20 },
    visible: {
      opacity: 1,
      y: 0,
      transition: {
        duration: 0.5,
        ease: "easeOut" as const
      }
    }
  }

  // Handle assistant selection from AssistantCreation
  const handleAssistantSelect = async (assistant: ServiceAssistant) => {
    const convertedAssistant = convertServiceAssistantToChatAssistant(assistant)
    
    try {
      // selectAssistant will load sessions from database and auto-select one
      await selectAssistant(convertedAssistant)
      
      // Note: selectAssistant now handles session loading and creation internally
      // No need for additional session creation logic here
      
    } catch (error) {
      console.error('Failed to select assistant:', error)
    }
  }

  // Handle message sending from ChatArea
  const handleSendMessage = async (content: string, files?: FileUploadItem[]) => {
    if (!selectedSession || !selectedAssistant) return

    // Add user message immediately
    addMessage({
      type: 'user',
      content,
      timestamp: new Date(),
      sessionId: selectedSession.id
    })

    try {
      // Import and use chat service
      const { chatService } = await import('../../../services/chatService')
      
      // Send message to backend
      const response = await chatService.chatWithAssistant({
        query: content,
        assistant_id: selectedAssistant.id,
        session_id: selectedSession.id,
        include_context: true,
        max_context_messages: 10
      })

      // Add AI response
      addMessage({
        type: 'assistant',
        content: response.answer,
        timestamp: new Date(),
        sessionId: selectedSession.id
      })

      // Optional: Handle file uploads if provided
      if (files && files.length > 0) {
        // TODO: Implement file upload to knowledge base if needed
      }

    } catch (error) {
      console.error('Chat error:', error)
      // Add error message
      addMessage({
        type: 'assistant',
        content: 'Sorry, I encountered an error processing your request. Please try again.',
        timestamp: new Date(),
        sessionId: selectedSession.id
      })
    }
  }
  return (
    <div className={`min-h-screen bg-gray-50 ${className || ''}`}>
      {/* Header */}
      <motion.div
        variants={headerVariants}
        initial="hidden"
        animate="visible"
      >
        <KnowledgeHeader />
      </motion.div>

      {/* Main Chat Interface - 3 Column Layout */}
      <motion.div
        className="px-6 py-6"
        variants={containerVariants}
        initial="hidden"
        animate="visible"
      >
        <Flex gap="6" className="h-[calc(100vh-120px)] w-full">
          {/* Left Section - Create Assistant (25% width) */}
          <motion.div
            className="flex-1 min-w-0"
            variants={sectionVariants}
            whileHover={{ scale: 1.01 }}
            transition={{ duration: 0.2 }}
          >
            <AssistantCreation onAssistantSelect={handleAssistantSelect} />
          </motion.div>

          {/* Middle Section - Chart (25% width) */}
          <motion.div
            className="flex-1 min-w-0"
            variants={sectionVariants}
            whileHover={{ scale: 1.01 }}
            transition={{ duration: 0.2 }}
          >
            <ChartSection
              selectedAssistant={selectedAssistant}
              sessions={sessions}
              selectedSession={selectedSession}
              onSelectSession={selectSession}
              onCreateNewSession={createNewSession}
            />
          </motion.div>

          {/* Right Section - ChatArea (50% width - 2x wider) */}
          <motion.div
            className="flex-[2] min-w-0"
            variants={sectionVariants}
            whileHover={{ scale: 1.005 }}
            transition={{ duration: 0.2 }}
          >
            <ChatArea
              selectedSession={selectedSession}
              messages={messages}
              onSendMessage={handleSendMessage}
            />
          </motion.div>
        </Flex>
      </motion.div>
    </div>
  )
}
