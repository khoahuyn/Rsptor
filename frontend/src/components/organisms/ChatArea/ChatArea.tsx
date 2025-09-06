import { Card, CardBody, Input, Button, Avatar, Popover, PopoverTrigger, PopoverContent } from '@heroui/react'
import { Heading, Text, Flex } from '@radix-ui/themes'
import { PaperPlaneIcon, ChatBubbleIcon, ClipboardIcon } from '@radix-ui/react-icons'
import { useState, useRef, useEffect } from 'react'
import { FileUpload } from '../../molecules'
import type { FileUploadItem } from '../../molecules'
import type { Message, ChatSession } from '../../../hooks/useChatState'

interface ChatAreaProps {
  className?: string
  selectedSession: ChatSession | null
  messages: Message[]
  onSendMessage: (content: string, files?: FileUploadItem[]) => void
}

export const ChatArea = ({ className, selectedSession, messages, onSendMessage }: ChatAreaProps) => {
  const [inputValue, setInputValue] = useState('')
  const [isLoading] = useState(false)
  const [selectedFiles, setSelectedFiles] = useState<FileUploadItem[]>([])
  const [isFileUploadOpen, setIsFileUploadOpen] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const handleSendMessage = async () => {
    if (!inputValue.trim() && selectedFiles.length === 0) return
    if (!selectedSession) return

    // Send message with files if any
    onSendMessage(inputValue.trim(), selectedFiles.length > 0 ? selectedFiles : undefined)

    // Clear input and files
    setInputValue('')
    setSelectedFiles([])
    setIsFileUploadOpen(false)
  }

  const handleFilesSelected = (files: FileUploadItem[]) => {
    setSelectedFiles(prev => [...prev, ...files])
  }

  const handleRemoveFile = (fileId: string) => {
    setSelectedFiles(prev => prev.filter(file => file.id !== fileId))
  }

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSendMessage()
    }
  }

  return (
    <div className={`h-full ${className || ''}`}>
      <Card className="h-full border border-gray-200 shadow-sm">
        <CardBody className="p-0 h-full flex flex-col">
          {/* Header */}
          <div className="p-6 border-b border-gray-200 bg-gray-50">
            <Flex align="center" gap="2">
              <ChatBubbleIcon className="w-5 h-5 text-indigo-600" />
              <Heading size="4" className="text-gray-900 font-bold">
                {selectedSession ? selectedSession.name : 'Conversation'}
              </Heading>
            </Flex>
            <Text className="text-gray-600 text-sm mt-1">
              {selectedSession
                ? 'Chat with your AI assistant'
                : 'Select a session to start chatting'
              }
            </Text>
          </div>

          {/* Messages Area */}
          <div className="flex-1 overflow-y-auto p-6">
            {!selectedSession ? (
              <div className="h-full flex items-center justify-center">
                <div className="text-center">
                  <ChatBubbleIcon className="w-16 h-16 text-gray-300 mx-auto mb-4" />
                  <Text className="text-gray-500 text-lg font-medium mb-2">
                    No Session Selected
                  </Text>
                  <Text className="text-gray-400 text-sm">
                    Create or select a chat session to start chatting
                  </Text>
                </div>
              </div>
            ) : messages.length === 0 ? (
              <div className="h-full flex items-center justify-center">
                <div className="text-center">
                  <ChatBubbleIcon className="w-12 h-12 text-gray-300 mx-auto mb-2" />
                  <Text className="text-gray-500 text-sm">
                    Start a conversation
                  </Text>
                  <Text className="text-gray-400 text-xs">
                    Ask anything about your knowledge bases
                  </Text>
                </div>
              </div>
            ) : (
              <div className="space-y-4">
                {messages.map((message) => (
                  <div
                    key={message.id}
                    className={`flex gap-3 ${
                      message.type === 'user' ? 'justify-end' : 'justify-start'
                    }`}
                  >
                    {message.type === 'assistant' && (
                      <Avatar
                        size="sm"
                        className="bg-indigo-600 text-white flex-shrink-0"
                        fallback="AI"
                      />
                    )}

                    <div
                      className={`max-w-[80%] p-4 rounded-lg ${
                        message.type === 'user'
                          ? 'bg-indigo-600 text-white'
                          : 'bg-gray-100 text-gray-900'
                      }`}
                    >
                      <Text className={`text-sm ${
                        message.type === 'user' ? 'text-white' : 'text-gray-900'
                      }`}>
                        {message.content}
                      </Text>
                      <Text className={`text-xs mt-2 ${
                        message.type === 'user' ? 'text-indigo-200' : 'text-gray-500'
                      }`}>
                        {message.timestamp.toLocaleTimeString()}
                      </Text>

                      {/* ✅ ADD: Simple Citation Format (from relevant excerpt) */}
                      {message.type === 'assistant' && message.contextPassages && message.contextPassages.length > 0 && (
                        <div className="mt-3 pt-2 border-t border-gray-200">
                          <Text className="text-xs text-gray-600 italic">
                            • {message.contextPassages[0].relevant_excerpt || 
                                  (message.contextPassages[0].content.length > 120 
                                    ? `${message.contextPassages[0].content.substring(0, 120)}...`
                                    : message.contextPassages[0].content
                                  )}
                          </Text>
                        </div>
                      )}
                    </div>

                    {message.type === 'user' && (
                      <Avatar
                        size="sm"
                        className="bg-gray-600 text-white flex-shrink-0"
                        fallback="U"
                      />
                    )}
                  </div>
                ))}

                {/* Loading Indicator */}
                {isLoading && (
                  <div className="flex gap-3 justify-start">
                    <Avatar
                      size="sm"
                      className="bg-indigo-600 text-white flex-shrink-0"
                      fallback="AI"
                    />
                    <div className="bg-gray-100 text-gray-900 p-4 rounded-lg">
                      <div className="flex space-x-1">
                        <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
                        <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
                        <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                      </div>
                    </div>
                  </div>
                )}

                <div ref={messagesEndRef} />
              </div>
            )}
          </div>

          {/* Input Area */}
          <div className="p-6 border-t border-gray-200 bg-white">
            {/* File Upload Section */}
            {selectedFiles.length > 0 && (
              <div className="mb-4">
                <FileUpload
                  onFilesSelected={handleFilesSelected}
                  selectedFiles={selectedFiles}
                  onRemoveFile={handleRemoveFile}
                  maxFiles={3}
                />
              </div>
            )}

            <Flex gap="2" align="end">
              <div className="flex-1">
                <Input
                  placeholder={selectedSession ? "Ask me anything about your knowledge bases..." : "Select a session to start chatting"}
                  value={inputValue}
                  onChange={(e) => setInputValue(e.target.value)}
                  onKeyPress={handleKeyPress}
                  variant="bordered"
                  size="lg"
                  className="w-full"
                  disabled={isLoading || !selectedSession}
                />
              </div>

              {/* File Upload Button */}
              <Popover isOpen={isFileUploadOpen} onOpenChange={setIsFileUploadOpen}>
                <PopoverTrigger>
                  <Button
                    variant="bordered"
                    size="lg"
                    isIconOnly
                    isDisabled={!selectedSession}
                    className={selectedFiles.length > 0 ? 'border-indigo-500 text-indigo-600' : ''}
                  >
                    <ClipboardIcon className="w-4 h-4" />
                  </Button>
                </PopoverTrigger>
                <PopoverContent className="w-80">
                  <div className="p-4">
                    <Text className="text-sm font-medium mb-3">Upload Files</Text>
                    <FileUpload
                      onFilesSelected={handleFilesSelected}
                      selectedFiles={selectedFiles}
                      onRemoveFile={handleRemoveFile}
                      maxFiles={3}
                    />
                  </div>
                </PopoverContent>
              </Popover>

              {/* Send Button */}
              <Button
                color="primary"
                variant="solid"
                size="lg"
                onClick={handleSendMessage}
                isDisabled={(!inputValue.trim() && selectedFiles.length === 0) || isLoading || !selectedSession}
                isIconOnly
              >
                <PaperPlaneIcon className="w-4 h-4" />
              </Button>
            </Flex>
            <Text className="text-xs text-gray-500 mt-2">
              {selectedSession
                ? "Press Enter to send, Shift+Enter for new line"
                : "Select a session to start chatting"
              }
            </Text>
          </div>
        </CardBody>
      </Card>
    </div>
  )
}
