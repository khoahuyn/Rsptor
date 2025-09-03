import { Card, CardBody, Button } from '@heroui/react'
import { Heading, Text, Flex } from '@radix-ui/themes'
import { ChatBubbleIcon, PlusIcon, ClockIcon } from '@radix-ui/react-icons'
import type { Assistant, ChatSession } from '../../../hooks/useChatState'

interface ChartSectionProps {
  className?: string
  selectedAssistant: Assistant | null
  sessions: ChatSession[]
  selectedSession: ChatSession | null
  onSelectSession: (session: ChatSession) => void
  onCreateNewSession: (assistantId: string) => void
}

export const ChartSection = ({
  className,
  selectedAssistant,
  sessions,
  selectedSession,
  onSelectSession,
  onCreateNewSession
}: ChartSectionProps) => {
  const handleCreateNewSession = () => {
    if (selectedAssistant) {
      onCreateNewSession(selectedAssistant.id)
    }
  }

  return (
    <div className={`h-full ${className || ''}`}>
      <Card className="h-full border border-gray-200 shadow-sm">
        <CardBody className="p-6">
          <Flex direction="column" gap="4" className="h-full">
            {/* Header */}
            <div>
              <Flex align="center" gap="2" className="mb-2">
                <ChatBubbleIcon className="w-5 h-5 text-indigo-600" />
                <Heading size="4" className="text-gray-900 font-bold">
                  Chat Sessions
                </Heading>
              </Flex>
              <Text className="text-gray-600 text-sm">
                {selectedAssistant
                  ? `Sessions with ${selectedAssistant.name}`
                  : 'Select an assistant to view sessions'
                }
              </Text>
            </div>

            {selectedAssistant ? (
              <>
                {/* New Session Button - Only show if no sessions exist or user explicitly wants new one */}
                {sessions.length === 0 ? (
                  <Button
                    color="primary"
                    variant="solid"
                    size="sm"
                    startContent={<PlusIcon className="w-4 h-4" />}
                    className="w-full"
                    onClick={handleCreateNewSession}
                  >
                    Start New Chat
                  </Button>
                ) : (
                  <Button
                    color="primary"
                    variant="bordered"
                    size="sm"
                    startContent={<PlusIcon className="w-4 h-4" />}
                    className="w-full"
                    onClick={handleCreateNewSession}
                  >
                    New Chat Session
                  </Button>
                )}

                {/* Sessions List */}
                <div className="flex-1 overflow-y-auto">
                  {sessions.length === 0 ? (
                    <div className="flex flex-col items-center justify-center h-32 text-center">
                      <ChatBubbleIcon className="w-12 h-12 text-gray-300 mb-2" />
                      <Text className="text-gray-500 text-sm">
                        No chat sessions yet
                      </Text>
                      <Text className="text-gray-400 text-xs">
                        Create a new session to get started
                      </Text>
                    </div>
                  ) : (
                    <div className="space-y-2">
                      <Text className="text-sm font-medium text-gray-700 mb-3">
                        Recent Sessions ({sessions.length})
                      </Text>
                      {sessions.map((session) => (
                        <div
                          key={session.id}
                          className={`p-3 rounded-lg border cursor-pointer transition-colors ${
                            selectedSession?.id === session.id
                              ? 'border-indigo-500 bg-indigo-50'
                              : 'border-gray-200 hover:border-gray-300 hover:bg-gray-50'
                          }`}
                          onClick={() => onSelectSession(session)}
                        >
                          <Text className="text-sm font-medium text-gray-900 mb-1 line-clamp-1">
                            {session.name}
                          </Text>
                          <Flex align="center" justify="between" className="mb-1">
                            <Text className="text-xs text-gray-500">
                              {session.messageCount} messages
                            </Text>
                            <Flex align="center" gap="1">
                              <ClockIcon className="w-3 h-3 text-gray-400" />
                              <Text className="text-xs text-gray-400">
                                {session.createdAt.toLocaleDateString()}
                              </Text>
                            </Flex>
                          </Flex>
                          {session.lastMessage && (
                            <Text className="text-xs text-gray-500 line-clamp-1">
                              "{session.lastMessage}"
                            </Text>
                          )}
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </>
            ) : (
              <div className="flex-1 flex items-center justify-center">
                <div className="text-center">
                  <ChatBubbleIcon className="w-16 h-16 text-gray-300 mx-auto mb-4" />
                  <Text className="text-gray-500 text-lg font-medium mb-2">
                    No Assistant Selected
                  </Text>
                  <Text className="text-gray-400 text-sm">
                    Choose an assistant from the left panel to start chatting
                  </Text>
                </div>
              </div>
            )}
          </Flex>
        </CardBody>
      </Card>
    </div>
  )
}
