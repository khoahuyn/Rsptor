import { useState } from 'react'
import { Button, Card, CardBody } from '@heroui/react'
import { Text, Heading } from '@radix-ui/themes'
import { UuidUtils } from '../../utils'

interface GeneratedUuid {
  id: string
  type: string
  version: number | null
  timestamp: string
}

export const UuidDemo = () => {
  const [generatedUuids, setGeneratedUuids] = useState<GeneratedUuid[]>([])

  const generateUuid = (type: string, generator: () => string) => {
    const uuid = generator()
    const newUuid: GeneratedUuid = {
      id: uuid,
      type,
      version: UuidUtils.getVersion(uuid),
      timestamp: new Date().toLocaleTimeString()
    }
    setGeneratedUuids(prev => [newUuid, ...prev.slice(0, 9)]) // Keep only last 10
  }

  const validateUuid = () => {
    const input = prompt('Enter a UUID to validate:')
    if (input) {
      const isValid = UuidUtils.isValid(input)
      const version = UuidUtils.getVersion(input)
      alert(`UUID: ${input}\nValid: ${isValid}\nVersion: ${version || 'Invalid'}`)
    }
  }

  const clearHistory = () => {
    setGeneratedUuids([])
  }

  return (
    <div className="p-6 max-w-4xl mx-auto">
      <Heading size="6" className="mb-6">UUID Generator Demo</Heading>
      
      {/* Generator Buttons */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
        <Button 
          color="primary" 
          onClick={() => generateUuid('Knowledge Base', UuidUtils.generateKnowledgeBaseId)}
        >
          Generate KB ID
        </Button>
        
        <Button 
          color="secondary" 
          onClick={() => generateUuid('Document', UuidUtils.generateDocumentId)}
        >
          Generate Doc ID
        </Button>
        
        <Button 
          color="success" 
          onClick={() => generateUuid('Session', UuidUtils.generateSessionId)}
        >
          Generate Session ID
        </Button>
        
        <Button 
          color="warning" 
          onClick={() => generateUuid('Message', UuidUtils.generateMessageId)}
        >
          Generate Message ID
        </Button>
        
        <Button 
          variant="bordered" 
          onClick={() => generateUuid('UUID v1', UuidUtils.generateV1)}
        >
          Generate UUID v1
        </Button>
        
        <Button 
          variant="bordered" 
          onClick={() => generateUuid('UUID v4', UuidUtils.generateV4)}
        >
          Generate UUID v4
        </Button>
        
        <Button 
          variant="bordered" 
          onClick={() => generateUuid('Short UUID', UuidUtils.generateShort)}
        >
          Generate Short
        </Button>
        
        <Button 
          variant="ghost" 
          onClick={validateUuid}
        >
          Validate UUID
        </Button>
      </div>

      {/* Control Buttons */}
      <div className="flex gap-2 mb-6">
        <Button 
          variant="flat" 
          onClick={clearHistory}
          disabled={generatedUuids.length === 0}
        >
          Clear History
        </Button>
        
        <Text className="self-center text-gray-600">
          Generated: {generatedUuids.length}/10
        </Text>
      </div>

      {/* Generated UUIDs History */}
      <div className="space-y-3">
        <Heading size="4">Generated UUIDs</Heading>
        
        {generatedUuids.length === 0 ? (
          <Card>
            <CardBody className="text-center py-8">
              <Text className="text-gray-500">No UUIDs generated yet. Click a button above to start!</Text>
            </CardBody>
          </Card>
        ) : (
          generatedUuids.map((uuid, index) => (
            <Card key={index} className="hover:shadow-md transition-shadow">
              <CardBody>
                <div className="flex flex-col md:flex-row md:items-center justify-between gap-2">
                  <div className="flex-1">
                    <div className="flex items-center gap-2 mb-1">
                      <Text className="font-semibold text-blue-600">{uuid.type}</Text>
                      <Text className="text-xs text-gray-500">v{uuid.version}</Text>
                      <Text className="text-xs text-gray-400">{uuid.timestamp}</Text>
                    </div>
                    <code className="text-sm bg-gray-100 p-1 rounded font-mono break-all">
                      {uuid.id}
                    </code>
                  </div>
                  
                  <div className="flex gap-2">
                    <Button 
                      size="sm" 
                      variant="flat"
                      onClick={() => navigator.clipboard.writeText(uuid.id)}
                    >
                      Copy
                    </Button>
                    
                    <Button 
                      size="sm" 
                      variant="flat"
                      onClick={() => {
                        const formatted = UuidUtils.formatForDisplay(uuid.id)
                        alert(`Formatted: ${formatted}\nNo hyphens: ${UuidUtils.removeHyphens(uuid.id)}`)
                      }}
                    >
                      Format
                    </Button>
                  </div>
                </div>
              </CardBody>
            </Card>
          ))
        )}
      </div>

      {/* Usage Examples */}
      <div className="mt-8 p-4 bg-gray-50 rounded-lg">
        <Heading size="4" className="mb-4">Usage Examples</Heading>
        <div className="space-y-2 text-sm">
          <code className="block bg-white p-2 rounded">
            import &#123; UuidUtils &#125; from './utils'
          </code>
          <code className="block bg-white p-2 rounded">
            const kbId = UuidUtils.generateKnowledgeBaseId()
          </code>
          <code className="block bg-white p-2 rounded">
            const isValid = UuidUtils.isValid(someUuid)
          </code>
          <code className="block bg-white p-2 rounded">
            const version = UuidUtils.getVersion(someUuid)
          </code>
        </div>
      </div>
    </div>
  )
}