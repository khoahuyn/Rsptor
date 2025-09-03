import { apiRequest } from './api'
import { KnowledgeBaseService } from './knowledgeBaseService'

export interface UploadDocumentData {
  file: File
  datasetId: string
  source?: string
  tags?: string[]
  extraMeta?: string // JSON string instead of object
  buildTree?: boolean
  summaryLLM?: string // Match the API field name
  vectorIndex?: string // JSON string instead of object
  upsertMode?: 'upsert' | 'replace' | 'skip_duplicates'
}

export interface DocumentUploadResponse {
  doc_id: string
  filename: string
  total_chunks: number
  total_embeddings: number
  processing_time: number
  status: string
  
  // RAPTOR-specific chunk counts
  original_chunks?: number
  summary_chunks?: number
  
  // Optional RAPTOR fields
  raptor_enabled: boolean
  raptor_summary_count?: number
  raptor_tree_levels?: number
  
  // Metadata
  tenant_id: string
  kb_id: string
  created_at?: string
}

export class DocumentService {
  // Upload a document to a knowledge base (using UUID as dataset_id)
  static async uploadDocument(data: UploadDocumentData): Promise<DocumentUploadResponse> {
    // Verify the knowledge base exists in localStorage
    if (!KnowledgeBaseService.exists(data.datasetId)) {
      throw new Error(`Knowledge base with ID "${data.datasetId}" not found`)
    }

    // Prepare form data for RAGFlow API
    const formData = new FormData()
    formData.append('file', data.file)
    
    // Extract tenant_id from datasetId (format: tenant_id::kb::name or just kb_id)
    const parts = data.datasetId.split('::')
    const tenant_id = parts.length >= 3 ? parts[0] : 'test_tenant'
    const kb_id = data.datasetId  // Use full datasetId as kb_id
    
    formData.append('tenant_id', tenant_id)
    formData.append('kb_id', kb_id)
    
    // RAGFlow RAPTOR settings
    formData.append('enable_raptor', (data.buildTree !== false).toString())
    formData.append('max_clusters', '64')
    formData.append('threshold', '0.1')
    formData.append('random_seed', '42')

    // Debug: Log form data before sending
    console.log('[DocumentService] RAGFlow Upload request data:', {
      file: data.file.name,
      fileSize: data.file.size,
      fileType: data.file.type,
      tenant_id: tenant_id,
      kb_id: kb_id,
      enable_raptor: (data.buildTree !== false).toString(),
      max_clusters: '64',
      threshold: '0.1',
      random_seed: '42'
    })

    // Debug: Log FormData entries
    console.log('[DocumentService] FormData entries:')
    for (const [key, value] of formData.entries()) {
      if (value instanceof File) {
        console.log(`  ${key}: File { name: ${value.name}, size: ${value.size}, type: ${value.type} }`)
      } else {
        console.log(`  ${key}: ${value}`)
      }
    }

    try {
      // Call the RAGFlow upload API
      const response = await fetch(`http://localhost:8081/v1/ragflow/process`, {
        method: 'POST',
        body: formData,
      })

      console.log('[DocumentService] Upload response:', {
        status: response.status,
        statusText: response.statusText
      })

      if (!response.ok) {
        let errorMessage: string
        try {
          const errorData = await response.json()
          errorMessage = errorData.detail || errorData.message || `HTTP ${response.status}: ${response.statusText}`
          console.error('[DocumentService] Upload error details:', errorData)
        } catch {
          errorMessage = `HTTP ${response.status}: ${response.statusText}`
          const errorText = await response.text()
          console.error('[DocumentService] Upload error text:', errorText)
        }
        throw new Error(`Upload failed: ${errorMessage}`)
      }

      const result: DocumentUploadResponse = await response.json()
      console.log('[DocumentService] RAGFlow Upload success:', {
        doc_id: result.doc_id,
        filename: result.filename,
        chunks: result.total_chunks,
        embeddings: result.total_embeddings,
        raptor_enabled: result.raptor_enabled,
        processing_time: result.processing_time,
        status: result.status
      })
      
      // Validate that the response contains expected data
      if (!result.doc_id || !result.filename) {
        throw new Error(`Invalid response from backend: missing required fields (doc_id: ${result.doc_id}, filename: ${result.filename})`)
      }
      
      if (result.status && result.status.toLowerCase().includes('error')) {
        throw new Error(`Backend processing failed: ${result.status}`)
      }

      // Update document count in localStorage
      KnowledgeBaseService.incrementDocumentCount(data.datasetId)

      return result
    } catch (error) {
      console.error('[DocumentService] Document upload failed:', error)
      throw error
    }
  }

  // Get documents for a knowledge base (this would call your existing API)
  static async getDocuments(datasetId: string, page: number = 1, pageSize: number = 20) {
    // Verify the knowledge base exists in localStorage
    if (!KnowledgeBaseService.exists(datasetId)) {
      throw new Error(`Knowledge base with ID "${datasetId}" not found`)
    }

    return apiRequest(`/kb/${datasetId}/documents?tenant_id=test_tenant&page=${page}&page_size=${pageSize}`)
  }

  // Validate if a file is a markdown file
  static validateMarkdownFile(file: File): boolean {
    const allowedTypes = ['text/markdown', 'text/x-markdown', 'application/x-markdown']
    const allowedExtensions = ['.md', '.markdown']
    
    // Check file type
    if (allowedTypes.includes(file.type)) {
      return true
    }
    
    // Check file extension
    const fileName = file.name.toLowerCase()
    return allowedExtensions.some(ext => fileName.endsWith(ext))
  }

  // Get file size in MB
  static getFileSizeMB(file: File): number {
    return file.size / (1024 * 1024)
  }
}