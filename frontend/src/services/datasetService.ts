import { apiRequest } from './api'

// Types based on your API response
export interface Document {
  doc_id: string
  source: string
  tags: string[] | null
  extra_meta: Record<string, unknown> | null
  checksum: string
  created_at: string
  chunk_count?: number  // Real chunk count from backend
}

export interface PaginationInfo {
  page: number
  page_size: number
  total: number
  pages: number
}

export interface DocumentsResponse {
  documents: Document[]
  pagination: PaginationInfo
}

export interface Dataset {
  id: string
  name: string
  description?: string
  document_count: number
  created_at: string
  last_updated?: string
}

export interface DatasetsResponse {
  datasets: Dataset[]
  total: number
}

export interface DatasetStatistics {
  document_count: number
  chunk_count: number
  embedding_count: number
  tree_node_count: number
  processing_status: string
  [key: string]: unknown
}

export interface DeleteDatasetResponse {
  message: string
  details: {
    documents_deleted: number
    chunks_deleted: number
    embeddings_deleted: number
    trees_deleted: number
  }
}

// KB Response interface from backend
interface KBResponse {
  kb_id: string
  tenant_id: string
  name: string
  description: string | null
  status: string
  document_count: number
  chunk_count: number
  total_tokens: number
  settings: Record<string, unknown> | null
  created_at: string
  updated_at: string | null
}

// Dataset API service
export class DatasetService {
  // Get all datasets
  static async getDatasets(): Promise<DatasetsResponse> {
    // Call KB list API instead of datasets API
    const kbList = await apiRequest<KBResponse[]>('/kb/list?tenant_id=test_tenant')
    
    // Transform KB format to Dataset format
    const datasets: Dataset[] = kbList.map(kb => ({
      id: kb.kb_id,
      name: kb.name,
      description: kb.description || '',
      document_count: kb.document_count || 0,
      created_at: kb.created_at,
      last_updated: kb.updated_at || undefined
    }))
    
    return {
      datasets,
      total: datasets.length
    }
  }

  // Get specific dataset details
  static async getDataset(datasetId: string): Promise<Dataset> {
    return apiRequest<Dataset>(`/datasets/${datasetId}`)
  }

  // Get documents in a dataset
  static async getDatasetDocuments(
    datasetId: string,
    page: number = 1,
    pageSize: number = 20
  ): Promise<DocumentsResponse> {
    return apiRequest<DocumentsResponse>(
      `/kb/${datasetId}/documents?tenant_id=test_tenant&page=${page}&page_size=${pageSize}`
    )
  }

  // Get dataset statistics
  static async getDatasetStatistics(datasetId: string): Promise<DatasetStatistics> {
    return apiRequest<DatasetStatistics>(`/datasets/${datasetId}/statistics`)
  }

  // Check if dataset exists
  static async datasetExists(datasetId: string): Promise<boolean> {
    try {
      const response = await fetch(`http://localhost:8081/v1/datasets/${datasetId}`, {
        method: 'HEAD',
      })
      return response.ok
    } catch {
      return false
    }
  }

  // Delete dataset
  static async deleteDataset(datasetId: string): Promise<DeleteDatasetResponse> {
    return apiRequest<DeleteDatasetResponse>(`/datasets/${datasetId}`, {
      method: 'DELETE',
    })
  }
}