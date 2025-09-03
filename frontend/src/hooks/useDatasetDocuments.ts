import { useState, useEffect } from 'react'
import { DatasetService, type Document, type DocumentsResponse, type PaginationInfo } from '../services'

export interface UseDatasetDocumentsState {
  documents: Document[]
  pagination: PaginationInfo | null
  loading: boolean
  error: string | null
}

export const useDatasetDocuments = (datasetId: string | null, page: number = 1, pageSize: number = 20) => {
  const [state, setState] = useState<UseDatasetDocumentsState>({
    documents: [],
    pagination: null,
    loading: false,
    error: null
  })

  const fetchDocuments = async () => {
    if (!datasetId) {
      setState(prev => ({ ...prev, documents: [], pagination: null, loading: false }))
      return
    }

    try {
      setState(prev => ({ ...prev, loading: true, error: null }))
      const response: DocumentsResponse = await DatasetService.getDatasetDocuments(datasetId, page, pageSize)
      
      setState({
        documents: response.documents,
        pagination: response.pagination,
        loading: false,
        error: null
      })
    } catch (error) {
      console.error('Failed to fetch documents:', error)
      setState(prev => ({
        ...prev,
        loading: false,
        error: error instanceof Error ? error.message : 'Failed to fetch documents'
      }))
    }
  }

  const refetch = () => {
    fetchDocuments()
  }

  useEffect(() => {
    fetchDocuments()
  }, [datasetId, page, pageSize])

  return {
    ...state,
    refetch
  }
}