import { useState, useEffect } from 'react'
import { DatasetService, type Dataset, type DatasetsResponse } from '../services'

export interface UseDatasetState {
  datasets: Dataset[]
  loading: boolean
  error: string | null
  total: number
}

export const useDatasets = () => {
  const [state, setState] = useState<UseDatasetState>({
    datasets: [],
    loading: true,
    error: null,
    total: 0
  })

  const fetchDatasets = async () => {
    try {
      setState(prev => ({ ...prev, loading: true, error: null }))
      const response: DatasetsResponse = await DatasetService.getDatasets()
      
      setState({
        datasets: response.datasets,
        total: response.total,
        loading: false,
        error: null
      })
    } catch (error) {
      console.error('Failed to fetch datasets:', error)
      setState(prev => ({
        ...prev,
        loading: false,
        error: error instanceof Error ? error.message : 'Failed to fetch datasets'
      }))
    }
  }

  const refetch = () => {
    fetchDatasets()
  }

  useEffect(() => {
    fetchDatasets()
  }, [])

  return {
    ...state,
    refetch
  }
}