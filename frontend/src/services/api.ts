// API configuration and base service
const API_BASE_URL = 'http://localhost:8081'
const API_VERSION = 'v1'

export const createApiUrl = (endpoint: string): string => {
  return `${API_BASE_URL}/${API_VERSION}${endpoint}`
}

export const apiRequest = async <T>(
  endpoint: string,
  options?: RequestInit
): Promise<T> => {
  const url = createApiUrl(endpoint)
  
  const defaultOptions: RequestInit = {
    headers: {
      'Content-Type': 'application/json',
    },
  }

  const config = { ...defaultOptions, ...options }

  try {
    const response = await fetch(url, config)
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`)
    }

    return await response.json()
  } catch (error) {
    console.error('API request failed:', error)
    throw error
  }
}