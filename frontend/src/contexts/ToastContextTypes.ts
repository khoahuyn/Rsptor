import { createContext } from 'react'

export interface ToastContextType {
  success: (title: string, description?: string) => void
  error: (title: string, description?: string) => void
  info: (title: string, description?: string) => void
  warning: (title: string, description?: string) => void
}

export const ToastContext = createContext<ToastContextType | undefined>(undefined)