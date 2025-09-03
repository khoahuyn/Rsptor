import React, { useState } from 'react'
import * as Toast from '@radix-ui/react-toast'
import { ToastContext, type ToastContextType } from './ToastContextTypes'

interface ToastItem {
  id: string
  type: 'success' | 'error' | 'info' | 'warning'
  title: string
  description?: string
}



interface ToastProviderProps {
  children: React.ReactNode
}

export const ToastProvider: React.FC<ToastProviderProps> = ({ children }) => {
  const [toasts, setToasts] = useState<ToastItem[]>([])

  const addToast = (type: ToastItem['type'], title: string, description?: string) => {
    const id = Math.random().toString(36).substr(2, 9)
    const newToast: ToastItem = { id, type, title, description }
    setToasts(prev => [...prev, newToast])
    
    // Auto remove after 5 seconds
    setTimeout(() => {
      setToasts(prev => prev.filter(toast => toast.id !== id))
    }, 5000)
  }

  const removeToast = (id: string) => {
    setToasts(prev => prev.filter(toast => toast.id !== id))
  }

  const contextValue: ToastContextType = {
    success: (title, description) => addToast('success', title, description),
    error: (title, description) => addToast('error', title, description),
    info: (title, description) => addToast('info', title, description),
    warning: (title, description) => addToast('warning', title, description),
  }

  const getToastStyles = (type: ToastItem['type']) => {
    const baseStyles = 'relative bg-white border rounded-lg shadow-lg p-4 max-w-sm w-full'
    
    switch (type) {
      case 'success':
        return `${baseStyles} border-green-200 bg-green-50`
      case 'error':
        return `${baseStyles} border-red-200 bg-red-50`
      case 'warning':
        return `${baseStyles} border-yellow-200 bg-yellow-50`
      case 'info':
        return `${baseStyles} border-blue-200 bg-blue-50`
      default:
        return baseStyles
    }
  }

  const getTitleStyles = (type: ToastItem['type']) => {
    switch (type) {
      case 'success':
        return 'text-green-800 font-semibold'
      case 'error':
        return 'text-red-800 font-semibold'
      case 'warning':
        return 'text-yellow-800 font-semibold'
      case 'info':
        return 'text-blue-800 font-semibold'
      default:
        return 'text-gray-800 font-semibold'
    }
  }

  const getDescriptionStyles = (type: ToastItem['type']) => {
    switch (type) {
      case 'success':
        return 'text-green-600 text-sm mt-1'
      case 'error':
        return 'text-red-600 text-sm mt-1'
      case 'warning':
        return 'text-yellow-600 text-sm mt-1'
      case 'info':
        return 'text-blue-600 text-sm mt-1'
      default:
        return 'text-gray-600 text-sm mt-1'
    }
  }

  return (
    <ToastContext.Provider value={contextValue}>
      {children}
      <Toast.Provider swipeDirection="right">
        {toasts.map((toast) => (
          <Toast.Root
            key={toast.id}
            className={getToastStyles(toast.type)}
            onOpenChange={(open) => !open && removeToast(toast.id)}
          >
            <div className="flex items-start justify-between">
              <div className="flex-1">
                <Toast.Title className={getTitleStyles(toast.type)}>
                  {toast.title}
                </Toast.Title>
                {toast.description && (
                  <Toast.Description className={getDescriptionStyles(toast.type)}>
                    {toast.description}
                  </Toast.Description>
                )}
              </div>
              <Toast.Close className="ml-4 opacity-70 hover:opacity-100 transition-opacity">
                <svg width="16" height="16" viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg">
                  <path d="M12 4L4 12M4 4L12 12" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                </svg>
              </Toast.Close>
            </div>
          </Toast.Root>
        ))}
        <Toast.Viewport className="fixed bottom-0 right-0 flex flex-col p-6 gap-2 w-96 max-w-[100vw] m-0 list-none z-50" />
      </Toast.Provider>
    </ToastContext.Provider>
  )
}