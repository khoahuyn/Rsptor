/**
 * File validation utilities for Markdown documents
 */

export interface FileValidationResult {
  isValid: boolean
  error?: string
  warnings?: string[]
}

/**
 * Validates if a file is a proper Markdown document
 * @param file The file to validate
 * @returns Validation result with status and optional error/warnings
 */
export const validateMarkdownFile = (file: File): FileValidationResult => {
  const result: FileValidationResult = { isValid: true, warnings: [] }

  // Check file extension
  const fileName = file.name.toLowerCase()
  const validExtensions = ['.md', '.markdown']
  const hasValidExtension = validExtensions.some(ext => fileName.endsWith(ext))
  
  if (!hasValidExtension) {
    return {
      isValid: false,
      error: 'File must have .md or .markdown extension'
    }
  }

  // Check file size (max 10MB)
  const maxSizeBytes = 10 * 1024 * 1024 // 10MB
  if (file.size > maxSizeBytes) {
    return {
      isValid: false,
      error: `File size (${(file.size / 1024 / 1024).toFixed(2)}MB) exceeds maximum limit of 10MB`
    }
  }

  // Check if file is empty
  if (file.size === 0) {
    return {
      isValid: false,
      error: 'File cannot be empty'
    }
  }

  // Check for suspicious content (basic checks)
  if (file.size > 100) {
    const fileNameWithoutExt = fileName.substring(0, fileName.lastIndexOf('.'))
    
    // Warn if filename contains suspicious patterns
    if (fileNameWithoutExt.includes('..') || fileNameWithoutExt.includes('/')) {
      result.warnings?.push('Filename contains unusual characters')
    }
  }

  return result
}

/**
 * Gets file size in MB
 * @param file The file to measure
 * @returns Size in megabytes
 */
export const getFileSizeMB = (file: File): number => {
  return file.size / (1024 * 1024)
}

/**
 * Gets file extension
 * @param file The file to check
 * @returns File extension (e.g., '.md')
 */
export const getFileExtension = (file: File): string => {
  const parts = file.name.split('.')
  return parts.length > 1 ? `.${parts[parts.length - 1].toLowerCase()}` : ''
}

/**
 * Checks if file is a Markdown file based on extension
 * @param file The file to check
 * @returns True if file has .md or .markdown extension
 */
export const isMarkdownFile = (file: File): boolean => {
  const ext = getFileExtension(file)
  return ext === '.md' || ext === '.markdown'
}