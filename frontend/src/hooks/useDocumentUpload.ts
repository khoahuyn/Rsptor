import { useState, useCallback } from 'react'
import { DocumentService } from '../services'
import { useToast } from './useToast'

export interface UploadFileData {
  file: File
  source?: string
  tags?: string[]
  extraMeta?: string // JSON string instead of Record<string, any>
  buildTree?: boolean
  summaryLLM?: string
  vectorIndex?: string // JSON string instead of Record<string, any>
  upsertMode?: 'upsert' | 'replace' | 'skip_duplicates'
}

export interface UseDocumentUploadOptions {
  datasetId: string
  onSuccess?: (uploadedFiles: string[]) => void
  onError?: (error: Error) => void
}

export interface UseDocumentUploadReturn {
  uploadDocuments: (files: UploadFileData[]) => Promise<void>
  isUploading: boolean
  uploadProgress: Record<string, number>
  validateMarkdownFile: (file: File) => { isValid: boolean; error?: string }
  getFileSizeMB: (file: File) => number
}

export const useDocumentUpload = ({
  datasetId,
  onSuccess,
  onError
}: UseDocumentUploadOptions): UseDocumentUploadReturn => {
  const [isUploading, setIsUploading] = useState(false)
  const [uploadProgress, setUploadProgress] = useState<Record<string, number>>({})
  const toast = useToast()

  // Frontend validation for Markdown files
  const validateMarkdownFile = useCallback((file: File): { isValid: boolean; error?: string } => {
    // Check file extension
    const fileName = file.name.toLowerCase()
    if (!fileName.endsWith('.md') && !fileName.endsWith('.markdown')) {
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

    // Basic MIME type check (though browsers may not set this correctly for .md files)
    const allowedMimeTypes = [
      'text/markdown',
      'text/plain',
      'text/x-markdown',
      'application/octet-stream' // Some browsers use this for .md files
    ]
    
    if (file.type && !allowedMimeTypes.includes(file.type)) {
      console.warn(`Unexpected MIME type: ${file.type} for file: ${file.name}`)
      // Don't fail validation based on MIME type alone, as it's unreliable for .md files
    }

    return { isValid: true }
  }, [])

  const getFileSizeMB = useCallback((file: File): number => {
    return file.size / (1024 * 1024)
  }, [])

  const uploadDocuments = useCallback(async (files: UploadFileData[]): Promise<void> => {
    if (!datasetId) {
      throw new Error('Dataset ID is required for upload')
    }

    if (files.length === 0) {
      throw new Error('No files selected for upload')
    }

    // Debug: Log input data
    console.log('[useDocumentUpload] 🚀 PARALLEL UploadDocuments called with:', { datasetId, filesCount: files.length, files })

    setIsUploading(true)
    setUploadProgress({})

    const uploadedFiles: string[] = []
    const failedFiles: { fileName: string; error: string }[] = []

    try {
      // Initialize progress for all files
      const initialProgress = files.reduce((acc, fileData) => {
        acc[fileData.file.name] = 0
        return acc
      }, {} as Record<string, number>)
      setUploadProgress(initialProgress)

      // Create upload function for individual files
      const uploadSingleFile = async (fileData: UploadFileData, index: number) => {
        const fileName = fileData.file.name

        try {
          // Frontend validation
          const validation = validateMarkdownFile(fileData.file)
          if (!validation.isValid) {
            throw new Error(validation.error || 'File validation failed')
          }

          // Debug: Log each file upload data
          console.log(`[useDocumentUpload] Processing file ${index + 1}/${files.length}:`, {
            fileName: fileName,
            fileSize: fileData.file.size,
            fileType: fileData.file.type,
            datasetId: datasetId,
            source: fileData.source,
            tags: fileData.tags,
            extraMeta: fileData.extraMeta,
            buildTree: fileData.buildTree,
            summaryLLM: fileData.summaryLLM,
            vectorIndex: fileData.vectorIndex,
            upsertMode: fileData.upsertMode
          })

          // Prepare upload data according to the API specification
          const uploadPayload = {
            file: fileData.file,
            datasetId: datasetId,
            source: fileData.source || `Frontend upload: ${fileName}`,
            tags: fileData.tags || ['frontend-upload'],
            extraMeta: fileData.extraMeta ? fileData.extraMeta : JSON.stringify({
              uploadedBy: 'frontend',
              timestamp: new Date().toISOString(),
              originalFileName: fileName
            }),
            buildTree: fileData.buildTree ?? true,
            summaryLLM: fileData.summaryLLM || 'DeepSeek-V3',
            vectorIndex: fileData.vectorIndex ? fileData.vectorIndex : undefined,
            upsertMode: fileData.upsertMode || 'upsert'
          }

          // Debug: Log prepared upload payload
          console.log(`[useDocumentUpload] Prepared upload payload for ${fileName}:`, uploadPayload)

          // Update progress to show upload started
          setUploadProgress(prev => ({ ...prev, [fileName]: 25 }))

          // Call the document service to upload
          const result = await DocumentService.uploadDocument(uploadPayload)
          
          // Complete progress
          setUploadProgress(prev => ({ ...prev, [fileName]: 100 }))
          
          console.log(`[useDocumentUpload] ${fileName} uploaded successfully:`, result)
          
          toast.success(
            `Document uploaded successfully`,
            `"${fileName}" processed: ${result.total_chunks} chunks, ${result.total_embeddings} embeddings${result.raptor_enabled ? ', RAPTOR enabled' : ''}`
          )

          return {
            fileName,
            result,
            success: true
          }

        } catch (error) {
          const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred'
          
          console.error(`Failed to upload ${fileName}:`, error)
          
          toast.error(
            `Failed to upload "${fileName}"`,
            errorMessage
          )

          return {
            fileName,
            error: errorMessage,
            success: false
          }
        }
      }

      // 🚀 PARALLEL UPLOAD: Process all files simultaneously
      console.log(`[useDocumentUpload] Starting PARALLEL upload of ${files.length} files...`)
      
      const uploadResults = await Promise.allSettled(
        files.map((fileData, index) => uploadSingleFile(fileData, index))
      )

      // Process results
      uploadResults.forEach((result, index) => {
        if (result.status === 'fulfilled') {
          const uploadResult = result.value
          if (uploadResult.success) {
            uploadedFiles.push(uploadResult.result.doc_id || uploadResult.fileName)
          } else {
            failedFiles.push({ 
              fileName: uploadResult.fileName, 
              error: uploadResult.error 
            })
          }
        } else {
          // Promise itself failed (shouldn't happen with our error handling)
          const fileName = files[index]?.file.name || `File ${index + 1}`
          failedFiles.push({ 
            fileName, 
            error: result.reason?.message || 'Upload promise failed' 
          })
        }
      })

      // Handle results immediately (no delay - backend handles async RAPTOR)
      console.log(`[useDocumentUpload] Upload results: ${uploadedFiles.length} success, ${failedFiles.length} failed`)
      
      if (uploadedFiles.length > 0) {
        console.log(`[useDocumentUpload] Successful uploads:`, uploadedFiles)
        onSuccess?.(uploadedFiles)
      }

      if (failedFiles.length > 0) {
        console.error(`[useDocumentUpload] Failed uploads:`, failedFiles)
        const errorSummary = `${failedFiles.length} file(s) failed to upload: ${failedFiles.map(f => f.fileName).join(', ')}`
        const combinedError = new Error(errorSummary)
        onError?.(combinedError)
        
        // Show detailed error for debugging
        failedFiles.forEach(f => {
          console.error(`[useDocumentUpload] ${f.fileName} failed:`, f.error)
        })
      }

      // Show final summary for multiple files (parallel processing)
      if (files.length > 1) {
        if (uploadedFiles.length === files.length) {
          toast.success(
            `🚀 All ${files.length} documents uploaded successfully!`,
            `Parallel processing completed - All files processed and added to the knowledge base`
          )
        } else if (uploadedFiles.length > 0) {
          toast.warning(
            '⚡ Partial upload completed',
            `${uploadedFiles.length} of ${files.length} file(s) uploaded successfully via parallel processing`
          )
        } else {
          toast.error(
            '❌ All uploads failed',
            `None of the ${files.length} files could be processed. Check individual error messages above.`
          )
        }
      }

    } catch (error) {
      console.error('Parallel upload process failed:', error)
      const errorMessage = error instanceof Error ? error.message : 'Parallel upload process failed'
      toast.error('Parallel upload failed', errorMessage)
      onError?.(error instanceof Error ? error : new Error(errorMessage))
    } finally {
      setIsUploading(false)
      // Clear progress immediately to remove "Uploading" UI
      setUploadProgress({})
    }
  }, [datasetId, validateMarkdownFile, toast, onSuccess, onError])

  return {
    uploadDocuments,
    isUploading,
    uploadProgress,
    validateMarkdownFile,
    getFileSizeMB
  }
}