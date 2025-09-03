import { Container, Section } from '@radix-ui/themes'
import { CreateKnowledgeModal, SearchAndCreateBar, KnowledgeCardsGrid } from '../../molecules'
import { WelcomeSection, EmptyState, ConfirmDialog } from '../../atoms'
import { type CreateKnowledgeFormData } from '../../../schemas'
import { useDatasets } from '../../../hooks'
import { KnowledgeBaseService, DatasetService, type KnowledgeBase } from '../../../services'
import { useToast } from '../../../hooks'
import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'

interface KnowledgeContentProps {
  userName?: string
  className?: string
}

export const KnowledgeContent = ({ userName = 'John', className }: KnowledgeContentProps) => {
  const [searchQuery, setSearchQuery] = useState('')
  const [isCreateModalOpen, setIsCreateModalOpen] = useState(false)
  const [localKnowledgeBases, setLocalKnowledgeBases] = useState<KnowledgeBase[]>([])
  const [isInitialLoading, setIsInitialLoading] = useState(true)
  const [deleteConfirm, setDeleteConfirm] = useState<{
    isOpen: boolean
    id: string | null
    name: string
  }>({ isOpen: false, id: null, name: '' })
  const [isDeleting, setIsDeleting] = useState(false)
  
  // Toast notifications
  const toast = useToast()
  
  // Use real backend API data for fetching (not used directly in UI but needed for sync)
  const { datasets, loading, error, refetch } = useDatasets()

  // IMPLEMENT USER'S FLOW: Fetch backend data on page load, clear localStorage, save new data
  useEffect(() => {
    const initializeKnowledgeData = async () => {
      try {
        setIsInitialLoading(true)
        
        // Step 1: Fetch data from backend and sync to localStorage
        const syncedData = await KnowledgeBaseService.syncWithBackend()
        setLocalKnowledgeBases(syncedData)
        
      } catch (error) {
        console.error('Failed to sync knowledge base data:', error)
        const existingData = KnowledgeBaseService.getAll()
        setLocalKnowledgeBases(existingData)
      } finally {
        setIsInitialLoading(false)
      }
    }

    initializeKnowledgeData()
  }, []) // Only run once on component mount

  // Update local state when new data is available
  useEffect(() => {
    if (!loading && !isInitialLoading) {
      const currentLocalData = KnowledgeBaseService.getAll()
      setLocalKnowledgeBases(currentLocalData)
    }
  }, [datasets, loading, isInitialLoading])

  // Use local knowledge bases from localStorage (synced with backend)
  const transformedDatasets = localKnowledgeBases.map(kb => ({
    id: kb.id,
    title: kb.name,
    description: kb.description,
    documentCount: kb.documentCount,
    createdAt: kb.createdAt
  }))

  const filteredKnowledgeBases = transformedDatasets.filter(kb =>
    kb.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
    kb.description.toLowerCase().includes(searchQuery.toLowerCase())
  )

  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        duration: 0.6,
        staggerChildren: 0.1
      }
    }
  }

  const handleCreateKnowledge = async (data: CreateKnowledgeFormData) => {
    try {
      // Create KB via Backend API + sync to localStorage
      const newKB = await KnowledgeBaseService.createKB({
        name: data.name,
        description: data.description
      })

      console.log('Knowledge base created via backend:', newKB)

      // Update local state to reflect the new knowledge base
      setLocalKnowledgeBases(prev => [...prev, newKB])

      // Show success toast notification
      toast.success(
        `Knowledge base "${data.name}" created!`,
        `ID: ${newKB.id} - Synced to database and ready for uploads`
      )

      // Close modal
      setIsCreateModalOpen(false)
    } catch (error) {
      console.error('Failed to create knowledge base:', error)
      const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred'
      toast.error(
        'Failed to create knowledge base',
        `Backend error: ${errorMessage}`
      )
      throw error // Re-throw to let the modal handle the error state
    }
  }

  const handleRename = async (id: string, newName: string) => {
    try {
      // Update in localStorage
      KnowledgeBaseService.updateKnowledgeBase(id, { name: newName })
      
      // Update local state
      setLocalKnowledgeBases(prev => 
        prev.map(kb => kb.id === id ? { ...kb, name: newName } : kb)
      )
      
      toast.success('Knowledge base renamed successfully', `Name updated to "${newName}"`)
    } catch (error) {
      console.error('Failed to rename knowledge base:', error)
      toast.error('Failed to rename knowledge base', 'Please try again')
    }
  }

  const handleDelete = (id: string) => {
    const kb = localKnowledgeBases.find(item => item.id === id)
    if (kb) {
      setDeleteConfirm({ isOpen: true, id, name: kb.name })
    }
  }

  const handleConfirmDelete = async () => {
    if (!deleteConfirm.id) return
    
    setIsDeleting(true)
    try {
      // Delete from backend
      await DatasetService.deleteDataset(deleteConfirm.id)
      
      // Remove from localStorage
      KnowledgeBaseService.removeKnowledgeBase(deleteConfirm.id)
      
      // Update local state
      setLocalKnowledgeBases(prev => prev.filter(kb => kb.id !== deleteConfirm.id))
      
      toast.success(
        'Knowledge base deleted successfully',
        `"${deleteConfirm.name}" has been permanently removed`
      )
    } catch (error) {
      console.error('Failed to delete knowledge base:', error)
      toast.error(
        'Failed to delete knowledge base',
        'Please try again or contact support'
      )
    } finally {
      setIsDeleting(false)
      setDeleteConfirm({ isOpen: false, id: null, name: '' })
    }
  }

  return (
    <Section className={`py-8 overflow-visible ${className || ''}`}>
      <Container size="4" className="overflow-visible">
        <motion.div
          variants={containerVariants}
          initial="hidden"
          animate="visible"
          className="overflow-visible"
        >
          {/* Welcome Section */}
          <WelcomeSection userName={userName} />

          {/* Search and Create Section */}
          <SearchAndCreateBar
            searchQuery={searchQuery}
            onSearchChange={setSearchQuery}
            onCreateClick={() => setIsCreateModalOpen(true)}
          />

          {/* Loading State - Show Skeleton Cards */}
          {(loading || isInitialLoading) && (
            <KnowledgeCardsGrid 
              knowledgeBases={[]}
              isLoading={true}
              onRename={handleRename}
              onDelete={handleDelete}
            />
          )}

          {/* Error State */}
          {error && !loading && !isInitialLoading && (
            <div className="flex justify-center items-center py-12">
              <div className="text-red-500">
                Error loading knowledge bases: {error}
                <button 
                  onClick={refetch}
                  className="ml-2 px-3 py-1 bg-blue-500 text-white rounded text-sm hover:bg-blue-600"
                >
                  Retry
                </button>
              </div>
            </div>
          )}

          {/* Knowledge Cards Grid */}
          {!loading && !isInitialLoading && !error && (
            <KnowledgeCardsGrid 
              knowledgeBases={filteredKnowledgeBases}
              isLoading={false}
              onRename={handleRename}
              onDelete={handleDelete}
            />
          )}

          {/* Empty State */}
          {!loading && !isInitialLoading && !error && filteredKnowledgeBases.length === 0 && (
            <EmptyState
              searchQuery={searchQuery}
              onClearSearch={() => setSearchQuery('')}
            />
          )}
        </motion.div>
      </Container>

      {/* Create Knowledge Modal */}
      <CreateKnowledgeModal
        isOpen={isCreateModalOpen}
        onClose={() => setIsCreateModalOpen(false)}
        onSubmit={handleCreateKnowledge}
      />

      {/* Delete Confirmation Dialog */}
      <ConfirmDialog
        isOpen={deleteConfirm.isOpen}
        onClose={() => setDeleteConfirm({ isOpen: false, id: null, name: '' })}
        onConfirm={handleConfirmDelete}
        title="Delete Knowledge Base"
        message={`Are you sure you want to delete "${deleteConfirm.name}"? This action cannot be undone and will permanently remove all associated data.`}
        confirmText="Delete"
        cancelText="Cancel"
        isLoading={isDeleting}
        confirmColor="danger"
      />
    </Section>
  )
}
