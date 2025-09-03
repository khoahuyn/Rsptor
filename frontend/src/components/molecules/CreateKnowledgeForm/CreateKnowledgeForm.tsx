import { useForm } from 'react-hook-form'
import { zodResolver } from '@hookform/resolvers/zod'
import { useState, useRef, forwardRef, useImperativeHandle } from 'react'
import { createKnowledgeSchema, type CreateKnowledgeFormData } from '../../../schemas'
import { FormField } from '../../atoms/FormField'

export interface CreateKnowledgeFormProps {
  onSubmit: (data: CreateKnowledgeFormData) => Promise<void>
  disabled?: boolean
  autoFocus?: boolean
}

export interface CreateKnowledgeFormRef {
  submit: () => Promise<void>
  reset: () => void
  isValid: boolean
  isSubmitting: boolean
}

export const CreateKnowledgeForm = forwardRef<CreateKnowledgeFormRef, CreateKnowledgeFormProps>(
  ({ onSubmit, disabled = false, autoFocus = true }, ref) => {
    const [isSubmitting, setIsSubmitting] = useState(false)
    const [hasAttemptedSubmit, setHasAttemptedSubmit] = useState(false)
    const hasAttemptedSubmitRef = useRef(false)

    const {
      register,
      handleSubmit,
      reset,
      trigger,
      formState: { errors, isValid }
    } = useForm<CreateKnowledgeFormData>({
      // @ts-expect-error - Zod version compatibility issue
      resolver: zodResolver(createKnowledgeSchema),
      mode: hasAttemptedSubmit ? 'onChange' : 'onSubmit', // Dynamic validation mode
      reValidateMode: 'onChange', // After first validation, revalidate on change
      defaultValues: {
        name: '',
        description: ''
      }
    })

    const onSubmitForm = async (data: CreateKnowledgeFormData) => {
      setIsSubmitting(true)
      try {
        await onSubmit(data)
      } catch (error) {
        console.error('Error creating knowledge base:', error)
        throw error
      } finally {
        setIsSubmitting(false)
      }
    }

    const handleFormSubmit = async () => {
      // Mark that user has attempted to submit
      hasAttemptedSubmitRef.current = true
      setHasAttemptedSubmit(true) // Force re-render

      // Validate all fields first
      const isFormValid = await trigger()
      
      if (isFormValid) {
        // If form is valid, submit it
        const submitHandler = handleSubmit(
          // onValid - this runs if form is valid
          onSubmitForm,
          // onInvalid - this runs if form has errors
          (errors) => {
            console.log('Form validation errors:', errors)
            // Errors will be automatically displayed through formState.errors
          }
        )
        
        await submitHandler()
      }
    }

    const resetForm = () => {
      reset()
      setIsSubmitting(false)
      setHasAttemptedSubmit(false)
      hasAttemptedSubmitRef.current = false
    }

    // Expose form methods through ref
    useImperativeHandle(ref, () => ({
      submit: handleFormSubmit,
      reset: resetForm,
      isValid,
      isSubmitting
    }))

    return (
      <div className="space-y-4">
        <div>
          <FormField
            {...register('name')}
            label="Name"
            placeholder="Enter a name for your knowledge base"
            isRequired
            autoFocus={autoFocus}
            disabled={disabled || isSubmitting}
            isInvalid={!!errors.name}
            errorMessage={errors.name?.message}
          />
        </div>

        <div>
          <FormField
            {...register('description')}
            type="textarea"
            label="Description"
            placeholder="Describe what this knowledge base contains..."
            minRows={3}
            maxRows={5}
            disabled={disabled || isSubmitting}
            isInvalid={!!errors.description}
            errorMessage={errors.description?.message}
          />
        </div>
      </div>
    )
  }
)

CreateKnowledgeForm.displayName = 'CreateKnowledgeForm'
