import { Input, Textarea } from '@heroui/react'
import { forwardRef } from 'react'

export interface BaseFormFieldProps {
  label: string
  placeholder?: string
  isRequired?: boolean
  disabled?: boolean
  isInvalid?: boolean
  errorMessage?: string
  className?: string
  autoFocus?: boolean
  name?: string
  value?: string
  onChange?: (e: React.ChangeEvent<HTMLInputElement>) => void
  onBlur?: (e: React.FocusEvent<HTMLInputElement>) => void
}

export interface InputFormFieldProps extends BaseFormFieldProps {
  type?: 'input'
}

export interface TextareaFormFieldProps extends BaseFormFieldProps {
  type: 'textarea'
  minRows?: number
  maxRows?: number
}

export type FormFieldProps = InputFormFieldProps | TextareaFormFieldProps

// Internal Input component
const InputField = forwardRef<HTMLInputElement, InputFormFieldProps>(
  ({ value, onChange, onBlur, name, ...props }, ref) => {
    return (
      <Input
        label={props.label}
        placeholder={props.placeholder}
        variant="bordered"
        size="lg"
        isRequired={props.isRequired}
        disabled={props.disabled}
        isInvalid={props.isInvalid}
        errorMessage={props.errorMessage}
        className={props.className || 'w-full'}
        autoFocus={props.autoFocus}
        name={name}
        value={value}
        onChange={onChange}
        onBlur={onBlur}
        ref={ref}
      />
    )
  }
)

// Internal Textarea component
const TextareaField = forwardRef<HTMLInputElement, TextareaFormFieldProps>(
  ({ minRows, maxRows, value, onChange, onBlur, name, ...props }, ref) => {
    return (
      <Textarea
        label={props.label}
        placeholder={props.placeholder}
        variant="bordered"
        size="lg"
        isRequired={props.isRequired}
        disabled={props.disabled}
        isInvalid={props.isInvalid}
        errorMessage={props.errorMessage}
        className={props.className || 'w-full'}
        autoFocus={props.autoFocus}
        minRows={minRows}
        maxRows={maxRows}
        name={name}
        value={value}
        onChange={onChange}
        onBlur={onBlur}
        // @ts-expect-error - HeroUI Textarea expects different ref type
        ref={ref}
      />
    )
  }
)

// Main FormField component
export const FormField = forwardRef<HTMLInputElement, FormFieldProps>(
  ({ type = 'input', ...props }, ref) => {
    if (type === 'textarea') {
      const textareaProps = props as TextareaFormFieldProps
      return <TextareaField {...textareaProps} ref={ref} />
    }

    const inputProps = props as InputFormFieldProps
    return <InputField {...inputProps} ref={ref} />
  }
)

InputField.displayName = 'InputField'
TextareaField.displayName = 'TextareaField'
FormField.displayName = 'FormField'
