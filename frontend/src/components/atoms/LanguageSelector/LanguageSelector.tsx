import { Button } from '@heroui/react'
import { ChevronDownIcon, GlobeIcon } from '@radix-ui/react-icons'
import { useState } from 'react'

interface LanguageSelectorProps {
  className?: string
}

const languages = [
  { code: 'en', name: 'English', flag: '🇺🇸' },
  { code: 'vi', name: 'Tiếng Việt', flag: '🇻🇳' }
]

export const LanguageSelector = ({ className }: LanguageSelectorProps) => {
  const [selectedLanguage, setSelectedLanguage] = useState(languages[0])
  const [isOpen, setIsOpen] = useState(false)

  return (
    <div className={`relative ${className || ''}`}>
      <Button
        variant="ghost"
        size="sm"
        className="gap-2 text-gray-600 hover:text-gray-900"
        onClick={() => setIsOpen(!isOpen)}
        endContent={<ChevronDownIcon className="w-4 h-4" />}
      >
        <GlobeIcon className="w-4 h-4" />
        <span className="hidden sm:inline">{selectedLanguage.name}</span>
        <span className="sm:hidden">{selectedLanguage.flag}</span>
      </Button>

      {isOpen && (
        <div className="absolute top-full right-0 mt-1 bg-white border border-gray-200 rounded-lg shadow-lg z-50 min-w-32">
          {languages.map((lang) => (
            <button
              key={lang.code}
              className="w-full px-3 py-2 text-left hover:bg-gray-50 flex items-center gap-2 first:rounded-t-lg last:rounded-b-lg"
              onClick={() => {
                setSelectedLanguage(lang)
                setIsOpen(false)
              }}
            >
              <span>{lang.flag}</span>
              <span className="text-sm">{lang.name}</span>
            </button>
          ))}
        </div>
      )}
    </div>
  )
}
