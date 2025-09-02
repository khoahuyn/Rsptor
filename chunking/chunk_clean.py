import re
from typing import List
from utils import token_count


def clean_content(text: str, preserve_structure: bool = True) -> str:
    """Clean text content - remove URLs, citations, markdown artifacts"""
    if not text:
        return text
    
    # Remove code blocks first (before other markdown processing)
    text = re.sub(r'```[\s\S]*?```', '', text)  # Code blocks
    text = re.sub(r'`[^`]+`', '', text)  # Inline code
    
    # Remove markdown headers (keep text)
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    
    # Remove emphasis markup (bold/italic)
    text = re.sub(r'\*\*([^\*]+)\*\*', r'\1', text)  # Bold **text**
    text = re.sub(r'\*([^\*]+)\*', r'\1', text)      # Italic *text*
    text = re.sub(r'__([^_]+)__', r'\1', text)       # Bold __text__
    text = re.sub(r'_([^_]+)_', r'\1', text)         # Italic _text_
    
    # Remove list markers
    text = re.sub(r'^\s*[-*+]\s+', '', text, flags=re.MULTILINE)  # Bullet lists
    text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)  # Numbered lists
    
    # Remove URLs (both markdown and plain)
    text = re.sub(r'\(https?://[^\)]+\)', '', text)  # Remove (https://...)
    text = re.sub(r'https?://\S+', '', text)  # Remove plain URLs
    
    # Remove markdown links [text](url) - keep text
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)  # Keep text, remove URL
    text = re.sub(r'<[^>]+>', '', text)  # Remove HTML tags
    
    # Remove citations and references [1], [2], [citation needed]
    text = re.sub(r'\[\d+\]', '', text)
    text = re.sub(r'\[citation needed\]', '', text)
    text = re.sub(r'\[edit\]', '', text)
    
    # Remove "Archived from the original" patterns
    text = re.sub(r'Archived[^.]+from the original[^.]*\.', '', text)
    text = re.sub(r'Retrieved[^.]*\.', '', text)
    
    # Clean up table formatting artifacts
    text = re.sub(r'\|\s*\|\s*', ' ', text)  # Remove empty table cells
    text = re.sub(r'\|---\|', '', text)  # Remove table separators
    text = re.sub(r'\|', ' ', text)  # Convert remaining pipes to spaces
    
    # Clean up excessive whitespace
    if preserve_structure:
        # Keep paragraph structure
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)  # Max 2 consecutive newlines
    else:
        # Collapse all whitespace
        text = re.sub(r'\s+', ' ', text)
    
    text = re.sub(r'[ \t]+', ' ', text)  # Multiple spaces to single space
    text = re.sub(r'^\s+|\s+$', '', text, flags=re.MULTILINE)  # Trim lines
    
    return text.strip()


def force_split_large_text(text: str, max_tokens: int) -> List[str]:
    """Force split text that's too large, even without delimiters"""
    if not text or token_count(text) <= max_tokens:
        return [text]
    
    parts = []
    
    # Try splitting by sentences first (support multiple languages)
    sentences = re.split(r'([.!?。！？])', text)
    current_part = ""
    
    for i in range(0, len(sentences), 2):  # Process sentence + delimiter pairs
        sentence = sentences[i] if i < len(sentences) else ""
        delimiter = sentences[i + 1] if i + 1 < len(sentences) else ""
        
        test_part = current_part + sentence + delimiter
        
        if token_count(test_part) <= max_tokens:
            current_part = test_part
        else:
            # Save current part and start new one
            if current_part:
                parts.append(current_part.strip())
            current_part = sentence + delimiter
            
            # If single sentence is still too long, split by words
            if token_count(current_part) > max_tokens:
                word_parts = split_by_words(current_part, max_tokens)
                parts.extend(word_parts[:-1])  # Add all but last
                current_part = word_parts[-1] if word_parts else ""
    
    # Add remaining part
    if current_part:
        parts.append(current_part.strip())
    
    return [p for p in parts if p.strip()]


def split_by_words(text: str, max_tokens: int) -> List[str]:
    """Split text by words when sentences are too long"""
    words = text.split()
    parts = []
    current_part = ""
    
    for word in words:
        test_part = current_part + " " + word if current_part else word
        
        if token_count(test_part) <= max_tokens:
            current_part = test_part
        else:
            # Save current part and start new one
            if current_part:
                parts.append(current_part)
            current_part = word
    
    # Add remaining part
    if current_part:
        parts.append(current_part)
    
    return parts







