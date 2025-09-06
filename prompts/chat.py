RAG_SYSTEM_PROMPT = """Answer questions based on the provided information from the knowledge base. Respond in the same language as the user's question."""

EXCERPT_EXTRACTION_PROMPT = """Given the following answer and source content, extract the most relevant excerpts from the content that support the answer.

ANSWER: {answer}

SOURCE CONTENT:
{content}

INSTRUCTIONS:
1. Find all sentences/phrases in the source content that directly relate to any part of the answer
2. Extract complete sentences, not partial phrases
3. Combine the excerpts naturally with spaces
4. Keep the total length under 300 characters
5. Return only the relevant excerpts, no explanations

RELEVANT EXCERPTS:"""