import { KnowledgeContent } from '../components/organisms'
import { KnowledgePageTemplate } from '../components/templates'

export const KnowledgePage = () => {
  return (
    <KnowledgePageTemplate>
      <KnowledgeContent userName="John" />
    </KnowledgePageTemplate>
  )
}
