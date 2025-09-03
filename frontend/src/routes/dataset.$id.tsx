import { createFileRoute } from '@tanstack/react-router'
import { DatasetPage } from '../pages'

export const Route = createFileRoute('/dataset/$id')({
  component: () => <DatasetPage />
})
