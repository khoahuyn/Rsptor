import { StatCard } from '../../atoms'

interface Stat {
  value: string
  label: string
}

interface StatsGridProps {
  stats: Stat[]
  className?: string
}

export const StatsGrid = ({ stats, className }: StatsGridProps) => {
  return (
    <div className={`grid grid-cols-1 md:grid-cols-3 gap-8 max-w-2xl mx-auto ${className || ''}`}>
      {stats.map((stat, index) => (
        <StatCard
          key={index}
          value={stat.value}
          label={stat.label}
        />
      ))}
    </div>
  )
}
