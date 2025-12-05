/**
 * Premium Stats Grid Component
 * 
 * Inspired by Maxton Bootstrap 5 Admin Dashboard
 * Features: Responsive grid, animated stats, gradient accents
 */

import { LucideIcon } from 'lucide-react';
import { PremiumMetricCard } from './PremiumMetricCard';

export interface StatItem {
  title: string;
  value: string | number;
  change?: string;
  changeType?: 'positive' | 'negative' | 'neutral';
  icon: LucideIcon;
  iconColor?: string;
  gradient?: string;
}

interface PremiumStatsGridProps {
  stats: StatItem[];
  columns?: 2 | 3 | 4;
}

export function PremiumStatsGrid({ stats, columns = 4 }: PremiumStatsGridProps) {
  const gridCols = {
    2: 'grid-cols-1 md:grid-cols-2',
    3: 'grid-cols-1 md:grid-cols-2 lg:grid-cols-3',
    4: 'grid-cols-1 md:grid-cols-2 lg:grid-cols-4',
  };

  return (
    <div className={`grid ${gridCols[columns]} gap-6`}>
      {stats.map((stat, index) => (
        <PremiumMetricCard
          key={index}
          title={stat.title}
          value={stat.value}
          change={stat.change}
          changeType={stat.changeType}
          icon={stat.icon}
          iconColor={stat.iconColor}
          gradient={stat.gradient}
        />
      ))}
    </div>
  );
}
