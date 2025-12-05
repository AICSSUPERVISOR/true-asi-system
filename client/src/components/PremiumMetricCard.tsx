/**
 * Premium Metric Card Component
 * 
 * Inspired by Maxton Bootstrap 5 Admin Dashboard
 * Features: Glass-morphism, gradient accents, animated counters
 */

import { LucideIcon } from 'lucide-react';
import { Card } from '@/components/ui/card';
import { cn } from '@/lib/utils';

interface PremiumMetricCardProps {
  title: string;
  value: string | number;
  change?: string;
  changeType?: 'positive' | 'negative' | 'neutral';
  icon: LucideIcon;
  iconColor?: string;
  gradient?: string;
  className?: string;
}

export function PremiumMetricCard({
  title,
  value,
  change,
  changeType = 'neutral',
  icon: Icon,
  iconColor = 'text-blue-500',
  gradient = 'from-blue-500/10 to-cyan-500/10',
  className,
}: PremiumMetricCardProps) {
  return (
    <Card
      className={cn(
        'relative overflow-hidden backdrop-blur-xl bg-card/50 border-border/50',
        'hover:bg-card/70 transition-all duration-300 hover:scale-[1.02]',
        'hover:shadow-lg hover:shadow-primary/20',
        className
      )}
    >
      {/* Gradient Background */}
      <div className={cn('absolute inset-0 bg-gradient-to-br opacity-50', gradient)} />

      {/* Content */}
      <div className="relative p-6">
        <div className="flex items-start justify-between">
          {/* Left Side - Metrics */}
          <div className="space-y-2">
            <p className="text-sm font-medium text-muted-foreground">{title}</p>
            <p className="text-3xl font-black text-foreground tracking-tight">
              {value}
            </p>
            {change && (
              <div className="flex items-center gap-1">
                <span
                  className={cn(
                    'text-xs font-semibold',
                    changeType === 'positive' && 'text-green-500',
                    changeType === 'negative' && 'text-red-500',
                    changeType === 'neutral' && 'text-muted-foreground'
                  )}
                >
                  {change}
                </span>
                <span className="text-xs text-muted-foreground">vs last period</span>
              </div>
            )}
          </div>

          {/* Right Side - Icon */}
          <div
            className={cn(
              'flex h-12 w-12 items-center justify-center rounded-xl',
              'bg-gradient-to-br from-background/50 to-background/30',
              'border border-border/50 backdrop-blur-sm'
            )}
          >
            <Icon className={cn('h-6 w-6', iconColor)} />
          </div>
        </div>

        {/* Bottom Accent Line */}
        <div className="absolute bottom-0 left-0 right-0 h-1 bg-gradient-to-r from-transparent via-primary/50 to-transparent" />
      </div>
    </Card>
  );
}
