/**
 * Premium Chart Card Component
 * 
 * Inspired by Maxton Bootstrap 5 Admin Dashboard
 * Features: Glass-morphism, gradient headers, chart containers
 */

import { ReactNode } from 'react';
import { Card } from '@/components/ui/card';
import { cn } from '@/lib/utils';
import { LucideIcon } from 'lucide-react';

interface PremiumChartCardProps {
  title: string;
  subtitle?: string;
  icon?: LucideIcon;
  children: ReactNode;
  headerAction?: ReactNode;
  gradient?: string;
  className?: string;
}

export function PremiumChartCard({
  title,
  subtitle,
  icon: Icon,
  children,
  headerAction,
  gradient = 'from-blue-500/5 to-cyan-500/5',
  className,
}: PremiumChartCardProps) {
  return (
    <Card
      className={cn(
        'relative overflow-hidden backdrop-blur-xl bg-card/50 border-border/50',
        'hover:bg-card/70 transition-all duration-300',
        className
      )}
    >
      {/* Header with Gradient */}
      <div className={cn('relative bg-gradient-to-r p-6 border-b border-border/50', gradient)}>
        <div className="flex items-start justify-between">
          <div className="space-y-1">
            <div className="flex items-center gap-2">
              {Icon && <Icon className="h-5 w-5 text-primary" />}
              <h3 className="text-lg font-bold text-foreground">{title}</h3>
            </div>
            {subtitle && (
              <p className="text-sm text-muted-foreground">{subtitle}</p>
            )}
          </div>
          {headerAction && <div>{headerAction}</div>}
        </div>
      </div>

      {/* Chart Content */}
      <div className="relative p-6">{children}</div>

      {/* Bottom Accent */}
      <div className="absolute bottom-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-primary/30 to-transparent" />
    </Card>
  );
}
