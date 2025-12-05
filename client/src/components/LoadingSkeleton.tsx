/**
 * LoadingSkeleton Component
 * Reusable skeleton screens for loading states
 */

import { Card } from "@/components/ui/card";

interface LoadingSkeletonProps {
  variant?: "card" | "table" | "chart" | "metric" | "list";
  count?: number;
}

export function LoadingSkeleton({ variant = "card", count = 1 }: LoadingSkeletonProps) {
  const skeletons = Array.from({ length: count }, (_, i) => i);

  if (variant === "card") {
    return (
      <>
        {skeletons.map((i) => (
          <Card
            key={i}
            className="bg-white/5 backdrop-blur-xl border-white/10 p-6 animate-pulse"
            style={{ animationDelay: `${i * 100}ms` }}
          >
            <div className="flex items-center justify-between mb-4">
              <div className="w-12 h-12 bg-white/10 rounded-xl" />
              <div className="space-y-2">
                <div className="w-24 h-8 bg-white/10 rounded" />
                <div className="w-16 h-4 bg-white/10 rounded ml-auto" />
              </div>
            </div>
            <div className="w-32 h-4 bg-white/10 rounded" />
          </Card>
        ))}
      </>
    );
  }

  if (variant === "metric") {
    return (
      <>
        {skeletons.map((i) => (
          <Card
            key={i}
            className="bg-white/5 backdrop-blur-xl border-white/10 p-6 animate-pulse"
            style={{ animationDelay: `${i * 100}ms` }}
          >
            <div className="space-y-3">
              <div className="w-24 h-4 bg-white/10 rounded" />
              <div className="w-32 h-10 bg-white/10 rounded" />
              <div className="w-20 h-3 bg-white/10 rounded" />
            </div>
          </Card>
        ))}
      </>
    );
  }

  if (variant === "table") {
    return (
      <Card className="bg-white/5 backdrop-blur-xl border-white/10 p-6">
        <div className="space-y-4">
          {/* Table Header */}
          <div className="flex gap-4 pb-4 border-b border-white/10">
            <div className="w-1/4 h-4 bg-white/10 rounded animate-pulse" />
            <div className="w-1/4 h-4 bg-white/10 rounded animate-pulse" style={{ animationDelay: "100ms" }} />
            <div className="w-1/4 h-4 bg-white/10 rounded animate-pulse" style={{ animationDelay: "200ms" }} />
            <div className="w-1/4 h-4 bg-white/10 rounded animate-pulse" style={{ animationDelay: "300ms" }} />
          </div>
          {/* Table Rows */}
          {skeletons.map((i) => (
            <div
              key={i}
              className="flex gap-4 py-3"
              style={{ animationDelay: `${(i + 4) * 100}ms` }}
            >
              <div className="w-1/4 h-4 bg-white/10 rounded animate-pulse" />
              <div className="w-1/4 h-4 bg-white/10 rounded animate-pulse" />
              <div className="w-1/4 h-4 bg-white/10 rounded animate-pulse" />
              <div className="w-1/4 h-4 bg-white/10 rounded animate-pulse" />
            </div>
          ))}
        </div>
      </Card>
    );
  }

  if (variant === "chart") {
    return (
      <Card className="bg-white/5 backdrop-blur-xl border-white/10 p-6">
        <div className="space-y-4">
          {/* Chart Title */}
          <div className="w-48 h-6 bg-white/10 rounded animate-pulse" />
          {/* Chart Area */}
          <div className="w-full h-64 bg-white/5 rounded-lg flex items-end gap-2 p-4">
            {Array.from({ length: 12 }, (_, i) => (
              <div
                key={i}
                className="flex-1 bg-white/10 rounded-t animate-pulse"
                style={{
                  height: `${Math.random() * 80 + 20}%`,
                  animationDelay: `${i * 50}ms`,
                }}
              />
            ))}
          </div>
          {/* Chart Legend */}
          <div className="flex gap-4 justify-center">
            <div className="w-20 h-3 bg-white/10 rounded animate-pulse" />
            <div className="w-20 h-3 bg-white/10 rounded animate-pulse" style={{ animationDelay: "100ms" }} />
            <div className="w-20 h-3 bg-white/10 rounded animate-pulse" style={{ animationDelay: "200ms" }} />
          </div>
        </div>
      </Card>
    );
  }

  if (variant === "list") {
    return (
      <div className="space-y-4">
        {skeletons.map((i) => (
          <Card
            key={i}
            className="bg-white/5 backdrop-blur-xl border-white/10 p-4 animate-pulse"
            style={{ animationDelay: `${i * 100}ms` }}
          >
            <div className="flex items-center gap-4">
              <div className="w-10 h-10 bg-white/10 rounded-full" />
              <div className="flex-1 space-y-2">
                <div className="w-3/4 h-4 bg-white/10 rounded" />
                <div className="w-1/2 h-3 bg-white/10 rounded" />
              </div>
              <div className="w-20 h-8 bg-white/10 rounded" />
            </div>
          </Card>
        ))}
      </div>
    );
  }

  return null;
}

/**
 * Shimmer effect for premium loading feel
 */
export function ShimmerSkeleton({ className = "" }: { className?: string }) {
  return (
    <div className={`relative overflow-hidden bg-white/5 rounded ${className}`}>
      <div className="absolute inset-0 -translate-x-full animate-[shimmer_2s_infinite] bg-gradient-to-r from-transparent via-white/10 to-transparent" />
    </div>
  );
}

/**
 * Pulse skeleton for simple loading states
 */
export function PulseSkeleton({ className = "" }: { className?: string }) {
  return (
    <div className={`bg-white/10 rounded animate-pulse ${className}`} />
  );
}
