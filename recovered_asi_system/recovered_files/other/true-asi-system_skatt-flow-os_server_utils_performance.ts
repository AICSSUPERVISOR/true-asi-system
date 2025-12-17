// ============================================================================
// PERFORMANCE MONITORING UTILITIES
// Track and report on application performance
// ============================================================================

interface PerformanceMetric {
  name: string;
  duration: number;
  timestamp: number;
  metadata?: Record<string, unknown>;
}

interface AggregatedMetrics {
  name: string;
  count: number;
  totalDuration: number;
  avgDuration: number;
  minDuration: number;
  maxDuration: number;
  p95Duration: number;
}

class PerformanceMonitor {
  private metrics: PerformanceMetric[] = [];
  private maxMetrics: number = 10000;

  /**
   * Record a performance metric
   */
  record(name: string, duration: number, metadata?: Record<string, unknown>): void {
    this.metrics.push({
      name,
      duration,
      timestamp: Date.now(),
      metadata,
    });

    // Trim old metrics if we exceed the limit
    if (this.metrics.length > this.maxMetrics) {
      this.metrics = this.metrics.slice(-this.maxMetrics);
    }
  }

  /**
   * Measure the duration of an async function
   */
  async measure<T>(name: string, fn: () => Promise<T>, metadata?: Record<string, unknown>): Promise<T> {
    const start = performance.now();
    try {
      const result = await fn();
      this.record(name, performance.now() - start, { ...metadata, success: true });
      return result;
    } catch (error) {
      this.record(name, performance.now() - start, { ...metadata, success: false, error: String(error) });
      throw error;
    }
  }

  /**
   * Get aggregated metrics for a specific operation
   */
  getAggregated(name: string, since?: number): AggregatedMetrics | null {
    const filtered = this.metrics.filter(
      (m) => m.name === name && (!since || m.timestamp >= since)
    );

    if (filtered.length === 0) return null;

    const durations = filtered.map((m) => m.duration).sort((a, b) => a - b);
    const totalDuration = durations.reduce((sum, d) => sum + d, 0);

    return {
      name,
      count: filtered.length,
      totalDuration,
      avgDuration: totalDuration / filtered.length,
      minDuration: durations[0],
      maxDuration: durations[durations.length - 1],
      p95Duration: durations[Math.floor(durations.length * 0.95)] || durations[durations.length - 1],
    };
  }

  /**
   * Get all aggregated metrics
   */
  getAllAggregated(since?: number): AggregatedMetrics[] {
    const names = new Set(this.metrics.map((m) => m.name));
    const results: AggregatedMetrics[] = [];

    for (const name of Array.from(names)) {
      const agg = this.getAggregated(name, since);
      if (agg) results.push(agg);
    }

    return results.sort((a, b) => b.avgDuration - a.avgDuration);
  }

  /**
   * Get recent metrics
   */
  getRecent(limit: number = 100): PerformanceMetric[] {
    return this.metrics.slice(-limit);
  }

  /**
   * Clear all metrics
   */
  clear(): void {
    this.metrics = [];
  }

  /**
   * Get health summary
   */
  getHealthSummary(): {
    totalOperations: number;
    avgResponseTime: number;
    slowOperations: number;
    errorRate: number;
  } {
    const last5Minutes = Date.now() - 5 * 60 * 1000;
    const recent = this.metrics.filter((m) => m.timestamp >= last5Minutes);

    if (recent.length === 0) {
      return {
        totalOperations: 0,
        avgResponseTime: 0,
        slowOperations: 0,
        errorRate: 0,
      };
    }

    const totalDuration = recent.reduce((sum, m) => sum + m.duration, 0);
    const slowThreshold = 1000; // 1 second
    const slowOperations = recent.filter((m) => m.duration > slowThreshold).length;
    const errors = recent.filter((m) => m.metadata?.success === false).length;

    return {
      totalOperations: recent.length,
      avgResponseTime: totalDuration / recent.length,
      slowOperations,
      errorRate: (errors / recent.length) * 100,
    };
  }
}

// Singleton instance
export const performanceMonitor = new PerformanceMonitor();

// Common operation names
export const operations = {
  dbQuery: "db.query",
  dbInsert: "db.insert",
  dbUpdate: "db.update",
  forvaltApi: "api.forvalt",
  altinnApi: "api.altinn",
  regnskapApi: "api.regnskap",
  aimlApi: "api.aiml",
  llmInvoke: "llm.invoke",
  documentProcess: "document.process",
  filingGenerate: "filing.generate",
  saftExport: "saft.export",
};

/**
 * Express middleware for request timing
 */
export function requestTimingMiddleware() {
  return (req: { method: string; path: string }, res: { on: (event: string, cb: () => void) => void }, next: () => void) => {
    const start = performance.now();
    
    res.on("finish", () => {
      const duration = performance.now() - start;
      performanceMonitor.record(`http.${req.method.toLowerCase()}`, duration, {
        path: req.path,
      });
    });

    next();
  };
}
