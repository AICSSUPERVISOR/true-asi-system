// ============================================================================
// PAGINATION UTILITIES
// Standard pagination for all list endpoints
// ============================================================================

export interface PaginationParams {
  page?: number;
  pageSize?: number;
  sortBy?: string;
  sortOrder?: "asc" | "desc";
}

export interface PaginatedResult<T> {
  data: T[];
  pagination: {
    page: number;
    pageSize: number;
    totalItems: number;
    totalPages: number;
    hasNextPage: boolean;
    hasPreviousPage: boolean;
  };
}

export const DEFAULT_PAGE_SIZE = 20;
export const MAX_PAGE_SIZE = 100;

/**
 * Normalize pagination parameters with defaults and limits
 */
export function normalizePaginationParams(params: PaginationParams): Required<PaginationParams> {
  const page = Math.max(1, params.page || 1);
  const pageSize = Math.min(MAX_PAGE_SIZE, Math.max(1, params.pageSize || DEFAULT_PAGE_SIZE));
  const sortBy = params.sortBy || "createdAt";
  const sortOrder = params.sortOrder || "desc";

  return { page, pageSize, sortBy, sortOrder };
}

/**
 * Calculate offset for SQL queries
 */
export function calculateOffset(page: number, pageSize: number): number {
  return (page - 1) * pageSize;
}

/**
 * Build paginated result object
 */
export function buildPaginatedResult<T>(
  data: T[],
  totalItems: number,
  params: Required<PaginationParams>
): PaginatedResult<T> {
  const totalPages = Math.ceil(totalItems / params.pageSize);

  return {
    data,
    pagination: {
      page: params.page,
      pageSize: params.pageSize,
      totalItems,
      totalPages,
      hasNextPage: params.page < totalPages,
      hasPreviousPage: params.page > 1,
    },
  };
}

/**
 * Zod schema for pagination input
 */
import { z } from "zod";

export const paginationSchema = z.object({
  page: z.number().int().positive().optional().default(1),
  pageSize: z.number().int().positive().max(MAX_PAGE_SIZE).optional().default(DEFAULT_PAGE_SIZE),
  sortBy: z.string().optional(),
  sortOrder: z.enum(["asc", "desc"]).optional().default("desc"),
});

export type PaginationInput = z.infer<typeof paginationSchema>;
