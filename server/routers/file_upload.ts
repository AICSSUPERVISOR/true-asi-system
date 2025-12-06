/**
 * File Upload Router
 * 
 * Handles massive file uploads (up to 500MB per file)
 * with AWS S3 storage and file management.
 */

import { z } from 'zod';
import { protectedProcedure, router } from '../_core/trpc';
import { storagePut } from '../storage';

// In-memory file storage (in production, use database)
const fileStore: Map<string, {
  id: string;
  userId: number;
  filename: string;
  fileSize: number;
  mimeType: string;
  fileKey: string;
  fileUrl: string;
  uploadedAt: Date;
}> = new Map();

export const fileUploadRouter = router({
  /**
   * Upload file metadata (actual file upload happens via multipart/form-data)
   */
  uploadFile: protectedProcedure
    .input(
      z.object({
        filename: z.string(),
        fileSize: z.number(),
        mimeType: z.string(),
        fileKey: z.string(),
        fileUrl: z.string(),
      })
    )
    .mutation(async ({ input, ctx }) => {
      try {
        const fileId = `file_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
        
        const file = {
          id: fileId,
          userId: ctx.user.id,
          filename: input.filename,
          fileSize: input.fileSize,
          mimeType: input.mimeType,
          fileKey: input.fileKey,
          fileUrl: input.fileUrl,
          uploadedAt: new Date(),
        };

        fileStore.set(fileId, file);

        return {
          success: true,
          file,
        };
      } catch (error) {
        console.error('[FileUpload] Error saving file metadata:', error);
        return {
          success: false,
          error: error instanceof Error ? error.message : 'Unknown error',
        };
      }
    }),

  /**
   * List user's uploaded files
   */
  listFiles: protectedProcedure
    .input(
      z.object({
        limit: z.number().default(50),
        offset: z.number().default(0),
      })
    )
    .query(async ({ input, ctx }) => {
      try {
        const userFiles = Array.from(fileStore.values())
          .filter(file => file.userId === ctx.user.id)
          .sort((a, b) => b.uploadedAt.getTime() - a.uploadedAt.getTime())
          .slice(input.offset, input.offset + input.limit);

        return {
          success: true,
          files: userFiles,
          total: userFiles.length,
        };
      } catch (error) {
        console.error('[FileUpload] Error listing files:', error);
        return {
          success: false,
          error: error instanceof Error ? error.message : 'Unknown error',
          files: [],
          total: 0,
        };
      }
    }),

  /**
   * Delete file
   */
  deleteFile: protectedProcedure
    .input(
      z.object({
        fileId: z.string(),
      })
    )
    .mutation(async ({ input, ctx }) => {
      try {
        const file = fileStore.get(input.fileId);

        if (!file || file.userId !== ctx.user.id) {
          return {
            success: false,
            error: 'File not found or access denied',
          };
        }

        fileStore.delete(input.fileId);

        return {
          success: true,
        };
      } catch (error) {
        console.error('[FileUpload] Error deleting file:', error);
        return {
          success: false,
          error: error instanceof Error ? error.message : 'Unknown error',
        };
      }
    }),

  /**
   * Get file by ID
   */
  getFile: protectedProcedure
    .input(
      z.object({
        fileId: z.string(),
      })
    )
    .query(async ({ input, ctx }) => {
      try {
        const file = fileStore.get(input.fileId);

        if (!file || file.userId !== ctx.user.id) {
          return {
            success: false,
            error: 'File not found or access denied',
            file: null,
          };
        }

        return {
          success: true,
          file,
        };
      } catch (error) {
        console.error('[FileUpload] Error getting file:', error);
        return {
          success: false,
          error: error instanceof Error ? error.message : 'Unknown error',
          file: null,
        };
      }
    }),

  /**
   * Search files by filename
   */
  searchFiles: protectedProcedure
    .input(
      z.object({
        query: z.string(),
        limit: z.number().default(20),
      })
    )
    .query(async ({ input, ctx }) => {
      try {
        const userFiles = Array.from(fileStore.values())
          .filter(file => file.userId === ctx.user.id);

        const filtered = userFiles.filter(file =>
          file.filename.toLowerCase().includes(input.query.toLowerCase())
        ).slice(0, input.limit);

        return {
          success: true,
          files: filtered,
          total: filtered.length,
        };
      } catch (error) {
        console.error('[FileUpload] Error searching files:', error);
        return {
          success: false,
          error: error instanceof Error ? error.message : 'Unknown error',
          files: [],
          total: 0,
        };
      }
    }),

  /**
   * Get storage statistics
   */
  getStorageStats: protectedProcedure.query(async ({ ctx }) => {
    try {
      const userFiles = Array.from(fileStore.values())
        .filter(file => file.userId === ctx.user.id);

      const totalSize = userFiles.reduce((sum, file) => sum + file.fileSize, 0);
      const fileCount = userFiles.length;

      // Group by mime type
      const byType: Record<string, number> = {};
      userFiles.forEach(file => {
        const type = file.mimeType.split('/')[0] || 'other';
        byType[type] = (byType[type] || 0) + 1;
      });

      return {
        success: true,
        stats: {
          totalSize,
          fileCount,
          byType,
        },
      };
    } catch (error) {
      console.error('[FileUpload] Error getting storage stats:', error);
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error',
        stats: {
          totalSize: 0,
          fileCount: 0,
          byType: {},
        },
      };
    }
  }),
});
