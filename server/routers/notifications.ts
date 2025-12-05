/**
 * Notifications Router
 * tRPC procedures for notification management
 */

import { z } from 'zod';
import { router, protectedProcedure } from '../_core/trpc';
import { getDb } from '../db';
import { notifications } from '../../drizzle/schema';
import { eq, and, desc } from 'drizzle-orm';
import { emitUserNotification } from '../_core/websocket';

export const notificationsRouter = router({
  /**
   * Get all notifications for current user
   */
  getAll: protectedProcedure.query(async ({ ctx }) => {
    const db = await getDb();
    if (!db) return [];

    const userNotifications = await db
      .select()
      .from(notifications)
      .where(eq(notifications.userId, ctx.user.id))
      .orderBy(desc(notifications.createdAt))
      .limit(50);

    return userNotifications;
  }),

  /**
   * Get unread notification count
   */
  getUnreadCount: protectedProcedure.query(async ({ ctx }) => {
    const db = await getDb();
    if (!db) return { count: 0 };

    const unread = await db
      .select()
      .from(notifications)
      .where(
        and(
          eq(notifications.userId, ctx.user.id),
          eq(notifications.isRead, 0)
        )
      );

    return { count: unread.length };
  }),

  /**
   * Mark notification as read
   */
  markAsRead: protectedProcedure
    .input(z.object({
      id: z.string(),
    }))
    .mutation(async ({ ctx, input }) => {
      const db = await getDb();
      if (!db) return { success: false };

      await db
        .update(notifications)
        .set({
          isRead: 1,
          readAt: new Date(),
        })
        .where(
          and(
            eq(notifications.id, input.id),
            eq(notifications.userId, ctx.user.id)
          )
        );

      return { success: true };
    }),

  /**
   * Mark all notifications as read
   */
  markAllAsRead: protectedProcedure.mutation(async ({ ctx }) => {
    const db = await getDb();
    if (!db) return { success: false };

    await db
      .update(notifications)
      .set({
        isRead: 1,
        readAt: new Date(),
      })
      .where(
        and(
          eq(notifications.userId, ctx.user.id),
          eq(notifications.isRead, 0)
        )
      );

    return { success: true };
  }),

  /**
   * Delete notification
   */
  delete: protectedProcedure
    .input(z.object({
      id: z.string(),
    }))
    .mutation(async ({ ctx, input }) => {
      const db = await getDb();
      if (!db) return { success: false };

      await db
        .delete(notifications)
        .where(
          and(
            eq(notifications.id, input.id),
            eq(notifications.userId, ctx.user.id)
          )
        );

      return { success: true };
    }),

  /**
   * Create notification (for testing or manual creation)
   */
  create: protectedProcedure
    .input(z.object({
      title: z.string(),
      message: z.string(),
      type: z.enum(['info', 'success', 'warning', 'error', 'analysis_complete', 'execution_complete']).default('info'),
      analysisId: z.string().optional(),
      workflowId: z.string().optional(),
      link: z.string().optional(),
    }))
    .mutation(async ({ ctx, input }) => {
      const notificationId = `notif_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;

      const db = await getDb();
      if (!db) return { success: false, id: '' };

      await db.insert(notifications).values({
        id: notificationId,
        userId: ctx.user.id,
        title: input.title,
        message: input.message,
        type: input.type,
        analysisId: input.analysisId,
        workflowId: input.workflowId,
        link: input.link,
        isRead: 0,
      });

      // Emit real-time notification
      emitUserNotification(ctx.user.id.toString(), {
        id: notificationId,
        title: input.title,
        message: input.message,
        type: input.type,
      });

      return { id: notificationId, success: true };
    }),
});
