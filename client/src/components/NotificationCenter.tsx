/**
 * NotificationCenter - Dropdown notification panel with real-time updates
 */

import React, { useState } from 'react';
import { trpc } from '@/lib/trpc';
import { useWebSocketEvent } from '@/contexts/WebSocketProvider';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Bell, Check, CheckCheck, Trash2 } from 'lucide-react';
import { useToast } from '@/hooks/use-toast';
import { useLocation } from 'wouter';

export function NotificationCenter() {
  const [, setLocation] = useLocation();
  const { toast } = useToast();
  const utils = trpc.useUtils();

  // Fetch notifications
  const { data: notifications = [], isLoading } = trpc.notifications.getAll.useQuery();

  // Mark as read mutation
  const { mutate: markAsRead } = trpc.notifications.markAsRead.useMutation({
    onSuccess: () => {
      utils.notifications.getAll.invalidate();
    },
  });

  // Mark all as read mutation
  const { mutate: markAllAsRead } = trpc.notifications.markAllAsRead.useMutation({
    onSuccess: () => {
      utils.notifications.getAll.invalidate();
      toast({
        title: 'All notifications marked as read',
      });
    },
  });

  // Delete notification mutation
  const { mutate: deleteNotification } = trpc.notifications.delete.useMutation({
    onSuccess: () => {
      utils.notifications.getAll.invalidate();
    },
  });

  // Real-time notification updates
  useWebSocketEvent('notification:new', (data: any) => {
    utils.notifications.getAll.invalidate();
    toast({
      title: data.title,
      description: data.message,
    });
  });

  useWebSocketEvent('analysis:complete', (data: any) => {
    utils.notifications.getAll.invalidate();
    toast({
      title: 'Analysis Complete',
      description: `Analysis for ${data.companyName} is ready`,
    });
  });

  // Count unread notifications
  const unreadCount = notifications.filter((n: any) => !n.isRead).length;

  // Format timestamp (e.g., "2 minutes ago")
  const formatTimestamp = (timestamp: string | Date) => {
    const date = new Date(timestamp);
    const now = new Date();
    const diff = now.getTime() - date.getTime();
    
    const seconds = Math.floor(diff / 1000);
    const minutes = Math.floor(seconds / 60);
    const hours = Math.floor(minutes / 60);
    const days = Math.floor(hours / 24);
    
    if (days > 0) return `${days}d ago`;
    if (hours > 0) return `${hours}h ago`;
    if (minutes > 0) return `${minutes}m ago`;
    return 'Just now';
  };

  // Get notification icon color based on type
  const getTypeColor = (type: string) => {
    switch (type) {
      case 'success': return 'text-green-400';
      case 'warning': return 'text-yellow-400';
      case 'error': return 'text-red-400';
      case 'analysis_complete': return 'text-cyan-400';
      case 'execution_complete': return 'text-purple-400';
      default: return 'text-blue-400';
    }
  };

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button
          variant="ghost"
          size="icon"
          className="relative text-slate-400 hover:text-white"
        >
          <Bell className="w-5 h-5" />
          {unreadCount > 0 && (
            <Badge
              variant="destructive"
              className="absolute -top-1 -right-1 h-5 w-5 flex items-center justify-center p-0 text-xs"
            >
              {unreadCount > 9 ? '9+' : unreadCount}
            </Badge>
          )}
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent
        align="end"
        className="w-96 max-h-[500px] overflow-y-auto bg-slate-900/95 backdrop-blur-xl border-slate-700"
      >
        <DropdownMenuLabel className="flex items-center justify-between text-white">
          <span>Notifications</span>
          {unreadCount > 0 && (
            <Button
              variant="ghost"
              size="sm"
              onClick={() => markAllAsRead()}
              className="text-xs text-slate-400 hover:text-white"
            >
              <CheckCheck className="w-4 h-4 mr-1" />
              Mark all read
            </Button>
          )}
        </DropdownMenuLabel>
        <DropdownMenuSeparator className="bg-slate-700" />
        
        {isLoading ? (
          <div className="p-4 text-center text-slate-400">Loading...</div>
        ) : notifications.length === 0 ? (
          <div className="p-8 text-center text-slate-400">
            <Bell className="w-12 h-12 mx-auto mb-2 opacity-20" />
            <p>No notifications yet</p>
          </div>
        ) : (
          notifications.map((notification: any) => (
            <DropdownMenuItem
              key={notification.id}
              className={`
                flex flex-col items-start gap-2 p-4 cursor-pointer
                ${!notification.isRead ? 'bg-slate-800/50' : ''}
                hover:bg-slate-800 transition-colors
              `}
              onClick={() => {
                if (!notification.isRead) {
                  markAsRead({ id: notification.id });
                }
                if (notification.link) {
                  setLocation(notification.link);
                }
              }}
            >
              <div className="flex items-start justify-between w-full">
                <div className="flex-1">
                  <div className="flex items-center gap-2 mb-1">
                    {!notification.isRead && (
                      <div className="w-2 h-2 rounded-full bg-cyan-400" />
                    )}
                    <span className={`font-semibold ${getTypeColor(notification.type)}`}>
                      {notification.title}
                    </span>
                  </div>
                  <p className="text-sm text-slate-300">{notification.message}</p>
                  <p className="text-xs text-slate-500 mt-1">
                    {formatTimestamp(notification.createdAt)}
                  </p>
                </div>
                <div className="flex items-center gap-1 ml-2">
                  {!notification.isRead && (
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-6 w-6 text-slate-400 hover:text-white"
                      onClick={(e) => {
                        e.stopPropagation();
                        markAsRead({ id: notification.id });
                      }}
                    >
                      <Check className="w-3 h-3" />
                    </Button>
                  )}
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-6 w-6 text-slate-400 hover:text-red-400"
                    onClick={(e) => {
                      e.stopPropagation();
                      deleteNotification({ id: notification.id });
                    }}
                  >
                    <Trash2 className="w-3 h-3" />
                  </Button>
                </div>
              </div>
            </DropdownMenuItem>
          ))
        )}
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
