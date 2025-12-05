/**
 * ConnectionStatus - Visual indicator for WebSocket connection
 */

import { useWebSocket } from '@/contexts/WebSocketProvider';
import { Wifi, WifiOff } from 'lucide-react';
import { Badge } from '@/components/ui/badge';

export function ConnectionStatus() {
  const { isConnected } = useWebSocket();

  return (
    <Badge
      variant="outline"
      className={`
        flex items-center gap-2 px-3 py-1 transition-all duration-300
        ${isConnected 
          ? 'bg-green-500/10 border-green-500/30 text-green-400' 
          : 'bg-red-500/10 border-red-500/30 text-red-400'
        }
      `}
    >
      {isConnected ? (
        <>
          <Wifi className="w-3 h-3 animate-pulse" />
          <span className="text-xs font-medium">Live</span>
        </>
      ) : (
        <>
          <WifiOff className="w-3 h-3" />
          <span className="text-xs font-medium">Offline</span>
        </>
      )}
    </Badge>
  );
}
