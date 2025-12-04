import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Text, Sphere, Line } from '@react-three/drei';
import { trpc } from "@/lib/trpc";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Network, Search, ZoomIn, ZoomOut, Maximize2, Activity } from "lucide-react";
import { useRef, useMemo, useState } from "react";
import type { Mesh } from 'three';
import * as THREE from 'three';

interface Node {
  id: number;
  position: [number, number, number];
  label: string;
  connections: number[];
  color: string;
}

function GraphNode({ node, isSelected, onClick }: { node: Node; isSelected: boolean; onClick: () => void }) {
  const meshRef = useRef<Mesh>(null);

  useFrame((state) => {
    if (meshRef.current && isSelected) {
      meshRef.current.scale.setScalar(1 + Math.sin(state.clock.elapsedTime * 3) * 0.1);
    } else if (meshRef.current) {
      meshRef.current.scale.setScalar(1);
    }
  });

  return (
    <group position={node.position} onClick={onClick}>
      <Sphere ref={meshRef} args={[0.3, 16, 16]}>
        <meshStandardMaterial
          color={isSelected ? "#00d9ff" : node.color}
          emissive={isSelected ? "#00d9ff" : node.color}
          emissiveIntensity={isSelected ? 0.5 : 0.2}
        />
      </Sphere>
      {isSelected && (
        <Text
          position={[0, 0.6, 0]}
          fontSize={0.3}
          color="#ffffff"
          anchorX="center"
          anchorY="middle"
        >
          {node.label}
        </Text>
      )}
    </group>
  );
}

function GraphEdge({ start, end }: { start: [number, number, number]; end: [number, number, number] }) {
  const points = useMemo(() => [
    new THREE.Vector3(...start),
    new THREE.Vector3(...end),
  ], [start, end]);

  return (
    <Line
      points={points}
      color="#4299e1"
      lineWidth={1}
      opacity={0.3}
      transparent
    />
  );
}

function KnowledgeGraphScene({ nodes, selectedNode, onNodeClick }: { 
  nodes: Node[]; 
  selectedNode: number | null;
  onNodeClick: (id: number) => void;
}) {
  return (
    <>
      <ambientLight intensity={0.5} />
      <pointLight position={[10, 10, 10]} intensity={1} />
      <pointLight position={[-10, -10, -10]} intensity={0.5} />

      {/* Render edges */}
      {nodes.map((node) =>
        node.connections.map((targetId) => {
          const target = nodes.find((n) => n.id === targetId);
          if (!target) return null;
          return (
            <GraphEdge
              key={`${node.id}-${targetId}`}
              start={node.position}
              end={target.position}
            />
          );
        })
      )}

      {/* Render nodes */}
      {nodes.map((node) => (
        <GraphNode
          key={node.id}
          node={node}
          isSelected={selectedNode === node.id}
          onClick={() => onNodeClick(node.id)}
        />
      ))}

      <OrbitControls
        enableZoom={true}
        enablePan={true}
        autoRotate={false}
        maxDistance={50}
        minDistance={5}
      />
    </>
  );
}

export default function KnowledgeGraph() {
  const { data: knowledgeGraph, isLoading } = trpc.asi.knowledgeGraph.useQuery();
  const [selectedNode, setSelectedNode] = useState<number | null>(null);
  const [searchTerm, setSearchTerm] = useState("");

  // Generate sample graph data (in production, this would come from the backend)
  const graphNodes = useMemo<Node[]>(() => {
    const nodes: Node[] = [];
    const colors = ["#4299e1", "#8b5cf6", "#00d9ff", "#10b981", "#f59e0b"];
    
    // Create a sphere of nodes
    for (let i = 0; i < 100; i++) {
      const phi = Math.acos(-1 + (2 * i) / 100);
      const theta = Math.sqrt(100 * Math.PI) * phi;
      
      const x = 15 * Math.cos(theta) * Math.sin(phi);
      const y = 15 * Math.sin(theta) * Math.sin(phi);
      const z = 15 * Math.cos(phi);

      const connections: number[] = [];
      // Connect to nearby nodes
      for (let j = Math.max(0, i - 3); j < Math.min(100, i + 3); j++) {
        if (j !== i && Math.random() > 0.5) {
          connections.push(j);
        }
      }

      nodes.push({
        id: i,
        position: [x, y, z],
        label: `Entity ${i}`,
        connections,
        color: colors[i % colors.length] || colors[0],
      });
    }

    return nodes;
  }, []);

  const selectedNodeData = selectedNode !== null ? graphNodes[selectedNode] : null;

  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <Activity className="w-12 h-12 animate-spin text-primary mx-auto mb-4" />
          <p className="text-muted-foreground">Loading knowledge graph...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-background flex flex-col">
      {/* Header */}
      <div className="border-b border-border bg-card">
        <div className="container mx-auto px-6 py-4">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-4">
              <Network className="w-8 h-8 text-primary" />
              <div>
                <h1 className="text-3xl font-bold">Knowledge Graph</h1>
                <p className="text-sm text-muted-foreground">
                  Visualizing {knowledgeGraph?.entities.toLocaleString()} entities and {knowledgeGraph?.relationships.toLocaleString()} relationships
                </p>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <Badge className="badge-success text-lg px-4 py-2">
                {knowledgeGraph?.size}
              </Badge>
            </div>
          </div>

          <div className="flex gap-4">
            <div className="relative flex-1">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-muted-foreground" />
              <Input
                type="text"
                placeholder="Search entities..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="pl-10"
              />
            </div>
            <Button variant="outline" className="gap-2">
              <ZoomIn className="w-4 h-4" />
              Zoom In
            </Button>
            <Button variant="outline" className="gap-2">
              <ZoomOut className="w-4 h-4" />
              Zoom Out
            </Button>
            <Button variant="outline" className="gap-2">
              <Maximize2 className="w-4 h-4" />
              Reset View
            </Button>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 flex">
        {/* 3D Visualization */}
        <div className="flex-1 relative">
          <Canvas camera={{ position: [0, 0, 30], fov: 60 }}>
            <KnowledgeGraphScene
              nodes={graphNodes}
              selectedNode={selectedNode}
              onNodeClick={setSelectedNode}
            />
          </Canvas>

          {/* Instructions Overlay */}
          <div className="absolute bottom-6 left-6 glass-effect p-4 rounded-lg max-w-xs">
            <h3 className="font-bold mb-2 text-sm">Controls</h3>
            <ul className="text-xs text-muted-foreground space-y-1">
              <li>• Click and drag to rotate</li>
              <li>• Scroll to zoom in/out</li>
              <li>• Click nodes to select</li>
              <li>• Right-click and drag to pan</li>
            </ul>
          </div>
        </div>

        {/* Side Panel */}
        <div className="w-96 border-l border-border bg-card p-6 overflow-y-auto">
          <h2 className="text-xl font-bold mb-4">Graph Statistics</h2>

          <div className="space-y-4 mb-6">
            <Card className="card-elevated p-4">
              <div className="text-sm text-muted-foreground mb-1">Total Entities</div>
              <div className="text-2xl font-bold text-gradient">
                {knowledgeGraph?.entities.toLocaleString()}
              </div>
            </Card>

            <Card className="card-elevated p-4">
              <div className="text-sm text-muted-foreground mb-1">Relationships</div>
              <div className="text-2xl font-bold text-gradient">
                {knowledgeGraph?.relationships.toLocaleString()}
              </div>
            </Card>

            <Card className="card-elevated p-4">
              <div className="text-sm text-muted-foreground mb-1">Total Files</div>
              <div className="text-2xl font-bold text-gradient">
                {knowledgeGraph?.files.toLocaleString()}
              </div>
            </Card>

            <Card className="card-elevated p-4">
              <div className="text-sm text-muted-foreground mb-1">Storage Size</div>
              <div className="text-2xl font-bold text-gradient">
                {knowledgeGraph?.size}
              </div>
            </Card>
          </div>

          {selectedNodeData && (
            <>
              <h2 className="text-xl font-bold mb-4 mt-8">Selected Entity</h2>
              <Card className="card-elevated p-4">
                <h3 className="font-bold mb-2">{selectedNodeData.label}</h3>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">ID:</span>
                    <span className="font-mono">{selectedNodeData.id}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Connections:</span>
                    <span className="font-bold">{selectedNodeData.connections.length}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Position:</span>
                    <span className="font-mono text-xs">
                      [{selectedNodeData.position.map(p => p.toFixed(1)).join(', ')}]
                    </span>
                  </div>
                </div>
                <Button className="w-full mt-4 btn-primary" size="sm">
                  View Details
                </Button>
              </Card>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
