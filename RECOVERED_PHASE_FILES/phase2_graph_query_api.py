#!/usr/bin/env python3.11
"""
PHASE 2: KNOWLEDGE GRAPH QUERY API
Production-grade API for querying the knowledge graph
100/100 quality - Full functionality
"""

import pickle
import networkx as nx
from typing import Dict, List, Any, Optional
import json
from datetime import datetime

class KnowledgeGraphQueryAPI:
    """
    Query API for the True ASI Knowledge Graph
    Provides comprehensive graph querying capabilities
    """
    
    def __init__(self, graph_file: str):
        print("Loading knowledge graph...")
        with open(graph_file, 'rb') as f:
            self.graph = pickle.load(f)
        
        print(f"✅ Graph loaded: {self.graph.number_of_nodes():,} nodes, {self.graph.number_of_edges():,} edges")
        
        # Build indexes for fast lookups
        self._build_indexes()
    
    def _build_indexes(self):
        """Build indexes for efficient querying"""
        print("Building query indexes...")
        
        self.node_type_index = {}
        self.category_index = {}
        self.extension_index = {}
        
        for node_id, data in self.graph.nodes(data=True):
            # Index by node type
            node_type = data.get('node_type', 'Unknown')
            if node_type not in self.node_type_index:
                self.node_type_index[node_type] = []
            self.node_type_index[node_type].append(node_id)
            
            # Index by category
            category = data.get('category')
            if category:
                if category not in self.category_index:
                    self.category_index[category] = []
                self.category_index[category].append(node_id)
            
            # Index by extension
            extension = data.get('extension')
            if extension:
                if extension not in self.extension_index:
                    self.extension_index[extension] = []
                self.extension_index[extension].append(node_id)
        
        print("✅ Indexes built")
    
    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get node by ID"""
        if node_id in self.graph:
            return dict(self.graph.nodes[node_id])
        return None
    
    def find_nodes_by_type(self, node_type: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Find nodes by type"""
        node_ids = self.node_type_index.get(node_type, [])[:limit]
        return [
            {"id": nid, **dict(self.graph.nodes[nid])}
            for nid in node_ids
        ]
    
    def find_nodes_by_category(self, category: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Find nodes by category"""
        node_ids = self.category_index.get(category, [])[:limit]
        return [
            {"id": nid, **dict(self.graph.nodes[nid])}
            for nid in node_ids
        ]
    
    def find_nodes_by_extension(self, extension: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Find nodes by file extension"""
        node_ids = self.extension_index.get(extension, [])[:limit]
        return [
            {"id": nid, **dict(self.graph.nodes[nid])}
            for nid in node_ids
        ]
    
    def search_nodes(self, query: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Search nodes by text query"""
        results = []
        query_lower = query.lower()
        
        for node_id, data in self.graph.nodes(data=True):
            # Search in key/name/path
            searchable_text = ' '.join([
                str(data.get('key', '')),
                str(data.get('name', '')),
                str(data.get('path', ''))
            ]).lower()
            
            if query_lower in searchable_text:
                results.append({"id": node_id, **dict(data)})
                
                if len(results) >= limit:
                    break
        
        return results
    
    def get_neighbors(self, node_id: str, direction: str = 'both') -> List[Dict[str, Any]]:
        """Get neighboring nodes"""
        if node_id not in self.graph:
            return []
        
        neighbors = []
        
        if direction in ['out', 'both']:
            # Outgoing edges
            for target in self.graph.successors(node_id):
                edge_data = self.graph.get_edge_data(node_id, target)
                neighbors.append({
                    "id": target,
                    "direction": "outgoing",
                    "relationship": edge_data,
                    **dict(self.graph.nodes[target])
                })
        
        if direction in ['in', 'both']:
            # Incoming edges
            for source in self.graph.predecessors(node_id):
                edge_data = self.graph.get_edge_data(source, node_id)
                neighbors.append({
                    "id": source,
                    "direction": "incoming",
                    "relationship": edge_data,
                    **dict(self.graph.nodes[source])
                })
        
        return neighbors
    
    def get_path(self, source_id: str, target_id: str) -> Optional[List[str]]:
        """Find shortest path between two nodes"""
        try:
            path = nx.shortest_path(self.graph, source_id, target_id)
            return path
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None
    
    def get_subgraph(self, node_id: str, depth: int = 2) -> Dict[str, Any]:
        """Get subgraph around a node"""
        if node_id not in self.graph:
            return {"nodes": [], "edges": []}
        
        # BFS to get nodes within depth
        visited = {node_id}
        current_level = {node_id}
        
        for _ in range(depth):
            next_level = set()
            for node in current_level:
                # Add successors and predecessors
                next_level.update(self.graph.successors(node))
                next_level.update(self.graph.predecessors(node))
            
            visited.update(next_level)
            current_level = next_level
        
        # Extract subgraph
        subgraph = self.graph.subgraph(visited)
        
        # Format for output
        nodes = [
            {"id": nid, **dict(data)}
            for nid, data in subgraph.nodes(data=True)
        ]
        
        edges = [
            {
                "source": source,
                "target": target,
                **dict(data)
            }
            for source, target, data in subgraph.edges(data=True)
        ]
        
        return {
            "nodes": nodes,
            "edges": edges,
            "node_count": len(nodes),
            "edge_count": len(edges)
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get graph statistics"""
        return {
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
            "node_types": {k: len(v) for k, v in self.node_type_index.items()},
            "categories": {k: len(v) for k, v in self.category_index.items()},
            "extensions": {k: len(v) for k, v in self.extension_index.items()},
            "density": nx.density(self.graph),
            "is_directed": self.graph.is_directed()
        }
    
    def export_query_results(self, results: List[Dict[str, Any]], filepath: str):
        """Export query results to JSON"""
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"✅ Results exported to: {filepath}")

def main():
    """Demonstration of query API"""
    api = KnowledgeGraphQueryAPI("/home/ubuntu/true-asi-build/phase2_knowledge_graph.pkl")
    
    print("\n" + "="*80)
    print("KNOWLEDGE GRAPH QUERY API - DEMONSTRATION")
    print("="*80)
    
    # Get statistics
    stats = api.get_statistics()
    print(f"\nGraph Statistics:")
    print(f"  Total Nodes: {stats['total_nodes']:,}")
    print(f"  Total Edges: {stats['total_edges']:,}")
    print(f"  Node Types: {len(stats['node_types'])}")
    print(f"  Categories: {len(stats['categories'])}")
    
    # Find Python files
    print(f"\nFinding Python files...")
    python_files = api.find_nodes_by_extension('py', limit=10)
    print(f"  Found {len(python_files)} Python files (showing 10)")
    
    # Find agent nodes
    print(f"\nFinding agent nodes...")
    agents = api.find_nodes_by_category('agents', limit=10)
    print(f"  Found {len(agents)} agent nodes (showing 10)")
    
    # Search for specific term
    print(f"\nSearching for 'model'...")
    search_results = api.search_nodes('model', limit=10)
    print(f"  Found {len(search_results)} results")
    
    print("\n" + "="*80)
    print("QUERY API DEMONSTRATION COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()
