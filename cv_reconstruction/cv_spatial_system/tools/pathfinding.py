import json
import numpy as np
import heapq

class AStarPathfinder:
    def __init__(self, graph_json_path):
        with open(graph_json_path, 'r') as f:
            data = json.load(f)
            
        self.nodes = data['nodes']
        # Build adjacency list
        self.graph = {node: [] for node in self.nodes}
        for edge in data['edges']:
            n1, n2, cost = edge
            # Bi-directional graph
            self.graph[n1].append((cost, n2))
            self.graph[n2].append((cost, n1))
            
    def heuristic(self, node_a, node_b):
        """Euclidean distance between two 3D nodes (Heuristic)."""
        pos_a = np.array(self.nodes[node_a])
        pos_b = np.array(self.nodes[node_b])
        return np.linalg.norm(pos_a - pos_b)
        
    def find_path(self, start_node, end_node):
        """Find shortest path using A* algorithm."""
        queue = []
        heapq.heappush(queue, (0, start_node))
        
        came_from = {}
        cost_so_far = {start_node: 0}
        
        while queue:
            current_priority, current_node = heapq.heappop(queue)
            
            if current_node == end_node:
                break
                
            for cost, next_node in self.graph[current_node]:
                new_cost = cost_so_far[current_node] + cost
                if next_node not in cost_so_far or new_cost < cost_so_far[next_node]:
                    cost_so_far[next_node] = new_cost
                    priority = new_cost + self.heuristic(next_node, end_node)
                    heapq.heappush(queue, (priority, next_node))
                    came_from[next_node] = current_node
                    
        # Reconstruct path
        path = []
        current = end_node
        if current not in came_from and current != start_node:
            return [] # No path found
            
        while current != start_node:
            path.append(current)
            current = came_from[current]
        path.append(start_node)
        path.reverse()
        
        # Return list of 3D coordinates
        return [self.nodes[n] for n in path]

if __name__ == "__main__":
    import os
    # Basic test
    graph_path = os.path.join(os.path.dirname(__file__), "../data/nav_graph.json")
    if os.path.exists(graph_path):
        finder = AStarPathfinder(graph_path)
        print("Finding path from 'hallway_start' to 'library_door'...")
        path = finder.find_path("hallway_start", "library_door")
        print("A* Path Coordinates:", path)
