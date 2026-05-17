import asyncio
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from core.database import db
from core.indoor_router import build_indoor_graph_from_db, IndoorRouter

async def test():
    await db.init()
    
    print('=== BUILDING GRAPH ===')
    graph = await build_indoor_graph_from_db('main_building')
    
    print(f'\nNodes: {len(graph.nodes)}')
    print(f'Total edges (directed): {sum(len(adj) for adj in graph.adj.values())}')
    
    print('\n=== NODES ===')
    for nid, node in sorted(graph.nodes.items(), key=lambda x: (x[1].floor, x[0])):
        name_safe = node.name.encode('ascii', 'replace').decode('ascii')
        print(f'  {nid:10s}: {name_safe:20s} Floor {node.floor}')
    
    print('\n=== ADJACENCY LIST ===')
    for nid in sorted(graph.adj.keys(), key=lambda x: (graph.nodes[x].floor, x)):
        node = graph.nodes[nid]
        neighbors = graph.adj[nid]
        name_safe = node.name.encode('ascii', 'replace').decode('ascii')
        print(f'\n{nid} ({name_safe}, Floor {node.floor}):')
        for neighbor_id, edge in neighbors:
            neighbor = graph.nodes[neighbor_id]
            neighbor_name_safe = neighbor.name.encode('ascii', 'replace').decode('ascii')
            print(f'  -> {neighbor_id} ({neighbor_name_safe}, Floor {neighbor.floor}) via {edge.edge_type} [{edge.distance_m:.1f}m]')
    
    print('\n=== TEST ROUTING ===')
    
    # Find nodes
    phong303 = None
    bep = None
    for nid, node in graph.nodes.items():
        if '303' in node.name and node.floor == 3:
            phong303 = nid
        # Try multiple patterns for kitchen
        name_lower = node.name.lower()
        if any(x in name_lower for x in ['bep', 'b?p', 'kitchen']) and node.floor == 1:
            bep = nid
    
    # If still not found, check by ID
    if not bep and 'loc_8' in graph.nodes:
        bep = 'loc_8'
    
    if not phong303:
        print('ERROR: Cannot find room 303!')
        # List all floor 3 nodes
        print('Floor 3 nodes:')
        for nid, node in graph.nodes.items():
            if node.floor == 3:
                print(f'  {nid}: {node.name}')
        return
    if not bep:
        print('ERROR: Cannot find kitchen!')
        # List all floor 1 nodes
        print('Floor 1 nodes:')
        for nid, node in graph.nodes.items():
            if node.floor == 1:
                print(f'  {nid}: {node.name}')
        return
    
    print(f'Origin: {phong303}')
    print(f'Dest: {bep}')
    
    router = IndoorRouter(graph)
    route = router.route(phong303, bep)
    
    if not route:
        print('\nERROR: NO ROUTE FOUND!')
        print('\nChecking connectivity:')
        # BFS to check connectivity
        from collections import deque
        visited = {phong303}
        queue = deque([phong303])
        while queue:
            current = queue.popleft()
            if current == bep:
                print(f'  => CAN reach {bep} from {phong303}')
                break
            for neighbor_id, _ in graph.adj.get(current, []):
                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    queue.append(neighbor_id)
        else:
            print(f'  => CANNOT reach {bep} from {phong303}!')
            print(f'  => Graph is NOT connected!')
        return
    
    print(f'\nROUTE FOUND!')
    print(f'Distance: {route.total_distance_m:.1f}m')
    print(f'Duration: {route.total_duration_s:.1f}s ({route.total_duration_s/60:.1f} min)')
    print(f'Floors: {route.floors_visited}')
    print(f'\nSteps ({len(route.steps)}):')
    for i, step in enumerate(route.steps, 1):
        instr_safe = step.instruction.encode('ascii', 'replace').decode('ascii')
        print(f'  {i}. {instr_safe}')
        print(f'     [{step.from_node_id} -> {step.to_node_id}] Floor {step.from_floor}->{step.to_floor} | {step.distance_m:.1f}m')

asyncio.run(test())
