import sqlite3

conn = sqlite3.connect('data/navbot.db')
cursor = conn.cursor()

print('=== LOCATIONS ===')
cursor.execute('SELECT id, name, lat, lon, floor FROM locations ORDER BY floor, id')
locs = {}
for r in cursor.fetchall():
    locs[r[0]] = {'name': r[1], 'lat': r[2], 'lon': r[3], 'floor': r[4]}
    print(f'  {r[0]}: {r[1]:20s} Tang {r[4]} ({r[2]:.6f}, {r[3]:.6f})')

print('\n=== EDGES ===')
cursor.execute('SELECT id, name, from_lat, from_lon, to_lat, to_lon, from_floor, to_floor, distance_m, is_bidirectional FROM custom_edges ORDER BY id')
edges = cursor.fetchall()

for e in edges:
    eid, name, flat, flon, tlat, tlon, ffloor, tfloor, dist, bidir = e
    
    # Find matching locations
    from_loc = None
    to_loc = None
    for lid, loc in locs.items():
        if abs(loc['lat'] - flat) < 0.00001 and abs(loc['lon'] - flon) < 0.00001:
            from_loc = f"{lid}:{loc['name']}"
        if abs(loc['lat'] - tlat) < 0.00001 and abs(loc['lon'] - tlon) < 0.00001:
            to_loc = f"{lid}:{loc['name']}"
    
    bidir_str = 'YES' if bidir else 'NO'
    print(f'  {eid}: {name:20s} | {from_loc or "?"}(T{ffloor}) -> {to_loc or "?"}(T{tfloor}) | {dist:.1f}m | Bidir:{bidir_str}')

print('\n=== KIEM TRA KET NOI ===')
print('Duong di can: Phong 303 (Tang 3) -> Bep (Tang 1)')
print('  1. Phong 303 -> sanh tang 3 (edge 11)')
print('  2. Sanh tang 3 -> sanh tang 2 (edge 10 reverse)')
print('  3. Sanh tang 2 -> nha xe tang 1 (edge 9 reverse)')
print('  4. Nha xe -> bep (edge 8 reverse)')

print('\nKiem tra edges:')
cursor.execute('SELECT id, name, from_floor, to_floor, is_bidirectional FROM custom_edges WHERE id IN (8, 9, 10, 11)')
for r in cursor.fetchall():
    bidir = 'YES' if r[4] else 'NO'
    print(f'  Edge {r[0]} ({r[1]}): Tang {r[2]}->{r[3]}, Bidirectional: {bidir}')

# Check edge 12
print('\nEdge 12 (cau thang 3-4):')
cursor.execute('SELECT id, name, from_lat, from_lon, to_lat, to_lon, from_floor, to_floor, is_bidirectional FROM custom_edges WHERE id=12')
r = cursor.fetchone()
if r:
    print(f'  From: ({r[2]:.6f}, {r[3]:.6f}) Tang {r[6]}')
    print(f'  To: ({r[4]:.6f}, {r[5]:.6f}) Tang {r[7]}')
    print(f'  Bidirectional: {"YES" if r[8] else "NO"}')
    print(f'  VAI TRO: Ten la "cau thang 3-4" nhung from_floor={r[6]}, to_floor={r[7]}')
    print(f'  => NGUOC! Can sua lai hoac dam bao bidirectional=1')

conn.close()
