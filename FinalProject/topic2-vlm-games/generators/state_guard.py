"""Audit exact Sokoban/Maze test-state overlap, independent of rendering.

Other games are outside this checker's scope, not automatically prohibited.
"""
import hashlib
import json
from pathlib import Path

DEFAULT_EXCLUSIONS = Path(__file__).resolve().parents[1] / 'test_data' / 'excluded-states.txt'

def state_key(game, state):
    if game not in ('sokoban', 'maze'):
        raise ValueError(f'State auditing supports only sokoban and maze; unsupported game: {game}')
    grid = state['grid' if game == 'sokoban' else 'maze']
    if not grid or not grid[0] or any(len(row) != len(grid[0]) for row in grid):
        raise ValueError('State must have a nonempty rectangular grid')
    allowed = range(7) if game == 'sokoban' else range(4)
    if any(type(v) is not int or v not in allowed for row in grid for v in row):
        raise ValueError('Invalid grid cell')
    # Grid encodes walls, player, boxes and targets (including overlaps).
    encoded = game + ':' + str(len(grid)) + ':' + str(len(grid[0])) + ':' + ';'.join(','.join(map(str,row)) for row in grid)
    return hashlib.sha256(encoded.encode('utf-8')).hexdigest()

def load_exclusions(path=DEFAULT_EXCLUSIONS):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'Missing test-state exclusion list: {path}')
    keys = set(path.read_text().splitlines())
    if any(len(k)!=64 or any(c not in '0123456789abcdef' for c in k) for k in keys):
        raise ValueError('Invalid state fingerprint list')
    return keys

def audit_dataset(data_path, exclusions=DEFAULT_EXCLUSIONS):
    data_path=Path(data_path)
    forbidden=load_exclusions(exclusions)
    rows=json.loads(data_path.read_text())
    collisions=[]
    for row in rows:
        game=row['game_name']
        if game not in ('sokoban','maze'):
            raise ValueError(f'State auditing supports only sokoban and maze; unsupported game: {game}. Audit those games separately.')
        state=json.loads((data_path.parent/row['state']).read_text())
        if state_key(game,state) in forbidden:
            collisions.append(row['data_id'])
    return {'samples':len(rows),'overlap_rows':len(collisions),'overlap_ids':collisions}

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('data',type=Path)
    p.add_argument('--exclude-states',type=Path,default=DEFAULT_EXCLUSIONS)
    a=p.parse_args()
    report=audit_dataset(a.data,a.exclude_states)
    print(json.dumps(report,indent=2))
    raise SystemExit(1 if report['overlap_rows'] else 0)
