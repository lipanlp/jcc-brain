"""List review/verified evidence or revoke a new sample. Never removes evidence."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from jccbrain.vision.eye.unit_learning import UnitLearningStore


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--root',type=Path,required=True,help='independent learning root (parent of s<season>)')
    p.add_argument('--season',required=True)
    sub=p.add_subparsers(dest='command',required=True)
    ls=sub.add_parser('list');ls.add_argument('--kind',choices=['verified','pending','rejected','conflict','all'],default='all')
    revoke=sub.add_parser('revoke');revoke.add_argument('sample_id')
    a=p.parse_args()
    # Listing/revoking does not require a pool and cannot promote anything.
    store=UnitLearningStore(a.root,a.season,set())
    if a.command=='revoke':
        answer={'revoked':store.revoke(a.sample_id),'sample_id':a.sample_id}
        print(json.dumps(answer,ensure_ascii=False,indent=2));return 0 if answer['revoked'] else 1
    rows=[]
    if a.kind in ('all','verified'):
        rows.extend(dict(r,kind='verified',evidence=str(store.verified/r['id']/'evidence.json')) for r in store.entries())
    if a.kind!='verified':
        rows.extend(dict(r,kind=r['status'],evidence=str(store.queue/r['id']/'evidence.json'))
                    for r in store.entries('review') if a.kind in ('all',r['status']))
    print(json.dumps(rows,ensure_ascii=False,indent=2));return 0

if __name__=='__main__':raise SystemExit(main())
