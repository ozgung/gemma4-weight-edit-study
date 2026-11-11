"""Descriptive, burned gate set: first attempts vs retries after a recorded not_found, by arm."""
import json, collections
from pathlib import Path
R=Path('outputs/weight_edit/claude')
allc=[json.loads(l) for l in open(R/'replay_ood/calls.jsonl')]
by=collections.defaultdict(list)
for c in allc: by[(c['run'],c['instance_id'])].append(c)
state={}
for t,cs in by.items():
    cs.sort(key=lambda c:(c['item'],c['tc']))
    failed=[]; 
    for c in cs:
        state[c['key']]=dict(after_fail=bool(failed), prev=[x for x in failed])
        if c['error_kind']=='not_found': failed.append(c['old_string'])
print('held-out recorded not_found:',sum(c['error_kind']=='not_found' for c in allc),' of which after an earlier not_found in the task:',sum(c['error_kind']=='not_found' and state[c['key']]['after_fail'] for c in allc))
use=set(l.strip() for l in open(R/'replay_ood/usable_keys.txt'))
print('usable first/retry calls:',sum(not state[k]['after_fail'] for k in use),sum(state[k]['after_fail'] for k in use))
print(f"{'arm':20s} {'first acc':>12s} {'retry acc':>12s} {'identical-to-a-failed-string':>30s} {'first exact-same-as-base':>26s}")
base={(x['key'],x['draw']):x for x in map(json.loads,open(R/'gate_r2/gen_base.jsonl'))}
for arm in ['base','opus_r2','fable_r2','opusinf_r2','fable_r2_rand1','opusinf_r2_rand1']:
    s={(x['key'],x['draw']):x for x in map(json.loads,open(R/f'gate_r2/gen_{arm}.scored.jsonl'))}
    g={(x['key'],x['draw']):x for x in map(json.loads,open(R/f'gate_r2/gen_{arm}.jsonl'))}
    f=[0,0];r=[0,0];same=0;fs=0
    for k,x in s.items():
        st=state[k[0]]
        (r if st['after_fail'] else f)[0]+=x['accepted']; (r if st['after_fail'] else f)[1]+=1
        if st['after_fail'] and g[k]['gen'] in st['prev']: same+=1
        if not st['after_fail'] and g[k]['gen']==base[k]['gen']: fs+=1
    print(f"{arm:20s} {f[0]:4d}/{f[1]:3d}={f[0]/f[1]:.2f} {r[0]:4d}/{r[1]:3d}={r[0]/r[1]:.2f} {same:10d} of {r[1]} ({same/r[1]:.0%}) {fs:18d} of {f[1]}")
