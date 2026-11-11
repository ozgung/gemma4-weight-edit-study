import json, collections, math
from pathlib import Path
R=Path('outputs/weight_edit/claude')
allc=[json.loads(l) for l in open(R/'replay_ood/calls.jsonl')]
by=collections.defaultdict(list)
for c in allc: by[(c['run'],c['instance_id'])].append(c)
st={}
for t,cs in by.items():
    cs.sort(key=lambda c:(c['item'],c['tc'])); f=[]
    for c in cs:
        st[c['key']]=list(f)
        if c['error_kind']=='not_found': f.append(c['old_string'])
s={(x['key'],x['draw']):x for x in map(json.loads,open(R/'gate_r2/gen_base.scored.jsonl'))}
g={(x['key'],x['draw']):x for x in map(json.loads,open(R/'gate_r2/gen_base.jsonl'))}
idn=[0,0];non=[0,0];tasks=set()
for k,x in s.items():
    p=st[k[0]]
    if not p: continue
    tasks.add(k[0].split('|')[1])
    b=idn if g[k]['gen'] in p else non
    b[0]+=x['accepted'];b[1]+=1
print('retry gens identical accepted',idn,'non-identical accepted',non,'tasks with usable retries',len(tasks))
def sign(a,b):
    n=a+b;return sum(math.comb(n,i) for i in range(a,n+1))/2**n
def rate(arm):
    t=collections.defaultdict(lambda:[0,0])
    for x in map(json.loads,open(R/f'gate_r2/gen_{arm}.scored.jsonl')):
        i=x['key'].split('|')[1]; t[i][0]+=x['exact_copy'];t[i][1]+=1
    return {k:v[0]/v[1] for k,v in t.items()}
b=rate('base')
for arm in ['opus_r2','fable_r2','opusinf_r2','fable_r2_rand1','opusinf_r2_rand1']:
    r=rate(arm);up=sum(r[k]>b[k]+1e-9 for k in b);dn=sum(r[k]<b[k]-1e-9 for k in b)
    print(f'{arm:18s} strict task-macro delta {sum(r[k]-b[k] for k in b)/len(b):+.3f} up/down {up}/{dn} p={sign(up,dn):.2g}')
