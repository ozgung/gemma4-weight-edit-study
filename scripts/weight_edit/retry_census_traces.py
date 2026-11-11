"""Descriptive census: what happens after an old_string-not-found, on held-out e001nt/e001tt traces."""
import json, collections, sys, re
from pathlib import Path
R=Path('results')
GROUPS={
 'e001nt held-out (e4t ood+oodb, wp7 base)':[('e4t','ood_e001nt_baseline_nothink_t06_e4b_cap12'),('e4t','oodb_e001nt_baseline_nothink_t06_e4b_cap12'),('wp7a','wp7_ood_e001nt_baseline_nothink_t06_e4b_cap12'),('wp7b','wp7_oodb_e001nt_baseline_nothink_t06_e4b_cap12')],
 'e001tt held-out (e4t ood+oodb)':[('e4t','ood_e001tt_think_t06_e4b_cap12'),('e4t','oodb_e001tt_think_t06_e4b_cap12')],
}
def obs(s):
    try: return json.loads((s.get('observation') or {}).get('content') or '{}')
    except Exception: return {}
def scan(p):
    t=json.load(open(p))
    ev=[]
    for s in t['steps']:
        if s.get('source')!='agent': continue
        o=obs(s); raw=json.dumps(o)[:400]
        for tc in s.get('tool_calls') or []:
            a=tc.get('arguments') or {}
            nf=('not found' in raw.lower()) and tc.get('function_name')=='edit_file' and o.get('status')!='ok'
            ev.append((tc.get('function_name'),a,o.get('status'),nf))
    return ev
for g,lst in GROUPS.items():
    tab=collections.Counter(); per=collections.Counter()
    for sess,name in lst:
        d=R/sess/'results'/name
        if not d.exists(): print('missing',d); continue
        res={json.loads(l).get('instance_id') or json.loads(l).get('task_id'):json.loads(l) for l in open(d/'task_results.jsonl')}
        for p in sorted((d/'traces').glob('*.json*')):
            try: ev=scan(p)
            except Exception as e: tab['bad']+=1; continue
            iid=p.stem.split('.')[0]
            per['traces']+=1
            edits=[e for e in ev if e[0]=='edit_file']
            if not edits: per['no_edit_calls']+=1
            ever_ok=any(e[2]=='ok' for e in edits)
            anynf=any(e[3] for e in edits)
            if anynf: per['traces_with_nf']+=1
            if anynf and not ever_ok: per['nf_and_no_edit_ever_landed']+=1
            # first not-found in the trace: what next
            for i,e in enumerate(ev):
                if not e[3]: continue
                tab['nf_total']+=1
                fp=e[1].get('filepath'); old=e[1].get('old_string')
                nxt=ev[i+1:]
                if not nxt: tab['nf_then_end']+=1; continue
                # gap until next edit_file
                j=next((k for k,x in enumerate(nxt) if x[0] in('edit_file','write_file')),None)
                if j is None: tab['nf_then_no_more_edits']+=1; continue
                between=nxt[:j]
                ne=nxt[j]
                readit=any(x[0]=='read_file' and (x[1].get('filepath') or x[1].get('path') or '')==fp for x in between)
                gotview=any(x[0] in('read_file','run_command','grep','search','find_in_files','grep_files') for x in between)
                k='reread' if readit else ('other_view' if gotview else 'no_view')
                same=ne[1].get('old_string')==old
                tab[k]+=1
                tab[(k,'same_old')]+=int(same)
                okn=ne[2]=='ok'
                tab[(k,'landed')]+=int(okn)
    print('==',g,dict(per))
    n=tab['nf_total']
    print(' not-found',n,'then_end',tab['nf_then_end'],'no_more_edits',tab['nf_then_no_more_edits'])
    for k in ('no_view','other_view','reread'):
        m=tab[k]
        if m: print(f"  {k:11s} n={m:4d} ({m/n:.0%}) next_edit_landed={tab[(k,'landed')]/m:.2f} same_old_string={tab[(k,'same_old')]/m:.2f}")
