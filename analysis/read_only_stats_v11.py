from __future__ import annotations
import csv, hashlib, math, json
from pathlib import Path
import numpy as np

ROOT=Path('/mnt/data')
ST31=ROOT/'review_st31'/'MASTER_STAGE30_E4_RESULTS.csv'
ST28=ROOT/'review_stage28'/'MASTER_STAGE28_RESULTS.csv'
OUT=ROOT/'photonics_v11_stats'
OUT.mkdir(exist_ok=True)
B=4000
TAU_FINE=435.96252114890945
TAU_ACCEPT=1e-4


def stable_seed(*parts):
    payload='|'.join(map(str,parts)).encode('utf-8')
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], 'little') % (2**32-1)

def pct_ci(vals, alpha=.05):
    return float(np.quantile(vals,alpha/2)), float(np.quantile(vals,1-alpha/2))

def boot_mean(x, seed, B=B):
    x=np.asarray(x,float); n=len(x); rng=np.random.default_rng(seed)
    vals=np.empty(B,float)
    for b in range(B):
        idx=rng.integers(0,n,size=n)
        vals[b]=x[idx].mean()
    return pct_ci(vals)

def wilson(k,n,z=1.959963984540054):
    if n==0: return (float('nan'),float('nan'))
    p=k/n; den=1+z*z/n
    center=(p+z*z/(2*n))/den
    half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return center-half,center+half

# ---------------- Stage31 ----------------
rows=[]
with ST31.open(newline='',encoding='utf-8') as f:
    for r in csv.DictReader(f):
        rr=dict(r)
        for key in ['e3_thickness_nm','e4_candidate_thickness_nm','objective_relative_improvement','tree_fine_energy_ratio','true_thickness_nm','absolute_error_e3_nm','absolute_error_e4_nm']:
            rr[key]=float(rr[key]) if rr.get(key,'') not in ('',None) else float('nan')
        rr['triggered']=rr['triggered'].lower()=='true'
        rr['inner_converged']=rr['inner_converged'].lower()=='true'
        rows.append(rr)
assert len(rows)==7200

def v2_eligible(r):
    return bool(r['triggered'] and r['inner_converged'] and np.isfinite(r['objective_relative_improvement']) and r['objective_relative_improvement']>=TAU_ACCEPT)
def fine_eligible(r):
    return bool(v2_eligible(r) and np.isfinite(r['tree_fine_energy_ratio']) and r['tree_fine_energy_ratio']<=TAU_FINE)

arr=[]
for r in rows:
    e3=r['absolute_error_e3_nm']
    cand=abs(r['e4_candidate_thickness_nm']-r['true_thickness_nm'])
    v2=v2_eligible(r); fine=fine_eligible(r)
    ae_v2=cand if v2 else e3
    ae_fine=cand if fine else e3
    arr.append((r['observation_id'],r['scenario'],r['material_id'],r['true_thickness_nm'],e3,cand,ae_v2,ae_fine,v2,fine))

# Validate canonical counts/means.
e3=np.array([x[4] for x in arr])
cand=np.array([x[5] for x in arr])
ae_v2=np.array([x[6] for x in arr]); ae_fine=np.array([x[7] for x in arr])
v2=np.array([x[8] for x in arr],bool); fine=np.array([x[9] for x in arr],bool)
dv2=ae_v2-e3; dfine=ae_fine-e3; dfineminv2=ae_fine-ae_v2
assert v2.sum()==579, v2.sum()
assert fine.sum()==485, fine.sum()
assert np.sum(dv2<0)==324 and np.sum(dv2>0)==255
assert np.sum(dfine<0)==299 and np.sum(dfine>0)==186
assert abs(dv2.mean()-(-0.001930143356463666))<1e-12
assert abs(dfine.mean()-(-0.0023350451067654554))<1e-12

stats=[]
for label,d in [('V2_final_minus_E3',dv2),('Fine_q75_final_minus_E3',dfine),('Fine_q75_final_minus_V2_final',dfineminv2)]:
    seed=stable_seed('stage31_v11','paired_mean',label)
    lo,hi=boot_mean(d,seed)
    stats.append({'contrast':label,'n':len(d),'mean_delta_AE_nm':float(d.mean()),'ci95_low_nm':lo,'ci95_high_nm':hi,'bootstrap_iterations':B,'bootstrap_seed':seed})

# Scenario-specific overall mean effects + CIs.
for scenario in ['gaussian','impulsive','baseline_drift','mixed']:
    idx=np.array([x[1]==scenario for x in arr])
    for label,d in [('V2_final_minus_E3',dv2[idx]),('Fine_q75_final_minus_E3',dfine[idx]),('Fine_q75_final_minus_V2_final',dfineminv2[idx])]:
        seed=stable_seed('stage31_v11','scenario',scenario,label)
        lo,hi=boot_mean(d,seed)
        stats.append({'contrast':f'{scenario}:{label}','n':len(d),'mean_delta_AE_nm':float(d.mean()),'ci95_low_nm':lo,'ci95_high_nm':hi,'bootstrap_iterations':B,'bootstrap_seed':seed})

# Improvement fraction Wilson intervals.
props=[]
for label,mask,d in [('V2',v2,dv2),('Frozen-Fine-q75',fine,dfine)]:
    accepted=int(mask.sum()); improved=int(np.sum(d[mask]<0)); worsened=int(np.sum(d[mask]>0))
    lo,hi=wilson(improved,accepted)
    props.append({'rule':label,'accepted':accepted,'improved':improved,'worsened':worsened,'improved_fraction':improved/accepted,'wilson95_low':lo,'wilson95_high':hi})

with (OUT/'Stage31_ReadOnly_Bootstrap_CI.csv').open('w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=list(stats[0].keys()));w.writeheader();w.writerows(stats)
with (OUT/'Stage31_Improvement_Fraction_Wilson_CI.csv').open('w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=list(props[0].keys()));w.writeheader();w.writerows(props)

# ---------------- Stage28 stratified sensitivity ----------------
# Build paired AEs by observation and method.
rec={}
with ST28.open(newline='',encoding='utf-8') as f:
    rdr=csv.DictReader(f)
    for r in rdr:
        oid=r['observation_id']; method=r['strategy']; scenario=r['scenario']; mat=r['material_id']; th=float(r['true_thickness_nm']); ae=float(r['absolute_error_nm'])
        z=rec.setdefault(oid,{'scenario':scenario,'material_id':mat,'true_thickness_nm':th})
        z[method]=ae
assert len(rec)==7200
for oid,z in rec.items():
    for m in ['E0_constrained_L2_profile','E1_full_design_one_step_score','E3_constrained_multistart_robust_profile']:
        assert m in z,(oid,m)

methods={'E0':'E0_constrained_L2_profile','E1':'E1_full_design_one_step_score','E3':'E3_constrained_multistart_robust_profile'}

def endpoint_delta(a,b,ep):
    if ep=='mean': return float(np.mean(a)-np.mean(b))
    q=.90 if ep=='q90' else .95
    return float(np.quantile(a,q)-np.quantile(b,q))

def strat_boot(scenario,ref,ep,seed,B=B):
    # fixed 9-cell design: resample 200 paired observations within each material x thickness cell.
    cells=[]
    for mat in ['A','B','C']:
        for th in [380.0,520.0,680.0]:
            zs=[z for z in rec.values() if z['scenario']==scenario and z['material_id']==mat and z['true_thickness_nm']==th]
            assert len(zs)==200,(scenario,mat,th,len(zs))
            a=np.array([z[methods['E3']] for z in zs],float)
            b=np.array([z[methods[ref]] for z in zs],float)
            cells.append((a,b))
    rng=np.random.default_rng(seed); vals=np.empty(B,float)
    for i in range(B):
        aa=[];bb=[]
        for a,b in cells:
            idx=rng.integers(0,len(a),size=len(a))
            aa.append(a[idx]);bb.append(b[idx])
        vals[i]=endpoint_delta(np.concatenate(aa),np.concatenate(bb),ep)
    return pct_ci(vals)

s28=[]
for scenario in ['gaussian','impulsive','baseline_drift','mixed']:
    zs=[z for z in rec.values() if z['scenario']==scenario]
    assert len(zs)==1800
    for ref in ['E0','E1']:
        a=np.array([z[methods['E3']] for z in zs],float)
        b=np.array([z[methods[ref]] for z in zs],float)
        for ep in ['mean','q90','q95']:
            est=endpoint_delta(a,b,ep)
            seed=stable_seed('stage28_v11','stratified',scenario,f'E3-{ref}',ep)
            lo,hi=strat_boot(scenario,ref,ep,seed)
            s28.append({'scenario':scenario,'comparison':f'E3-{ref}','endpoint':ep,'paired_n':1800,'delta_AE_nm':est,'stratified_ci95_low_nm':lo,'stratified_ci95_high_nm':hi,'bootstrap_iterations':B,'bootstrap_seed':seed})
with (OUT/'Stage28_Stratified_Bootstrap_Sensitivity.csv').open('w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=list(s28[0].keys()));w.writeheader();w.writerows(s28)

# concise report
lines=[]
lines.append('PHOTONICS v1.1 READ-ONLY STATISTICAL REANALYSIS')
lines.append('No estimator was rerun. Stage31 final rules were replayed on frozen candidate fields; Stage28 sensitivity resampled frozen paired AEs.')
lines.append('')
lines.append('Stage31 pooled paired mean contrasts (4000 bootstrap resamples):')
for s in stats[:3]:
    lines.append(f"- {s['contrast']}: mean={s['mean_delta_AE_nm']:+.9f} nm; 95% CI [{s['ci95_low_nm']:+.9f}, {s['ci95_high_nm']:+.9f}], n={s['n']}, seed={s['bootstrap_seed']}")
lines.append('')
lines.append('Stage31 improvement fractions among accepted (Wilson 95% CI):')
for p in props:
    lines.append(f"- {p['rule']}: {p['improved']}/{p['accepted']}={100*p['improved_fraction']:.2f}%; 95% CI [{100*p['wilson95_low']:.2f}%, {100*p['wilson95_high']:.2f}%]; worsened={p['worsened']}")
lines.append('')
lines.append('Stage28 material×thickness-stratified bootstrap sensitivity generated for all 24 scenario/comparison/endpoint rows. Point estimates are unchanged by construction; the CSV contains sensitivity CIs.')
(OUT/'V11_READ_ONLY_STATS_SUMMARY.txt').write_text('\n'.join(lines),encoding='utf-8')

manifest={
    'stage31_source':str(ST31),'stage31_rows':len(rows),'tau_accept':TAU_ACCEPT,'tau_fine':TAU_FINE,
    'stage31_v2_accepted':int(v2.sum()),'stage31_fine_accepted':int(fine.sum()),
    'stage28_source':str(ST28),'stage28_observations':len(rec),'bootstrap_iterations':B,
    'seed_rule':'little-endian uint64 prefix of SHA-256 labels modulo 2^32-1',
    'estimator_rerun':False,
}
(OUT/'V11_READ_ONLY_STATS_AUDIT.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print('\n'.join(lines))
