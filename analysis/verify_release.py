#!/usr/bin/env python3
"""Portable read-only verification of the manuscript release package.

This script does not rerun E0/E1/E3/E4. It checks the released display/source tables
for the frozen counts and key numerical identities reported in the manuscript.
It uses only the Python standard library.
"""
from __future__ import annotations
import csv, math, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def read_csv(rel):
    p = ROOT / rel
    with p.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def f(x): return float(x)
def i(x): return int(float(x))
def close(a,b,tol=5e-9): return abs(a-b) <= tol

def ok(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print('PASS', msg)

# Selected-material/thickness full cell table
s5 = read_csv('tables/supplement/TableS5_Full_Cell_Level_AE_and_Failure_Summaries.csv')
ok(len(s5) == 108, 'Table S5 contains 108 material x thickness x scenario x method rows')
ok(sum(i(r['n']) for r in s5) == 21600, 'Table S5 represents 21,600 estimator records')
ok(sum(i(r['non_ok_count']) for r in s5) == 1, 'Exactly one non-ok selected-material estimator record is retained')
ok(sum(i(r['boundary_count']) for r in s5) == 1, 'Exactly one selected-material boundary event is retained')
bad = [r for r in s5 if i(r['non_ok_count'])]
ok(len(bad)==1 and bad[0]['material_id']=='C' and f(bad[0]['true_thickness_nm'])==380.0 and bad[0]['scenario']=='mixed' and bad[0]['method']=='E0',
   'The retained non-ok record is Material C / 380 nm / mixed / E0')
ok(close(f(bad[0]['max_AE_nm']), 319.99997973861395, 1e-6), 'The retained boundary-event AE is ~319.99998 nm')

# Development gate
s6a = read_csv('tables/supplement/TableS6a_Development_FineGate_Sensitivity_Overall.csv')
by_rule={r['rule_id']:r for r in s6a}
ok(i(by_rule['V2']['accepted'])==565, 'Development V2 accepted count = 565')
ok(i(by_rule['Fine-q75']['accepted'])==472, 'Development Fine-q75 accepted count = 472')

s8b = read_csv('tables/supplement/TableS8b_DevelopmentV1_CandidateAudit_FallbackReasons.csv')
dev_v1=sum(i(r['count']) for r in s8b if r['fallback_reason']=='development_v1_accepted')
ok(dev_v1==78, 'Development-v1 initial gate accepted count = 78')
ok(sum(i(r['count']) for r in s8b)==7200, 'Development-v1 fallback-reason table accounts for all 7,200 observations')

# Independent validation
s7a = read_csv('tables/supplement/TableS7a_IndependentValidation_Overall.csv')
vr={r['rule_id']:r for r in s7a}
ok(i(vr['V2']['accepted'])==579 and i(vr['V2']['improved'])==324 and i(vr['V2']['worsened'])==255,
   'Independent validation V2 counts = 579 accepted / 324 improved / 255 worsened')
# Handle either historical label or journalized label
fine_key=next(k for k in vr if 'Fine-q75' in k)
ok(i(vr[fine_key]['accepted'])==485 and i(vr[fine_key]['improved'])==299 and i(vr[fine_key]['worsened'])==186,
   'Independent validation Fine-q75 counts = 485 accepted / 299 improved / 186 worsened')
ok(close(f(vr['V2']['mean_delta_AE_vs_E3_nm']), -0.001930143356463666, 1e-12), 'V2 mean AE contrast matches manuscript')
ok(close(f(vr[fine_key]['mean_delta_AE_vs_E3_nm']), -0.0023350451067654554, 1e-12), 'Fine-q75 mean AE contrast matches manuscript')

s8c = read_csv('tables/supplement/TableS8c_Operational_Trigger_Accept_Fallback_Counts.csv')
ok(sum(i(r['observations']) for r in s8c)==7200, 'Operational validation accounting totals 7,200 observations')
ok(sum(i(r['triggered']) for r in s8c)==5654, 'Independent validation triggered count = 5,654')
ok(sum(i(r['V2_eligible']) for r in s8c)==579, 'Operational V2 eligible count = 579')
ok(sum(i(r['Fine_q75_eligible']) for r in s8c)==485, 'Operational Fine-q75 eligible count = 485')
ok(sum(i(r['final_E3_fallback']) for r in s8c)==6715, 'Final E3 fallback count = 6,715')

# Primary contrast table
m3 = read_csv('tables/main/Table3_Stage28_Primary_Paired_Statistics.csv')
ok(len(m3)==24, 'Primary selected-material contrast table contains 24 endpoints')
def find(scenario,comparison,endpoint):
    return next(r for r in m3 if r['scenario']==scenario and r['comparison']==comparison and r['endpoint']==endpoint)
ok(close(f(find('impulsive','E3-E0','mean')['delta_AE_nm']), -1.249558, 1e-6), 'Impulsive E3-E0 mean contrast = -1.249558 nm')
ok(close(f(find('impulsive','E3-E0','q95')['delta_AE_nm']), -2.040323, 1e-6), 'Impulsive E3-E0 q95 contrast = -2.040323 nm')
ok(close(f(find('baseline_drift','E3-E0','mean')['delta_AE_nm']), 0.161383, 1e-6), 'Baseline-drift E3-E0 mean contrast = +0.161383 nm')

print('\nRELEASE VERIFICATION PASS')
print('No estimator was rerun.')
