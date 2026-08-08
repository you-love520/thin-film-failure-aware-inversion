from __future__ import annotations

from pathlib import Path
import zipfile, hashlib, json, shutil, math, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path('/mnt/data')
OUT = ROOT / 'photonics_supplement_v01'
FIG = OUT / 'figures'
TAB = OUT / 'tables'
DAT = OUT / 'figure_data'
PROV = OUT / 'provenance'
for d in [OUT,FIG,TAB,DAT,PROV]:
    d.mkdir(parents=True, exist_ok=True)

STAGE28_ZIP = ROOT/'STAGE28_RESULTS_HANDOFF.zip'
STAGE30_ZIP = ROOT/'STAGE30_E4_RESULTS_HANDOFF (1).zip'
STAGE31_ZIP = ROOT/'STAGE31_E4_FINEQ75_VALIDATION_HANDOFF.zip'
CLOUD_ZIP = ROOT/'STAGE30_E4_HORP_CLOUD_PACKAGE.zip'
REF_PKG = ROOT/'E4_MANUSCRIPT_REFERENCE_PACKAGE(1).zip'
PRIMARY_STATS = ROOT/'photonics_v08_artifacts/tables/Table3_Stage28_Primary_Paired_Statistics.csv'
STRAT_STATS = ROOT/'photonics_v11_stats/Stage28_Stratified_Bootstrap_Sensitivity.csv'
ST31_CI = ROOT/'photonics_v11_stats/Stage31_ReadOnly_Bootstrap_CI.csv'
OLD_SHEFFIELD = ROOT/'_docx_media/image8.png'

TAU_FINE = 435.96252114890945
TAU_V2 = 1e-4


def sha256_file(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def save_figure(fig, stem: str, dpi=800):
    png=FIG/f'{stem}.png'; pdf=FIG/f'{stem}.pdf'; svg=FIG/f'{stem}.svg'; tif=FIG/f'{stem}.tiff'
    fig.savefig(png,dpi=dpi,bbox_inches='tight')
    fig.savefig(pdf,bbox_inches='tight')
    fig.savefig(svg,bbox_inches='tight')
    fig.savefig(tif,dpi=dpi,bbox_inches='tight',pil_kwargs={'compression':'tiff_lzw'})
    plt.close(fig)
    return [png,pdf,svg,tif]

# -------------------- Read authoritative results --------------------
use28=['observation_id','scenario','strategy','true_thickness_nm','estimate_nm','absolute_error_nm','runtime_ms','boundary_hit','status','material_id','film_material','substrate_material','converged','failure_code','candidate_count','valid_candidate_count']
with zipfile.ZipFile(STAGE28_ZIP) as z:
    df28=pd.read_csv(z.open('MASTER_STAGE28_RESULTS.csv'), usecols=use28)
    aud28=pd.read_csv(z.open('MASTER_STAGE28_OBSERVATION_AUDIT.csv'))
    exec28=json.load(z.open('STAGE28_EXECUTION_REPORT.json'))

cols_e4=['observation_id','scenario','triggered','accepted','fallback_reason','objective_relative_improvement','baseline_gain_gb','baseline_amplitude_ratio_ab','first_order_projection_qc','second_order_leakage','identifiability_retention_ratio','trust_radius_nm','profile_curvature','drift_effective_rank','drift_condition_number','drift_active_columns','second_order_protection_active','inner_converged','inner_iterations','outer_nfev','runtime_e4_seconds','tree_total_energy','tree_fine_energy','tree_fine_energy_ratio','material_id','true_thickness_nm','e3_thickness_nm','e4_candidate_thickness_nm','absolute_error_e3_nm','absolute_error_e4_nm']
with zipfile.ZipFile(STAGE30_ZIP) as z:
    d30=pd.read_csv(z.open('MASTER_STAGE30_E4_RESULTS.csv'), usecols=cols_e4)
    sens30=pd.read_csv(z.open('E4_FINE_SCALE_GATE_SENSITIVITY.csv'))
    sens30_sc=pd.read_csv(z.open('E4_FINE_SCALE_GATE_BY_SCENARIO.csv'))
    rep30=json.load(z.open('STAGE30_E4_EXECUTION_REPORT.json'))
with zipfile.ZipFile(STAGE31_ZIP) as z:
    d31=pd.read_csv(z.open('MASTER_STAGE30_E4_RESULTS.csv'), usecols=cols_e4)
    val31=pd.read_csv(z.open('E4_FINEQ75_INDEPENDENT_VALIDATION.csv'))
    val31_sc=pd.read_csv(z.open('E4_FINEQ75_INDEPENDENT_VALIDATION_BY_SCENARIO.csv'))
    rep31=json.load(z.open('STAGE30_E4_EXECUTION_REPORT.json'))

primary=pd.read_csv(PRIMARY_STATS)
strat=pd.read_csv(STRAT_STATS)
ci31=pd.read_csv(ST31_CI)

# -------------------- Table S1: material / physics provenance --------------------
S1=pd.DataFrame([
    ['A','fused silica','N-BK7','analytic Sellmeier','Malitson fused-silica coefficients; SCHOTT N-BK7 coefficients','420–780','TMM, coherent single layer','8','0.6'],
    ['B','SiO2','Si','frozen n,k tables','KLA/Filmetrics optical-constants package; Si source record cites Palik','420–780','TMM, coherent single layer','8','0.6'],
    ['C','SiO2 buffer','soda-lime glass','epsilon-to-(n,k) normalized tables','Zenodo 10.5281/zenodo.15055400; passive k>=0 conversion','420–780','TMM, coherent single layer','8','0.6'],
], columns=['material_id','film','substrate','optical_constant_mode','source/provenance','wavelength_nm','forward_model','incidence_deg','spectral_blur_sigma_nm'])
S1.to_csv(TAB/'TableS1_Material_and_ForwardModel_Provenance.csv',index=False)

# -------------------- Table S2: E0/E1/E3 numerical settings --------------------
S2=pd.DataFrame([
    ['Common','thickness bounds','350–700 nm'],['Common','coarse grid step','5 nm'],['Common','local x tolerance','1e-6 nm'],['Common','gain bounds','0–2'],['Common','baseline basis','Legendre P0,P1,P2 on x in [-1,1]'],['Common','gain starts','0, 0.5, 1.0, 1.5, 2.0'],['Common','adaptive scale','preliminary constrained-L2 profile -> full-design residual -> MAD'],['Common','MAD factor / floor','1.4826 / 1e-6'],['E1/E3','Tukey tuning c','4.685'],['E3','maximum IRLS iterations','100'],['E3','coefficient-relative tolerance','1e-8'],['E3','objective-relative tolerance','1e-10'],['E3','local-minimum relative tolerance','1e-12'],['E3','candidate deduplication','1e-4 nm'],['E3','candidate rule','all finite E3 robust-profile local minima; no fixed top-k'],['E3','selection rule','lowest finite eligible successful E3 objective; failed candidate retained when no eligible success'],
], columns=['scope','parameter','frozen_value'])
S2.to_csv(TAB/'TableS2_E0_E1_E3_Numerical_Settings.csv',index=False)

# -------------------- Table S3: E4 frozen parameters --------------------
S3=pd.DataFrame([
    ['derivatives','first_derivative_step_nm',0.25],['derivatives','second_derivative_step_nm',0.25],['derivatives','second_order_rank_tolerance',1e-10],
    ['residual_tree','tree_depth',2],['residual_tree','window_overlap_fraction',0.25],['residual_tree','basis_per_node','local_constant; local_linear; local_quadratic'],['residual_tree','rank_tolerance',1e-10],['residual_tree','condition_limit',1e8],['residual_tree','level_weights','1.0; 1.5; 2.25'],
    ['physical_projection','eta2',0.5],['physical_projection','first_order_projection_tolerance',1e-8],
    ['optimization','lambda_eta',0.05],['optimization','lambda_tree',0.02],['optimization','inner_maxiter',60],['optimization','inner_tolerance',1e-7],['optimization','outer_solver','bounded scalar minimization'],['optimization','outer_xatol_nm',0.001],['optimization','outer_max_nfev',80],
    ['trust_region','R_min_nm',2.0],['trust_region','R_max_nm',15.0],['trust_region','curvature_step_nm',0.5],['trust_region','fallback_radius_nm',10.0],['trust_region','boundary_tolerance_nm',0.001],
    ['trigger','G_B_min',0.002],['trigger','A_B_max',1.0],['trigger','R_I_min',0.5],['trigger','L2_max',0.25],['trigger','Q1_max',1e-8],
    ['operational_V2','G_A_min',1e-4],['Fine-q75','I_fine_max',TAU_FINE],
], columns=['block','parameter','frozen_value'])
S3.to_csv(TAB/'TableS3_E4_Frozen_Parameters.csv',index=False)

# -------------------- Table S4: simulation generator / RNG --------------------
S4=pd.DataFrame([
    ['gaussian',0.0008,0,0,0,0.005,0.0002,0],
    ['impulsive',0.0004,0.04,0.012,0,0.005,0,14],
    ['baseline_drift',0.0004,0,0,0.0035,0.005,0,0],
    ['mixed',0.0008,0.06,0.015,0.0035,0.008,0.0003,22],
],columns=['scenario','gaussian_sd','outlier_ratio','nominal_outlier_magnitude','baseline_amplitude','gain_sd','offset_sd','registered_outlier_count_361'])
S4.to_csv(TAB/'TableS4_Contamination_and_RNG_Design.csv',index=False)

# -------------------- Table S5: full cell-level Stage28 stats --------------------
method_map={'E0_constrained_L2_profile':'E0','E1_full_design_one_step_score':'E1','E3_constrained_multistart_robust_profile':'E3'}
df28['method']=df28['strategy'].map(method_map)
cell=(df28.groupby(['material_id','true_thickness_nm','scenario','method'],sort=True)
      .agg(n=('absolute_error_nm','size'), mean_AE_nm=('absolute_error_nm','mean'), median_AE_nm=('absolute_error_nm','median'), q90_AE_nm=('absolute_error_nm',lambda x: x.quantile(.90)), q95_AE_nm=('absolute_error_nm',lambda x: x.quantile(.95)), max_AE_nm=('absolute_error_nm','max'), boundary_count=('boundary_hit','sum'), non_ok_count=('status',lambda x:(x!='ok').sum()), convergence_count=('converged','sum'))
      .reset_index())
cell.to_csv(TAB/'TableS5_Full_Cell_Level_AE_and_Failure_Summaries.csv',index=False)

# -------------------- Table S6: development gate sensitivity --------------------
sens30.to_csv(TAB/'TableS6a_Development_FineGate_Sensitivity_Overall.csv',index=False)
sens30_sc.to_csv(TAB/'TableS6b_Development_FineGate_Sensitivity_ByScenario.csv',index=False)

# -------------------- Table S7: independent validation + CI --------------------
# overall rows
ci_over=ci31[~ci31['contrast'].str.contains(':')].copy()
# Create readable validation table
val31_aug=val31.copy()
ci_map={r['contrast']:r for _,r in ci_over.iterrows()}
# map only vs E3 rows to V2/Fine; plus direct difference as separate row
rows=[]
for _,r in val31_aug.iterrows():
    c='V2_final_minus_E3' if r.rule_id=='V2' else 'Fine_q75_final_minus_E3'
    cr=ci_map[c]
    rows.append([r.rule_id,r.frozen_tau_fine,r.accepted,r.improved,r.worsened,r.unchanged,r.improved_fraction_among_accepted,r.mean_delta_AE,cr.ci95_low_nm,cr.ci95_high_nm])
S7a=pd.DataFrame(rows,columns=['rule_id','tau_fine','accepted','improved','worsened','unchanged','improved_fraction_among_accepted','mean_delta_AE_vs_E3_nm','ci95_low_nm','ci95_high_nm'])
S7a.to_csv(TAB/'TableS7a_IndependentValidation_Overall.csv',index=False)
# scenario + scenario CI
rows=[]
for _,r in val31_sc.iterrows():
    prefix='V2_final_minus_E3' if r.rule_id=='V2' else 'Fine_q75_final_minus_E3'
    c=f"{r.scenario}:{prefix}"
    cr=ci31[ci31.contrast==c].iloc[0]
    rows.append([r.rule_id,r.scenario,r.accepted,r.improved,r.worsened,r.accepted_only_mean_delta_AE,cr.mean_delta_AE_nm,cr.ci95_low_nm,cr.ci95_high_nm])
S7b=pd.DataFrame(rows,columns=['rule_id','scenario','accepted','improved','worsened','accepted_only_mean_delta_AE_nm','overall_mean_delta_AE_vs_E3_nm','ci95_low_nm','ci95_high_nm'])
S7b.to_csv(TAB/'TableS7b_IndependentValidation_ByScenario.csv',index=False)
ci31.to_csv(TAB/'TableS7c_IndependentValidation_All_ReadOnly_Bootstrap_CI.csv',index=False)

# -------------------- Table S8: failures / fallbacks --------------------
st28_fail=(df28.groupby(['method','status']).size().rename('count').reset_index())
st28_fail.to_csv(TAB/'TableS8a_Stage28_Status_Counts.csv',index=False)
# Stage31 full candidate-generation fallback reasons + operational replay counts
fb31=d31.groupby(['scenario','fallback_reason']).size().rename('count').reset_index()
fb31.to_csv(TAB/'TableS8b_E4_CandidateGeneration_FallbackReasons.csv',index=False)
v2mask=d31.triggered & d31.inner_converged & (d31.objective_relative_improvement>=TAU_V2)
finemask=v2mask & (d31.tree_fine_energy_ratio<=TAU_FINE)
oper=[]
for sc,g in d31.groupby('scenario'):
    idx=g.index
    oper.append([sc,len(g),int(d31.loc[idx,'triggered'].sum()),int(v2mask.loc[idx].sum()),int(finemask.loc[idx].sum()),int((~finemask.loc[idx]).sum())])
S8c=pd.DataFrame(oper,columns=['scenario','observations','triggered','V2_eligible','Fine_q75_eligible','final_E3_fallback'])
S8c.to_csv(TAB/'TableS8c_Operational_Trigger_Accept_Fallback_Counts.csv',index=False)

# -------------------- Table S9: runtime summaries --------------------
runtime_rows=[]
for m,g in df28.groupby('method'):
    sec=g.runtime_ms/1000.0
    runtime_rows.append([m,len(sec),sec.median(),sec.quantile(.90),sec.quantile(.95),sec.max()])
trig31=d31[d31.triggered]
for sc,g in trig31.groupby('scenario'):
    sec=g.runtime_e4_seconds
    runtime_rows.append([f'E4_incremental_triggered:{sc}',len(sec),sec.median(),sec.quantile(.90),sec.quantile(.95),sec.max()])
S9=pd.DataFrame(runtime_rows,columns=['group','n','median_s','q90_s','q95_s','max_s'])
S9.to_csv(TAB/'TableS9_Reference_Implementation_Runtime_Summaries.csv',index=False)

# -------------------- Table S10: Sheffield summary (frozen manuscript source) --------------------
S10=pd.DataFrame([
    ['SAMPLE1',476.3,'E0',18,18,482.44,4.75,57.12,0],
    ['SAMPLE1',476.3,'E1',18,18,481.78,6.10,46.91,0],
    ['SAMPLE1',476.3,'E3',18,18,481.94,5.95,57.39,0],
    ['SAMPLE2',198.7,'E0',18,14,391.36,np.nan,244.8,4],
    ['SAMPLE2',198.7,'E1',18,15,392.98,np.nan,269.6,3],
    ['SAMPLE2',198.7,'E3',18,18,391.53,np.nan,283.2,0],
],columns=['sample','vendor_reference_nm','method','acquisitions','valid_completions','median_valid_estimate_nm','IQR_nm_if_frozen_summary_available','conditional_mean_AE_nm','upper_boundary_hits'])
S10.to_csv(TAB/'TableS10_Sheffield_FailureMode_Summary.csv',index=False)

# -------------------- Table S11: bootstrap sensitivity --------------------
strat.to_csv(TAB/'TableS11_Stage28_Stratified_Bootstrap_Sensitivity.csv',index=False)

# -------------------- Figure S1: Sheffield full acquisition-level visualization --------------------
# Reuse the 800-dpi canonical visualization embedded in the prior Photonics draft; no data are reconstructed from pixels.
if not OLD_SHEFFIELD.exists():
    raise FileNotFoundError(OLD_SHEFFIELD)
shutil.copy2(OLD_SHEFFIELD, FIG/'FigureS1_Sheffield_AcquisitionLevel_FailureMode.png')
im=Image.open(OLD_SHEFFIELD)
im.save(FIG/'FigureS1_Sheffield_AcquisitionLevel_FailureMode.tiff',dpi=(800,800),compression='tiff_lzw')
im.convert('RGB').save(FIG/'FigureS1_Sheffield_AcquisitionLevel_FailureMode.pdf',resolution=800.0)

# -------------------- Figure S2: pooled vs stratified bootstrap sensitivity --------------------
merge=primary.merge(strat,on=['scenario','comparison','endpoint','paired_n','delta_AE_nm'],suffixes=('_pooled','_strat'))
merge['label']=merge.apply(lambda r:f"{r.scenario.replace('_',' ')} | {r.comparison} | {r.endpoint}",axis=1)
merge.to_csv(DAT/'FigureS2_BootstrapSensitivity_Data.csv',index=False)
# order by scenario, comparison, endpoint
scenario_order=['gaussian','impulsive','baseline_drift','mixed']; comp_order=['E3-E0','E3-E1']; ep_order=['mean','q90','q95']
merge['ord']=merge.apply(lambda r:scenario_order.index(r.scenario)*6+comp_order.index(r.comparison)*3+ep_order.index(r.endpoint),axis=1)
merge=merge.sort_values('ord').reset_index(drop=True)
y=np.arange(len(merge))
fig,ax=plt.subplots(figsize=(9.2,11.5))
ax.axvline(0,ls='--',lw=1)
for i,r in merge.iterrows():
    ax.errorbar(r.delta_AE_nm,i-0.12,xerr=[[r.delta_AE_nm-r.ci95_low_nm],[r.ci95_high_nm-r.delta_AE_nm]],fmt='o',capsize=2,label='Pooled observation bootstrap' if i==0 else None)
    ax.errorbar(r.delta_AE_nm,i+0.12,xerr=[[r.delta_AE_nm-r.stratified_ci95_low_nm],[r.stratified_ci95_high_nm-r.delta_AE_nm]],fmt='s',capsize=2,label='Material×thickness-stratified bootstrap' if i==0 else None)
ax.set_yticks(y); ax.set_yticklabels(merge.label)
ax.invert_yaxis(); ax.set_xlabel('AE contrast (nm); negative favors E3'); ax.set_title('Supplementary Figure S2. Bootstrap-resampling sensitivity')
ax.legend(frameon=False,loc='lower right'); ax.grid(axis='x',alpha=.2)
save_figure(fig,'FigureS2_Stage28_Bootstrap_Resampling_Sensitivity')

# -------------------- Figure S3: I_fine vs candidate AE change --------------------
v2=d31[v2mask].copy()
v2['candidate_delta_AE_nm']=np.abs(v2.e4_candidate_thickness_nm-v2.true_thickness_nm)-np.abs(v2.e3_thickness_nm-v2.true_thickness_nm)
v2['log10_1plus_Ifine']=np.log10(1+np.maximum(v2.tree_fine_energy_ratio,0))
v2.to_csv(DAT/'FigureS3_FineIndex_vs_CandidateDeltaAE_Data.csv',index=False)
fig,ax=plt.subplots(figsize=(8.8,6.2))
markers={'gaussian':'o','impulsive':'^','baseline_drift':'s','mixed':'D'}
for sc,g in v2.groupby('scenario'):
    ax.scatter(g.log10_1plus_Ifine,g.candidate_delta_AE_nm,s=24,alpha=.65,marker=markers.get(sc,'o'),label=sc.replace('_',' '))
ax.axhline(0,ls='--',lw=1)
ax.axvline(np.log10(1+TAU_FINE),ls=':',lw=1.5,label='Frozen q75 threshold')
ax.set_xlabel(r'$\log_{10}(1+I_{fine})$')
ax.set_ylabel('Candidate AE − E3 AE (nm)')
ax.set_title('Supplementary Figure S3. Fine-scale index versus candidate error change')
ax.legend(frameon=False,ncol=2); ax.grid(alpha=.2)
save_figure(fig,'FigureS3_FineIndex_vs_Candidate_AE_Change')

# -------------------- Figure S4: development gate sensitivity --------------------
sens30.to_csv(DAT/'FigureS4_DevelopmentGateSensitivity_Data.csv',index=False)
fig,ax=plt.subplots(figsize=(8.8,5.8))
x=np.arange(len(sens30)); width=.24
ax.bar(x-width,sens30.improved,width,label='Improved')
ax.bar(x,sens30.worsened,width,label='Worsened')
ax.bar(x+width,sens30.accepted,width,label='Accepted')
ax.set_xticks(x); ax.set_xticklabels(sens30.rule_id)
ax.set_ylabel('Number of observations')
ax.set_title('Supplementary Figure S4. Fine-scale gate sensitivity on development candidates')
ax.legend(frameon=False); ax.grid(axis='y',alpha=.2)
save_figure(fig,'FigureS4_Development_FineGate_Sensitivity')

# -------------------- Figure S5: reference runtime distributions --------------------
rt=[]
for m,g in df28.groupby('method'):
    rt.append((m,(g.runtime_ms/1000).to_numpy()))
rt.append(('E4 incremental\n(triggered validation)',trig31.runtime_e4_seconds.to_numpy()))
# save long data
pd.concat([pd.DataFrame({'group':name,'runtime_s':vals}) for name,vals in rt],ignore_index=True).to_csv(DAT/'FigureS5_Runtime_Data.csv',index=False)
fig,ax=plt.subplots(figsize=(8.5,6.2))
ax.boxplot([vals for _,vals in rt],tick_labels=[name for name,_ in rt],showfliers=False)
ax.set_yscale('log'); ax.set_ylabel('Wall-clock time per execution (s; log scale)')
ax.set_title('Supplementary Figure S5. Reference-implementation runtime distributions')
ax.grid(axis='y',alpha=.2,which='both')
plt.setp(ax.get_xticklabels(),rotation=15,ha='right')
save_figure(fig,'FigureS5_Reference_Runtime_Distributions')

# -------------------- Supplementary text --------------------
si = r'''SUPPLEMENTARY INFORMATION

Failure-Mechanism-Aware Robust Profiling with Selective Structured-Residual Refinement for Thin-Film Thickness Inversion

This Supplementary Information documents numerical implementation details, complete simulation settings, sensitivity analyses, failure accounting, and provenance that are intentionally condensed in the main manuscript. Internal project labels are included only where needed to connect the public scientific description to frozen computational artifacts.

S1. Forward-model numerical details
-----------------------------------
The forward model is a coherent characteristic-matrix calculation for one homogeneous isotropic film on a semi-infinite substrate. The ambient index is 1. The wavelength grid is 420–780 nm inclusive with 361 equally spaced samples (1 nm spacing), and the nominal incidence angle is 8 degrees. Unpolarized reflectance is the arithmetic mean of the s- and p-polarized intensity reflectances.

For a complex refractive index n_j, the transmitted-angle cosine is evaluated from

    sin(theta_j) = n_0 sin(theta_0) / n_j,
    cos(theta_j) = sqrt(1 - sin(theta_j)^2 + 0j).

The square-root branch is flipped when Re(cos(theta_j)) < 0, or when Re(cos(theta_j)) is numerically zero and Im(cos(theta_j)) < 0. This implements the forward/passive branch convention used by the frozen optics code. Optical admittances are eta_s=n cos(theta) and eta_p=n/cos(theta).

Material A uses the analytic fused-silica and N-BK7 Sellmeier relations listed in the main references. Material B uses frozen SiO2 and Si n,k tables. Material C uses frozen SiO2-buffer and soda-lime-glass tables converted from epsilon1,epsilon2 through

    rho = sqrt(epsilon1^2 + epsilon2^2),
    n = sqrt((rho + epsilon1)/2),
    k = sqrt((rho - epsilon1)/2), k>=0.

Tabulated optical constants are used only inside their frozen wavelength support; no extrapolation is permitted by the material adapter. After reflectance calculation, Gaussian spectral blur is applied with sigma=0.6 nm. With the 1 nm wavelength grid, the code passes sigma=0.6 samples to scipy.ndimage.gaussian_filter1d and uses mode="nearest" at the wavelength boundaries. Table S1 records the material provenance.

S2. E0/E1/E3 implementation details
------------------------------------
The shared nuisance design at a trial thickness d is [f(d), B2], where B2 contains Legendre P0,P1,P2 on the normalized wavelength coordinate. Gain is bounded to [0,2]; the three baseline coefficients are unbounded.

Observation-level robust scale. A preliminary complete constrained-L2 thickness profile is solved first. At that preliminary thickness, the full gain+baseline design is refitted by bounded L2 and the residual scale is

    s_MAD = 1.4826 median |r - median(r)|.

If s_MAD is nonfinite or below 1e-6, the implementation uses centered RMS,

    s_RMS = sqrt(mean((r-median(r))^2)),
    s = max(s_RMS,1e-6).

The selected-material/thickness observations did not require this degenerate-scale fallback. The resulting scale is fixed for the observation's E1/E3 comparisons; it is not re-estimated at every thickness.

Robust IRLS. For Tukey c=4.685, the mathematical weight is [1-(u/c)^2]^2 for |u|<=c and zero otherwise. In the E3 weighted-L2 subproblem only, numerical weights are clipped to [1e-12,1] before taking square roots. The objective used for selection remains the exact Tukey rho sum, so the weight floor is a numerical linear-solve safeguard rather than a change to the reported objective.

E3 candidate discovery. The repaired E3 implementation evaluates E3's own robust profile on the 5 nm grid, identifies every finite discrete local-minimum plateau, and chooses a deterministic representative for each plateau. A run is a local minimum if its representative is no greater than finite neighboring runs and is strictly lower than at least one neighbor, with boundary runs retained when appropriate. Overlapping refinement intervals are split at the midpoint of their representative grid locations. Each candidate is refined between immediate coarse-grid neighbors, and refined candidates within 1e-4 nm are deduplicated. No fixed top-k number of basins is imposed. Every retained refined thickness is validated with deterministic gain starts {0,0.5,1.0,1.5,2.0}; the lowest finite eligible successful robust objective is selected. If no eligible candidate succeeds, the lowest finite unique candidate is retained and marked failed instead of being deleted. The full frozen settings are in Table S2.

S3. E4-HORP physical protection, derivatives, and residual tree
---------------------------------------------------------------
E4 starts from the E3 anchor (d3,g3,beta3,s3). First- and second-order finite differences use 0.25 nm nominal steps. For the first derivative, a centered difference is used when both sides lie inside the global thickness interval; a one-sided difference is used at a global boundary. For the second derivative, equal centered spacing uses the standard three-point second difference; clipped unequal spacing uses

    f''(d) = 2[ f(left)/(h1(h1+h2)) - f(d)/(h1 h2) + f(right)/(h2(h1+h2)) ].

If both sides of d are not available, the implementation returns a zero second-derivative vector and second-order protection is inactive for that observation.

The raw residual tree has depth 2. Level l contains 2^l wavelength nodes. Each node uses a smooth local window and three atoms: local constant, local linear, and centered local quadratic. The local coordinate is xi=(x-center)/halfwidth clipped to [-1,1], and the quadratic atom subtracts its window-weighted mean xi^2. After first- and partial second-order physical projection, columns with weighted norm below 1e-10 are removed and surviving columns are scaled to weighted unit norm.

Let T1=[f(d3),B2,g3 f'(d3)]. The raw residual dictionary is weighted-residualized against T1. The residualized second-order direction h2_perp is then used for partial projection with eta2=0.5. The effective rank is computed from singular values of W^(1/2)S_phys using a relative tolerance of 1e-10; the corresponding weighted condition number is stored. The first-order projection QC is

    Q1 = ||T1^T W S_phys||_F /
         (||W^(1/2)T1||_F ||W^(1/2)S_phys||_F + eps).

For the fitted residual-tree correction b=S_phys gamma, the second-order leakage diagnostic is

    L2 = (h2_perp^T W b)^2 /
         [(h2_perp^T W h2_perp)(b^T W b)+eps].

The tree graph penalty contains one level-weighted diagonal penalty row per retained atom and an additional parent-child same-basis difference row whenever the parent atom survives projection. Level weights are [1,1.5,2.25]. Complete E4 settings are in Table S3.

S4. Development-v1 safety gate and final operational replay
------------------------------------------------------------
Candidate generation used the frozen trigger

    G_B >= 0.002,
    A_B <= 1,
    R_I >= 0.5,
    L2 <= 0.25,
    Q1 <= 1e-8.

The original development-v1 acceptance gate then required a finite candidate strictly inside the trust region, successful inner convergence, candidate gain in [0,2], objective relative improvement >=5e-4, R_I>=0.5, L2<=0.25, and nonnegative four-window consensus.

For four overlapping wavelength windows, local Tukey data-fit losses J3,k and J4,k were computed from the fixed E3 and candidate predictions. Window improvement was

    Delta_k=(J3,k-J4,k)/max(|J3,k|,eps),

and the recorded consensus statistics were median(Delta_k) and MAD(Delta_k). The windows used 25% overlap. These diagnostics belong to development-v1 and are not additional predicates in the final V2 replay.

The operational V2 replay that exactly reproduces the frozen development and independent-seed counts is

    V2_eligible = triggered AND inner_converged AND G_A>=1e-4,

where G_A=(J_anchor-J_candidate)/max(|J_anchor|,eps). The replay acts on already generated candidates; it does not rerun the optimizer.

Fine-q75. For retained residual-tree level l, let b_l denote the fitted correction contributed by that level after physical projection and scaling. The audited fine-scale index is

    A_total = ||W_3^(1/2) sum_l b_l||_2^2,
    A_fine  = sum_{l in L_fine} ||W_3^(1/2)b_l||_2^2,
    I_fine  = A_fine/(A_total+1e-12),

where L_fine is the deepest one or two retained levels (levels[-2:] when two or more levels survive; otherwise the deepest surviving level). Because level contributions are not treated as an orthogonal energy decomposition, I_fine is not bounded by one. The development q75 among triggered and inner-converged candidates was 435.96252114890945. The final rule is

    Fine_q75_eligible = V2_eligible AND I_fine<=435.96252114890945.

Table S6 and Figure S4 report the development sensitivity. The independent-seed validation outcomes are in Table S7.

S5. Simulation generator, RNG, and observation audit
----------------------------------------------------
For wavelength index j,

    y_j = G R_j(d_true,M) + O + B_j + N_j + I_j.

Here G=1+Normal(0,gain_sd), O=Normal(0,offset_sd), and N_j are iid Normal(0,gaussian_sd). No clipping, renormalization, or post-generation correction is applied.

With x mapping wavelength to [-1,1], smooth drift is generated as

    raw(x)=c0+c1 x+c2 x^2+c3 sin(1.35 pi x+phi),
    c ~ Normal([0.15,0.55,0.30,0.18],[0.05,0.08,0.08,0.04]),
    phi ~ Uniform(-0.5,0.5).

After removing the mean, raw is divided by max(max|raw|,1e-12) and multiplied by the registered baseline amplitude. The drift is therefore intentionally not exactly contained in the fitted quadratic Legendre nuisance space.

The impulsive count is round(outlier_ratio*361). Positions are sampled uniformly without replacement, signs are independent equiprobable +/-1, and absolute magnitudes equal nominal_outlier_magnitude*Uniform(0.75,1.25). This yields 14 outliers in the impulsive condition and 22 in mixed contamination.

For the selected-material/thickness study, master_seed=20260726 and

    scenario_id = 280000 + 1000*material_index + 100*thickness_index + scenario_index,
    SeedSequence entropy=[master_seed,scenario_id,trial_id],
    trial_id=0,...,199,
    bit generator=PCG64.

All E0/E1/E3 rows for an observation reuse the same realized spectrum. The execution audit stores SHA-256 hashes for clean spectrum, observed spectrum, Gaussian component, baseline, outlier indices and values, together with realized gain/offset, component summaries, and the independently recomputed adaptive scales. Table S4 lists the scenario parameters.

Supplementary Results
=====================

SR1. Sheffield acquisition-level failure-mode visualization
-----------------------------------------------------------
Supplementary Figure S1 reproduces the complete acquisition-level thickness-estimate visualization from the frozen Sheffield analysis. Filled markers denote valid completions; open X symbols denote retained 1000 nm upper-boundary failures; dotted lines denote vendor-reported references. The figure is diagnostic rather than metrological: the inputs are coated-to-uncoated relative curves whereas the frozen forward model represents absolute reflectance, and the vendor reference method/uncertainty are incomplete. Table S10 reports only frozen summary quantities; no unreported acquisition value is reconstructed from the plotted pixels.

SR2. Bootstrap-resampling sensitivity
-------------------------------------
Supplementary Figure S2 compares the prespecified pooled observation-level percentile bootstrap with the material-by-thickness-stratified sensitivity bootstrap for every primary E3-E0 and E3-E1 mean/q90/q95 contrast. Large impulsive, baseline-drift, and mixed effects retain their direction. The most visible sensitivity occurs for near-zero endpoints such as Gaussian E3-E0 q95, whose pooled interval crosses zero whereas the stratified interval is slightly positive. The stratified analysis is therefore treated as sensitivity analysis and does not replace the pooled primary analysis. Full values are in Table S11.

SR3. Fine-scale index versus candidate error change
---------------------------------------------------
Supplementary Figure S3 plots I_fine against the candidate-minus-E3 absolute-error change for all 579 V2-eligible observations in the independent-seed validation. Baseline-drift candidates occupy the lower-I_fine region and all 200 are retained by Fine-q75. Gaussian and impulsive candidates extend to much larger I_fine, and Fine-q75 removes many of them. The scatter also shows why I_fine is not a perfect outcome classifier: both beneficial and harmful candidates occur on each side of the threshold. The final gate is therefore interpreted as selective risk reduction rather than a deterministic guarantee of improvement.

SR4. Fine-gate development sensitivity
--------------------------------------
Supplementary Figure S4 and Tables S6a-S6b report V2, Fine-q50, Fine-q75, and Fine-q90 replay on frozen development candidate fields. Accepting all triggered candidates is not part of this figure because it represents a different accept-all sensitivity; the archived audit showed that such unrestricted intervention was harmful in overall mean error. Fine-q75 was frozen before the independent-seed validation and was not retuned there.

SR5. Reference-implementation runtime distributions
---------------------------------------------------
Supplementary Figure S5 shows wall-clock distributions from the frozen reference implementation. E0 and E1 are tens-of-milliseconds procedures, whereas E3 is orders of magnitude slower because it evaluates a robust multistart profile and refines all finite local minima. Once E3 is available, incremental E4 candidate generation is much smaller. These measurements are implementation- and hardware-dependent and are not hardware-independent algorithmic complexity constants. Quantiles are reported in Table S9.

SR6. Complete cell-level and failure accounting
-----------------------------------------------
Table S5 reports n, mean, median, q90, q95, maximum AE, boundary count, non-ok count, and convergence count for all 3 materials x 3 thicknesses x 4 contamination scenarios x 3 estimators. The single non-ok result in the 21,600-row selected-material/thickness table is Material C / 380 nm / mixed / E0, which reached 699.99998 nm and AE=319.99998 nm. The record is retained in all unconditional summaries. Tables S8a-S8c provide status/fallback accounting for the selected-material and E4 analyses.

Supplementary Figure Captions
=============================

Figure S1. Complete Sheffield acquisition-level thickness estimates. Filled estimator-coded markers denote valid completions; open X symbols denote retained upper-boundary failures at their numerical estimates. Dotted lines show vendor-reported reference thicknesses (476.3 and 198.7 nm). The figure is a failure-mode diagnostic and does not establish traceable physical accuracy.

Figure S2. Sensitivity of the selected-material/thickness AE contrasts to bootstrap resampling. Circles show the prespecified pooled observation-level percentile bootstrap intervals and squares show material-by-thickness-stratified sensitivity intervals. Point estimates are identical because only the resampling scheme changes. Negative contrasts favor E3. Near-zero interval crossing can depend on resampling structure, whereas the large failure-mechanism effects are directionally stable.

Figure S3. Fine-scale normalized energy index versus candidate-minus-E3 absolute-error change for V2-eligible independent-seed validation candidates. The vertical line is the frozen q75 threshold I_fine=435.96252114890945 and the horizontal line indicates no AE change. The index is plotted as log10(1+I_fine) because it is not a 0-1 fraction and spans several orders of magnitude.

Figure S4. Development-set sensitivity of the fine-scale gate. Bars show improved, worsened, and total accepted counts for V2 and Fine-q50/q75/q90 replays on the frozen candidate table. Candidate thicknesses were not recomputed during these gate replays. Fine-q75 was frozen before the independent-seed validation.

Figure S5. Wall-clock distributions for the reference implementation. E0/E1/E3 distributions come from the selected-material/thickness run; incremental E4 runtime is shown for triggered observations in the independent-seed validation. The vertical axis is logarithmic. Values characterize the present implementation and execution environment rather than hardware-independent complexity.

Supplementary Table Captions
============================

Table S1. Material systems, optical-constant provenance, and common forward-model settings.
Table S2. Frozen E0/E1/E3 numerical settings.
Table S3. Frozen E4-HORP geometry, projection, optimization, trust-region, trigger, V2, and Fine-q75 parameters.
Table S4. Contamination parameters and registered outlier counts for the selected-material/thickness simulation study.
Table S5. Complete cell-level AE and failure summaries for all material x thickness x scenario x estimator combinations.
Table S6a. Overall Fine-q50/q75/q90 acceptance sensitivity on the development candidates.
Table S6b. Scenario-specific Fine-q50/q75/q90 development sensitivity.
Table S7a. Overall independent-seed validation outcomes with read-only paired-bootstrap intervals.
Table S7b. Scenario-specific independent-seed validation outcomes and mean-contrast intervals.
Table S7c. Complete read-only paired-bootstrap contrast table for V2, Frozen Fine-q75, and their direct comparison.
Table S8a. Selected-material/thickness estimator status counts.
Table S8b. E4 candidate-generation fallback-reason counts by scenario.
Table S8c. Operational trigger, V2 eligibility, Fine-q75 eligibility, and final E3 fallback counts.
Table S9. Reference-implementation runtime quantiles.
Table S10. Frozen Sheffield failure-mode summary. IQR values are included only where they were frozen in the source summary; missing SAMPLE2 IQRs are not reconstructed.
Table S11. Material-by-thickness-stratified bootstrap sensitivity for the selected-material/thickness primary contrasts.

Supplementary Reproducibility
=============================
The public scientific labels in the manuscript correspond internally to: mechanism benchmark (Stage26), selected-material/thickness study (Stage28), acceptance-rule development (Stage30), and independent-seed validation (Stage31). The provenance labels are retained here solely to make archived computational artifacts traceable.

Frozen environment: Python 3.12.0; NumPy 2.5.1; SciPy 1.18.0; pandas 2.2.2; pyarrow 17.0.0; matplotlib 3.9.1. The selected-material package used PCG64/SeedSequence and the acceptance-development cloud script requested 32 workers with OMP/MKL/OpenBLAS/NumExpr thread counts set to one per worker. Runtime values should not be generalized to other hardware.

All figure and table artifacts in this supplement are generated from frozen result packages or copied from a frozen canonical visualization. No E0/E1/E3/E4 estimator was rerun to produce the supplementary statistical figures and tables.
'''
(OUT/'PHOTONICS_SUPPLEMENTARY_INFORMATION_EN_V01.txt').write_text(si,encoding='utf-8-sig')

# -------------------- Provenance manifest --------------------
source_files=[STAGE28_ZIP,STAGE30_ZIP,STAGE31_ZIP,CLOUD_ZIP,REF_PKG,PRIMARY_STATS,STRAT_STATS,ST31_CI,OLD_SHEFFIELD]
manifest={'schema':'photonics-supplement-v01/1.0','generated_from_frozen_outputs_only':True,'estimator_rerun':False,'tau_accept_v2':TAU_V2,'tau_fine':TAU_FINE,'source_files':[],'key_assertions':{}}
for p in source_files:
    manifest['source_files'].append({'path':str(p),'sha256':sha256_file(p),'bytes':p.stat().st_size})
manifest['key_assertions']={
    'stage28_rows':int(len(df28)),
    'stage28_observations':int(df28.observation_id.nunique()),
    'stage28_non_ok_rows':int((df28.status!='ok').sum()),
    'stage28_non_ok_identity':df28.loc[df28.status!='ok',['observation_id','material_id','true_thickness_nm','scenario','method','estimate_nm','absolute_error_nm','status']].to_dict('records'),
    'stage31_rows':int(len(d31)),
    'stage31_triggered':int(d31.triggered.sum()),
    'stage31_v2_eligible':int(v2mask.sum()),
    'stage31_fine_q75_eligible':int(finemask.sum()),
}
# output hashes
manifest['output_files']=[]
for p in sorted(OUT.rglob('*')):
    if p.is_file() and p.name!='SUPPLEMENT_PROVENANCE_MANIFEST.json':
        manifest['output_files'].append({'relative_path':str(p.relative_to(OUT)),'sha256':sha256_file(p),'bytes':p.stat().st_size})
(PROV/'SUPPLEMENT_PROVENANCE_MANIFEST.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')

# README
readme=f'''PHOTONICS Supplement v0.1\n\nPurpose: first complete Supplementary Methods + Figures/Tables package aligned to manuscript core v1.4.\n\nContents:\n- PHOTONICS_SUPPLEMENTARY_INFORMATION_EN_V01.txt\n- figures/: Figure S1-S5 in submission formats\n- tables/: Table S1-S11 source tables\n- figure_data/: exact data used to generate S2-S5\n- provenance/: source/output SHA-256 manifest\n\nCritical counts:\n- selected-material/thickness results: {len(df28)} rows, {df28.observation_id.nunique()} observations, {(df28.status!='ok').sum()} non-ok row\n- independent-seed E4 results: {len(d31)} observations, {int(d31.triggered.sum())} triggered, {int(v2mask.sum())} V2 eligible, {int(finemask.sum())} Fine-q75 eligible\n\nNo estimator was rerun.\n'''
(OUT/'README.txt').write_text(readme,encoding='utf-8-sig')

# Zip package
zipout=ROOT/'PHOTONICS_SUPPLEMENTARY_INFORMATION_V01.zip'
if zipout.exists(): zipout.unlink()
with zipfile.ZipFile(zipout,'w',compression=zipfile.ZIP_DEFLATED) as z:
    for p in OUT.rglob('*'):
        if p.is_file(): z.write(p,arcname=str(p.relative_to(OUT)))

print('CREATED',OUT)
print('ZIP',zipout)
print('Stage28',len(df28),df28.observation_id.nunique(),(df28.status!='ok').sum())
print('Stage31',len(d31),int(d31.triggered.sum()),int(v2mask.sum()),int(finemask.sum()))
print('Figures',len(list(FIG.glob('*'))),'Tables',len(list(TAB.glob('*.csv'))))
