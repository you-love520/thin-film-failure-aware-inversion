from pathlib import Path
import math, json, shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter, LogLocator
from PIL import Image

ROOT=Path('/mnt/data')
SRC=ROOT/'photonics_v08_artifacts'
FD=SRC/'figure_data'
WORK=ROOT/'work_v08'
OUT=ROOT/'photonics_v09_figures'
FIG=OUT/'figures'
AUD=OUT/'audit'
for p in [FIG,AUD]: p.mkdir(parents=True, exist_ok=True)

# restrained, colorblind-safe palette; applied consistently across all figures
C_BLUE='#0072B2'; C_ORANGE='#D55E00'; C_GREEN='#009E73'; C_PURPLE='#CC79A7'; C_GRAY='#666666'; C_LIGHT='#E9EEF2'; C_DARK='#222222'
MAT_COL={'A':C_BLUE,'B':C_GREEN,'C':C_PURPLE}
SCEN_LABEL={'gaussian':'Gaussian','impulsive':'Impulsive','baseline_drift':'Baseline drift','mixed':'Mixed'}
SCENS=['gaussian','impulsive','baseline_drift','mixed']

plt.rcParams.update({
    'font.family':'Liberation Sans',
    'font.size':8.0,
    'axes.titlesize':8.6,
    'axes.labelsize':8.3,
    'legend.fontsize':7.1,
    'xtick.labelsize':7.6,
    'ytick.labelsize':7.6,
    'axes.linewidth':0.8,
    'lines.linewidth':1.25,
    'lines.markersize':4.2,
    'xtick.major.width':0.8,
    'ytick.major.width':0.8,
    'xtick.major.size':3.0,
    'ytick.major.size':3.0,
    'pdf.fonttype':42,
    'ps.fonttype':42,
    'savefig.facecolor':'white',
})

def clean_ax(ax, grid=None):
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    if grid=='x': ax.grid(axis='x', linestyle=':', linewidth=0.45, alpha=0.65)
    elif grid=='y': ax.grid(axis='y', linestyle=':', linewidth=0.45, alpha=0.65)
    elif grid=='both': ax.grid(linestyle=':', linewidth=0.45, alpha=0.65)

def panel_label(ax, label, x=-0.12, y=1.04):
    ax.text(x,y,label,transform=ax.transAxes,ha='left',va='bottom',fontsize=9.5,fontweight='bold',clip_on=False)

def save_all(fig, stem, dpi=800):
    fig.savefig(FIG/f'{stem}.svg', bbox_inches='tight', pad_inches=0.03)
    fig.savefig(FIG/f'{stem}.pdf', bbox_inches='tight', pad_inches=0.03)
    fig.savefig(FIG/f'{stem}.png', dpi=dpi, bbox_inches='tight', pad_inches=0.03)
    fig.savefig(FIG/f'{stem}.tiff', dpi=dpi, bbox_inches='tight', pad_inches=0.03,
                pil_kwargs={'compression':'tiff_lzw'})
    plt.close(fig)

# ================= Figure 1 =================
fig=plt.figure(figsize=(7.08,5.15))
gs=fig.add_gridspec(2,2,left=0.045,right=0.985,bottom=0.06,top=0.97,wspace=0.22,hspace=0.34)

# (a) physical stack + forward model
ax=fig.add_subplot(gs[0,0]); ax.set_axis_off(); panel_label(ax,'(a)',x=-0.02,y=1.00)
ax.set_title('Physics-based forward model',pad=5,fontweight='bold')
# stack
x0,y0,w=0.08,0.14,0.40
ax.add_patch(Rectangle((x0,y0+0.32),w,0.18,facecolor='#F5F5F5',edgecolor=C_DARK,lw=0.9))
ax.add_patch(Rectangle((x0,y0+0.18),w,0.14,facecolor='#D9EAF7',edgecolor=C_DARK,lw=0.9))
ax.add_patch(Rectangle((x0,y0),w,0.18,facecolor='#EEEEEE',edgecolor=C_DARK,lw=0.9))
ax.text(x0+w/2,y0+0.41,'Air',ha='center',va='center')
ax.text(x0+w/2,y0+0.25,'Film, thickness $d$',ha='center',va='center')
ax.text(x0+w/2,y0+0.09,'Substrate',ha='center',va='center')
# incidence/reflection arrows
ax.add_patch(FancyArrowPatch((0.20,0.82),(0.30,0.66),arrowstyle='-|>',mutation_scale=10,lw=1.1,color=C_BLUE))
ax.add_patch(FancyArrowPatch((0.30,0.66),(0.40,0.82),arrowstyle='-|>',mutation_scale=10,lw=1.1,color=C_ORANGE))
ax.text(0.16,0.83,'Incident',fontsize=7.2,ha='center'); ax.text(0.44,0.83,'Reflected',fontsize=7.2,ha='center')
ax.text(0.29,0.71,'8°',fontsize=7.2)
# formulas / flow
ax.add_patch(FancyArrowPatch((0.50,0.39),(0.60,0.39),arrowstyle='-|>',mutation_scale=10,lw=1.0,color=C_GRAY))
box=FancyBboxPatch((0.61,0.19),0.34,0.42,boxstyle='round,pad=0.018,rounding_size=0.018',facecolor='white',edgecolor=C_DARK,lw=0.9)
ax.add_patch(box)
ax.text(0.78,0.50,'TMM / Fresnel model',ha='center',va='center',fontweight='bold')
ax.text(0.78,0.39,'$f(d;M)$',ha='center',va='center',fontsize=9.3)
ax.text(0.78,0.28,'$y=g f(d;M)+B_2\\beta+\\varepsilon$',ha='center',va='center',fontsize=7.4)
ax.set_xlim(0,1); ax.set_ylim(0,1)

# (b) failure modes schematic
ax=fig.add_subplot(gs[0,1]); panel_label(ax,'(b)');
ax.set_title('Distinct residual failure mechanisms',pad=5,fontweight='bold')
x=np.linspace(0,1,220); clean=0.5+0.15*np.sin(2*np.pi*1.35*x+0.3)
ax.plot(x,clean,color=C_GRAY,lw=1.2,label='Clean-like spectrum')
imp=clean.copy(); inds=np.array([25,62,95,154,186]); imp[inds]+=np.array([.23,-.20,.18,-.24,.20])
ax.plot(x,imp,color=C_ORANGE,lw=1.0,label='Impulsive contamination')
base=clean+0.11*(x-0.4)+0.055*np.sin(np.pi*x)
ax.plot(x,base,color=C_BLUE,lw=1.2,label='Smooth baseline drift')
ax.set_xlabel('Normalized wavelength'); ax.set_ylabel('Relative reflectance')
ax.set_xticks([]); ax.set_yticks([]); clean_ax(ax)
ax.legend(frameon=False,loc='lower left',handlelength=2.2)
ax.text(0.70,0.86,'Sparse spikes\n→ robust-loss failure',transform=ax.transAxes,fontsize=7.2,ha='center')
ax.text(0.76,0.14,'Smooth structure\n→ residual-model failure',transform=ax.transAxes,fontsize=7.2,ha='center')

# (c) estimator hierarchy
ax=fig.add_subplot(gs[1,0]); ax.set_axis_off(); panel_label(ax,'(c)',x=-0.02,y=1.00)
ax.set_title('Estimator hierarchy',pad=5,fontweight='bold')
xs=[0.04,0.36,0.68]; widths=[0.24]*3
labels=[('E0','Constrained L2\nprofile'),('E1','One-step Tukey\nscore'),('E3','Full Tukey nuisance\nprofiling +\nmultistart')]
for i,(title,sub) in enumerate(labels):
    fc=['#F4F4F4','#FFF3E8','#E8F3FA'][i]
    b=FancyBboxPatch((xs[i],0.38),widths[i],0.34,boxstyle='round,pad=0.02,rounding_size=0.02',facecolor=fc,edgecolor=C_DARK,lw=1.0)
    ax.add_patch(b); ax.text(xs[i]+widths[i]/2,0.61,title,ha='center',va='center',fontweight='bold',fontsize=9.2)
    ax.text(xs[i]+widths[i]/2,0.48,sub,ha='center',va='center',fontsize=7.6)
    if i<2: ax.add_patch(FancyArrowPatch((xs[i]+widths[i],0.55),(xs[i+1],0.55),arrowstyle='-|>',mutation_scale=10,lw=1.0,color=C_GRAY))
ax.set_xlim(0,1); ax.set_ylim(0,1)

# (d) E4 selection + evidence chain
ax=fig.add_subplot(gs[1,1]); ax.set_axis_off(); panel_label(ax,'(d)',x=-0.02,y=1.00)
ax.set_title('Selective E4 refinement and frozen validation',pad=5,fontweight='bold')
# main flow
positions=[(0.02,0.18,'E3 anchor'),(0.24,0.20,'E4-HORP\nlocal candidate'),(0.48,0.20,'V2 safety\neligibility'),(0.72,0.25,'Fine-q75\n$I_{fine}\\leq\\tau_{fine}$')]
for x0,w,lab in positions:
    b=FancyBboxPatch((x0,0.55),w,0.22,boxstyle='round,pad=0.012,rounding_size=0.015',facecolor='white',edgecolor=C_DARK,lw=0.9)
    ax.add_patch(b); ax.text(x0+w/2,0.66,lab,ha='center',va='center',fontsize=7.0,fontweight='bold' if 'E4' in lab else 'normal')
for a,bx in [(0.20,0.24),(0.44,0.48),(0.68,0.72)]: ax.add_patch(FancyArrowPatch((a,0.66),(bx,0.66),arrowstyle='-|>',mutation_scale=9,lw=1.0,color=C_GRAY))
# outcomes
acc=FancyBboxPatch((0.56,0.24),0.18,0.17,boxstyle='round,pad=0.012',facecolor='#E6F4EE',edgecolor=C_GREEN,lw=1.0)
fb=FancyBboxPatch((0.78,0.24),0.18,0.17,boxstyle='round,pad=0.012',facecolor='#F2F2F2',edgecolor=C_GRAY,lw=1.0)
ax.add_patch(acc); ax.add_patch(fb); ax.text(0.65,0.325,'Accept $d_4$',ha='center',va='center',fontsize=7.8); ax.text(0.87,0.325,'Fallback $d_3$',ha='center',va='center',fontsize=7.8)
ax.add_patch(FancyArrowPatch((0.86,0.55),(0.65,0.41),arrowstyle='-|>',mutation_scale=9,lw=1.0,color=C_GREEN))
ax.add_patch(FancyArrowPatch((0.89,0.55),(0.87,0.41),arrowstyle='-|>',mutation_scale=9,lw=1.0,color=C_GRAY))
ax.set_xlim(0,1); ax.set_ylim(0,1)
save_all(fig,'Figure1_FailureAware_Inversion_Workflow_v09')

# ================= Figure 2 =================
R=pd.read_csv(FD/'Figure2_spectra_data.csv')
C=pd.read_csv(FD/'Figure2_E3_candidate_minima_data.csv')
D28=pd.read_csv(WORK/'stage28'/'MASTER_STAGE28_RESULTS.csv',low_memory=False)
D30=pd.read_csv(WORK/'stage30fine'/'MASTER_STAGE30_E4_RESULTS.csv',low_memory=False)
fig,axs=plt.subplots(2,2,figsize=(7.08,5.45),constrained_layout=False)
fig.subplots_adjust(left=0.095,right=0.985,bottom=0.09,top=0.96,wspace=0.30,hspace=0.38)
# a
ax=axs[0,0]; r=R[R.panel=='impulsive']; o=r[r.is_outlier_index==1]
ax.plot(r.wavelength_nm,r.clean_reflectance,color=C_GRAY,label='Clean forward spectrum',lw=1.25)
ax.plot(r.wavelength_nm,r.observed_reflectance,color=C_ORANGE,label='Observed spectrum',lw=0.95)
ax.scatter(o.wavelength_nm,o.observed_reflectance,s=13,color=C_BLUE,label='Injected outlier locations',zorder=5)
ax.set_xlabel('Wavelength (nm)'); ax.set_ylabel('Reflectance'); ax.set_title('Impulsive contamination')
clean_ax(ax,'both'); panel_label(ax,'(a)'); ax.legend(frameon=False,loc='upper left')
# b
ax=axs[0,1]; r=R[R.panel=='baseline_drift']
ax.plot(r.wavelength_nm,r.observed_reflectance-r.clean_reflectance,color=C_BLUE,lw=1.0,label='Observed − clean')
ax.plot(r.wavelength_nm,r.baseline_component,color=C_ORANGE,lw=1.4,label='Injected smooth baseline')
ax.axhline(0,color=C_GRAY,lw=0.7)
ax.set_xlabel('Wavelength (nm)'); ax.set_ylabel('Reflectance difference'); ax.set_title('Smooth baseline drift')
clean_ax(ax,'both'); panel_label(ax,'(b)'); ax.legend(frameon=False,loc='upper left')
# c
ax=axs[1,0]; c=C[C.panel=='impulsive'].sort_values('refined_thickness_nm'); oid=c.observation_id.iloc[0]; rrr=D28[D28.observation_id==oid].set_index('strategy'); true=float(rrr.true_thickness_nm.iloc[0])
ax.plot(c.refined_thickness_nm,c.normalized_objective_across_E3_candidates,color=C_BLUE,marker='o',ms=4.0,lw=1.1)
vals=[('True',true,'--',C_DARK),('E0',float(rrr.loc['E0_constrained_L2_profile','estimate_nm']),':',C_ORANGE),('E1',float(rrr.loc['E1_full_design_one_step_score','estimate_nm']),'-.',C_GREEN),('E3',float(rrr.loc['E3_constrained_multistart_robust_profile','estimate_nm']),'-',C_BLUE)]
for lab,xv,ls,col in vals: ax.axvline(xv,ls=ls,lw=1.0,color=col,label=lab)
ax.set_xlabel('Thickness (nm)'); ax.set_ylabel('Normalized E3 candidate objective')
ax.set_title('Competing thickness basins'); clean_ax(ax,'both'); panel_label(ax,'(c)')
ax.legend(frameon=False,ncol=2,loc='lower left',handlelength=2.1,columnspacing=0.9)
ax.text(0.02,-0.23,'Normalization is within this observation only.',transform=ax.transAxes,fontsize=6.8,color=C_GRAY)
# d
ax=axs[1,1]; c=C[C.panel=='baseline_drift'].sort_values('refined_thickness_nm'); oid=c.observation_id.iloc[0]; r30=D30[D30.observation_id==oid].iloc[0]; true=float(r30.true_thickness_nm); anchor=float(r30.e3_thickness_nm); cand=float(r30.e4_candidate_thickness_nm); rad=float(r30.trust_radius_nm)
ax.plot(c.refined_thickness_nm,c.normalized_objective_across_E3_candidates,color=C_BLUE,marker='o',ms=4.0,lw=1.1)
ax.axvspan(anchor-rad,anchor+rad,color=C_LIGHT,alpha=0.85)
ax.axvline(true,ls='--',lw=1.0,color=C_DARK)
ax.axvline(anchor,ls=':',lw=1.2,color=C_BLUE)
ax.axvline(cand,ls='-.',lw=1.2,color=C_ORANGE)
ax.text(true,1.01,'True',ha='center',va='bottom',fontsize=6.6,color=C_DARK)
ax.text(anchor,0.94,'E3',ha='right',va='top',fontsize=6.6,color=C_BLUE)
ax.text(cand,0.88,'E4',ha='left',va='top',fontsize=6.6,color=C_ORANGE)
ax.text(anchor-rad+0.2,0.80,'trust region',fontsize=6.6,color=C_GRAY)
# Anchor/candidate objective values are on a different scale from the normalized E3-candidate axis; show thickness movement only.
ax.annotate('',xy=(cand,0.12),xytext=(anchor,0.12),arrowprops=dict(arrowstyle='->',lw=1.0,color=C_ORANGE))
ax.text((anchor+cand)/2,0.16,f'{abs(cand-anchor):.2f} nm refinement',ha='center',va='bottom',fontsize=6.8,color=C_ORANGE)
ax.annotate(f'$d_3$ = {anchor:.2f} nm\n$d_4$ = {cand:.2f} nm',xy=(cand,0.12),xytext=(0.58,0.62),textcoords='axes fraction',arrowprops=dict(arrowstyle='->',lw=0.8,color=C_GRAY),fontsize=7.0)
ax.set_xlim(max(350,anchor-rad-12),min(700,anchor+rad+12)); ax.set_ylim(-0.05,1.05)
ax.set_xlabel('Thickness (nm)'); ax.set_ylabel('Normalized E3 candidate objective'); ax.set_title('E3-anchored local refinement')
clean_ax(ax,'both'); panel_label(ax,'(d)')
save_all(fig,'Figure2_FailureMechanisms_and_ObjectiveLandscapes_v09')

# ================= Figure 3 =================
T3=pd.read_csv(SRC/'tables'/'Table3_Stage28_Primary_Paired_Statistics.csv')
endpoint_order=['mean','q90','q95']; endpoint_mark={'mean':'o','q90':'s','q95':'D'}
fig,axs=plt.subplots(1,2,figsize=(7.08,4.2),sharey=True)
fig.subplots_adjust(left=0.19,right=0.985,bottom=0.13,top=0.92,wspace=0.12)
# positions grouped by scenario: 3 endpoints per scenario
ys=[]; ylabels=[]; y=0
for scen in SCENS:
    for ep in endpoint_order:
        ys.append(y); ylabels.append(f'{SCEN_LABEL[scen]}  {ep if ep!="mean" else "mean"}'); y+=1
    y+=0.7
# reverse numeric positions so the first scenario (Gaussian) appears at the top
maxy=max(ys)
ys=[maxy-v for v in ys]
for ax,comp,label in zip(axs,['E3-E0','E3-E1'],['(a)','(b)']):
    yy=[]; ests=[]; los=[]; his=[]; marks=[]
    for scen in SCENS:
        for ep in endpoint_order:
            r=T3[(T3.scenario==scen)&(T3.comparison==comp)&(T3.endpoint==ep)].iloc[0]
            yy.append(ys[len(yy)]); ests.append(r.delta_AE_nm); los.append(r.ci95_low_nm); his.append(r.ci95_high_nm); marks.append(endpoint_mark[ep])
    # light separators between scenario groups
    for sep in [maxy-2.85, maxy-6.55, maxy-10.25]: ax.axhline(sep,color='#DDDDDD',lw=0.6,zorder=0)
    for yi,e,lo,hi,m in zip(yy,ests,los,his,marks):
        ax.errorbar(e,yi,xerr=[[e-lo],[hi-e]],fmt=m,color=C_BLUE,ecolor=C_BLUE,capsize=2.0,elinewidth=0.9,ms=4.2,zorder=3)
    ax.axvline(0,color=C_DARK,ls='--',lw=0.9)
    clean_ax(ax,'x'); ax.text(0.01,0.98,label,transform=ax.transAxes,ha='left',va='top',fontsize=9.5,fontweight='bold',bbox=dict(facecolor='white',edgecolor='none',pad=1.2))
    ax.set_title(comp.replace('-',' − '),fontweight='bold')
    ax.set_xlabel('Paired ΔAE (nm)\nnegative = E3 lower error')
axs[0].set_yticks(ys); axs[0].set_yticklabels(ylabels)
axs[1].tick_params(labelleft=False)
# set independent x limits
for ax,comp in zip(axs,['E3-E0','E3-E1']):
    s=T3[T3.comparison==comp]; lo=s.ci95_low_nm.min(); hi=s.ci95_high_nm.max(); span=hi-lo
    ax.set_xlim(lo-0.06*span,hi+0.06*span)
legend=[Line2D([0],[0],marker=endpoint_mark['mean'],color='none',markerfacecolor=C_BLUE,markeredgecolor=C_BLUE,label='Mean'),Line2D([0],[0],marker=endpoint_mark['q90'],color='none',markerfacecolor=C_BLUE,markeredgecolor=C_BLUE,label='q90'),Line2D([0],[0],marker=endpoint_mark['q95'],color='none',markerfacecolor=C_BLUE,markeredgecolor=C_BLUE,label='q95')]
fig.legend(handles=legend,loc='upper center',bbox_to_anchor=(0.60,0.995),ncol=3,frameon=False,handletextpad=0.4,columnspacing=1.0)
save_all(fig,'Figure3_Stage28_Paired_Forest_v09')

# ================= Figure 4 =================
C4=pd.read_csv(FD/'Figure4_data.csv')
fig,axs=plt.subplots(2,2,figsize=(7.08,5.25),sharex=True)
fig.subplots_adjust(left=0.095,right=0.985,bottom=0.15,top=0.95,wspace=0.27,hspace=0.34)
for ax,scen,label in zip(axs.ravel(),SCENS,['(a)','(b)','(c)','(d)']):
    s=C4[C4.scenario==scen]
    for mat in ['A','B','C']:
        m=s[s.material_id==mat].sort_values('true_thickness_nm')
        y0=m.E3_minus_E0_mean_delta_AE_nm.to_numpy().copy(); y1=m.E3_minus_E1_mean_delta_AE_nm.to_numpy().copy()
        if scen=='mixed' and mat=='C':
            exact=float(y0[0]); y0[0]=-3.0
        ax.plot(m.true_thickness_nm,y0,color=MAT_COL[mat],marker='o',lw=1.2,ls='-')
        ax.plot(m.true_thickness_nm,y1,color=MAT_COL[mat],marker='s',lw=1.05,ls='--',mfc='white')
    ax.axhline(0,color=C_DARK,ls=':',lw=0.85)
    ax.set_title(SCEN_LABEL[scen],fontweight='bold'); ax.set_xticks([380,520,680]); ax.set_xlabel('True thickness (nm)'); ax.set_ylabel('Mean ΔAE (nm)'); clean_ax(ax,'y'); panel_label(ax,label)
    if scen=='mixed':
        ax.set_ylim(-3.18,0.55)
        ax.annotate(f'C/380, E3−E0 = {exact:.2f} nm\n(off-scale; retained boundary event)',xy=(380,-3.0),xytext=(435,-2.18),arrowprops=dict(arrowstyle='->',lw=0.8,color=C_GRAY),fontsize=6.8)
# global legends
mat_handles=[Line2D([0],[0],color=MAT_COL[m],marker='o',lw=1.2,label=f'Material {m}') for m in ['A','B','C']]
comp_handles=[Line2D([0],[0],color=C_DARK,marker='o',lw=1.2,ls='-',label='E3 − E0'),Line2D([0],[0],color=C_DARK,marker='s',mfc='white',lw=1.05,ls='--',label='E3 − E1')]
fig.legend(handles=mat_handles+comp_handles,loc='lower center',bbox_to_anchor=(0.5,0.015),ncol=5,frameon=False,columnspacing=1.2,handletextpad=0.5)
save_all(fig,'Figure4_Material_Thickness_Generalization_v09')

# ================= Figure 5 =================
S30=pd.read_csv(FD/'Figure5A_Stage30_gate_development_data.csv')
V31=pd.read_csv(FD/'Figure5C_Stage31_validation_data.csv')
SC31=pd.read_csv(FD/'Figure5D_Stage31_scenario_data.csv')
fig,axs=plt.subplots(2,2,figsize=(7.08,5.0))
fig.subplots_adjust(left=0.095,right=0.985,bottom=0.11,top=0.91,wspace=0.28,hspace=0.42)
# a accepted counts dev
ax=axs[0,0]; x=np.arange(len(S30)); bars=ax.bar(x,S30.accepted,color=C_BLUE,width=0.70); ax.set_xticks(x); ax.set_xticklabels(S30.rule_id,rotation=24); ax.set_ylabel('Accepted candidates'); ax.set_title('Intervention count'); clean_ax(ax,'y'); panel_label(ax,'(a)')
for b,v in zip(bars,S30.accepted): ax.text(b.get_x()+b.get_width()/2,b.get_height()+max(S30.accepted)*0.025,f'{int(v)}',ha='center',va='bottom',fontsize=6.9)
# b mean effect dev
ax=axs[0,1]; ax.plot(x,S30.mean_delta_AE,color=C_BLUE,marker='o',lw=1.2); ax.axhline(0,color=C_DARK,ls='--',lw=0.85); ax.set_xticks(x); ax.set_xticklabels(S30.rule_id,rotation=24); ax.set_ylabel('Overall mean ΔAE (nm)'); ax.set_title('Net error effect'); clean_ax(ax,'y'); panel_label(ax,'(b)')
# c validation stacked outcomes
ax=axs[1,0]; x2=np.arange(len(V31)); imp=V31.improved.to_numpy(); wors=V31.worsened.to_numpy(); ax.bar(x2,imp,color=C_GREEN,label='Improved'); ax.bar(x2,wors,bottom=imp,color=C_ORANGE,label='Worsened'); ax.set_xticks(x2); ax.set_xticklabels(['V2','Frozen Fine-q75']); ax.set_ylabel('Accepted candidates'); ax.set_title('Accepted outcomes'); ax.set_ylim(0,680); clean_ax(ax,'y'); panel_label(ax,'(c)')
ax.text(x2[0],imp[0]/2,'Improved',ha='center',va='center',fontsize=6.8,color='white',fontweight='bold')
ax.text(x2[0],imp[0]+wors[0]/2,'Worsened',ha='center',va='center',fontsize=6.8,color='white',fontweight='bold')
for xi,tot,frac in zip(x2,V31.accepted,V31.improved_fraction_among_accepted): ax.text(xi,tot+10,f'n={int(tot)}; {100*frac:.1f}% improved',ha='center',va='bottom',fontsize=6.8)
# d scenario effects validation
ax=axs[1,1]; xs=np.arange(4); labels=[SCEN_LABEL[s] for s in SCENS]
for rule,col,mark in [('V2',C_BLUE,'o'),('Frozen-Fine-q75',C_ORANGE,'s')]:
    ss=SC31[SC31.rule_id==rule].set_index('scenario').loc[SCENS]; ax.plot(xs,ss.accepted_only_mean_delta_AE,color=col,marker=mark,lw=1.2,label='Frozen Fine-q75' if 'Fine' in rule else 'V2')
ax.axhline(0,color=C_DARK,ls='--',lw=0.85); ax.set_xticks(xs); ax.set_xticklabels(labels,rotation=18); ax.set_ylabel('Accepted-only mean ΔAE (nm)'); ax.set_title('Scenario effect'); clean_ax(ax,'y'); panel_label(ax,'(d)'); ax.legend(frameon=False,loc='upper right')
fig.text(0.54,0.965,'DEVELOPMENT (old seed; gate replay)',ha='center',va='center',fontsize=8.3,fontweight='bold',color=C_GRAY)
fig.text(0.54,0.505,'INDEPENDENT VALIDATION (new seed)',ha='center',va='center',fontsize=8.3,fontweight='bold',color=C_GRAY)
save_all(fig,'Figure5_Development_to_IndependentValidation_v09')

# ================= Figure 6 =================
D31=pd.read_csv(WORK/'stage31'/'MASTER_STAGE30_E4_RESULTS.csv',low_memory=False)
SC31v=pd.read_csv(WORK/'stage31'/'E4_FINEQ75_INDEPENDENT_VALIDATION_BY_SCENARIO.csv')
tau=435.96252114890945
v2=D31[D31.triggered.astype(bool)&D31.inner_converged.astype(bool)&(D31.objective_relative_improvement>=1e-4)].copy()
fig,axs=plt.subplots(2,2,figsize=(7.08,5.0))
fig.subplots_adjust(left=0.10,right=0.985,bottom=0.12,top=0.96,wspace=0.30,hspace=0.40)
# a boxplot I_fine
ax=axs[0,0]; series=[v2.loc[v2.scenario==s,'tree_fine_energy_ratio'].to_numpy() for s in SCENS]; bp=ax.boxplot(series,labels=[SCEN_LABEL[s] for s in SCENS],showfliers=False,patch_artist=True)
for box in bp['boxes']: box.set_facecolor('#F0F0F0'); box.set_edgecolor(C_DARK)
for med in bp['medians']: med.set_color(C_ORANGE); med.set_linewidth(1.2)
ax.axhline(tau,color=C_BLUE,ls='--',lw=1.0,label=r'Frozen $\tau_{fine}=435.96$'); ax.set_yscale('log'); ax.set_ylabel(r'Fine-scale normalized energy index, $I_{fine}$'); ax.set_title('Fine-scale structure among V2-eligible candidates'); clean_ax(ax,'y'); panel_label(ax,'(a)'); ax.legend(frameon=False,loc='upper left'); ax.tick_params(axis='x',rotation=15)
# b retention
ax=axs[0,1]; flow=[]
for s in SCENS:
    ss=v2[v2.scenario==s]; acc=(ss.tree_fine_energy_ratio<=tau).sum(); flow.append((len(ss),acc,acc/max(len(ss),1)))
frac=[z[2] for z in flow]; bars=ax.bar(np.arange(4),frac,color=C_BLUE); ax.set_xticks(np.arange(4)); ax.set_xticklabels([SCEN_LABEL[s] for s in SCENS],rotation=15); ax.set_ylim(0,1.16); ax.set_ylabel('Retention among V2-eligible'); ax.set_title('Selective retention after Fine-q75'); clean_ax(ax,'y'); panel_label(ax,'(b)')
for b,(n,a,f) in zip(bars,flow): ax.text(b.get_x()+b.get_width()/2,f+0.025,f'{100*f:.0f}%\n({a}/{n})',ha='center',va='bottom',fontsize=6.8)
# c stacked proportions accepted outcomes
ax=axs[1,0]; fsc=SC31v[SC31v.rule_id=='Frozen-Fine-q75'].set_index('scenario').loc[SCENS]; imp=(fsc.improved/fsc.accepted).to_numpy(); wor=(fsc.worsened/fsc.accepted).to_numpy(); x3=np.arange(4); ax.bar(x3,imp,color=C_GREEN,label='Improved'); ax.bar(x3,wor,bottom=imp,color=C_ORANGE,label='Worsened'); ax.set_xticks(x3); ax.set_xticklabels([SCEN_LABEL[s] for s in SCENS],rotation=15); ax.set_ylim(0,1.05); ax.set_ylabel('Fraction of accepted candidates'); ax.set_title('Outcome quality after Frozen Fine-q75'); clean_ax(ax,'y'); panel_label(ax,'(c)'); ax.legend(frameon=False,loc='lower right')
for xi,n in zip(x3,fsc.accepted): ax.text(xi,1.01,f'n={int(n)}',ha='center',va='bottom',fontsize=6.7)
# d runtime triggered
ax=axs[1,1]; tr=D31[D31.triggered.astype(bool)]; vals=[tr.loc[tr.scenario==s,'runtime_e4_seconds'].to_numpy() for s in SCENS]; bp=ax.boxplot(vals,labels=[SCEN_LABEL[s] for s in SCENS],showfliers=False,patch_artist=True)
for box in bp['boxes']: box.set_facecolor('#F0F0F0'); box.set_edgecolor(C_DARK)
for med in bp['medians']: med.set_color(C_ORANGE); med.set_linewidth(1.2)
ax.set_ylabel('Incremental E4-HORP runtime (s)'); ax.set_title('Incremental runtime among triggered observations'); clean_ax(ax,'y'); panel_label(ax,'(d)'); ax.tick_params(axis='x',rotation=15)
save_all(fig,'Figure6_SelectiveRefinement_Diagnostics_v09')

# contact sheet previews
stems=[
'Figure1_FailureAware_Inversion_Workflow_v09','Figure2_FailureMechanisms_and_ObjectiveLandscapes_v09','Figure3_Stage28_Paired_Forest_v09','Figure4_Material_Thickness_Generalization_v09','Figure5_Development_to_IndependentValidation_v09','Figure6_SelectiveRefinement_Diagnostics_v09']
imgs=[]
for s in stems:
    im=Image.open(FIG/f'{s}.png').convert('RGB'); im.thumbnail((1800,1300)); imgs.append(im)
cols=2; gap=60; margin=60; maxw=max(im.width for im in imgs); maxh=max(im.height for im in imgs)
canvas=Image.new('RGB',(cols*maxw+(cols-1)*gap+2*margin,3*maxh+2*gap+2*margin),'white')
for i,im in enumerate(imgs):
    x=margin+(i%2)*(maxw+gap)+(maxw-im.width)//2; y=margin+(i//2)*(maxh+gap)+(maxh-im.height)//2; canvas.paste(im,(x,y))
canvas.save(OUT/'V09_Figure_Contact_Sheet.png',dpi=(300,300))

# audit dimensions + dpi + files
rows=[]
for s in stems:
    for ext in ['svg','pdf','png','tiff']:
        p=FIG/f'{s}.{ext}'; rec={'figure':s,'format':ext,'bytes':p.stat().st_size}
        if ext in ['png','tiff']:
            im=Image.open(p); rec.update({'width_px':im.width,'height_px':im.height,'dpi_x':im.info.get('dpi',(None,None))[0] if im.info.get('dpi') else None,'dpi_y':im.info.get('dpi',(None,None))[1] if im.info.get('dpi') else None})
        rows.append(rec)
pd.DataFrame(rows).to_csv(AUD/'V09_FIGURE_FILE_AUDIT.csv',index=False)
(AUD/'V09_FIGURE_STYLE_GUIDE.txt').write_text('''Photonics v0.9 figure freeze style\n\nTarget: full-width multi-panel figures (~180 mm).\nFont: Liberation Sans; 7.6 pt ticks, 8.3 pt axis labels, 8.6 pt panel titles, 9.5 pt panel letters.\nPanel labels: (a), (b), ...\nLine width: ~1.1–1.25 pt; axes 0.8 pt.\nPalette: restrained colorblind-safe blue/orange/green/purple + neutral gray; semantics are consistent within each figure.\nMasters: SVG + PDF.\nSubmission raster: PNG + LZW TIFF, 800 dpi.\nNo manual numerical editing; all data plots are generated from v0.8 canonical figure-data/results tables.\nFigure 1 is a schematic and contains no numerical claim beyond frozen method labels/parameters.\nFigure 2 representative observations and candidate basins are inherited from the v0.8 audited deterministic reconstruction/data selection.\n''',encoding='utf-8')

# captions draft
(AUD/'V09_FIGURE_CAPTIONS_CN.txt').write_text('''Figure 1. Failure-aware thin-film thickness inversion framework. (a) Physics-based single-layer reflectance model and shared gain-plus-baseline observation formulation. (b) Distinct sparse-impulsive and smooth-structured residual failure mechanisms. (c) E0/E1/E3 estimator hierarchy. (d) E3-anchored E4-HORP candidate generation followed by V2 eligibility, Frozen Fine-q75 filtering, and accept/fallback logic; the evidence chain separates mechanism benchmarking, selected-material/thickness generalization, gate development, and independent-seed validation.\n\nFigure 2. Representative failure mechanisms and local objective structure. (a) A deterministically reconstructed Stage28 impulsive observation with injected outlier locations. (b) A representative baseline-drift observation showing the measured-minus-clean residual and the injected smooth component. (c) Refined E3 candidate minima under impulsive contamination, with the true thickness and E0/E1/E3 estimates. Candidate objectives are normalized only within the selected observation. (d) E3-anchored E4 local refinement under baseline drift; the shaded region denotes the curvature-adaptive trust region. Representative spectra and candidate data are inherited from the audited v0.8 selection/reconstruction pipeline.\n\nFigure 3. Stage28 pooled paired absolute-error contrasts across the selected material systems and thicknesses. (a) E3−E0 and (b) E3−E1 contrasts for mean, q90, and q95 absolute error; error bars are 95% observation-level percentile-bootstrap intervals. Negative values indicate lower error for E3.\n\nFigure 4. Thickness-stratified mean paired effects across the three selected material systems. Solid circles denote E3−E0 and open dashed squares denote E3−E1. The mixed Material-C/380-nm E3−E0 point is shown off-scale and annotated with its exact boundary-driven value rather than being hidden.\n\nFigure 5. Separation of Stage30 development from Stage31 independent-seed validation. (a,b) Stage30 intervention count and overall mean error effect during gate development on frozen candidate fields. (c,d) Stage31 accepted outcomes and scenario-specific accepted-only effects after candidates were recomputed from independent-seed observations while the V2+Fine-q75 rule remained frozen.\n\nFigure 6. Selective-refinement diagnostics for Frozen Fine-q75. (a) Fine-scale normalized energy index among V2-eligible candidates with the frozen threshold. (b) Scenario-specific retention after Fine-q75. (c) Improved/worsened composition among accepted candidates. (d) Incremental E4-HORP runtime among triggered observations.\n''',encoding='utf-8')

# zip
zipbase=ROOT/'PHOTONICS_V09_PUBLICATION_FIGURES'
if (zipbase.with_suffix('.zip')).exists(): (zipbase.with_suffix('.zip')).unlink()
shutil.make_archive(str(zipbase),'zip',root_dir=OUT)
print('created',OUT)
