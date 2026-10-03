"""Static, source-backed notebook figures for the S8 diagnostic."""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.stats import chi2

METHODS=('all_6_equal_gcc_wls','equal_weight_srp_phat')
NAMES={METHODS[0]:'GCC-PHAT / WLS',METHODS[1]:'SRP-PHAT'}
COLORS={'broadband':'#2166ac','nonstationary_harmonic':'#d97726'}
STATION_COLORS={'S0':'#2166ac','S1':'#b35806','S2':'#637d3c'}


def style():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.titlesize':12,
                         'axes.labelsize':11,'figure.dpi':120,'savefig.dpi':150,
                         'axes.spines.top':False,'axes.spines.right':False,
                         'axes.grid':True,'grid.alpha':.18})


def bearing_comparison(metrics):
    style();fig,axes=plt.subplots(2,2,figsize=(12,8),sharex=True,sharey=True,layout='constrained')
    positions={0:0,1:1,2:2,3:3,4:4,6:0,7:1,8:2,9:3,10:4}
    markers={'S0':'o','S1':'s','S2':'^'}
    for row,subset in enumerate(('all_frames','tracker_selected')):
        for col,method in enumerate(METHODS):
            ax=axes[row,col];data=metrics[(metrics['subset']==subset)&(metrics.method==method)]
            for source,color in COLORS.items():
                for si,(station,marker) in enumerate(markers.items()):
                    part=data[(data.source_class==source)&(data.station_id==station)]
                    offset=(-.16 if source=='broadband' else .16)+(si-1)*.045
                    ax.scatter([positions[i]+offset for i in part['index']],part.p95_error_deg,
                               marker=marker,color=color,s=34,
                               label=f'{source.replace("nonstationary_","")} / {station}' if row==0 and col==0 else None)
            ax.set_yscale('log');ax.set_ylim(.1,180);ax.set_yticks([.1,1,10,100],labels=['0.1','1','10','100'])
            ax.set_title(f'{NAMES[method]} · '+('все кадры' if row==0 else 'передано трекеру'))
            ax.set_xticks(range(5),labels=['200/r0','200/r1','700/r0','700/r1','1000/r0'])
            if col==0:ax.set_ylabel('P95 угловой ошибки, ° (log)')
            if row==1:ax.set_xlabel('Дальность, м / replicate; одна точка = одна станция')
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc='outside lower center',ncol=3,fontsize=9)
    fig.suptitle('Пять пар harmonic/broadband: отдельные станции, без независимых испытаний по кадрам',fontsize=13)
    return fig


def calibration_qq(residuals):
    style();fig,axes=plt.subplots(1,2,figsize=(12,5),sharey=True,layout='constrained')
    probabilities=np.linspace(.01,.99,99);reference=chi2.ppf(probabilities,2)
    for ax,method in zip(axes,METHODS):
        for index,color,linestyle,label in ((0,'#2166ac','-','broadband 200 m'),(6,'#d97726','-','harmonic 200 m'),
                                            (2,'#2166ac','--','broadband 700 m'),(8,'#d97726','--','harmonic 700 m')):
            x=residuals[(residuals['index']==index)&(residuals.method==method)&(residuals.station_id=='S0')&residuals.tracker_selected]
            ax.plot(reference,np.quantile(x.d_R2,probabilities),color=color,ls=linestyle,label=label,lw=1.8)
        ax.plot([.01,20],[.01,20],color='#333333',ls=':',label='номинальное совпадение',lw=1.4)
        ax.set_xscale('log');ax.set_yscale('log');ax.set_title(NAMES[method]+' · S0 · stride 32')
        ax.set_xlabel('Квантиль номинального χ²₂ (log)');ax.set_ylabel('Квантиль bearing d²_R после bias (log)')
        ax.legend(fontsize=9)
    fig.suptitle('Применимость прежней R: Q–Q диагностический график, не тест независимых кадров')
    return fig


def error_consistency(residuals):
    style();fig,axes=plt.subplots(2,2,figsize=(12,7),sharex=True,sharey=True,layout='constrained')
    for row,index in enumerate((6,8)):
        for col,method in enumerate(METHODS):
            ax=axes[row,col]
            for station,color in STATION_COLORS.items():
                part=residuals[(residuals['index']==index)&(residuals.method==method)&
                               (residuals.station_id==station)&residuals.tracker_selected]
                ax.plot(part.reception_time_s,part.angular_error_deg,color=color,marker='.',lw=1,label=station)
            ax.axhline(10,color='#555555',ls='--',lw=1)
            ax.set_ylim(0,180);ax.set_title(f'{NAMES[method]} · harmonic {200 if index==6 else 700} м, r0')
            if col==0:ax.set_ylabel('Угловая ошибка, °')
            if row==1:ax.set_xlabel('Время приёма после начала окна, с')
    axes[0,0].legend(ncol=3,loc='upper right')
    fig.suptitle('Временная и межстанционная структура ошибок: фактически переданные кадры')
    return fig


def failure_timeline(timeline,hypotheses,residuals):
    style();fig,axes=plt.subplots(6,2,figsize=(13,16),sharex=True,layout='constrained',
                               gridspec_kw={'height_ratios':[1.3,1,1.2,1.2,.65,.65]})
    state_colors={'uninitialized':'#bdbdbd','tentative':'#e6b450','confirmed':'#2166ac',
                  'budget':'#8866aa','hypothesis_limit':'#d97726','other':'#637d3c'}
    for col,index in enumerate((0,6)):
        title='Broadband контроль · индекс 0' if index==0 else 'Harmonic · индекс 6'
        for station,color in STATION_COLORS.items():
            x=residuals[(residuals['index']==index)&(residuals.method==METHODS[0])&
                        (residuals.station_id==station)&residuals.tracker_selected]
            axes[0,col].plot(x.reception_time_s,x.angular_error_deg,color=color,marker='.',lw=1,label=station)
        axes[0,col].set_yscale('log');axes[0,col].set_ylim(max(1e-6, .8*residuals[(residuals['index'].isin([0,6])) & (residuals.method==METHODS[0]) & residuals.tracker_selected].angular_error_deg.min()),180);axes[0,col].set_title(title+' · 200 м, replicate 0')
        part=timeline[(timeline['index']==index)&(timeline.method==METHODS[0])&
                      (timeline.confirmation_variant=='three_station_confirmation')]
        axes[1,col].step(part.processing_time_s,part.available_event_count,where='post',color='#333333',lw=1.7)
        for mi,method in enumerate(METHODS):
            ax=axes[2+mi,col];state_ax=axes[4+mi,col]
            for vi,variant in enumerate(('baseline','three_station_confirmation')):
                t=timeline[(timeline['index']==index)&(timeline.method==method)&(timeline.confirmation_variant==variant)].sort_values('processing_time_s')
                ax.step(t.processing_time_s,t.executed_optimizations_cumulative,where='post',
                        color='#2166ac' if vi==0 else '#d97726',ls='-' if vi==0 else '--',lw=1.7,label='2 станции' if vi==0 else '3 станции')
                h=hypotheses[(hypotheses['index']==index)&(hypotheses.method==method)&(hypotheses.confirmation_variant==variant)]
                if vi==1:
                    for action,marker in (('tentative_created','^'),('tentative_rejected','x'),('confirmed','o')):
                        hs=h[h.action==action]
                        if len(hs):
                            y=np.interp(hs.processing_time_s,t.processing_time_s,t.executed_optimizations_cumulative)
                            ax.scatter(hs.processing_time_s,y,marker=marker,color='#333333',s=32,
                                       label={'tentative_created':'создана гипотеза','tentative_rejected':'отклонена','confirmed':'подтверждена'}[action])
                times=t.processing_time_s.to_numpy();widths=np.diff(np.r_[times,times[-1]+np.median(np.diff(times))])
                for (_,r),left,width in zip(t.iterrows(),times,widths):
                    reason=str(r.failure_reason)
                    category=('confirmed' if r.valid else 'hypothesis_limit' if 'hypothesis_limit' in reason else
                              'budget' if 'budget' in reason else 'tentative' if r.status=='tentative' else 'uninitialized')
                    state_ax.broken_barh([(left,width)],(vi-.3,.6),facecolors=state_colors[category])
            ax.axhline(128,color='#666666',ls=':',lw=1.2);ax.set_ylim(-3,138)
            ax.set_title(NAMES[method]+' · выполненные batch-оптимизации')
            state_ax.set_yticks([0,1],labels=['2 станции','3 станции']);state_ax.set_ylim(-.5,1.5)
            state_ax.set_title(NAMES[method]+' · статус публикаций',fontsize=10);state_ax.grid(False)
            if col==0:ax.set_ylabel('Число выполненных fits')
    axes[0,0].set_ylabel('Ошибка GCC, ° (log)');axes[0,0].legend(ncol=3,fontsize=9)
    axes[1,0].set_ylabel('Доступно событий');axes[1,0].set_ylim(0,185);axes[1,1].set_ylim(0,185)
    axes[1,0].set_title('Общее расписание · три станции · 177 событий')
    axes[1,1].set_title('То же расписание и состав измерений')
    axes[2,1].legend(fontsize=8,ncol=2,loc='upper left');axes[2,0].legend(fontsize=8,loc='upper left')
    for ax in axes[-1]:ax.set_xlabel('Время доступности после начала приёма, с');ax.set_xlim(0,20.1)
    fig.legend(handles=[Patch(color=state_colors[x],label=y) for x,y in
        (('uninitialized','Нет гипотезы'),('tentative','Предварительный трек'),('confirmed','Подтверждён'),
         ('budget','Предел fits'),('hypothesis_limit','Предел гипотез'))],loc='outside lower center',ncol=3,fontsize=9)
    fig.suptitle('Цепочка отказа в 200 м: исходные журналы; пунктир 128 — предел fits на поколение',fontsize=13)
    return fig
