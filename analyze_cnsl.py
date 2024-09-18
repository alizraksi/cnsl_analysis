#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Example code for bioinformatics portfolio.

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def run():
    df = pd.read_csv('cnsl_data.csv', index_col=0)
    df_cn = normalize_and_convert_to_cn(df)
    df_bkpts = find_bkpts(df_cn)
    df_cnv_types = categorize_cnv_type(df_bkpts)
    summarize_cnv_results(df_cnv_types)
    plot_cnsl_probes(df_cn)


def normalize_and_convert_to_cn(df):
# Normalize NGS read depth data to account for differences in coverage between samples, and also normalize for
# differences in the DNA capture efficiency across probes. Then filter out problematic probes and scale data to CN=2
    df_probes = df.drop('ethnicity', axis=1)
    df_normalized = normalize_coverage(df_probes)
    df_normalized = normalize_probes(df_normalized)
    df_cn = df_normalized * 2
    df_cn_cleaned = drop_misbehaving_probes(df_cn)
    df_cn_cleaned.insert(loc=0, column='ethnicity', value=df['ethnicity'])
    df_cn_cleaned.to_csv('output' / Path('cnsl_data_normalized.csv'))
    return df_cn_cleaned


def normalize_coverage(df):
# Correct for the difference in overall coverage between samples by taking the mean of the non_CNSL probe values and
# using this as a scaling factor for all probe values

    def scale_to_cn2(x):
        bg_probes = x[[col for col in x.index if col.startswith('non_')]]
        return x / bg_probes.mean()

    df_scaled = df.apply(scale_to_cn2, axis=1)
    return df_scaled


def normalize_probes(df):
# Normalize coverage values per probe. Use median (instead of mean) to account for the possibility of CNVs in samples.
# Assumption is that the majority of samples are copy neutral at each individual probe
    df_scaled = df.apply(lambda x: x / x.median(), axis=0)
    return df_scaled


def drop_misbehaving_probes(df):
# Most probes have variance < 0.05 across samples, with several clear outliers. Drop outlier misbehaving probes.
    var = df.var(axis=0)

    # plot variance as a histogram
    plot_variance(var)

    probes_to_drop = var[var >= 0.1].index.tolist()
    print(f'The following probes have high variance between samples:\n{var[var > 0.1]}.\nDropping probes.')
    df.drop(probes_to_drop, axis=1, inplace=True)
    return df


def find_bkpts(df):
# For every potential known breakpoint, get CN to the left and to the right of the breakpoint,
# and if the difference is close to 1, call that breakpoint found
    bkpts = ['CNSL_probe_10', 'CNSL_probe_20', 'CNSL_probe_27', 'CNSL_probe_32', 'CNSL_probe_34',
             'CNSL_probe_38', 'CNSL_probe_40']
    bkpts_results = []
    for bkpt in bkpts:
        ix_bkpt = df.columns.tolist().index(bkpt)
        probes_left = df.columns.tolist()[(ix_bkpt-4):ix_bkpt]
        probes_right = df.columns.tolist()[ix_bkpt:(ix_bkpt+4)]
        cn_left = df[probes_left].mean(axis=1)
        cn_right = df[probes_right].mean(axis=1)
        bkpt_present = abs(cn_right - cn_left).round(decimals=0).astype(bool)
        bkpts_results.append(bkpt_present)
    colnames = [f'{s}_bkpt_present' for s in bkpts]
    df_bkpts = pd.DataFrame(bkpts_results, index=colnames, columns=df.index).T
    return pd.concat([df, df_bkpts], axis=1)


def categorize_cnv_type(df):
# Determine the CNV type that is present for each sample, if any
    def cnv_is_present_between_probes(x, bkpt1:int, bkpt2:int, cn:int):
        if (x[f'CNSL_probe_{bkpt1}_bkpt_present'] and x[f'CNSL_probe_{bkpt2}_bkpt_present'] and
                cn_between_probes(x, bkpt1, bkpt2) == cn):
            return True
        return False

    def categorize_cnv_type_for_sample(x):
        if cnv_is_present_between_probes(x, bkpt1=32, bkpt2=38, cn=1):
            return 'cnv1_del'
        if cnv_is_present_between_probes(x, bkpt1=32, bkpt2=38, cn=3):
            return 'cnv1_dup'
        if cnv_is_present_between_probes(x, bkpt1=27, bkpt2=34, cn=1):
            return 'cnv2_del'
        if cnv_is_present_between_probes(x, bkpt1=27, bkpt2=34, cn=3):
            return 'cnv2_dup'
        if cnv_is_present_between_probes(x, bkpt1=20, bkpt2=40, cn=1):
            return 'cnv3_del'
        if cnv_is_present_between_probes(x, bkpt1=20, bkpt2=40, cn=3):
            return 'cnv3_dup'
        if cnv_is_present_between_probes(x, bkpt1=10, bkpt2=40, cn=1):
            return 'cnv4_del'
        if cnv_is_present_between_probes(x, bkpt1=10, bkpt2=40, cn=3):
            return 'cnv4_dup'
        return 'None'

    df_annot_cnv = df.apply(categorize_cnv_type_for_sample, axis=1)
    df_annot_cnv.name = 'cnv_type'
    df_annot_cnv = pd.concat([df['ethnicity'], df_annot_cnv], axis=1)
    df_annot_cnv.to_csv('output' / Path('cnsl_cnvs.csv'))
    return df_annot_cnv


def cn_between_probes(x, bkpt1, bkpt2):
# Return mean of probe values between two breakpoints (inclusive)
    ix_bkpt1 = x.index.tolist().index(f'CNSL_probe_{bkpt1}')
    ix_bkpt2 = x.index.tolist().index(f'CNSL_probe_{bkpt2}')
    probes = x.index.tolist()[ix_bkpt1:ix_bkpt2+1]
    return x[probes].mean().round(decimals=0)


def summarize_cnv_results(df):
# Count the number of samples with each CNV type present in each ethnicity
    df_summary = df.groupby('ethnicity').value_counts()
    df_summary.to_csv('output' / Path('cnsl_cnvs_summary.csv'))


def plot_variance(var):
# Plot histogram of variance values across "clean" probes
    well_behaved_probes_var = [x for x in var if x < 1]
    plt.figure(figsize=(30, 6))
    fig, ax = plt.subplots(1, 1)
    ax.hist(well_behaved_probes_var, bins=30)
    ax.set_xlim(0, 0.1)
    plt.savefig('output' / Path('cnsl_probes_variance.png'))


def plot_cnsl_probes(df):
# Plot normalized CNSL probe values for each sample. Display a subplot for each ethnicity.
    plt.figure(figsize=(20,14))
    fig, axs = plt.subplots(3,1)

    for ax, ethnicity in zip(axs, ['A', 'B', 'C']):
        df_cnsl_probes = df[df['ethnicity'] == ethnicity][[col for col in df.columns if col.startswith('CNSL_')]]
        ax.plot(np.arange(0, len(df_cnsl_probes.columns)), df_cnsl_probes.T, alpha=0.2)
        ax.set_ylim(0, 4.0)
        ax.set_xlabel('CNSL Probe')
        ax.set_ylabel(ethnicity)

    plt.savefig('output' / Path(f'cnsl_cvg.png'))
    plt.savefig('output' / Path(f'cnsl_cvg.pdf'))


def main():
    run()
    return 0


if __name__ == "__main__": sys.exit(main())
