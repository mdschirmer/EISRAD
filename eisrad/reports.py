#!/usr/bin/env python
"""
:Summary:
Utilities for generating terminal summaries and CSV exports of
low‐Dice and high‐volume‐difference cases.

:Authors:
    Markus D. Schirmer
    MGH / Harvard Medical School

:Version: 1.0
:Date: 2025-06-24
:License: MIT
"""

import pandas as pd

def write_low_dice(df, thresh, out_csv=None):
    """
    Print % of cases with Dice < thresh; optional CSV of those rows.
    """
    pct = (df['Dice'] < thresh).mean() * 100
    print(f"ℹ️  {pct:.1f}% of cases have Dice < {thresh}")
    if out_csv:
        df[df['Dice'] < thresh][['manual','auto','Dice']].to_csv(out_csv,
                                                                  index=False)

def write_high_vol_diff(df, thresh, out_csv=None):
    """
    Print count of cases where relative volume diff > thresh;
    optional CSV of those rows.
    """
    high = df[df['vol_rel_diff'] > thresh]
    print(f"ℹ️  {len(high)} cases have relative volume-diff > {thresh}")
    if out_csv:
        high[['manual','auto','vol_rel_diff']].to_csv(out_csv, index=False)
