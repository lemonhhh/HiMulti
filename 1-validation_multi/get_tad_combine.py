import os
import sys
import numpy as np
import pandas as pd


from itertools import combinations

import pickle


resolution = 40000
ref = "mm9"
df_mm = pd.read_csv(f"/shareb/mliu/HiMulti/ref_data/{ref}.chrom.sizes",sep="\t",header=None)
df_mm.columns = ['chrom','size']
df_mm = df_mm.query("chrom!='chrX' and chrom != 'chrY'")


bins = []

# 遍历每个染色体，生成对应的bins
for index, row in df_mm.iterrows():
    chrom = row['chrom']
    size = row['size']
    for start in range(0, size, resolution):
        stop = min(start + resolution, size)
        pos = (start + stop) // 2
        bins.append([chrom, start, stop, pos])

# 将bins列表转换为DataFrame
bin_table_df = pd.DataFrame(bins, columns=['chrom', 'start', 'stop', 'pos'])
bin_table_df_mat = bin_table_df.copy()
bin_table_df_mat['chrom'] = bin_table_df_mat['chrom'].apply(lambda x: x+"(mat)")
bin_table_df_pat = bin_table_df.copy()
bin_table_df_pat['chrom'] = bin_table_df_pat['chrom'].apply(lambda x: x+"(pat)")

df_bin_mm9_40kb = pd.concat([bin_table_df_mat,bin_table_df_pat],axis=0)
df_bin_mm9_40kb.index = range(len(df_bin_mm9_40kb))


# 将bins列表转换为DataFrame
bin_table_df = pd.DataFrame(bins, columns=['chrom', 'start', 'stop', 'pos'])
chr_list = list(bin_table_df['chrom'].unique())


df_tad = pd.read_csv("/shareb/mliu/HiMulti/data/tads_by_se_and_groseq.csv")
df_tad['chr'] = df_tad['tad'].apply(lambda x: x.split(":")[0])
df_tad['start'] = df_tad['tad'].apply(lambda x: int(x.split(":")[1].split("-")[0]))
df_tad['end'] = df_tad['tad'].apply(lambda x: int(x.split(":")[1].split("-")[1]))
df_tad['mid'] = (df_tad['start'] + df_tad['end']) // 2
df_tad['size'] = (df_tad['end'] - df_tad['start']) // 40000



#分染色体做
for chr in chr_list:
    df_tad_chr = df_tad.query(f"chr=='{chr}'")
    df_tad_chr.index = range(len(df_tad_chr))
    combs = list(combinations(df_tad_chr.index, 3))
    new_data = []
    for comb in combs:
        tad1, tad2, tad3 = df_tad_chr.loc[comb[0]], df_tad_chr.loc[comb[1]], df_tad_chr.loc[comb[2]]
        new_data.append({
            'chr': tad1['chr'],
            'start1': tad1['start'],
            'end1': tad1['end'],
            'start2': tad2['start'],
            'end2': tad2['end'],
            'start3': tad3['start'],
            'end3': tad3['end'],
        })

    df_combinations = pd.DataFrame(new_data)
    df_combinations.to_csv(f"tad_combine_{chr}.csv")