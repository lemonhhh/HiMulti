import numpy as np
import os
import pandas as pd
import math
import random
import h5py
import pickle
from joblib import Parallel, delayed
import cooler

def compute_max_distance(d_ab, d_bc, d_ca):
    return np.max([d_ab, d_bc, d_ca])

def compute_mean_distance(d_ab, d_bc, d_ca):
    return np.mean([d_ab, d_bc, d_ca])

def circumcircle_radius(d_ab, d_bc, d_ca):
    s = (d_ab + d_bc + d_ca) / 2
    area = math.sqrt(s * (s - d_ab) * (s - d_bc) * (s - d_ca))
    radius = (d_ab * d_bc * d_ca) / (4 * area)
    return radius

def generate_random_indices(chr_idx_list, span_range, start1_idx, end1_idx, start2_idx, end2_idx, start3_idx, end3_idx):
    random_indices = []
    for _ in range(100):
        start_index1_random = random.choice(chr_idx_list[:-span_range])
        end_index3_random = start_index1_random + span_range
        end_index1_random = start_index1_random + (end1_idx - start1_idx)
        start_index2_random = start_index1_random + (start2_idx - start1_idx)
        end_index2_random = start_index1_random + (end2_idx - start1_idx)
        start_index3_random = start_index1_random + (start3_idx - start1_idx)
        random_indices.append([(start_index1_random, end_index1_random), (start_index2_random, end_index2_random), (start_index3_random, end_index3_random)])
    return random_indices

def compute_distances(distance_array, indices):
    (start1_idx, end1_idx), (start2_idx, end2_idx), (start3_idx, end3_idx) = indices
    # print(f"start1_idx {start1_idx},end1_idx {end1_idx},start2_idx {start2_idx} end2_idx {end2_idx} start3_idx {start3_idx}  end3_idx{end3_idx}")
    distance_ab = np.nanmean(distance_array[start1_idx:end1_idx, start2_idx:end2_idx])
    distance_bc = np.nanmean(distance_array[start2_idx:end2_idx, start3_idx:end3_idx])
    distance_ca = np.nanmean(distance_array[start3_idx:end3_idx, start1_idx:end1_idx])
    return distance_ab, distance_bc, distance_ca

def process_cell(distance_array, hic_array,random_index_list, start1_idx, end1_idx, start2_idx, end2_idx, start3_idx, end3_idx):
    print("cell")
    #distance_array是一个矩阵
    distance_ab, distance_bc, distance_ca = compute_distances(distance_array, ((start1_idx, end1_idx), (start2_idx, end2_idx), (start3_idx, end3_idx)))
    contact_ab, contact_bc, contact_ca = compute_distances(hic_array, ((start1_idx, end1_idx), (start2_idx, end2_idx), (start3_idx, end3_idx)))
    
    mean_distance = compute_mean_distance(distance_ab, distance_bc, distance_ca)
    max_distance = compute_max_distance(distance_ab, distance_bc, distance_ca)
    circumcircle_distance = circumcircle_radius(distance_ab, distance_bc, distance_ca)
    #取最大的
    hic = compute_max_distance(contact_ab, contact_bc, contact_ca)

    random_mean_list_points, random_max_list_points, random_circle_list_points,random_hic_list_points = [], [], [],[]
    for random_indices in random_index_list:
        random_distance_ab, random_distance_bc, random_distance_ca = compute_distances(distance_array, random_indices)
        random_contact_ab, random_contact_bc, random_contact_ca = compute_distances(hic_array, random_indices)
        
        mean_distance_random = compute_mean_distance(random_distance_ab, random_distance_bc, random_distance_ca)
        max_distance_random = compute_max_distance(random_distance_ab, random_distance_bc, random_distance_ca)
        circumcircle_distance_random = circumcircle_radius(random_distance_ab, random_distance_bc, random_distance_ca)
        hic_random_point = compute_max_distance(random_contact_ab,random_contact_bc,random_contact_ca)

        random_mean_list_points.append(mean_distance_random)
        random_max_list_points.append(max_distance_random)
        random_circle_list_points.append(circumcircle_distance_random)
        random_hic_list_points.append(hic_random_point)

    mean_distance_random = np.nanmean(random_mean_list_points)
    max_distance_random = np.nanmean(random_max_list_points)
    circumcircle_distance_random = np.nanmean(random_circle_list_points)
    hic_random = np.nanmean(random_hic_list_points)
    distance = np.mean(end1_idx-start1_idx,end2_idx-start2_idx,end3_idx-start3_idx)
    return mean_distance, max_distance, circumcircle_distance, mean_distance_random, max_distance_random, circumcircle_distance_random,hic,hic_random,distance

def process_chr(chr, df_true_chr, chr_idx_list):
    spatial_distance_mean_list = []
    spatial_distance_max_list = []
    spatial_distance_circle_list = []
    contact_hic_list = []

    spatial_distance_mean_random_bg_list = []
    spatial_distance_max_random_bg_list = []
    spatial_distance_circle_random_bg_list = []
    contact_hic_random_bg_list = []
    distance_list = []

    
    with h5py.File(f'distance_cells_{chr}.h5', 'r') as f: 
        # n_cells = f[chr].shape[0]
        n_cells = 10
        for i in range(len(df_true_chr)):
            print(f"Processing {i}/{len(df_true_chr)} for {chr}")
            start1_idx, end1_idx = df_true_chr['start1_idx'].iat[i], df_true_chr['end1_idx'].iat[i]
            start2_idx, end2_idx = df_true_chr['start2_idx'].iat[i], df_true_chr['end2_idx'].iat[i]
            start3_idx, end3_idx = df_true_chr['start3_idx'].iat[i], df_true_chr['end3_idx'].iat[i]
            span_range = end3_idx - start1_idx

            # print(f"start1_idx {start1_idx},end1_idx {end1_idx},start2_idx {start2_idx} end2_idx {end2_idx} start3_idx {start3_idx}  end3_idx{end3_idx}")
            random_index_list = generate_random_indices(chr_idx_list, span_range, start1_idx, end1_idx, start2_idx, end2_idx, start3_idx, end3_idx)
            # print(f"random_index_list, {random_index_list}")
            

         
            #细胞并行 cell是num
            results = Parallel(n_jobs=20)(delayed(process_cell)(f[chr][cell, :, :]
                                , cooler.Cooler(f"{mesc_path}/{cellnames[cell]}.{surfix}::/resolutions/40000").matrix(balance=False).fetch(f"{chr}(mat)")
                                ,random_index_list, start1_idx, end1_idx, start2_idx, end2_idx, start3_idx, end3_idx
            ) for cell in range(n_cells))

            mean_distance_list, max_distance_list, circumcircle_distance_list, mean_distance_list_random, max_distance_list_random, circumcircle_distance_list_random,hic_list,hic_list_random,distance = zip(*results)

            spatial_distance_mean_list.append(np.nanmean(mean_distance_list))
            spatial_distance_max_list.append(np.nanmean(max_distance_list))
            spatial_distance_circle_list.append(np.nanmean(circumcircle_distance_list))
            contact_hic_list.append(np.nanmean(hic_list))

            spatial_distance_mean_random_bg_list.append(np.nanmean(mean_distance_list_random))
            spatial_distance_max_random_bg_list.append(np.nanmean(max_distance_list_random))
            spatial_distance_circle_random_bg_list.append(np.nanmean(circumcircle_distance_list_random))
            contact_hic_random_bg_list.append(np.nanmean(hic_list_random))
            distance_list.append(distance)

            

            
    print(f"{chr}:{spatial_distance_circle_list}")
    print(f"{chr}:{spatial_distance_circle_random_bg_list}")        
    return (spatial_distance_mean_list, spatial_distance_max_list, spatial_distance_circle_list, 
            spatial_distance_mean_random_bg_list, spatial_distance_max_random_bg_list, spatial_distance_circle_random_bg_list,contact_hic_list,contact_hic_random_bg_list)

resolution = 40000
ref = "mm9"
df_mm = pd.read_csv(f"/shareb/mliu/HiMulti/ref_data/{ref}.chrom.sizes", sep="\t", header=None)
df_mm.columns = ['chrom', 'size']
df_mm = df_mm.query("chrom!='chrX' and chrom != 'chrY'")

bins = []
for index, row in df_mm.iterrows():
    chrom = row['chrom']
    size = row['size']
    for start in range(0, size, resolution):
        stop = min(start + resolution, size)
        pos = (start + stop) // 2
        bins.append([chrom, start, stop, pos])

bin_table_df = pd.DataFrame(bins, columns=['chrom', 'start', 'stop', 'pos'])

metadata = pd.read_csv("/share/Data/hxie/project/202209/esc_xwliu/esc0728/stat/stat_0728.csv")




mesc1_path = "/shareb/mliu/HiMulti/data/mESC/mcool"
mesc2_mm9_path = "/shareb/mliu/HiMulti/data/mESC2/mm9/mcool"
surfix1="balanced.mcool"
surfix2_mm9 = "mm9.mcool"
cellnames = os.listdir(mesc2_mm9_path)
cellnames = [cellname.split(".")[0] for cellname in cellnames]
mesc_path = mesc2_mm9_path
surfix = surfix2_mm9



df_all_gam = pd.read_csv("common_gam_triplet.csv")



# df_all_gam_top = df_all_gam_top.head(100)
df_all_gam['start1_idx'] = df_all_gam['start1'] // resolution
df_all_gam['end1_idx'] = df_all_gam['end1'] // resolution
df_all_gam['start2_idx'] = df_all_gam['start2'] // resolution
df_all_gam['end2_idx'] = df_all_gam['end2'] // resolution
df_all_gam['start3_idx'] = df_all_gam['start3'] // resolution
df_all_gam['end3_idx'] = df_all_gam['end3'] // resolution
chr_list = df_all_gam['chr1'].unique()



results = Parallel(n_jobs=20)(delayed(process_chr)(chr, df_all_gam.query("chr1==@chr"), np.arange(0, bin_table_df.query("chrom==@chr").shape[0])) for chr in chr_list)

# Unpack the results
(spatial_distance_mean_list, spatial_distance_max_list, spatial_distance_circle_list, 
 spatial_distance_mean_random_bg_list, spatial_distance_max_random_bg_list, spatial_distance_circle_random_bg_list,contact_hic_list,contact_hic_random_bg_list) = map(list, zip(*results))

# Save the results
with open("all_gam_spatial_distance_mean_list.pickle", "wb") as f:
    pickle.dump(spatial_distance_mean_list, f)
with open("all_gam_spatial_distance_max_list.pickle", "wb") as f:
    pickle.dump(spatial_distance_max_list, f)
with open("all_gam_spatial_distance_circle_list.pickle", "wb") as f:
    pickle.dump(spatial_distance_circle_list, f)
with open("all_gam_spatial_distance_mean_random_bg_list.pickle", "wb") as f:
    pickle.dump(spatial_distance_mean_random_bg_list, f)
with open("all_gam_spatial_distance_max_random_bg_list.pickle", "wb") as f:
    pickle.dump(spatial_distance_max_random_bg_list, f)
with open("all_gam_spatial_distance_circle_random_bg_list.pickle", "wb") as f:
    pickle.dump(spatial_distance_circle_random_bg_list, f)

with open("all_gam_hic_list.pickle", "wb") as f:
    pickle.dump(contact_hic_list, f)
with open("all_gam_hic_random_bg_list.pickle", "wb") as f:
    pickle.dump(contact_hic_random_bg_list, f)
