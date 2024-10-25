# Standard libraries
import os
import sys
import errno
import random
import gzip
import pickle

# Numerical and data handling
import numpy as np
import pandas as pd

# Sparse matrices and spatial operations
from scipy.sparse import csr_matrix, lil_matrix
from scipy.spatial import KDTree, ConvexHull  # KDTree for spatial queries, ConvexHull for convex hull processing
from scipy.spatial.distance import pdist, squareform
from scipy import ndimage, stats
from scipy.signal import convolve2d
from scipy.ndimage import convolve
# Machine learning and clustering
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import LabelEncoder

# Image processing and similarity metrics
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error

# Plotting and visualization
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
from matplotlib.ticker import EngFormatter

# Data visualization settings
sns.set_theme(style="whitegrid")
fruitpunch = sns.blend_palette(['white', 'red'], as_cmap=True)

# Parallel processing
from multiprocessing import Pool
from functools import partial
import concurrent.futures

# Progress tracking
import tqdm

# Hi-C data processing
import cooler

sys.path.append('/share/home/mliu/shareb/mliu/HiMulti/3-sprite/CHARMtools')
from CHARMtools import Cell3Ddev as Cell3D
from CHARMtools import MultiCell3D

def point_cloud_rotation(point_cloud, x_angle=None,y_angle=None,z_angle=None):
    if x_angle:
        rotation_matrix = np.array([[1,0,0],[0,np.cos(x_angle),-np.sin(x_angle)],[0,np.sin(x_angle),np.cos(x_angle)]])
        point_cloud = np.dot(point_cloud,rotation_matrix)
    if y_angle:
        rotation_matrix = np.array([[np.cos(y_angle),0,np.sin(y_angle)],[0,1,0],[-np.sin(y_angle),0,np.cos(y_angle)]])
        point_cloud = np.dot(point_cloud,rotation_matrix)
    if z_angle:
        rotation_matrix = np.array([[np.cos(z_angle),-np.sin(z_angle),0],[np.sin(z_angle),np.cos(z_angle),0],[0,0,1]])
        point_cloud = np.dot(point_cloud,rotation_matrix)

    return point_cloud

def plot_metrics(x,xlabel,pearson,spearman,ssmi,rms):
    # line plot of corrs ssmis and rms vs width
    fig, axes = plt.subplots(1, 4, figsize=(10, 3),dpi=120)

    axes[0].plot(x, pearson, color='tab:blue')
    axes[0].set_xlabel(xlabel)
    axes[0].set_ylabel('Pearson corrs')
    axes[0].set_title('Pearson corrs')

    axes[1].plot(x, spearman, color='tab:blue')
    axes[1].set_xlabel(xlabel)
    axes[1].set_ylabel('Spearman corrs')
    axes[1].set_title('Spearman corrs')

    axes[2].plot(x, ssmi, color='tab:blue')
    axes[2].set_xlabel(xlabel)
    axes[2].set_ylabel('SSMIs')
    axes[2].set_title('SSMIs')

    axes[3].plot(x, rms, color='tab:blue')
    axes[3].set_xlabel(xlabel)
    axes[3].set_ylabel('RMSs')
    axes[3].set_title('RMSs')

    plt.tight_layout()
    plt.show()
    
bp_formatter = EngFormatter('b')

def format_ticks(ax, x=True, y=True, rotate=True):
    if y:
        ax.yaxis.set_major_formatter(bp_formatter)
    if x:
        ax.xaxis.set_major_formatter(bp_formatter)
        ax.xaxis.tick_bottom()
    if rotate:
        ax.tick_params(axis='x',rotation=45)

    

def generate_slices_uniform(tdg_path,CpG_df,slice_width=6,slice_num=1):
    tdg = pd.read_csv(tdg_path,sep="\t",header = None)
    tdg.columns = ["chrom","start","x","y","z"]
    
    # print(CpG_df.head())
    tdg["start"] = tdg["start"].astype(int)
    tdg["pos"] = tdg["start"] + 20000
    # print(tdg.head())

    slices = []
    for slice in range(slice_num):
        rotated = point_cloud_rotation(tdg[["x","y","z"]].values,
                                        x_angle=np.random.uniform(0, 2*np.pi),
                                        y_angle=np.random.uniform(0, 2*np.pi),
                                        z_angle=np.random.uniform(0, 2*np.pi)
                                        )

        tdg_temp = tdg[["chrom","pos"]]
        tdg_temp = tdg_temp.assign(x=rotated[:,0],y=rotated[:,1],z=rotated[:,2])
        
        xrange = (tdg_temp["x"].min(),tdg_temp["x"].max())
        
        slice_upper = np.random.uniform(xrange[0]+slice_width,xrange[1])
        slice_lower = slice_upper - slice_width

        slice_tmp = tdg_temp.query(f'x > @slice_lower & x < @slice_upper').assign(inslice = lambda x: 1)[["chrom","pos","inslice"]]

        
        slice = pd.merge(CpG_df,slice_tmp,how="left").fillna(0)  
        slices.append(slice["inslice"].values)
    return slices

def generate_slices(tdg_path,CpG_df,slice_width=6,slice_num=1):
    tdg = pd.read_csv(tdg_path,sep="\t",header = None)
    tdg.columns = ["chrom","pos","x","y","z"]
    tdg["pos"] = tdg["pos"].astype(int) - 20000

    slices = []
    for slice in range(slice_num):

        rotated = point_cloud_rotation(tdg[["x","y","z"]].values,
                                        x_angle=np.random.uniform(0, 2*np.pi),
                                        y_angle=np.random.uniform(0, 2*np.pi),
                                        z_angle=np.random.uniform(0, 2*np.pi)
                                        )

        tdg_temp = tdg[["chrom","pos"]]
        tdg_temp = tdg_temp.assign(x=rotated[:,0],y=rotated[:,1],z=rotated[:,2])
        
        xrange = (tdg_temp["x"].min(),tdg_temp["x"].max())
        
        # 计算点云的中心
        center_x = tdg_temp["x"].mean()
        slice_center = np.random.normal(center_x, slice_width / 2)
        slice_lower = max(slice_center - slice_width / 2, tdg_temp["x"].min())
        slice_upper = min(slice_center + slice_width / 2, tdg_temp["x"].max())

        # slice_upper = np.random.uniform(xrange[0]+slice_width,xrange[1])
        # slice_lower = slice_upper - slice_width

        slice = tdg_temp.query(f'x > @slice_lower & x < @slice_upper').assign(inslice = lambda x: 1)[["chrom","pos","inslice"]]
        slice = pd.merge(CpG_df,slice,how="left").fillna(0)
        slices.append(slice["inslice"].values)
    return slices



def plot_single_matrix(matrix,cmap="fall",title=None,vmax=1,vmin=0):
    plt.figure(figsize=(8, 3),dpi=120)
    plt.subplot(1, 2, 1)
    plt.imshow(matrix, cmap=cmap,vmax=vmax,vmin=vmin)
    plt.colorbar(label=title)
    plt.title(title)

    
    
def mat_cor_with_na(mat1,mat2,sample):
    # Calculate distance matrices
    distance_matrix_1 = mat1.flatten()
    distance_matrix_2 = mat2.flatten()

    # Replace inf values with nan
    distance_matrix_1 = np.where(np.isinf(distance_matrix_1), np.nan, distance_matrix_1)
    distance_matrix_2 = np.where(np.isinf(distance_matrix_2), np.nan, distance_matrix_2)

    # Remove any NaN values from both arrays (only where both have NaNs in the same position)
    mask = ~np.isnan(distance_matrix_1) & ~np.isnan(distance_matrix_2)

    distance_matrix_1 = distance_matrix_1[mask]
    distance_matrix_2 = distance_matrix_2[mask]

    #sample 
    if sample:
        sample_index = np.random.choice(len(distance_matrix_1), sample, replace=False)
        distance_matrix_1 = distance_matrix_1[sample_index]
        distance_matrix_2 = distance_matrix_2[sample_index]

    # Check if there are any remaining NaNs or infs
    if not np.isfinite(distance_matrix_1).all() or not np.isfinite(distance_matrix_2).all():
        raise ValueError("The input arrays contain infs or NaNs after preprocessing.")

    # Now you can safely call pearsonr
    print(f"length is {len(distance_matrix_1)}")
    pearsonr_value,_ = stats.pearsonr(distance_matrix_1, distance_matrix_2)
    spearmanr_value,_ = stats.spearmanr(distance_matrix_1, distance_matrix_2)
    normalized_mat1 = (mat1 - np.nanmean(mat1)) / np.nanstd(mat1)
    normalized_mat2 = (mat2 - np.nanmean(mat2)) / np.nanstd(mat2)
    normalized_mat1[np.isnan(normalized_mat1)] = 0
    normalized_mat2[np.isnan(normalized_mat2)] = 0
    ssmis = ssim(normalized_mat1, normalized_mat2)


    return [pearsonr_value,spearmanr_value,ssmis]


    
def Calculating_diagonal_data(matrix):
    N, M = len(matrix), len(matrix[0])
    Diagonal_mean = np.full(M, 0.0)
    Diagonal_std = np.full(M, 0.0)
    std = []
    for d in range(N):
        intermediate = []
        c = d
        r = 0
        while r < N - d:
            intermediate.append(matrix[r][c])
            r += 1
            c += 1
        intermediate = np.array(intermediate)
        Diagonal_mean[d] = (np.nanmean(intermediate))
        Diagonal_std[d] = (np.nanstd(intermediate))
    return Diagonal_mean, Diagonal_std


def Distance_normalization(rawmatrix):
    
    Diagonal_mean, Diagonal_std = Calculating_diagonal_data(rawmatrix)
    N, M = len(rawmatrix), len(rawmatrix[0])
    matrix = np.full((N, M), np.nan)
    for d in range(N):
        c = d
        r = 0
        while r < N - d:
            if Diagonal_mean[d] != np.nan and Diagonal_std[d] != 0:
                matrix[r][c] = (rawmatrix[r][c] - Diagonal_mean[d]) / Diagonal_std[d]
            r += 1
            c += 1
    for r in range(N):
        for c in range(r+1, M):
            matrix[c][r] = matrix[r][c]
    return matrix


def plot_metrics(x, xlabel, pearson, spearman, ssmi, rms, title, color):
    # line plot of corrs, ssmis, and rms vs width
    fig, axes = plt.subplots(1, 4, figsize=(10, 3), dpi=120)

    for ax in axes:
        # 去除右边框和上边框
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        # 不显示网格
        ax.grid(False)
    
    axes[0].plot(x, pearson, color=color)
    axes[0].set_xlabel(xlabel)
    axes[0].set_ylabel('Pearson')
    axes[0].set_title('Pearson')

    axes[1].plot(x, spearman, color=color)
    axes[1].set_xlabel(xlabel)
    axes[1].set_ylabel('Spearman')
    axes[1].set_title('Spearman')

    axes[2].plot(x, ssmi, color=color)
    axes[2].set_xlabel(xlabel)
    axes[2].set_ylabel('SSIM')
    axes[2].set_title('SSIM')

    axes[3].plot(x, rms, color=color)
    axes[3].set_xlabel(xlabel)
    axes[3].set_ylabel('SSC')
    axes[3].set_title('SSC')

    plt.tight_layout()
    plt.savefig(f"{title}.png")

    
bp_formatter = EngFormatter('b')

def get_merged(original_segmentation, merge=2):
    column_indexes = list(range(len(original_segmentation.columns)))

    # Shuffle column indexes in place
    np.random.shuffle(column_indexes)

    # Group them in threes
    index_sets = list(utils.grouper(column_indexes, merge))

    # Merge random columns of the segmentation in groups of three
    return pd.concat(
        [original_segmentation.iloc[:,indexes].any(axis=1).astype(int) for indexes in index_sets],
        axis=1)

def normalize_gam_matrix(gam_matrix, mean_slices):
    # 定义标准化后的矩阵
    normalized_matrix = np.zeros_like(gam_matrix)
    # 获取矩阵的行数和列数
    rows, cols = gam_matrix.shape
    # 遍历矩阵中的每个元素
    for i in range(rows):
        for j in range(cols):
            # 提取当前元素
            D = gam_matrix[i, j]
            # 计算 f_a 和 f_b
            f_a = mean_slices[i]
            f_b = mean_slices[j]

            # 应用标准化规则
            if D < 0:
                normalized_matrix[i, j] = min(f_a * f_b, (1 - f_a) * (1 - f_b))
            else:
                normalized_matrix[i, j] = min(f_b * (1 - f_a), f_a * (1 - f_b))

    return normalized_matrix

from typing import Union
from contextlib import suppress

def meanFilter(a: np.ndarray, h: int):
   
    # Create a filter kernel (2h+1 by 2h+1) with all ones
    fSize = 2 * h + 1
    kernel = np.ones((fSize, fSize)) / (fSize * fSize)  # Normalized kernel for mean filter

    # Apply convolution to the input matrix
    filtered = convolve(a, kernel, mode='constant', cval=0.0)
    return filtered


def trimDiags(a: np.ndarray, iDiagMax: int, bKeepMain: bool):
    rows, cols = np.indices(a.shape)
    gDist = np.abs(rows - cols)
    mask = (gDist < iDiagMax) & (bKeepMain | (gDist != 0))
    result = np.where(mask, a, 0)
    
    return result


epsilon = 1e-10
def sccByDiag(m1: np.ndarray, m2: np.ndarray, nDiags: int):
    rho_list = []
    weight_list = []

    for diag in range(nDiags):
        # Extract the diagonal elements with offset 'diag'
        m1_diag = np.diag(m1, k=diag)
        m2_diag = np.diag(m2, k=diag)

        # Remove common zero entries
        mask = ((np.abs(m1_diag) > epsilon) | (np.abs(m2_diag) > epsilon)) & ~np.isnan(m1_diag) & ~np.isnan(m2_diag)


        m1_diag_nonzero = m1_diag[mask]
        m2_diag_nonzero = m2_diag[mask]
       

        # If there are fewer than 3 non-zero values, skip this diagonal
        if len(m1_diag_nonzero) < 3:
            continue

        # Compute Pearson correlation for the diagonal
        rho = np.corrcoef(m1_diag_nonzero, m2_diag_nonzero)[0, 1]
        if np.isnan(rho):
            rho = 0  # Handle NaN values

        # Compute the weight based on the number of non-zero samples
        num_samples = len(m1_diag_nonzero)
        weight = num_samples * varVstran(num_samples)

        rho_list.append(rho)
        weight_list.append(weight)

    # Convert lists to arrays
    rho_array = np.array(rho_list)
    weight_array = np.array(weight_list)

    # Handle cases with no valid diagonals
    if weight_array.sum() == 0:
        return 0
    scc_score = np.sum(rho_array * weight_array) / np.sum(weight_array)
    
    return scc_score

def varVstran(n: Union[int, np.ndarray]):
    with suppress(ZeroDivisionError), np.errstate(divide='ignore', invalid='ignore'):
        return np.where(n < 2, np.nan, (1 + 1.0 / n) / 12.0)
    
def hicrepSCC(m1: np.ndarray, m2: np.ndarray,
              h: int, dBPMax: int):
    # 获取接触矩阵的尺寸
    nDiags = m1.shape[0] if dBPMax < 0 else min(dBPMax, m1.shape[0])
    
    # 初始化相关数组
    rho = np.full(nDiags, np.nan)
    ws = np.full(nDiags, np.nan)
    

    n1 = np.nansum(m1)
    n2 = np.nansum(m2)
   
    m1 = m1.astype(float) / n1
    m2 = m2.astype(float) / n2
    
    # 如果提供了窗口大小 h，应用平滑滤波
    if h > 0:
        m1 = meanFilter(m1, h)
        m2 = meanFilter(m2, h)


    # 去除主对角线及以上的对角线
    m1 = trimDiags(m1, nDiags, False)
    m2 = trimDiags(m2, nDiags, False)
    
    # 计算每个对角线的SCC分数
    scc = sccByDiag(m1, m2, nDiags)
    return scc

def plot_matrix(matrix, title, region_start, region_end, cmap, vmax, vmin):
    plt.figure(figsize=(4, 4), facecolor='white')
    plt.imshow(matrix,
               extent=(region_start, region_end, region_end, region_start),
               interpolation='none', vmin=vmin, vmax=vmax, cmap=cmap)

    plt.title(title)
    # 不显示网格
    plt.grid(False)
    ax = plt.gca()

    ax.yaxis.set_ticks_position('left')  # 设置y轴刻度仅在左侧显示
    ax.xaxis.set_ticks_position('none')  # 禁用x轴底部刻度显示
    ax.set_xticks([])  # 移除x轴的所有刻度
    ax.set_xticklabels([])  # 移除x轴的刻度标签

    ax.set_yticks([region_start, region_end])  # 仅设置最小值和最大值作为刻度
    ax.set_yticklabels([str(region_start), str(region_end)])  # 设置刻度标签


    format_ticks(ax)

    # 显示 colorbar
    cbar = plt.colorbar(shrink=0.8)
    cbar.set_label(title)
    cbar.set_ticks([vmin,vmax])
    cbar.set_ticklabels([vmin,vmax])

    plt.show()


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


chr_list = list(bin_table_df['chrom'].unique())

resolution = 40000
all_cells = pd.read_csv("/shareb/mliu/HiMulti/data/mESC/all_mESC_cellname.txt",sep="\t",header=None).values.flatten()

def _load_cell(cellname,resolution):
    cell = Cell3D.Cell3D(
        cellname = cellname,
        resolution = 40000,
        tdg_path = f"/shareb/mliu/HiMulti/data/mESC/tdg/{cellname}.40kb.3dg",
    )
    cell.add_chrom_length(chrom_length_path = "/shareb/mliu/HiMulti/ref_data/dip.len.mm9")
    cell.build_kdtree()
    return cell

with concurrent.futures.ProcessPoolExecutor(20) as executor:
   cells = list(tqdm.tqdm(executor.map(_load_cell, all_cells, resolution*np.ones(len(all_cells))), total=len(all_cells)))

esc_cells = MultiCell3D.MultiCell3D(cells)

for chr in chr_list:
    print(chr)
    start = 0 #还是挺慢的
    end = df_mm.query("chrom==@chr")['size'].iat[0]
    in_silico_gam_mat_chr = esc_cells.calc_insilico_GAM(genome_coord=f"{chr}a:{start}-{end}",slice_width=2.5,num_slices=200)
    np.save("/share/home/mliu/shareb/mliu/HiMulti/2-gam/silico_dprime/insilico_GAM_mESC_{chr}_dprime.npy",in_silico_gam_mat_chr)
    