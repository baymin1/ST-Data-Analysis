import json
import scanpy as sc
import matplotlib.pyplot as plt
import pandas as pd
import os

'''
读取数据
'''
library_id = 'BC_06066'
data_dir = r'F:\algorithm\python\feature-mapping\mapping_related'  # 根目录
ST_dir = os.path.join(data_dir, 'ST')  # ST目录

BC_dir = os.path.join(ST_dir, library_id)  # ST其中1例ST数据
h5_path = os.path.join(BC_dir, 'filtered_feature_bc_matrix.h5')  # h5文件路径
spatial_dir = os.path.join(BC_dir, 'spatial')  # 空间信息路径
position_path = os.path.join(spatial_dir, 'tissue_positions_list.csv')  # 空间坐标信息路径
hires_image_path = os.path.join(spatial_dir, 'tissue_hires_image.png')  # 高分辨率图像路径
scalefactors_json_path = os.path.join(spatial_dir, 'scalefactors_json.json')  # 缩放因子路径

# 读取基因表达矩阵，读取未过滤的自己进行质控。
adata = sc.read_10x_h5(h5_path)
adata.var_names_make_unique()
print(adata)

'''
obs：细胞相关的元数据（行索引是细胞ID）。
var：基因相关的元数据（行索引是基因ID）。
uns：非结构化的注释数据，存储各种辅助信息和参数。
obsm：细胞级别的多维数据，例如空间坐标。
'''

# 读取空间坐标表
positions = pd.read_csv(position_path, header=None)
positions.columns = ["barcode", "in_tissue", "array_row", "array_col", "pxl_row_in_fullres", "pxl_col_in_fullres"]
# 将barcode设置为索引
positions.set_index('barcode', inplace=True)
# 将空间坐标信息左连接添加到adata.obs中
adata.obs = adata.obs.join(positions, how='left')
# 将空间坐标添加到adata.obsm中
adata.obsm['spatial'] = adata.obs[['pxl_row_in_fullres', 'pxl_col_in_fullres']].to_numpy()

# 其他的空间信息在adata.uns中
hires_image = plt.imread(hires_image_path)  # 高分辨率图像地址
with open(scalefactors_json_path, 'r') as f:  # 加载json文件中的缩放因子
    scalefactors = json.load(f)
adata.uns['spatial'] = {
    library_id: {
        'images': {
            'hires': hires_image
        },
        'scalefactors': scalefactors
    }
}

'''
数据预处理与质控
'''
# 过滤掉那些表达的基因数量少于200个的低质量的细胞，添加细胞的表达基因数n_genes到obs中
sc.pp.filter_cells(adata, min_genes=200)
# 过滤掉那些在少于3个细胞中表达的基因，添加基因对应的细胞数n_cells到var中
sc.pp.filter_genes(adata, min_cells=3)
# 识别所有mt开头的基因即线粒体基因，添加mt属性(bool值)到var中，但给的数据里没有线粒体基因
adata.var['mt'] = adata.var_names.str.startswith('MT-')
# 计算质量控制（QC）指标，将结果属性添加到obs和var中
sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)

# # 线粒体基因占比的可视化 - 小提琴图
# sc.pl.violin(adata, ['n_genes_by_counts', 'total_counts', 'pct_counts_mt'],jitter=0.4, multi_panel=True)
# # 线粒体基因占比的可视化 - 散点图
# sc.pl.scatter(adata, x='total_counts', y='pct_counts_mt')
# sc.pl.scatter(adata, x='total_counts', y='n_genes_by_counts')

# 过滤掉线粒体基因比例超过5%的细胞
adata = adata[adata.obs.pct_counts_mt < 5, :]
# 过滤掉组织外的数据
adata = adata[adata.obs['in_tissue'] == 1, :]
# 进行数据的总量归一化，将每个细胞的总UMI数归一化为10,000。
sc.pp.normalize_total(adata, target_sum=1e4)
# 对数据进行对数变换，使得表达数据更接近正态分布
sc.pp.log1p(adata)

# # 绘制前20个表达最高的基因
# sc.pl.highest_expr_genes(adata, n_top=20, )

'''
空间位置可视化
'''
sc.pl.spatial(adata, img_key="hires", color="total_counts", frameon=False,
              title="BC_06066 - Total UMI", cmap="viridis")
sc.pl.spatial(adata, img_key="hires", color="n_genes", frameon=False,
              title="BC_06066 - Total Gene", cmap="Spectral_r")

'''
降维、聚类
'''
# 获取前2000个高变基因
sc.pp.highly_variable_genes(adata, flavor='seurat', n_top_genes=2000)
# 过滤非高变基因(可选)
adata = adata[:, adata.var['highly_variable']]

# 获取前20个高变基因打印名称
# top_var_genes = adata.var_names[adata.var.highly_variable.values][:20]
# print(top_var_genes)
# # 高变基因可视化
# sc.pl.highly_variable_genes(adata)

# 对数据进行回归处理，移除总UMI数量 (total_counts) 和线粒体基因比例 (pct_counts_mt) 对基因表达的影响。
sc.pp.regress_out(adata, ['total_counts', 'pct_counts_mt'])
# 对每个基因的表达量进行标准化（均值为0，标准差为1），保留表达量在[-10, 10] 之间的数据。
sc.pp.scale(adata, max_value=10)

# PCA降维
sc.tl.pca(adata, n_comps=50, svd_solver='arpack')
sc.pl.pca_variance_ratio(adata, log=True, n_pcs=50) # 绘制前20个主成分的方差贡献率

# 图聚类算法leiden,并保存聚类结果到obs
sc.pp.neighbors(adata, n_neighbors=20, n_pcs=50)  # 根据pca结果构建邻居图
sc.tl.leiden(adata)
sc.pl.spatial(adata, img_key="hires", color="leiden", frameon=False)

# t-SNE降维
sc.tl.tsne(adata)
sc.pl.tsne(adata, color='leiden')

# UMAP降维
sc.tl.umap(adata)
sc.pl.umap(adata, color='leiden')

'''
差异性分析
'''
# 采用三种方法寻找25个maker基因分别是t-test、wilcoxon、logreg
sc.settings.verbosity = 2  # 设置日志输出

sc.tl.rank_genes_groups(adata, 'leiden', method='t-test')
sc.pl.rank_genes_groups(adata, n_genes=25, sharey=False)

# sc.tl.rank_genes_groups(adata, 'leiden', method='wilcoxon')
# sc.pl.rank_genes_groups(adata, n_genes=25, sharey=False)
#
# sc.tl.rank_genes_groups(adata, 'leiden', method='logreg')
# sc.pl.rank_genes_groups(adata, n_genes=25, sharey=False)

# 查看排名前几的差异基因
result = adata.uns['rank_genes_groups']
groups = result['names'].dtype.names

# 打印每个聚类的前几个差异基因
for group in groups:
    print(f"Cluster {group}:")
    for gene in result['names'][group][:5]:  # 打印前5个基因
        print(f"  {gene}")

'''
确定相应的细胞类型，并分配给簇
'''

new_cluster_names = ['IGHG2', 'PPAN', 'SPR', 'APOD', 'BGN', 'C1R', 'C3', 'CAV1', 'EGR1',
                     'FAS', 'LRP1', 'PER1']
adata.rename_categories('leiden', new_cluster_names)
sc.pl.umap(adata, color='leiden', legend_loc='on data', title='', frameon=False, save='.pdf')
