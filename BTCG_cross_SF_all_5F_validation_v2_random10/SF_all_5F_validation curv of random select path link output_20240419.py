import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D #绘制3D坐标的函数
plt.rcParams['savefig.dpi']=100
plt.rcParams['figure.dpi']=100
import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.font_manager as fm
from matplotlib.lines import Line2D
import subprocess

# 1、关于path_proportion变量输出结果 在不同验证集中的 变量值AE误差的 中位数折线图（而不是损失误差MSE×）
#
# for j in range(1,5):       # 20%/40%/60%/80%
#     for i in range(1,11):   # 随机选取20%数据，藏住信息，作为验证集，共随机10次
#         df_path_20 = pd.read_csv(r'F:\SEU\MasterThesis\02BTCG_cross_SF64\15BTCG_cross_SF_all_5F_validation_v2_random10\input_path.csv')
#         num_samples = int(len(df_path_20) * 0.2*j)
#         random_indices = np.random.choice(df_path_20.index, num_samples, replace=False)
#         df_path_20.loc[random_indices, 'is_observed'] = False
#
#         df_path_float_20 = pd.read_csv(r'F:\SEU\MasterThesis\02BTCG_cross_SF64\15BTCG_cross_SF_all_5F_validation_v2_random10\input_path_float.csv')
#         false_path_ids = df_path_20.loc[random_indices, 'path_id']
#         df_path_float_20.loc[df_path_float_20['path_id'].isin(false_path_ids), 'target_path_proportion'] = -1
#         df_path_20.to_csv(f'input_path_{20*j}_{i}.csv', index=False)
#         df_path_float_20.to_csv(f'input_path_float_{20*j}_{i}.csv', index=False)
#     # 调用另一个Python程序
# subprocess.call(['python', r'F:\SEU\MasterThesis\02BTCG_cross_SF64\15BTCG_cross_SF_all_5F_validation_v2_random10/BTCG_HFN_cross_SF64_5F.py'])
# 跑py程序时，更改input_path、input_path_float、6个output、2个png图片的名字，+后缀eg. _20_1_20_1，共改10个名字
# 接下来，分别调用多个BTCG程序


# 2、分别绘制20%、40%、60%、80%验证集的5次随机AE结果
# 2.1、绘制20%验证集，5次随机的absolute error
# (2.1) draw box plan:
df_path_20_1 = pd.read_csv(r'F:\SEU\MasterThesis\02BTCG_cross_SF64\15BTCG_cross_SF_all_5F_validation_v2_random10\output_path_20_1.csv')
df_path_20_2 = pd.read_csv(r'F:\SEU\MasterThesis\02BTCG_cross_SF64\15BTCG_cross_SF_all_5F_validation_v2_random10\output_path_20_2.csv')
df_path_20_3 = pd.read_csv(r'F:\SEU\MasterThesis\02BTCG_cross_SF64\15BTCG_cross_SF_all_5F_validation_v2_random10\output_path_20_3.csv')
df_path_20_4 = pd.read_csv(r'F:\SEU\MasterThesis\02BTCG_cross_SF64\15BTCG_cross_SF_all_5F_validation_v2_random10\output_path_20_4.csv')
df_path_20_5 = pd.read_csv(r'F:\SEU\MasterThesis\02BTCG_cross_SF64\15BTCG_cross_SF_all_5F_validation_v2_random10\output_path_20_5.csv')
df_path_20_6 = pd.read_csv(r'F:\SEU\MasterThesis\02BTCG_cross_SF64\15BTCG_cross_SF_all_5F_validation_v2_random10\output_path_20_6.csv')
df_path_20_7 = pd.read_csv(r'F:\SEU\MasterThesis\02BTCG_cross_SF64\15BTCG_cross_SF_all_5F_validation_v2_random10\output_path_20_7.csv')
df_path_20_8 = pd.read_csv(r'F:\SEU\MasterThesis\02BTCG_cross_SF64\15BTCG_cross_SF_all_5F_validation_v2_random10\output_path_20_8.csv')
df_path_20_9 = pd.read_csv(r'F:\SEU\MasterThesis\02BTCG_cross_SF64\15BTCG_cross_SF_all_5F_validation_v2_random10\output_path_20_9.csv')
df_path_20_10 = pd.read_csv(r'F:\SEU\MasterThesis\02BTCG_cross_SF64\15BTCG_cross_SF_all_5F_validation_v2_random10\output_path_20_10.csv')

estimate_path_proportion_20_1 = df_path_20_1.iloc[:,4].tolist()
target_proportion_20_1 = df_path_20_1.iloc[:,6].tolist()
loss_path_20_1 = list()
for i in range(len(target_proportion_20_1)):
    if target_proportion_20_1[i] != -1:
        loss_path_20_1.append(list(map(abs,[target_proportion_20_1[i] - estimate_path_proportion_20_1[i]])))
loss_path_20_1 = [item for sublist in loss_path_20_1 for item in sublist]   # 将二维列表转换为一维列表

estimate_path_proportion_20_2 = df_path_20_2.iloc[:,4].tolist()
target_proportion_20_2 = df_path_20_2.iloc[:,6].tolist()
loss_path_20_2 = list()
for i in range(len(target_proportion_20_2)):
    if target_proportion_20_2[i] != -1:
        loss_path_20_2.append(list(map(abs,[target_proportion_20_2[i] - estimate_path_proportion_20_2[i]])))
loss_path_20_2 = [item for sublist in loss_path_20_2 for item in sublist]   # 将二维列表转换为一维列表

estimate_path_proportion_20_3 = df_path_20_3.iloc[:,4].tolist()
target_proportion_20_3 = df_path_20_3.iloc[:,6].tolist()
loss_path_20_3 = list()
for i in range(len(target_proportion_20_3)):
    if target_proportion_20_3[i] != -1:
        loss_path_20_3.append(list(map(abs,[target_proportion_20_3[i] - estimate_path_proportion_20_3[i]])))
loss_path_20_3 = [item for sublist in loss_path_20_3 for item in sublist]   # 将二维列表转换为一维列表

estimate_path_proportion_20_4 = df_path_20_4.iloc[:,4].tolist()
target_proportion_20_4 = df_path_20_4.iloc[:,6].tolist()
loss_path_20_4 = list()
for i in range(len(target_proportion_20_4)):
    if target_proportion_20_4[i] != -1:
        loss_path_20_4.append(list(map(abs,[target_proportion_20_4[i] - estimate_path_proportion_20_4[i]])))
loss_path_20_4 = [item for sublist in loss_path_20_4 for item in sublist]   # 将二维列表转换为一维列表

estimate_path_proportion_20_5 = df_path_20_5.iloc[:,4].tolist()
target_proportion_20_5 = df_path_20_5.iloc[:,6].tolist()
loss_path_20_5 = list()
for i in range(len(target_proportion_20_5)):
    if target_proportion_20_5[i] != -1:
        loss_path_20_5.append(list(map(abs,[target_proportion_20_5[i] - estimate_path_proportion_20_5[i]])))
loss_path_20_5 = [item for sublist in loss_path_20_5 for item in sublist]   # 将二维列表转换为一维列表

estimate_path_proportion_20_6 = df_path_20_6.iloc[:,4].tolist()
target_proportion_20_6 = df_path_20_6.iloc[:,6].tolist()
loss_path_20_6 = list()
for i in range(len(target_proportion_20_6)):
    if target_proportion_20_6[i] != -1:
        loss_path_20_6.append(list(map(abs,[target_proportion_20_6[i] - estimate_path_proportion_20_6[i]])))
loss_path_20_6 = [item for sublist in loss_path_20_6 for item in sublist]   # 将二维列表转换为一维列表

estimate_path_proportion_20_7 = df_path_20_7.iloc[:,4].tolist()
target_proportion_20_7 = df_path_20_7.iloc[:,6].tolist()
loss_path_20_7 = list()
for i in range(len(target_proportion_20_7)):
    if target_proportion_20_7[i] != -1:
        loss_path_20_7.append(list(map(abs,[target_proportion_20_7[i] - estimate_path_proportion_20_7[i]])))
loss_path_20_7 = [item for sublist in loss_path_20_7 for item in sublist]   # 将二维列表转换为一维列表

estimate_path_proportion_20_8 = df_path_20_8.iloc[:,4].tolist()
target_proportion_20_8 = df_path_20_8.iloc[:,6].tolist()
loss_path_20_8 = list()
for i in range(len(target_proportion_20_8)):
    if target_proportion_20_8[i] != -1:
        loss_path_20_8.append(list(map(abs,[target_proportion_20_8[i] - estimate_path_proportion_20_8[i]])))
loss_path_20_8 = [item for sublist in loss_path_20_8 for item in sublist]   # 将二维列表转换为一维列表

estimate_path_proportion_20_9 = df_path_20_9.iloc[:,4].tolist()
target_proportion_20_9 = df_path_20_9.iloc[:,6].tolist()
loss_path_20_9 = list()
for i in range(len(target_proportion_20_9)):
    if target_proportion_20_9[i] != -1:
        loss_path_20_9.append(list(map(abs,[target_proportion_20_9[i] - estimate_path_proportion_20_9[i]])))
loss_path_20_9 = [item for sublist in loss_path_20_9 for item in sublist]   # 将二维列表转换为一维列表

estimate_path_proportion_20_10 = df_path_20_10.iloc[:,4].tolist()
target_proportion_20_10 = df_path_20_10.iloc[:,6].tolist()
loss_path_20_10 = list()
for i in range(len(target_proportion_20_10)):
    if target_proportion_20_10[i] != -1:
        loss_path_20_10.append(list(map(abs,[target_proportion_20_10[i] - estimate_path_proportion_20_10[i]])))
loss_path_20_10 = [item for sublist in loss_path_20_10 for item in sublist]   # 将二维列表转换为一维列表

loss_path_20 = (loss_path_20_1, loss_path_20_2, loss_path_20_3, loss_path_20_4, loss_path_20_5, loss_path_20_6, loss_path_20_7, loss_path_20_8, loss_path_20_9, loss_path_20_10)

box_path = plt.boxplot(loss_path_20, patch_artist = True,
                         labels = ['sample 1', 'sample 2', 'sample 3', 'sample 4', 'sample 5', 'sample 6', 'sample 7', 'sample 8', 'sample 9', 'sample 10'],
                        flierprops={'marker': 'o', 'markerfacecolor': 'w', 'markeredgecolor': 'b', 'alpha': 0.5},   # 异常值设置为蓝色
                         medianprops = {'linewidth': 1.5, 'color': 'b'})  # 一定要有patch_artist = True，否则不识别facecolor这个关键词

color = 'b'
for box in box_path['boxes']:
    box.set(facecolor='w', edgecolor='b', linestyle='dashed')

for whisker in box_path['whiskers']:
    whisker.set(color='b', linewidth=1)

for cap in box_path['caps']:
    cap.set(color='b', linewidth=1)

for median in box_path['medians']:
    median.set(color='b', linewidth=1.5)

# （2.2）将5个箱型图的中位线处连接成折线图：
median_loss_path_20_1 = np.median(loss_path_20_1)
median_loss_path_20_2 = np.median(loss_path_20_2)
median_loss_path_20_3 = np.median(loss_path_20_3)
median_loss_path_20_4 = np.median(loss_path_20_4)
median_loss_path_20_5 = np.median(loss_path_20_5)
median_loss_path_20_6 = np.median(loss_path_20_6)
median_loss_path_20_7 = np.median(loss_path_20_7)
median_loss_path_20_8 = np.median(loss_path_20_8)
median_loss_path_20_9 = np.median(loss_path_20_9)
median_loss_path_20_10 = np.median(loss_path_20_10)

median_loss_path_20 = np.array([median_loss_path_20_1, median_loss_path_20_2, median_loss_path_20_3, median_loss_path_20_4, median_loss_path_20_5, median_loss_path_20_6, median_loss_path_20_7, median_loss_path_20_8, median_loss_path_20_9, median_loss_path_20_10])

plt.plot(range(1, len(median_loss_path_20) + 1), median_loss_path_20, marker='o', markersize=10, linewidth=3.0, color='dodgerblue')  # 绘制中位数折线图

plt.text(1.0, median_loss_path_20_1+0.002, 'Median: {:.5f}'.format(median_loss_path_20_1), fontproperties='Times New Roman', size=18, ha='center', va='bottom')     # 将箱型图的中位数标记出来
plt.text(2.0, median_loss_path_20_2-0.006, ' {:.5f}'.format(median_loss_path_20_2), fontproperties='Times New Roman', size=18, ha='center', va='bottom')     # 将箱型图的中位数标记出来
plt.text(3.0, median_loss_path_20_3+0.002, ' {:.5f}'.format(median_loss_path_20_3), fontproperties='Times New Roman', size=18, ha='center', va='bottom')     # 将箱型图的中位数标记出来
plt.text(4.0, median_loss_path_20_4-0.006, ' {:.5f}'.format(median_loss_path_20_4), fontproperties='Times New Roman', size=18, ha='center', va='bottom')     # 将箱型图的中位数标记出来
plt.text(5.0, median_loss_path_20_5+0.002, ' {:.5f}'.format(median_loss_path_20_5), fontproperties='Times New Roman', size=18, ha='center', va='bottom')     # 将箱型图的中位数标记出来
plt.text(6.0, median_loss_path_20_6-0.006, ' {:.5f}'.format(median_loss_path_20_6), fontproperties='Times New Roman', size=18, ha='center', va='bottom')     # 将箱型图的中位数标记出来
plt.text(7.0, median_loss_path_20_7+0.002, ' {:.5f}'.format(median_loss_path_20_7), fontproperties='Times New Roman', size=18, ha='center', va='bottom')     # 将箱型图的中位数标记出来
plt.text(8.0, median_loss_path_20_8-0.006, ' {:.5f}'.format(median_loss_path_20_8), fontproperties='Times New Roman', size=18, ha='center', va='bottom')     # 将箱型图的中位数标记出来
plt.text(9.0, median_loss_path_20_9+0.002, ' {:.5f}'.format(median_loss_path_20_9), fontproperties='Times New Roman', size=18, ha='center', va='bottom')     # 将箱型图的中位数标记出来
plt.text(10.0, median_loss_path_20_10-0.006, ' {:.5f}'.format(median_loss_path_20_10), fontproperties='Times New Roman', size=18, ha='center', va='bottom')     # 将箱型图的中位数标记出来

fig = plt.gcf()
fig.set_size_inches(8, 6)  # 设置保存图像的显示大小 宽度为 10英寸，高度为 8英寸
plt.xlabel('(a) 20% OBD validation set', labelpad=7.5, fontdict={'family': 'Times New Roman', 'size': 18, 'color': 'k'})  # 改变横坐标轴标题字体  labelpad=10表示坐标标题距离坐标轴的垂直距离
plt.ylabel('Absolute Error of path selection rate', labelpad=7.5, fontdict={'family': 'Times New Roman', 'size': 18, 'color': 'k'})  # 改变纵坐标轴标题字体
plt.xticks(fontproperties='Times New Roman', size=18, rotation=20)
plt.yticks(fontproperties='Times New Roman', size=18)
handles = [Line2D([0], [0], color='dodgerblue', marker='o', lw=2)]
labels = ['Median']   # 自定义图例标签
plt.legend(handles=handles, labels=labels, loc='upper left', prop={'family': 'Times New Roman', 'size': 18})   # 添加图例
plt.subplots_adjust(right=0.95, bottom=0.2, hspace=0.3, wspace=10.5)
# plt.subplots_adjust(left=0.125, right=0.9, bottom=0.1, top=0.9, wspace=0.2, hspace=0.2)  # 默认值
plt.savefig('20% path validation set box.png', dpi=300, format='png')
plt.show()

# 2.2、绘制40%验证集，5次随机的absolute error
# (2.1) draw box plan:
df_path_40_1 = pd.read_csv(r'F:\SEU\MasterThesis\02BTCG_cross_SF64\15BTCG_cross_SF_all_5F_validation_v2_random10\output_path_40_1.csv')
df_path_40_2 = pd.read_csv(r'F:\SEU\MasterThesis\02BTCG_cross_SF64\15BTCG_cross_SF_all_5F_validation_v2_random10\output_path_40_2.csv')
df_path_40_3 = pd.read_csv(r'F:\SEU\MasterThesis\02BTCG_cross_SF64\15BTCG_cross_SF_all_5F_validation_v2_random10\output_path_40_3.csv')
df_path_40_4 = pd.read_csv(r'F:\SEU\MasterThesis\02BTCG_cross_SF64\15BTCG_cross_SF_all_5F_validation_v2_random10\output_path_40_4.csv')
df_path_40_5 = pd.read_csv(r'F:\SEU\MasterThesis\02BTCG_cross_SF64\15BTCG_cross_SF_all_5F_validation_v2_random10\output_path_40_5.csv')
df_path_40_6 = pd.read_csv(r'F:\SEU\MasterThesis\02BTCG_cross_SF64\15BTCG_cross_SF_all_5F_validation_v2_random10\output_path_40_6.csv')
df_path_40_7 = pd.read_csv(r'F:\SEU\MasterThesis\02BTCG_cross_SF64\15BTCG_cross_SF_all_5F_validation_v2_random10\output_path_40_7.csv')
df_path_40_8 = pd.read_csv(r'F:\SEU\MasterThesis\02BTCG_cross_SF64\15BTCG_cross_SF_all_5F_validation_v2_random10\output_path_40_8.csv')
df_path_40_9 = pd.read_csv(r'F:\SEU\MasterThesis\02BTCG_cross_SF64\15BTCG_cross_SF_all_5F_validation_v2_random10\output_path_40_9.csv')
df_path_40_10 = pd.read_csv(r'F:\SEU\MasterThesis\02BTCG_cross_SF64\15BTCG_cross_SF_all_5F_validation_v2_random10\output_path_40_10.csv')

estimate_path_proportion_40_1 = df_path_40_1.iloc[:,4].tolist()
target_proportion_40_1 = df_path_40_1.iloc[:,6].tolist()
loss_path_40_1 = list()
for i in range(len(target_proportion_40_1)):
    if target_proportion_40_1[i] != -1:
        loss_path_40_1.append(list(map(abs,[target_proportion_40_1[i] - estimate_path_proportion_40_1[i]])))
loss_path_40_1 = [item for sublist in loss_path_40_1 for item in sublist]   # 将二维列表转换为一维列表

estimate_path_proportion_40_2 = df_path_40_2.iloc[:,4].tolist()
target_proportion_40_2 = df_path_40_2.iloc[:,6].tolist()
loss_path_40_2 = list()
for i in range(len(target_proportion_40_2)):
    if target_proportion_40_2[i] != -1:
        loss_path_40_2.append(list(map(abs,[target_proportion_40_2[i] - estimate_path_proportion_40_2[i]])))
loss_path_40_2 = [item for sublist in loss_path_40_2 for item in sublist]   # 将二维列表转换为一维列表

estimate_path_proportion_40_3 = df_path_40_3.iloc[:,4].tolist()
target_proportion_40_3 = df_path_40_3.iloc[:,6].tolist()
loss_path_40_3 = list()
for i in range(len(target_proportion_40_3)):
    if target_proportion_40_3[i] != -1:
        loss_path_40_3.append(list(map(abs,[target_proportion_40_3[i] - estimate_path_proportion_40_3[i]])))
loss_path_40_3 = [item for sublist in loss_path_40_3 for item in sublist]   # 将二维列表转换为一维列表

estimate_path_proportion_40_4 = df_path_40_4.iloc[:,4].tolist()
target_proportion_40_4 = df_path_40_4.iloc[:,6].tolist()
loss_path_40_4 = list()
for i in range(len(target_proportion_40_4)):
    if target_proportion_40_4[i] != -1:
        loss_path_40_4.append(list(map(abs,[target_proportion_40_4[i] - estimate_path_proportion_40_4[i]])))
loss_path_40_4 = [item for sublist in loss_path_40_4 for item in sublist]   # 将二维列表转换为一维列表

estimate_path_proportion_40_5 = df_path_40_5.iloc[:,4].tolist()
target_proportion_40_5 = df_path_40_5.iloc[:,6].tolist()
loss_path_40_5 = list()
for i in range(len(target_proportion_40_5)):
    if target_proportion_40_5[i] != -1:
        loss_path_40_5.append(list(map(abs,[target_proportion_40_5[i] - estimate_path_proportion_40_5[i]])))
loss_path_40_5 = [item for sublist in loss_path_40_5 for item in sublist]   # 将二维列表转换为一维列表

estimate_path_proportion_40_6 = df_path_40_6.iloc[:,4].tolist()
target_proportion_40_6 = df_path_40_6.iloc[:,6].tolist()
loss_path_40_6 = list()
for i in range(len(target_proportion_40_6)):
    if target_proportion_40_6[i] != -1:
        loss_path_40_6.append(list(map(abs,[target_proportion_40_6[i] - estimate_path_proportion_40_6[i]])))
loss_path_40_6 = [item for sublist in loss_path_40_6 for item in sublist]   # 将二维列表转换为一维列表

estimate_path_proportion_40_7 = df_path_40_7.iloc[:,4].tolist()
target_proportion_40_7 = df_path_40_7.iloc[:,6].tolist()
loss_path_40_7 = list()
for i in range(len(target_proportion_40_7)):
    if target_proportion_40_7[i] != -1:
        loss_path_40_7.append(list(map(abs,[target_proportion_40_7[i] - estimate_path_proportion_40_7[i]])))
loss_path_40_7 = [item for sublist in loss_path_40_7 for item in sublist]   # 将二维列表转换为一维列表

estimate_path_proportion_40_8 = df_path_40_8.iloc[:,4].tolist()
target_proportion_40_8 = df_path_40_8.iloc[:,6].tolist()
loss_path_40_8 = list()
for i in range(len(target_proportion_40_8)):
    if target_proportion_40_8[i] != -1:
        loss_path_40_8.append(list(map(abs,[target_proportion_40_8[i] - estimate_path_proportion_40_8[i]])))
loss_path_40_8 = [item for sublist in loss_path_40_8 for item in sublist]   # 将二维列表转换为一维列表

estimate_path_proportion_40_9 = df_path_40_9.iloc[:,4].tolist()
target_proportion_40_9 = df_path_40_9.iloc[:,6].tolist()
loss_path_40_9 = list()
for i in range(len(target_proportion_40_9)):
    if target_proportion_40_9[i] != -1:
        loss_path_40_9.append(list(map(abs,[target_proportion_40_9[i] - estimate_path_proportion_40_9[i]])))
loss_path_40_9 = [item for sublist in loss_path_40_9 for item in sublist]   # 将二维列表转换为一维列表

estimate_path_proportion_40_10 = df_path_40_10.iloc[:,4].tolist()
target_proportion_40_10 = df_path_40_10.iloc[:,6].tolist()
loss_path_40_10 = list()
for i in range(len(target_proportion_40_10)):
    if target_proportion_40_10[i] != -1:
        loss_path_40_10.append(list(map(abs,[target_proportion_40_10[i] - estimate_path_proportion_40_10[i]])))
loss_path_40_10 = [item for sublist in loss_path_40_10 for item in sublist]   # 将二维列表转换为一维列表

loss_path_40 = (loss_path_40_1, loss_path_40_2, loss_path_40_3, loss_path_40_4, loss_path_40_5, loss_path_40_6, loss_path_40_7, loss_path_40_8, loss_path_40_9, loss_path_40_10)

box_path = plt.boxplot(loss_path_40, patch_artist = True,
                         labels = ['sample 1', 'sample 2', 'sample 3', 'sample 4', 'sample 5', 'sample 6', 'sample 7', 'sample 8', 'sample 9', 'sample 10'],
                        flierprops={'marker': 'o', 'markerfacecolor': 'w', 'markeredgecolor': 'b', 'alpha': 0.5},   # 异常值设置为蓝色
                         medianprops = {'linewidth': 1.5, 'color': 'b'})  # 一定要有patch_artist = True，否则不识别facecolor这个关键词

color = 'b'
for box in box_path['boxes']:
    box.set(facecolor='w', edgecolor='b', linestyle='dashed')

for whisker in box_path['whiskers']:
    whisker.set(color='b', linewidth=1)

for cap in box_path['caps']:
    cap.set(color='b', linewidth=1)

for median in box_path['medians']:
    median.set(color='b', linewidth=1.5)

# （2.2）将5个箱型图的中位线处连接成折线图：
median_loss_path_40_1 = np.median(loss_path_40_1)
median_loss_path_40_2 = np.median(loss_path_40_2)
median_loss_path_40_3 = np.median(loss_path_40_3)
median_loss_path_40_4 = np.median(loss_path_40_4)
median_loss_path_40_5 = np.median(loss_path_40_5)
median_loss_path_40_6 = np.median(loss_path_40_6)
median_loss_path_40_7 = np.median(loss_path_40_7)
median_loss_path_40_8 = np.median(loss_path_40_8)
median_loss_path_40_9 = np.median(loss_path_40_9)
median_loss_path_40_10 = np.median(loss_path_40_10)

median_loss_path_40 = np.array([median_loss_path_40_1, median_loss_path_40_2, median_loss_path_40_3, median_loss_path_40_4, median_loss_path_40_5, median_loss_path_40_6, median_loss_path_40_7, median_loss_path_40_8, median_loss_path_40_9, median_loss_path_40_10])

plt.plot(range(1, len(median_loss_path_40) + 1), median_loss_path_40, marker='o', markersize=10, linewidth=3.0, color='dodgerblue')  # 绘制中位数折线图

plt.text(1.0, median_loss_path_40_1+0.002, 'Median: {:.5f}'.format(median_loss_path_40_1), fontproperties='Times New Roman', size=18, ha='center', va='bottom')     # 将箱型图的中位数标记出来
plt.text(2.0, median_loss_path_40_2-0.006, ' {:.5f}'.format(median_loss_path_40_2), fontproperties='Times New Roman', size=18, ha='center', va='bottom')     # 将箱型图的中位数标记出来
plt.text(3.0, median_loss_path_40_3+0.002, ' {:.5f}'.format(median_loss_path_40_3), fontproperties='Times New Roman', size=18, ha='center', va='bottom')     # 将箱型图的中位数标记出来
plt.text(4.0, median_loss_path_40_4-0.006, ' {:.5f}'.format(median_loss_path_40_4), fontproperties='Times New Roman', size=18, ha='center', va='bottom')     # 将箱型图的中位数标记出来
plt.text(5.0, median_loss_path_40_5+0.002, ' {:.5f}'.format(median_loss_path_40_5), fontproperties='Times New Roman', size=18, ha='center', va='bottom')     # 将箱型图的中位数标记出来
plt.text(6.0, median_loss_path_40_6-0.006, ' {:.5f}'.format(median_loss_path_40_6), fontproperties='Times New Roman', size=18, ha='center', va='bottom')     # 将箱型图的中位数标记出来
plt.text(7.0, median_loss_path_40_7+0.002, ' {:.5f}'.format(median_loss_path_40_7), fontproperties='Times New Roman', size=18, ha='center', va='bottom')     # 将箱型图的中位数标记出来
plt.text(8.0, median_loss_path_40_8-0.006, ' {:.5f}'.format(median_loss_path_40_8), fontproperties='Times New Roman', size=18, ha='center', va='bottom')     # 将箱型图的中位数标记出来
plt.text(9.0, median_loss_path_40_9+0.002, ' {:.5f}'.format(median_loss_path_40_9), fontproperties='Times New Roman', size=18, ha='center', va='bottom')     # 将箱型图的中位数标记出来
plt.text(10.0, median_loss_path_40_10-0.006, ' {:.5f}'.format(median_loss_path_40_10), fontproperties='Times New Roman', size=18, ha='center', va='bottom')     # 将箱型图的中位数标记出来

fig = plt.gcf()
fig.set_size_inches(8, 6)  # 设置保存图像的显示大小 宽度为 10英寸，高度为 8英寸
plt.xlabel('(b) 40% OBD validation set', labelpad=7.5, fontdict={'family': 'Times New Roman', 'size': 18, 'color': 'k'})  # 改变横坐标轴标题字体  labelpad=10表示坐标标题距离坐标轴的垂直距离
plt.ylabel('Absolute Error of path selection rate', labelpad=7.5, fontdict={'family': 'Times New Roman', 'size': 18, 'color': 'k'})  # 改变纵坐标轴标题字体
plt.xticks(fontproperties='Times New Roman', size=18, rotation=20)
plt.yticks(fontproperties='Times New Roman', size=18)
handles = [Line2D([0], [0], color='dodgerblue', marker='o', lw=2)]
labels = ['Median']   # 自定义图例标签
plt.legend(handles=handles, labels=labels, loc='upper left', prop={'family': 'Times New Roman', 'size': 18})   # 添加图例
plt.subplots_adjust(right=0.95, bottom=0.2, hspace=0.3, wspace=10.5)
plt.savefig('40% path validation set box.png', dpi=300, format='png')
plt.show()

# 2.3、绘制60%验证集，5次随机的absolute error
# (2.1) draw box plan:
df_path_60_1 = pd.read_csv(r'F:\SEU\MasterThesis\02BTCG_cross_SF64\15BTCG_cross_SF_all_5F_validation_v2_random10\output_path_60_1.csv')
df_path_60_2 = pd.read_csv(r'F:\SEU\MasterThesis\02BTCG_cross_SF64\15BTCG_cross_SF_all_5F_validation_v2_random10\output_path_60_2.csv')
df_path_60_3 = pd.read_csv(r'F:\SEU\MasterThesis\02BTCG_cross_SF64\15BTCG_cross_SF_all_5F_validation_v2_random10\output_path_60_3.csv')
df_path_60_4 = pd.read_csv(r'F:\SEU\MasterThesis\02BTCG_cross_SF64\15BTCG_cross_SF_all_5F_validation_v2_random10\output_path_60_4.csv')
df_path_60_5 = pd.read_csv(r'F:\SEU\MasterThesis\02BTCG_cross_SF64\15BTCG_cross_SF_all_5F_validation_v2_random10\output_path_60_5.csv')
df_path_60_6 = pd.read_csv(r'F:\SEU\MasterThesis\02BTCG_cross_SF64\15BTCG_cross_SF_all_5F_validation_v2_random10\output_path_60_6.csv')
df_path_60_7 = pd.read_csv(r'F:\SEU\MasterThesis\02BTCG_cross_SF64\15BTCG_cross_SF_all_5F_validation_v2_random10\output_path_60_7.csv')
df_path_60_8 = pd.read_csv(r'F:\SEU\MasterThesis\02BTCG_cross_SF64\15BTCG_cross_SF_all_5F_validation_v2_random10\output_path_60_8.csv')
df_path_60_9 = pd.read_csv(r'F:\SEU\MasterThesis\02BTCG_cross_SF64\15BTCG_cross_SF_all_5F_validation_v2_random10\output_path_60_9.csv')
df_path_60_10 = pd.read_csv(r'F:\SEU\MasterThesis\02BTCG_cross_SF64\15BTCG_cross_SF_all_5F_validation_v2_random10\output_path_60_10.csv')

estimate_path_proportion_60_1 = df_path_60_1.iloc[:,4].tolist()
target_proportion_60_1 = df_path_60_1.iloc[:,6].tolist()
loss_path_60_1 = list()
for i in range(len(target_proportion_60_1)):
    if target_proportion_60_1[i] != -1:
        loss_path_60_1.append(list(map(abs,[target_proportion_60_1[i] - estimate_path_proportion_60_1[i]])))
loss_path_60_1 = [item for sublist in loss_path_60_1 for item in sublist]   # 将二维列表转换为一维列表

estimate_path_proportion_60_2 = df_path_60_2.iloc[:,4].tolist()
target_proportion_60_2 = df_path_60_2.iloc[:,6].tolist()
loss_path_60_2 = list()
for i in range(len(target_proportion_60_2)):
    if target_proportion_60_2[i] != -1:
        loss_path_60_2.append(list(map(abs,[target_proportion_60_2[i] - estimate_path_proportion_60_2[i]])))
loss_path_60_2 = [item for sublist in loss_path_60_2 for item in sublist]   # 将二维列表转换为一维列表

estimate_path_proportion_60_3 = df_path_60_3.iloc[:,4].tolist()
target_proportion_60_3 = df_path_60_3.iloc[:,6].tolist()
loss_path_60_3 = list()
for i in range(len(target_proportion_60_3)):
    if target_proportion_60_3[i] != -1:
        loss_path_60_3.append(list(map(abs,[target_proportion_60_3[i] - estimate_path_proportion_60_3[i]])))
loss_path_60_3 = [item for sublist in loss_path_60_3 for item in sublist]   # 将二维列表转换为一维列表

estimate_path_proportion_60_4 = df_path_60_4.iloc[:,4].tolist()
target_proportion_60_4 = df_path_60_4.iloc[:,6].tolist()
loss_path_60_4 = list()
for i in range(len(target_proportion_60_4)):
    if target_proportion_60_4[i] != -1:
        loss_path_60_4.append(list(map(abs,[target_proportion_60_4[i] - estimate_path_proportion_60_4[i]])))
loss_path_60_4 = [item for sublist in loss_path_60_4 for item in sublist]   # 将二维列表转换为一维列表

estimate_path_proportion_60_5 = df_path_60_5.iloc[:,4].tolist()
target_proportion_60_5 = df_path_60_5.iloc[:,6].tolist()
loss_path_60_5 = list()
for i in range(len(target_proportion_60_5)):
    if target_proportion_60_5[i] != -1:
        loss_path_60_5.append(list(map(abs,[target_proportion_60_5[i] - estimate_path_proportion_60_5[i]])))
loss_path_60_5 = [item for sublist in loss_path_60_5 for item in sublist]   # 将二维列表转换为一维列表

estimate_path_proportion_60_6 = df_path_60_6.iloc[:,4].tolist()
target_proportion_60_6 = df_path_60_6.iloc[:,6].tolist()
loss_path_60_6 = list()
for i in range(len(target_proportion_60_6)):
    if target_proportion_60_6[i] != -1:
        loss_path_60_6.append(list(map(abs,[target_proportion_60_6[i] - estimate_path_proportion_60_6[i]])))
loss_path_60_6 = [item for sublist in loss_path_60_6 for item in sublist]   # 将二维列表转换为一维列表

estimate_path_proportion_60_7 = df_path_60_7.iloc[:,4].tolist()
target_proportion_60_7 = df_path_60_7.iloc[:,6].tolist()
loss_path_60_7 = list()
for i in range(len(target_proportion_60_7)):
    if target_proportion_60_7[i] != -1:
        loss_path_60_7.append(list(map(abs,[target_proportion_60_7[i] - estimate_path_proportion_60_7[i]])))
loss_path_60_7 = [item for sublist in loss_path_60_7 for item in sublist]   # 将二维列表转换为一维列表

estimate_path_proportion_60_8 = df_path_60_8.iloc[:,4].tolist()
target_proportion_60_8 = df_path_60_8.iloc[:,6].tolist()
loss_path_60_8 = list()
for i in range(len(target_proportion_60_8)):
    if target_proportion_60_8[i] != -1:
        loss_path_60_8.append(list(map(abs,[target_proportion_60_8[i] - estimate_path_proportion_60_8[i]])))
loss_path_60_8 = [item for sublist in loss_path_60_8 for item in sublist]   # 将二维列表转换为一维列表

estimate_path_proportion_60_9 = df_path_60_9.iloc[:,4].tolist()
target_proportion_60_9 = df_path_60_9.iloc[:,6].tolist()
loss_path_60_9 = list()
for i in range(len(target_proportion_60_9)):
    if target_proportion_60_9[i] != -1:
        loss_path_60_9.append(list(map(abs,[target_proportion_60_9[i] - estimate_path_proportion_60_9[i]])))
loss_path_60_9 = [item for sublist in loss_path_60_9 for item in sublist]   # 将二维列表转换为一维列表

estimate_path_proportion_60_10 = df_path_60_10.iloc[:,4].tolist()
target_proportion_60_10 = df_path_60_10.iloc[:,6].tolist()
loss_path_60_10 = list()
for i in range(len(target_proportion_60_10)):
    if target_proportion_60_10[i] != -1:
        loss_path_60_10.append(list(map(abs,[target_proportion_60_10[i] - estimate_path_proportion_60_10[i]])))
loss_path_60_10 = [item for sublist in loss_path_60_10 for item in sublist]   # 将二维列表转换为一维列表

loss_path_60 = (loss_path_60_1, loss_path_60_2, loss_path_60_3, loss_path_60_4, loss_path_60_5, loss_path_60_6, loss_path_60_7, loss_path_60_8, loss_path_60_9, loss_path_60_10)

box_path = plt.boxplot(loss_path_60, patch_artist = True,
                         labels = ['sample 1', 'sample 2', 'sample 3', 'sample 4', 'sample 5', 'sample 6', 'sample 7', 'sample 8', 'sample 9', 'sample 10'],
                        flierprops={'marker': 'o', 'markerfacecolor': 'w', 'markeredgecolor': 'b', 'alpha': 0.5},   # 异常值设置为蓝色
                         medianprops = {'linewidth': 1.5, 'color': 'b'})  # 一定要有patch_artist = True，否则不识别facecolor这个关键词

color = 'b'
for box in box_path['boxes']:
    box.set(facecolor='w', edgecolor='b', linestyle='dashed')

for whisker in box_path['whiskers']:
    whisker.set(color='b', linewidth=1)

for cap in box_path['caps']:
    cap.set(color='b', linewidth=1)

for median in box_path['medians']:
    median.set(color='b', linewidth=1.5)

# （2.2）将5个箱型图的中位线处连接成折线图：
median_loss_path_60_1 = np.median(loss_path_60_1)
median_loss_path_60_2 = np.median(loss_path_60_2)
median_loss_path_60_3 = np.median(loss_path_60_3)
median_loss_path_60_4 = np.median(loss_path_60_4)
median_loss_path_60_5 = np.median(loss_path_60_5)
median_loss_path_60_6 = np.median(loss_path_60_6)
median_loss_path_60_7 = np.median(loss_path_60_7)
median_loss_path_60_8 = np.median(loss_path_60_8)
median_loss_path_60_9 = np.median(loss_path_60_9)
median_loss_path_60_10 = np.median(loss_path_60_10)

median_loss_path_60 = np.array([median_loss_path_60_1, median_loss_path_60_2, median_loss_path_60_3, median_loss_path_60_4, median_loss_path_60_5, median_loss_path_60_6, median_loss_path_60_7, median_loss_path_60_8, median_loss_path_60_9, median_loss_path_60_10])

plt.plot(range(1, len(median_loss_path_60) + 1), median_loss_path_60, marker='o', markersize=10, linewidth=3.0, color='dodgerblue')  # 绘制中位数折线图

plt.text(1.0, median_loss_path_60_1+0.002, 'Median: {:.5f}'.format(median_loss_path_60_1), fontproperties='Times New Roman', size=18, ha='center', va='bottom')     # 将箱型图的中位数标记出来
plt.text(2.0, median_loss_path_60_2-0.006, ' {:.5f}'.format(median_loss_path_60_2), fontproperties='Times New Roman', size=18, ha='center', va='bottom')     # 将箱型图的中位数标记出来
plt.text(3.0, median_loss_path_60_3+0.002, ' {:.5f}'.format(median_loss_path_60_3), fontproperties='Times New Roman', size=18, ha='center', va='bottom')     # 将箱型图的中位数标记出来
plt.text(4.0, median_loss_path_60_4-0.006, ' {:.5f}'.format(median_loss_path_60_4), fontproperties='Times New Roman', size=18, ha='center', va='bottom')     # 将箱型图的中位数标记出来
plt.text(5.0, median_loss_path_60_5+0.002, ' {:.5f}'.format(median_loss_path_60_5), fontproperties='Times New Roman', size=18, ha='center', va='bottom')     # 将箱型图的中位数标记出来
plt.text(6.0, median_loss_path_60_6-0.006, ' {:.5f}'.format(median_loss_path_60_6), fontproperties='Times New Roman', size=18, ha='center', va='bottom')     # 将箱型图的中位数标记出来
plt.text(7.0, median_loss_path_60_7+0.002, ' {:.5f}'.format(median_loss_path_60_7), fontproperties='Times New Roman', size=18, ha='center', va='bottom')     # 将箱型图的中位数标记出来
plt.text(8.0, median_loss_path_60_8-0.006, ' {:.5f}'.format(median_loss_path_60_8), fontproperties='Times New Roman', size=18, ha='center', va='bottom')     # 将箱型图的中位数标记出来
plt.text(9.0, median_loss_path_60_9+0.002, ' {:.5f}'.format(median_loss_path_60_9), fontproperties='Times New Roman', size=18, ha='center', va='bottom')     # 将箱型图的中位数标记出来
plt.text(10.0, median_loss_path_60_10-0.006, ' {:.5f}'.format(median_loss_path_60_10), fontproperties='Times New Roman', size=18, ha='center', va='bottom')     # 将箱型图的中位数标记出来

fig = plt.gcf()
fig.set_size_inches(8, 6)  # 设置保存图像的显示大小 宽度为 10英寸，高度为 8英寸
plt.xlabel('(c) 60% OBD validation set', labelpad=7.5, fontdict={'family': 'Times New Roman', 'size': 18, 'color': 'k'})  # 改变横坐标轴标题字体  labelpad=10表示坐标标题距离坐标轴的垂直距离
plt.ylabel('Absolute Error of path selection rate', labelpad=7.5, fontdict={'family': 'Times New Roman', 'size': 18, 'color': 'k'})  # 改变纵坐标轴标题字体
plt.xticks(fontproperties='Times New Roman', size=18, rotation=20)
plt.yticks(fontproperties='Times New Roman', size=18)
handles = [Line2D([0], [0], color='dodgerblue', marker='o', lw=2)]
labels = ['Median']   # 自定义图例标签
plt.legend(handles=handles, labels=labels, loc='upper left', prop={'family': 'Times New Roman', 'size': 18})   # 添加图例
plt.subplots_adjust(right=0.95, bottom=0.2, hspace=0.3, wspace=10.5)
plt.savefig('60% path validation set box.png', dpi=300, format='png')
plt.show()

# 2.4、绘制80%验证集，5次随机的absolute error
# (2.1) draw box plan:
df_path_80_1 = pd.read_csv(r'F:\SEU\MasterThesis\02BTCG_cross_SF64\15BTCG_cross_SF_all_5F_validation_v2_random10\output_path_80_1.csv')
df_path_80_2 = pd.read_csv(r'F:\SEU\MasterThesis\02BTCG_cross_SF64\15BTCG_cross_SF_all_5F_validation_v2_random10\output_path_80_2.csv')
df_path_80_3 = pd.read_csv(r'F:\SEU\MasterThesis\02BTCG_cross_SF64\15BTCG_cross_SF_all_5F_validation_v2_random10\output_path_80_3.csv')
df_path_80_4 = pd.read_csv(r'F:\SEU\MasterThesis\02BTCG_cross_SF64\15BTCG_cross_SF_all_5F_validation_v2_random10\output_path_80_4.csv')
df_path_80_5 = pd.read_csv(r'F:\SEU\MasterThesis\02BTCG_cross_SF64\15BTCG_cross_SF_all_5F_validation_v2_random10\output_path_80_5.csv')
df_path_80_6 = pd.read_csv(r'F:\SEU\MasterThesis\02BTCG_cross_SF64\15BTCG_cross_SF_all_5F_validation_v2_random10\output_path_80_6.csv')
df_path_80_7 = pd.read_csv(r'F:\SEU\MasterThesis\02BTCG_cross_SF64\15BTCG_cross_SF_all_5F_validation_v2_random10\output_path_80_7.csv')
df_path_80_8 = pd.read_csv(r'F:\SEU\MasterThesis\02BTCG_cross_SF64\15BTCG_cross_SF_all_5F_validation_v2_random10\output_path_80_8.csv')
df_path_80_9 = pd.read_csv(r'F:\SEU\MasterThesis\02BTCG_cross_SF64\15BTCG_cross_SF_all_5F_validation_v2_random10\output_path_80_9.csv')
df_path_80_10 = pd.read_csv(r'F:\SEU\MasterThesis\02BTCG_cross_SF64\15BTCG_cross_SF_all_5F_validation_v2_random10\output_path_80_10.csv')

estimate_path_proportion_80_1 = df_path_80_1.iloc[:,4].tolist()
target_proportion_80_1 = df_path_80_1.iloc[:,6].tolist()
loss_path_80_1 = list()
for i in range(len(target_proportion_80_1)):
    if target_proportion_80_1[i] != -1:
        loss_path_80_1.append(list(map(abs,[target_proportion_80_1[i] - estimate_path_proportion_80_1[i]])))
loss_path_80_1 = [item for sublist in loss_path_80_1 for item in sublist]   # 将二维列表转换为一维列表

estimate_path_proportion_80_2 = df_path_80_2.iloc[:,4].tolist()
target_proportion_80_2 = df_path_80_2.iloc[:,6].tolist()
loss_path_80_2 = list()
for i in range(len(target_proportion_80_2)):
    if target_proportion_80_2[i] != -1:
        loss_path_80_2.append(list(map(abs,[target_proportion_80_2[i] - estimate_path_proportion_80_2[i]])))
loss_path_80_2 = [item for sublist in loss_path_80_2 for item in sublist]   # 将二维列表转换为一维列表

estimate_path_proportion_80_3 = df_path_80_3.iloc[:,4].tolist()
target_proportion_80_3 = df_path_80_3.iloc[:,6].tolist()
loss_path_80_3 = list()
for i in range(len(target_proportion_80_3)):
    if target_proportion_80_3[i] != -1:
        loss_path_80_3.append(list(map(abs,[target_proportion_80_3[i] - estimate_path_proportion_80_3[i]])))
loss_path_80_3 = [item for sublist in loss_path_80_3 for item in sublist]   # 将二维列表转换为一维列表

estimate_path_proportion_80_4 = df_path_80_4.iloc[:,4].tolist()
target_proportion_80_4 = df_path_80_4.iloc[:,6].tolist()
loss_path_80_4 = list()
for i in range(len(target_proportion_80_4)):
    if target_proportion_80_4[i] != -1:
        loss_path_80_4.append(list(map(abs,[target_proportion_80_4[i] - estimate_path_proportion_80_4[i]])))
loss_path_80_4 = [item for sublist in loss_path_80_4 for item in sublist]   # 将二维列表转换为一维列表

estimate_path_proportion_80_5 = df_path_80_5.iloc[:,4].tolist()
target_proportion_80_5 = df_path_80_5.iloc[:,6].tolist()
loss_path_80_5 = list()
for i in range(len(target_proportion_80_5)):
    if target_proportion_80_5[i] != -1:
        loss_path_80_5.append(list(map(abs,[target_proportion_80_5[i] - estimate_path_proportion_80_5[i]])))
loss_path_80_5 = [item for sublist in loss_path_80_5 for item in sublist]   # 将二维列表转换为一维列表

estimate_path_proportion_80_6 = df_path_80_6.iloc[:,4].tolist()
target_proportion_80_6 = df_path_80_6.iloc[:,6].tolist()
loss_path_80_6 = list()
for i in range(len(target_proportion_80_6)):
    if target_proportion_80_6[i] != -1:
        loss_path_80_6.append(list(map(abs,[target_proportion_80_6[i] - estimate_path_proportion_80_6[i]])))
loss_path_80_6 = [item for sublist in loss_path_80_6 for item in sublist]   # 将二维列表转换为一维列表

estimate_path_proportion_80_7 = df_path_80_7.iloc[:,4].tolist()
target_proportion_80_7 = df_path_80_7.iloc[:,6].tolist()
loss_path_80_7 = list()
for i in range(len(target_proportion_80_7)):
    if target_proportion_80_7[i] != -1:
        loss_path_80_7.append(list(map(abs,[target_proportion_80_7[i] - estimate_path_proportion_80_7[i]])))
loss_path_80_7 = [item for sublist in loss_path_80_7 for item in sublist]   # 将二维列表转换为一维列表

estimate_path_proportion_80_8 = df_path_80_8.iloc[:,4].tolist()
target_proportion_80_8 = df_path_80_8.iloc[:,6].tolist()
loss_path_80_8 = list()
for i in range(len(target_proportion_80_8)):
    if target_proportion_80_8[i] != -1:
        loss_path_80_8.append(list(map(abs,[target_proportion_80_8[i] - estimate_path_proportion_80_8[i]])))
loss_path_80_8 = [item for sublist in loss_path_80_8 for item in sublist]   # 将二维列表转换为一维列表

estimate_path_proportion_80_9 = df_path_80_9.iloc[:,4].tolist()
target_proportion_80_9 = df_path_80_9.iloc[:,6].tolist()
loss_path_80_9 = list()
for i in range(len(target_proportion_80_9)):
    if target_proportion_80_9[i] != -1:
        loss_path_80_9.append(list(map(abs,[target_proportion_80_9[i] - estimate_path_proportion_80_9[i]])))
loss_path_80_9 = [item for sublist in loss_path_80_9 for item in sublist]   # 将二维列表转换为一维列表

estimate_path_proportion_80_10 = df_path_80_10.iloc[:,4].tolist()
target_proportion_80_10 = df_path_80_10.iloc[:,6].tolist()
loss_path_80_10 = list()
for i in range(len(target_proportion_80_10)):
    if target_proportion_80_10[i] != -1:
        loss_path_80_10.append(list(map(abs,[target_proportion_80_10[i] - estimate_path_proportion_80_10[i]])))
loss_path_80_10 = [item for sublist in loss_path_80_10 for item in sublist]   # 将二维列表转换为一维列表

loss_path_80 = (loss_path_80_1, loss_path_80_2, loss_path_80_3, loss_path_80_4, loss_path_80_5, loss_path_80_6, loss_path_80_7, loss_path_80_8, loss_path_80_9, loss_path_80_10)

box_path = plt.boxplot(loss_path_80, patch_artist = True,
                         labels = ['sample 1', 'sample 2', 'sample 3', 'sample 4', 'sample 5', 'sample 6', 'sample 7', 'sample 8', 'sample 9', 'sample 10'],
                        flierprops={'marker': 'o', 'markerfacecolor': 'w', 'markeredgecolor': 'b', 'alpha': 0.5},   # 异常值设置为蓝色
                         medianprops = {'linewidth': 1.5, 'color': 'b'})  # 一定要有patch_artist = True，否则不识别facecolor这个关键词

color = 'b'
for box in box_path['boxes']:
    box.set(facecolor='w', edgecolor='b', linestyle='dashed')

for whisker in box_path['whiskers']:
    whisker.set(color='b', linewidth=1)

for cap in box_path['caps']:
    cap.set(color='b', linewidth=1)

for median in box_path['medians']:
    median.set(color='b', linewidth=1.5)

# （2.2）将5个箱型图的中位线处连接成折线图：
median_loss_path_80_1 = np.median(loss_path_80_1)
median_loss_path_80_2 = np.median(loss_path_80_2)
median_loss_path_80_3 = np.median(loss_path_80_3)
median_loss_path_80_4 = np.median(loss_path_80_4)
median_loss_path_80_5 = np.median(loss_path_80_5)
median_loss_path_80_6 = np.median(loss_path_80_6)
median_loss_path_80_7 = np.median(loss_path_80_7)
median_loss_path_80_8 = np.median(loss_path_80_8)
median_loss_path_80_9 = np.median(loss_path_80_9)
median_loss_path_80_10 = np.median(loss_path_80_10)

median_loss_path_80 = np.array([median_loss_path_80_1, median_loss_path_80_2, median_loss_path_80_3, median_loss_path_80_4, median_loss_path_80_5, median_loss_path_80_6, median_loss_path_80_7, median_loss_path_80_8, median_loss_path_80_9, median_loss_path_80_10])

plt.plot(range(1, len(median_loss_path_80) + 1), median_loss_path_80, marker='o', markersize=10, linewidth=3.0, color='dodgerblue')  # 绘制中位数折线图

plt.text(1.0, median_loss_path_80_1+0.002, 'Median: {:.5f}'.format(median_loss_path_80_1), fontproperties='Times New Roman', size=18, ha='center', va='bottom')     # 将箱型图的中位数标记出来
plt.text(2.0, median_loss_path_80_2-0.006, ' {:.5f}'.format(median_loss_path_80_2), fontproperties='Times New Roman', size=18, ha='center', va='bottom')     # 将箱型图的中位数标记出来
plt.text(3.0, median_loss_path_80_3+0.002, ' {:.5f}'.format(median_loss_path_80_3), fontproperties='Times New Roman', size=18, ha='center', va='bottom')     # 将箱型图的中位数标记出来
plt.text(4.0, median_loss_path_80_4-0.006, ' {:.5f}'.format(median_loss_path_80_4), fontproperties='Times New Roman', size=18, ha='center', va='bottom')     # 将箱型图的中位数标记出来
plt.text(5.0, median_loss_path_80_5+0.002, ' {:.5f}'.format(median_loss_path_80_5), fontproperties='Times New Roman', size=18, ha='center', va='bottom')     # 将箱型图的中位数标记出来
plt.text(6.0, median_loss_path_80_6-0.006, ' {:.5f}'.format(median_loss_path_80_6), fontproperties='Times New Roman', size=18, ha='center', va='bottom')     # 将箱型图的中位数标记出来
plt.text(7.0, median_loss_path_80_7+0.002, ' {:.5f}'.format(median_loss_path_80_7), fontproperties='Times New Roman', size=18, ha='center', va='bottom')     # 将箱型图的中位数标记出来
plt.text(8.0, median_loss_path_80_8-0.006, ' {:.5f}'.format(median_loss_path_80_8), fontproperties='Times New Roman', size=18, ha='center', va='bottom')     # 将箱型图的中位数标记出来
plt.text(9.0, median_loss_path_80_9+0.002, ' {:.5f}'.format(median_loss_path_80_9), fontproperties='Times New Roman', size=18, ha='center', va='bottom')     # 将箱型图的中位数标记出来
plt.text(10.0, median_loss_path_80_10-0.006, ' {:.5f}'.format(median_loss_path_80_10), fontproperties='Times New Roman', size=18, ha='center', va='bottom')     # 将箱型图的中位数标记出来

fig = plt.gcf()
fig.set_size_inches(8, 6)  # 设置保存图像的显示大小 宽度为 10英寸，高度为 8英寸
plt.xlabel('(d) 80% OBD validation set', labelpad=7.5, fontdict={'family': 'Times New Roman', 'size': 18, 'color': 'k'})  # 改变横坐标轴标题字体  labelpad=10表示坐标标题距离坐标轴的垂直距离
plt.ylabel('Absolute Error of path selection rate', labelpad=7.5, fontdict={'family': 'Times New Roman', 'size': 18, 'color': 'k'})  # 改变纵坐标轴标题字体
plt.xticks(fontproperties='Times New Roman', size=18, rotation=20)
plt.yticks(fontproperties='Times New Roman', size=18)
handles = [Line2D([0], [0], color='dodgerblue', marker='o', lw=2)]
labels = ['Median']   # 自定义图例标签
plt.legend(handles=handles, labels=labels, loc='upper left', prop={'family': 'Times New Roman', 'size': 18})   # 添加图例
plt.subplots_adjust(right=0.95, bottom=0.2, hspace=0.3, wspace=10.5)
plt.savefig('80% path validation set box.png', dpi=300, format='png')
plt.show()

# 3、计算0%、20%、40%、60%、80%验证集 5次的平均值，分别绘制5个箱型图，每个箱型图是eg.20%的5个平均值，横坐标是0%、20%、40%、60%、80%，纵坐标是AE平均值
# （3.1）calculate the AE average of 0%
df_path_00 = pd.read_csv(r'F:\SEU\MasterThesis\02BTCG_cross_SF64\14BTCG_cross_SF_all_5F\output_path.csv')

estimate_path_proportion_00 = df_path_00.iloc[:,4].tolist()
target_proportion_00 = df_path_00.iloc[:,6].tolist()
loss_path_00 = list(map(abs,[target_proportion_00[i] - estimate_path_proportion_00[i] for i in range(0,len(target_proportion_00))]))

ave_loss_path_00 = sum(loss_path_00) / len(loss_path_00)

ave_loss_path_20_1 = sum(loss_path_20_1) / len(loss_path_20_1)
ave_loss_path_20_2 = sum(loss_path_20_2) / len(loss_path_20_2)
ave_loss_path_20_3 = sum(loss_path_20_3) / len(loss_path_20_3)
ave_loss_path_20_4 = sum(loss_path_20_4) / len(loss_path_20_4)
ave_loss_path_20_5 = sum(loss_path_20_5) / len(loss_path_20_5)
ave_loss_path_20_6 = sum(loss_path_20_6) / len(loss_path_20_6)
ave_loss_path_20_7 = sum(loss_path_20_7) / len(loss_path_20_7)
ave_loss_path_20_8 = sum(loss_path_20_8) / len(loss_path_20_8)
ave_loss_path_20_9 = sum(loss_path_20_9) / len(loss_path_20_9)
ave_loss_path_20_10 = sum(loss_path_20_10) / len(loss_path_20_10)
ave_loss_path_20 = [ave_loss_path_20_1, ave_loss_path_20_2, ave_loss_path_20_3, ave_loss_path_20_4, ave_loss_path_20_5, ave_loss_path_20_6, ave_loss_path_20_7, ave_loss_path_20_8, ave_loss_path_20_9, ave_loss_path_20_10]

ave_loss_path_40_1 = sum(loss_path_40_1) / len(loss_path_40_1)
ave_loss_path_40_2 = sum(loss_path_40_2) / len(loss_path_40_2)
ave_loss_path_40_3 = sum(loss_path_40_3) / len(loss_path_40_3)
ave_loss_path_40_4 = sum(loss_path_40_4) / len(loss_path_40_4)
ave_loss_path_40_5 = sum(loss_path_40_5) / len(loss_path_40_5)
ave_loss_path_40_6 = sum(loss_path_40_6) / len(loss_path_40_6)
ave_loss_path_40_7 = sum(loss_path_40_7) / len(loss_path_40_7)
ave_loss_path_40_8 = sum(loss_path_40_8) / len(loss_path_40_8)
ave_loss_path_40_9 = sum(loss_path_40_9) / len(loss_path_40_9)
ave_loss_path_40_10 = sum(loss_path_40_10) / len(loss_path_40_10)
ave_loss_path_40 = [ave_loss_path_40_1, ave_loss_path_40_2, ave_loss_path_40_3, ave_loss_path_40_4, ave_loss_path_40_5, ave_loss_path_40_6, ave_loss_path_40_7, ave_loss_path_40_8, ave_loss_path_40_9, ave_loss_path_40_10]

ave_loss_path_60_1 = sum(loss_path_60_1) / len(loss_path_60_1)
ave_loss_path_60_2 = sum(loss_path_60_2) / len(loss_path_60_2)
ave_loss_path_60_3 = sum(loss_path_60_3) / len(loss_path_60_3)
ave_loss_path_60_4 = sum(loss_path_60_4) / len(loss_path_60_4)
ave_loss_path_60_5 = sum(loss_path_60_5) / len(loss_path_60_5)
ave_loss_path_60_6 = sum(loss_path_60_6) / len(loss_path_60_6)
ave_loss_path_60_7 = sum(loss_path_60_7) / len(loss_path_60_7)
ave_loss_path_60_8 = sum(loss_path_60_8) / len(loss_path_60_8)
ave_loss_path_60_9 = sum(loss_path_60_9) / len(loss_path_60_9)
ave_loss_path_60_10 = sum(loss_path_60_10) / len(loss_path_60_10)
ave_loss_path_60 = [ave_loss_path_60_1, ave_loss_path_60_2, ave_loss_path_60_3, ave_loss_path_60_4, ave_loss_path_60_5, ave_loss_path_60_6, ave_loss_path_60_7, ave_loss_path_60_8, ave_loss_path_60_9, ave_loss_path_60_10]

ave_loss_path_80_1 = sum(loss_path_80_1) / len(loss_path_80_1)
ave_loss_path_80_2 = sum(loss_path_80_2) / len(loss_path_80_2)
ave_loss_path_80_3 = sum(loss_path_80_3) / len(loss_path_80_3)
ave_loss_path_80_4 = sum(loss_path_80_4) / len(loss_path_80_4)
ave_loss_path_80_5 = sum(loss_path_80_5) / len(loss_path_80_5)
ave_loss_path_80_6 = sum(loss_path_80_6) / len(loss_path_80_6)
ave_loss_path_80_7 = sum(loss_path_80_7) / len(loss_path_80_7)
ave_loss_path_80_8 = sum(loss_path_80_8) / len(loss_path_80_8)
ave_loss_path_80_9 = sum(loss_path_80_9) / len(loss_path_80_9)
ave_loss_path_80_10 = sum(loss_path_80_10) / len(loss_path_80_10)
ave_loss_path_80 = [ave_loss_path_80_1, ave_loss_path_80_2, ave_loss_path_80_3, ave_loss_path_80_4, ave_loss_path_80_5, ave_loss_path_80_6, ave_loss_path_80_7, ave_loss_path_80_8, ave_loss_path_80_9, ave_loss_path_80_10]

ave_loss_path = ( ave_loss_path_20, ave_loss_path_40, ave_loss_path_60, ave_loss_path_80)

box_path = plt.boxplot(ave_loss_path, patch_artist = True,
                         labels = ['20%', '40%', '60%', '80%'],
                        flierprops={'marker': 'o', 'markerfacecolor': 'w', 'markeredgecolor': 'b', 'alpha': 0.5},   # 异常值设置为蓝色
                         medianprops = {'linewidth': 1.5, 'color': 'b'})  # 一定要有patch_artist = True，否则不识别facecolor这个关键词
color = 'b'
for box in box_path['boxes']:
    box.set(facecolor='w', edgecolor='b', linestyle='dashed')

for whisker in box_path['whiskers']:
    whisker.set(color='b', linewidth=1)

for cap in box_path['caps']:
    cap.set(color='b', linewidth=1)

for median in box_path['medians']:
    median.set(color='b', linewidth=1.5)

median_loss_path_20 = np.median(ave_loss_path_20)
median_loss_path_40 = np.median(ave_loss_path_40)
median_loss_path_60 = np.median(ave_loss_path_60)
median_loss_path_80 = np.median(ave_loss_path_80)

median_loss_path = np.array([median_loss_path_20, median_loss_path_40, median_loss_path_60, median_loss_path_80])

plt.plot(range(1, len(median_loss_path) + 1), median_loss_path, marker='o', markersize=10, linewidth=3.0, color='dodgerblue')  # 绘制中位数折线图

plt.text(1.0, median_loss_path_20+0.001, 'Median: {:.5f}'.format(median_loss_path_20), color='dodgerblue', fontproperties='Times New Roman', size=18, ha='center', va='bottom')     # 将箱型图的中位数标记出来
plt.text(2.05, median_loss_path_40+0.00013, ' {:.5f}'.format(median_loss_path_40), color='dodgerblue', fontproperties='Times New Roman', size=18, ha='center', va='bottom')     # 将箱型图的中位数标记出来
plt.text(3.02, median_loss_path_60+0.00055, ' {:.5f}'.format(median_loss_path_60), color='dodgerblue', fontproperties='Times New Roman', size=18, ha='center', va='bottom')     # 将箱型图的中位数标记出来
plt.text(4.02, median_loss_path_80+0.0006, ' {:.5f}'.format(median_loss_path_80), color='dodgerblue', fontproperties='Times New Roman', size=18, ha='center', va='bottom')     # 将箱型图的中位数标记出来

# calcualte mean:
mean_20  = np.mean(ave_loss_path_20)
mean_40  = np.mean(ave_loss_path_40)
mean_60  = np.mean(ave_loss_path_60)
mean_80  = np.mean(ave_loss_path_80)
# draw mean -----
plt.axhline(mean_20, color='darkgrey', linestyle='-.', linewidth = 1, label='Mean_20%')
plt.axhline(mean_40, color='darkgrey', linestyle='-.', linewidth = 1, label='Mean_40%')
plt.axhline(mean_60, color='darkgrey', linestyle='-.', linewidth = 1, label='Mean_60%')
plt.axhline(mean_80, color='darkgrey', linestyle='-.', linewidth = 1, label='Mean_80%')

plt.text(1.2, mean_20-0.001, 'Mean: {:.5f}'.format(mean_20), color='grey', fontproperties='Times New Roman', size=18, ha='center', va='bottom')     # 将箱型图的中位数标记出来
plt.text(2.5, mean_40-0.0008, ' {:.5f}'.format(mean_40), color='grey', fontproperties='Times New Roman', size=18, ha='center', va='bottom')     # 将箱型图的中位数标记出来
plt.text(3.5, mean_60-0.001, ' {:.5f}'.format(mean_60), color='grey', fontproperties='Times New Roman', size=18, ha='center', va='bottom')     # 将箱型图的中位数标记出来
plt.text(4.2, mean_80+0.001, ' {:.5f}'.format(mean_80), color='grey', fontproperties='Times New Roman', size=18, ha='center', va='bottom')     # 将箱型图的中位数标记出来
fig = plt.gcf()
fig.set_size_inches(8, 6)  # 设置保存图像的显示大小 宽度为 10英寸，高度为 8英寸
plt.xlabel('(a) OBD validation set', labelpad=7.5, fontdict={'family': 'Times New Roman', 'size': 18, 'color': 'k'})  # 改变横坐标轴标题字体  labelpad=10表示坐标标题距离坐标轴的垂直距离
plt.ylabel('Absolute Error Average of path selection rate', labelpad=5.5, fontdict={'family': 'Times New Roman', 'size': 18, 'color': 'k'})  # 改变纵坐标轴标题字体
plt.xticks(fontproperties='Times New Roman', size=18)
plt.yticks(fontproperties='Times New Roman', size=18)
# handles = [Line2D([0], [0], color='dodgerblue', marker='o', lw=2), Line2D([0], [0], linestyle='-.', color='dodgerblue', lw=2)]
handles = [Line2D([0], [0], color='dodgerblue', marker='o', lw=2), Line2D([0], [0], linestyle='-.', color='darkgrey', lw=2)]
labels = ['Median', 'Mean']   # 自定义图例标签
# plt.legend(handles=handles, labels=labels, loc='upper left')   # 添加图例
plt.legend(handles=handles, labels=labels, loc='upper left', prop={'family': 'Times New Roman', 'size': 18})   # 添加图例
plt.subplots_adjust(right=0.95, bottom=0.2, hspace=0.3, wspace=10.5)
plt.savefig('path validation set box.png', dpi=300, format='png')
plt.show()

# 对AE_path四舍五入，保留小数点后两位，否则频率分布图中，每个数据都不同，其频率都很小，没有规律
rounded_loss_path_20_1 = [round(num, 2) for num in loss_path_20_1]
rounded_loss_path_20_2 = [round(num, 2) for num in loss_path_20_2]
rounded_loss_path_20_3 = [round(num, 2) for num in loss_path_20_3]
rounded_loss_path_20_4 = [round(num, 2) for num in loss_path_20_4]
rounded_loss_path_20_5 = [round(num, 2) for num in loss_path_20_5]
rounded_loss_path_20_6 = [round(num, 2) for num in loss_path_20_6]
rounded_loss_path_20_7 = [round(num, 2) for num in loss_path_20_7]
rounded_loss_path_20_8 = [round(num, 2) for num in loss_path_20_8]
rounded_loss_path_20_9 = [round(num, 2) for num in loss_path_20_9]
rounded_loss_path_20_10 = [round(num, 2) for num in loss_path_20_10]
rounded_loss_path_40_1 = [round(num, 2) for num in loss_path_40_1]
rounded_loss_path_40_2 = [round(num, 2) for num in loss_path_40_2]
rounded_loss_path_40_3 = [round(num, 2) for num in loss_path_40_3]
rounded_loss_path_40_4 = [round(num, 2) for num in loss_path_40_4]
rounded_loss_path_40_5 = [round(num, 2) for num in loss_path_40_5]
rounded_loss_path_40_6 = [round(num, 2) for num in loss_path_40_6]
rounded_loss_path_40_7 = [round(num, 2) for num in loss_path_40_7]
rounded_loss_path_40_8 = [round(num, 2) for num in loss_path_40_8]
rounded_loss_path_40_9 = [round(num, 2) for num in loss_path_40_9]
rounded_loss_path_40_10 = [round(num, 2) for num in loss_path_40_10]
rounded_loss_path_60_1 = [round(num, 2) for num in loss_path_60_1]
rounded_loss_path_60_2 = [round(num, 2) for num in loss_path_60_2]
rounded_loss_path_60_3 = [round(num, 2) for num in loss_path_60_3]
rounded_loss_path_60_4 = [round(num, 2) for num in loss_path_60_4]
rounded_loss_path_60_5 = [round(num, 2) for num in loss_path_60_5]
rounded_loss_path_60_6 = [round(num, 2) for num in loss_path_60_6]
rounded_loss_path_60_7 = [round(num, 2) for num in loss_path_60_7]
rounded_loss_path_60_8 = [round(num, 2) for num in loss_path_60_8]
rounded_loss_path_60_9 = [round(num, 2) for num in loss_path_60_9]
rounded_loss_path_60_10 = [round(num, 2) for num in loss_path_60_10]
rounded_loss_path_80_1 = [round(num, 2) for num in loss_path_80_1]
rounded_loss_path_80_2 = [round(num, 2) for num in loss_path_80_2]
rounded_loss_path_80_3 = [round(num, 2) for num in loss_path_80_3]
rounded_loss_path_80_4 = [round(num, 2) for num in loss_path_80_4]
rounded_loss_path_80_5 = [round(num, 2) for num in loss_path_80_5]
rounded_loss_path_80_6 = [round(num, 2) for num in loss_path_80_6]
rounded_loss_path_80_7 = [round(num, 2) for num in loss_path_80_7]
rounded_loss_path_80_8 = [round(num, 2) for num in loss_path_80_8]
rounded_loss_path_80_9 = [round(num, 2) for num in loss_path_80_9]
rounded_loss_path_80_10 = [round(num, 2) for num in loss_path_80_10]
# frequencies, bins = np.histogram(rounded_loss_path_20_1, bins='auto')    #绘制频率分布直方图
# plt.hist(rounded_loss_path_20_1, bins=bins, density=True, alpha=0.5, color='aquamarine', label='Frequency_20_1')
# mu, sigma = norm.fit(rounded_loss_path_20_1)   # 拟合频率分布曲线
# x = np.linspace(min(rounded_loss_path_20_1), max(rounded_loss_path_20_1), 100)
# y = norm.pdf(x, mu, sigma)
# plt.plot(x, y, 'aquamarine', label='Fitted Curve_20_1')

# 只绘制频率分布曲线，不绘制分布直方图
plt.rc('font', family='Times New Roman', size = 12)    # 设置全局字体样式，坐标轴的数据在这里会被设置为相应的字体
mu, sigma = norm.fit(rounded_loss_path_20_1)   # 拟合频率分布曲线
x = np.linspace(min(rounded_loss_path_20_1), max(rounded_loss_path_20_1), 100)
y = norm.pdf(x, mu, sigma)
plt.plot(x, y, 'deepskyblue', label='Fitted Curve_20_1')

mu, sigma = norm.fit(rounded_loss_path_20_2)   # 拟合频率分布曲线
x = np.linspace(min(rounded_loss_path_20_2), max(rounded_loss_path_20_2), 100)
y = norm.pdf(x, mu, sigma)
plt.plot(x, y, 'lightskyblue', label='Fitted Curve_20_2')

mu, sigma = norm.fit(rounded_loss_path_20_3)   # 拟合频率分布曲线
x = np.linspace(min(rounded_loss_path_20_3), max(rounded_loss_path_20_3), 100)
y = norm.pdf(x, mu, sigma)
plt.plot(x, y, 'steelblue', label='Fitted Curve_20_3')

mu, sigma = norm.fit(rounded_loss_path_20_4)   # 拟合频率分布曲线
x = np.linspace(min(rounded_loss_path_20_4), max(rounded_loss_path_20_4), 100)
y = norm.pdf(x, mu, sigma)
plt.plot(x, y, 'powderblue', label='Fitted Curve_20_4')

mu, sigma = norm.fit(rounded_loss_path_20_5)   # 拟合频率分布曲线
x = np.linspace(min(rounded_loss_path_20_5), max(rounded_loss_path_20_5), 100)
y = norm.pdf(x, mu, sigma)
plt.plot(x, y, 'dodgerblue', label='Fitted Curve_20_5')

plt.rc('font', family='Times New Roman', size = 12)    # 设置全局字体样式，坐标轴的数据在这里会被设置为相应的字体
mu, sigma = norm.fit(rounded_loss_path_20_6)   # 拟合频率分布曲线
x = np.linspace(min(rounded_loss_path_20_6), max(rounded_loss_path_20_6), 100)
y = norm.pdf(x, mu, sigma)
plt.plot(x, y, 'cadetblue', label='Fitted Curve_20_6')

mu, sigma = norm.fit(rounded_loss_path_20_7)   # 拟合频率分布曲线
x = np.linspace(min(rounded_loss_path_20_7), max(rounded_loss_path_20_7), 100)
y = norm.pdf(x, mu, sigma)
plt.plot(x, y, 'skyblue', label='Fitted Curve_20_7')

mu, sigma = norm.fit(rounded_loss_path_20_8)   # 拟合频率分布曲线
x = np.linspace(min(rounded_loss_path_20_8), max(rounded_loss_path_20_8), 100)
y = norm.pdf(x, mu, sigma)
plt.plot(x, y, 'aliceblue', label='Fitted Curve_20_8')

mu, sigma = norm.fit(rounded_loss_path_20_9)   # 拟合频率分布曲线
x = np.linspace(min(rounded_loss_path_20_9), max(rounded_loss_path_20_9), 100)
y = norm.pdf(x, mu, sigma)
plt.plot(x, y, 'lightsteelblue', label='Fitted Curve_20_9')

mu, sigma = norm.fit(rounded_loss_path_20_10)   # 拟合频率分布曲线
x = np.linspace(min(rounded_loss_path_20_10), max(rounded_loss_path_20_10), 100)
y = norm.pdf(x, mu, sigma)
plt.plot(x, y, 'cornflowerblue', label='Fitted Curve_20_10')

mu, sigma = norm.fit(rounded_loss_path_40_1)   # 拟合频率分布曲线
x = np.linspace(min(rounded_loss_path_40_1), max(rounded_loss_path_40_1), 100)
y = norm.pdf(x, mu, sigma)
plt.plot(x, y, 'thistle', linestyle='-.', label='Fitted Curve_40_1')

mu, sigma = norm.fit(rounded_loss_path_40_2)   # 拟合频率分布曲线
x = np.linspace(min(rounded_loss_path_40_2), max(rounded_loss_path_40_2), 100)
y = norm.pdf(x, mu, sigma)
plt.plot(x, y, 'violet', linestyle='-.', label='Fitted Curve_40_2')

mu, sigma = norm.fit(rounded_loss_path_40_3)   # 拟合频率分布曲线
x = np.linspace(min(rounded_loss_path_40_3), max(rounded_loss_path_40_3), 100)
y = norm.pdf(x, mu, sigma)
plt.plot(x, y, 'm', linestyle='-.', label='Fitted Curve_40_3')

mu, sigma = norm.fit(rounded_loss_path_40_4)   # 拟合频率分布曲线
x = np.linspace(min(rounded_loss_path_40_4), max(rounded_loss_path_40_4), 100)
y = norm.pdf(x, mu, sigma)
plt.plot(x, y, 'fuchsia', linestyle='-.', label='Fitted Curve_40_4')

mu, sigma = norm.fit(rounded_loss_path_40_5)   # 拟合频率分布曲线
x = np.linspace(min(rounded_loss_path_40_5), max(rounded_loss_path_40_5), 100)
y = norm.pdf(x, mu, sigma)
plt.plot(x, y, 'darkviolet', linestyle='-.', label='Fitted Curve_40_5')

mu, sigma = norm.fit(rounded_loss_path_40_6)   # 拟合频率分布曲线
x = np.linspace(min(rounded_loss_path_40_6), max(rounded_loss_path_40_6), 100)
y = norm.pdf(x, mu, sigma)
plt.plot(x, y, 'plum', linestyle='-.', label='Fitted Curve_40_6')

mu, sigma = norm.fit(rounded_loss_path_40_7)   # 拟合频率分布曲线
x = np.linspace(min(rounded_loss_path_40_7), max(rounded_loss_path_40_7), 100)
y = norm.pdf(x, mu, sigma)
plt.plot(x, y, 'mediumorchid', linestyle='-.', label='Fitted Curve_40_7')

mu, sigma = norm.fit(rounded_loss_path_40_8)   # 拟合频率分布曲线
x = np.linspace(min(rounded_loss_path_40_8), max(rounded_loss_path_40_8), 100)
y = norm.pdf(x, mu, sigma)
plt.plot(x, y, 'purple', linestyle='-.', label='Fitted Curve_40_8')

mu, sigma = norm.fit(rounded_loss_path_40_9)   # 拟合频率分布曲线
x = np.linspace(min(rounded_loss_path_40_9), max(rounded_loss_path_40_9), 100)
y = norm.pdf(x, mu, sigma)
plt.plot(x, y, 'orchid', linestyle='-.', label='Fitted Curve_40_9')

mu, sigma = norm.fit(rounded_loss_path_40_10)   # 拟合频率分布曲线
x = np.linspace(min(rounded_loss_path_40_10), max(rounded_loss_path_40_10), 100)
y = norm.pdf(x, mu, sigma)
plt.plot(x, y, 'darkmagenta', linestyle='-.', label='Fitted Curve_40_10')

mu, sigma = norm.fit(rounded_loss_path_60_1)   # 拟合频率分布曲线
x = np.linspace(min(rounded_loss_path_60_1), max(rounded_loss_path_60_1), 100)
y = norm.pdf(x, mu, sigma)
plt.plot(x, y, 'yellow', linestyle='--', label='Fitted Curve_60_1')

mu, sigma = norm.fit(rounded_loss_path_60_2)   # 拟合频率分布曲线
x = np.linspace(min(rounded_loss_path_60_2), max(rounded_loss_path_60_2), 100)
y = norm.pdf(x, mu, sigma)
plt.plot(x, y, 'gold', linestyle='--', label='Fitted Curve_60_2')

mu, sigma = norm.fit(rounded_loss_path_60_3)   # 拟合频率分布曲线
x = np.linspace(min(rounded_loss_path_60_3), max(rounded_loss_path_60_3), 100)
y = norm.pdf(x, mu, sigma)
plt.plot(x, y, 'y', linestyle='--', label='Fitted Curve_60_3')

mu, sigma = norm.fit(rounded_loss_path_60_4)   # 拟合频率分布曲线
x = np.linspace(min(rounded_loss_path_60_4), max(rounded_loss_path_60_4), 100)
y = norm.pdf(x, mu, sigma)
plt.plot(x, y, 'khaki', linestyle='--', label='Fitted Curve_60_4')

mu, sigma = norm.fit(rounded_loss_path_60_5)   # 拟合频率分布曲线
x = np.linspace(min(rounded_loss_path_60_5), max(rounded_loss_path_60_5), 100)
y = norm.pdf(x, mu, sigma)
plt.plot(x, y, 'goldenrod', linestyle='--', label='Fitted Curve_60_5')

mu, sigma = norm.fit(rounded_loss_path_60_6)   # 拟合频率分布曲线
x = np.linspace(min(rounded_loss_path_60_6), max(rounded_loss_path_60_6), 100)
y = norm.pdf(x, mu, sigma)
plt.plot(x, y, 'cornsilk', linestyle='--', label='Fitted Curve_60_6')

mu, sigma = norm.fit(rounded_loss_path_60_7)   # 拟合频率分布曲线
x = np.linspace(min(rounded_loss_path_60_7), max(rounded_loss_path_60_7), 100)
y = norm.pdf(x, mu, sigma)
plt.plot(x, y, 'lemonchiffon', linestyle='--', label='Fitted Curve_60_7')

mu, sigma = norm.fit(rounded_loss_path_60_8)   # 拟合频率分布曲线
x = np.linspace(min(rounded_loss_path_60_8), max(rounded_loss_path_60_8), 100)
y = norm.pdf(x, mu, sigma)
plt.plot(x, y, 'ivory', linestyle='--', label='Fitted Curve_60_8')

mu, sigma = norm.fit(rounded_loss_path_60_9)   # 拟合频率分布曲线
x = np.linspace(min(rounded_loss_path_60_9), max(rounded_loss_path_60_9), 100)
y = norm.pdf(x, mu, sigma)
plt.plot(x, y, 'olive', linestyle='--', label='Fitted Curve_60_9')

mu, sigma = norm.fit(rounded_loss_path_60_10)   # 拟合频率分布曲线
x = np.linspace(min(rounded_loss_path_60_10), max(rounded_loss_path_60_10), 100)
y = norm.pdf(x, mu, sigma)
plt.plot(x, y, 'darkkhaki', linestyle='--', label='Fitted Curve_60_10')

mu, sigma = norm.fit(rounded_loss_path_80_1)   # 拟合频率分布曲线
x = np.linspace(min(rounded_loss_path_80_1), max(rounded_loss_path_80_1), 100)
y = norm.pdf(x, mu, sigma)
plt.plot(x, y, 'lightgreen', linestyle=':', label='Fitted Curve_80_1')

mu, sigma = norm.fit(rounded_loss_path_80_2)   # 拟合频率分布曲线
x = np.linspace(min(rounded_loss_path_80_2), max(rounded_loss_path_80_2), 100)
y = norm.pdf(x, mu, sigma)
plt.plot(x, y, 'g', linestyle=':', label='Fitted Curve_80_2')

mu, sigma = norm.fit(rounded_loss_path_80_3)   # 拟合频率分布曲线
x = np.linspace(min(rounded_loss_path_80_3), max(rounded_loss_path_80_3), 100)
y = norm.pdf(x, mu, sigma)
plt.plot(x, y, 'limegreen', linestyle=':', label='Fitted Curve_80_3')

mu, sigma = norm.fit(rounded_loss_path_80_4)   # 拟合频率分布曲线
x = np.linspace(min(rounded_loss_path_80_4), max(rounded_loss_path_80_4), 100)
y = norm.pdf(x, mu, sigma)
plt.plot(x, y, 'darkgreen', linestyle=':', label='Fitted Curve_80_4')

mu, sigma = norm.fit(rounded_loss_path_80_5)   # 拟合频率分布曲线
x = np.linspace(min(rounded_loss_path_80_5), max(rounded_loss_path_80_5), 100)
y = norm.pdf(x, mu, sigma)
plt.plot(x, y, 'lime', linestyle=':', label='Fitted Curve_80_5')

mu, sigma = norm.fit(rounded_loss_path_80_6)   # 拟合频率分布曲线
x = np.linspace(min(rounded_loss_path_80_6), max(rounded_loss_path_80_6), 100)
y = norm.pdf(x, mu, sigma)
plt.plot(x, y, 'forestgreen', linestyle=':', label='Fitted Curve_80_6')

mu, sigma = norm.fit(rounded_loss_path_80_7)   # 拟合频率分布曲线
x = np.linspace(min(rounded_loss_path_80_7), max(rounded_loss_path_80_7), 100)
y = norm.pdf(x, mu, sigma)
plt.plot(x, y, 'green', linestyle=':', label='Fitted Curve_80_7')

mu, sigma = norm.fit(rounded_loss_path_80_8)   # 拟合频率分布曲线
x = np.linspace(min(rounded_loss_path_80_8), max(rounded_loss_path_80_8), 100)
y = norm.pdf(x, mu, sigma)
plt.plot(x, y, 'seagreen', linestyle=':', label='Fitted Curve_80_8')

mu, sigma = norm.fit(rounded_loss_path_80_9)   # 拟合频率分布曲线
x = np.linspace(min(rounded_loss_path_80_9), max(rounded_loss_path_80_9), 100)
y = norm.pdf(x, mu, sigma)
plt.plot(x, y, 'mediumseagreen', linestyle=':', label='Fitted Curve_80_9')

mu, sigma = norm.fit(rounded_loss_path_80_10)   # 拟合频率分布曲线
x = np.linspace(min(rounded_loss_path_80_10), max(rounded_loss_path_80_10), 100)
y = norm.pdf(x, mu, sigma)
plt.plot(x, y, 'springgreen', linestyle=':', label='Fitted Curve_80_10')

fig = plt.gcf()
fig.set_size_inches(8, 6)  # 设置保存图像的显示大小 宽度为 10英寸，高度为 8英寸
plt.xlabel('Absolute Error Average of path_proportion', fontdict={'family': 'Times New Roman', 'size': 12, 'color': 'k'})
plt.ylabel('Frequency', fontdict={'family': 'Times New Roman', 'size': 12, 'color': 'k'})
font = fm.FontProperties(family='Times New Roman', style='normal', size=12)
plt.legend(loc='upper right', ncol=2, prop=font)
plt.savefig('path-Frequency Distribution with Fitted Curve.png', dpi=300, format='png')
plt.show()











# # 4、关于link_count变量输出结果 在不同验证集中的 变量值AE误差的 中位数折线图（而不是损失误差MSE×）
# for j in range(1,5):       # 20%/40%/60%/80%
#     for i in range(1,11):   # 随机选取20%数据，藏住信息，作为验证集，共随机10次
#         df_link = pd.read_csv(r'F:\SEU\MasterThesis\02BTCG_cross_SF64\15BTCG_cross_SF_all_5F_validation_v2_random10\input_link.csv')
#         df_link['sensor_name_copy'] = df_link['sensor_name'].copy()     # 先复制一列，便于后面input_link_sensor.csv文件的生成
#         num_samples_link = int(len(df_link) * 0.2*j)
#         random_indices_link = np.random.choice(df_link.index, num_samples_link, replace=False)
#         df_link.loc[random_indices_link, 'sensor_name'] = -1
#
#         df_link_sensor = pd.read_csv(r'F:\SEU\MasterThesis\02BTCG_cross_SF64\15BTCG_cross_SF_all_5F_validation_v2_random10\input_link_sensor.csv')
#         false_link_ids = df_link.loc[random_indices_link, 'sensor_name_copy']
#         df_link_sensor.loc[df_link_sensor['sensor_name'].isin(false_link_ids), 'sensor_count'] = -1
#         new_df_link_sensor = df_link_sensor[df_link_sensor['sensor_count'] != -1].copy()
#         new_df_link_sensor['id'] = range(1, len(new_df_link_sensor) + 1)   # 将id列重新按照12345……排序
#         new_df_link_sensor.to_csv(f'input_link_sensor_{20*j}_{i}.csv', index=False)
#
#         df_link = df_link.iloc[:,:8]  # 只保存input_link的前8列，即不保存sensor_name_copy列
#         df_link.to_csv(f'input_link_{20 * j}_{i}.csv', index=False)
# # 接下来，分别调用多个BTCG程序

# 只生成20%的验证集，随机10次
for i in range(1,11):   # 随机选取20%数据，藏住信息，作为验证集，共随机10次
    df_link = pd.read_csv(r'F:\SEU\MasterThesis\02BTCG_cross_SF64\15BTCG_cross_SF_all_5F_validation_v2_random10\input_link.csv')
    df_link['sensor_name_copy'] = df_link['sensor_name'].copy()     # 先复制一列，便于后面input_link_sensor.csv文件的生成
    num_samples_link = int(len(df_link) * 0.2*1)
    random_indices_link = np.random.choice(df_link.index, num_samples_link, replace=False)
    df_link.loc[random_indices_link, 'sensor_name'] = -1

    df_link_sensor = pd.read_csv(r'F:\SEU\MasterThesis\02BTCG_cross_SF64\15BTCG_cross_SF_all_5F_validation_v2_random10\input_link_sensor.csv')
    false_link_ids = df_link.loc[random_indices_link, 'sensor_name_copy']
    df_link_sensor.loc[df_link_sensor['sensor_name'].isin(false_link_ids), 'sensor_count'] = -1
    new_df_link_sensor = df_link_sensor[df_link_sensor['sensor_count'] != -1].copy()
    new_df_link_sensor['id'] = range(1, len(new_df_link_sensor) + 1)   # 将id列重新按照12345……排序
    new_df_link_sensor.to_csv(f'input_link_sensor_{20*1}_{i}.csv', index=False)

    df_link = df_link.iloc[:,:8]  # 只保存input_link的前8列，即不保存sensor_name_copy列
    df_link.to_csv(f'input_link_{20 * 1}_{i}.csv', index=False)


# 5、分别绘制link的20%、40%、60%、80%验证集的5次随机AE结果
# 2.1、绘制20%验证集，5次随机的absolute error
# (2.1) draw box plan:
df_link_20_1 = pd.read_csv(r'F:\SEU\MasterThesis\02BTCG_cross_SF64\15BTCG_cross_SF_all_5F_validation_v2_random10\output_link_20_1_link.csv')
df_link_20_2 = pd.read_csv(r'F:\SEU\MasterThesis\02BTCG_cross_SF64\15BTCG_cross_SF_all_5F_validation_v2_random10\output_link_20_2_link.csv')
df_link_20_3 = pd.read_csv(r'F:\SEU\MasterThesis\02BTCG_cross_SF64\15BTCG_cross_SF_all_5F_validation_v2_random10\output_link_20_3_link.csv')
df_link_20_4 = pd.read_csv(r'F:\SEU\MasterThesis\02BTCG_cross_SF64\15BTCG_cross_SF_all_5F_validation_v2_random10\output_link_20_4_link.csv')
df_link_20_5 = pd.read_csv(r'F:\SEU\MasterThesis\02BTCG_cross_SF64\15BTCG_cross_SF_all_5F_validation_v2_random10\output_link_20_5_link.csv')
df_link_20_6 = pd.read_csv(r'F:\SEU\MasterThesis\02BTCG_cross_SF64\15BTCG_cross_SF_all_5F_validation_v2_random10\output_link_20_6_link.csv')
df_link_20_7 = pd.read_csv(r'F:\SEU\MasterThesis\02BTCG_cross_SF64\15BTCG_cross_SF_all_5F_validation_v2_random10\output_link_20_7_link.csv')
df_link_20_8 = pd.read_csv(r'F:\SEU\MasterThesis\02BTCG_cross_SF64\15BTCG_cross_SF_all_5F_validation_v2_random10\output_link_20_8_link.csv')
df_link_20_9 = pd.read_csv(r'F:\SEU\MasterThesis\02BTCG_cross_SF64\15BTCG_cross_SF_all_5F_validation_v2_random10\output_link_20_9_link.csv')
df_link_20_10 = pd.read_csv(r'F:\SEU\MasterThesis\02BTCG_cross_SF64\15BTCG_cross_SF_all_5F_validation_v2_random10\output_link_20_10_link.csv')

estimate_link_count_20_1 = df_link_20_1.iloc[:,3].tolist()
target_count_20_1 = df_link_20_1.iloc[:,4].tolist()
loss_link_20_1 = list()
for i in range(len(target_count_20_1)):
    if target_count_20_1[i] != -1:
        loss_link_20_1.append(list(map(abs,[target_count_20_1[i] - estimate_link_count_20_1[i]])))
loss_link_20_1 = [item for sublist in loss_link_20_1 for item in sublist]   # 将二维列表转换为一维列表

estimate_link_count_20_2 = df_link_20_2.iloc[:,3].tolist()
target_count_20_2 = df_link_20_2.iloc[:,4].tolist()
loss_link_20_2 = list()
for i in range(len(target_count_20_2)):
    if target_count_20_2[i] != -1:
        loss_link_20_2.append(list(map(abs,[target_count_20_2[i] - estimate_link_count_20_2[i]])))
loss_link_20_2 = [item for sublist in loss_link_20_2 for item in sublist]   # 将二维列表转换为一维列表

estimate_link_count_20_3 = df_link_20_3.iloc[:,3].tolist()
target_count_20_3 = df_link_20_3.iloc[:,4].tolist()
loss_link_20_3 = list()
for i in range(len(target_count_20_3)):
    if target_count_20_3[i] != -1:
        loss_link_20_3.append(list(map(abs,[target_count_20_3[i] - estimate_link_count_20_3[i]])))
loss_link_20_3 = [item for sublist in loss_link_20_3 for item in sublist]   # 将二维列表转换为一维列表

estimate_link_count_20_4 = df_link_20_4.iloc[:,3].tolist()
target_count_20_4 = df_link_20_4.iloc[:,4].tolist()
loss_link_20_4 = list()
for i in range(len(target_count_20_4)):
    if target_count_20_4[i] != -1:
        loss_link_20_4.append(list(map(abs,[target_count_20_4[i] - estimate_link_count_20_4[i]])))
loss_link_20_4 = [item for sublist in loss_link_20_4 for item in sublist]   # 将二维列表转换为一维列表

estimate_link_count_20_5 = df_link_20_5.iloc[:,3].tolist()
target_count_20_5 = df_link_20_5.iloc[:,4].tolist()
loss_link_20_5 = list()
for i in range(len(target_count_20_5)):
    if target_count_20_5[i] != -1:
        loss_link_20_5.append(list(map(abs,[target_count_20_5[i] - estimate_link_count_20_5[i]])))
loss_link_20_5 = [item for sublist in loss_link_20_5 for item in sublist]   # 将二维列表转换为一维列表

estimate_link_count_20_6 = df_link_20_6.iloc[:,3].tolist()
target_count_20_6 = df_link_20_6.iloc[:,4].tolist()
loss_link_20_6 = list()
for i in range(len(target_count_20_6)):
    if target_count_20_6[i] != -1:
        loss_link_20_6.append(list(map(abs,[target_count_20_6[i] - estimate_link_count_20_6[i]])))
loss_link_20_6 = [item for sublist in loss_link_20_6 for item in sublist]   # 将二维列表转换为一维列表

estimate_link_count_20_7 = df_link_20_7.iloc[:,3].tolist()
target_count_20_7 = df_link_20_7.iloc[:,4].tolist()
loss_link_20_7 = list()
for i in range(len(target_count_20_7)):
    if target_count_20_7[i] != -1:
        loss_link_20_7.append(list(map(abs,[target_count_20_7[i] - estimate_link_count_20_7[i]])))
loss_link_20_7 = [item for sublist in loss_link_20_7 for item in sublist]   # 将二维列表转换为一维列表

estimate_link_count_20_8 = df_link_20_8.iloc[:,3].tolist()
target_count_20_8 = df_link_20_8.iloc[:,4].tolist()
loss_link_20_8 = list()
for i in range(len(target_count_20_8)):
    if target_count_20_8[i] != -1:
        loss_link_20_8.append(list(map(abs,[target_count_20_8[i] - estimate_link_count_20_8[i]])))
loss_link_20_8 = [item for sublist in loss_link_20_8 for item in sublist]   # 将二维列表转换为一维列表

estimate_link_count_20_9 = df_link_20_9.iloc[:,3].tolist()
target_count_20_9 = df_link_20_9.iloc[:,4].tolist()
loss_link_20_9 = list()
for i in range(len(target_count_20_9)):
    if target_count_20_9[i] != -1:
        loss_link_20_9.append(list(map(abs,[target_count_20_9[i] - estimate_link_count_20_9[i]])))
loss_link_20_9 = [item for sublist in loss_link_20_9 for item in sublist]   # 将二维列表转换为一维列表

estimate_link_count_20_10 = df_link_20_10.iloc[:,3].tolist()
target_count_20_10 = df_link_20_10.iloc[:,4].tolist()
loss_link_20_10 = list()
for i in range(len(target_count_20_10)):
    if target_count_20_10[i] != -1:
        loss_link_20_10.append(list(map(abs,[target_count_20_10[i] - estimate_link_count_20_10[i]])))
loss_link_20_10 = [item for sublist in loss_link_20_10 for item in sublist]   # 将二维列表转换为一维列表

loss_link_20 = (loss_link_20_1, loss_link_20_2, loss_link_20_3, loss_link_20_4, loss_link_20_5, loss_link_20_6, loss_link_20_7, loss_link_20_8, loss_link_20_9, loss_link_20_10)

box_link = plt.boxplot(loss_link_20, patch_artist = True,
                         labels = ['sample 1', 'sample 2', 'sample 3', 'sample 4', 'sample 5', 'sample 6', 'sample 7', 'sample 8', 'sample 9', 'sample 10'],
                        flierprops={'marker': 'o', 'markerfacecolor': 'w', 'markeredgecolor': 'red', 'alpha': 0.5},   # 异常值设置为蓝色
                         medianprops = {'linewidth': 1.5, 'color': 'red'})  # 一定要有patch_artist = True，否则不识别facecolor这个关键词

color = 'red'
for box in box_link['boxes']:
    box.set(facecolor='w', edgecolor='red', linestyle='dashed')

for whisker in box_link['whiskers']:
    whisker.set(color='red', linewidth=1)

for cap in box_link['caps']:
    cap.set(color='red', linewidth=1)

for median in box_link['medians']:
    median.set(color='red', linewidth=1.5)

# （2.2）将5个箱型图的中位线处连接成折线图：
median_loss_link_20_1 = np.median(loss_link_20_1)
median_loss_link_20_2 = np.median(loss_link_20_2)
median_loss_link_20_3 = np.median(loss_link_20_3)
median_loss_link_20_4 = np.median(loss_link_20_4)
median_loss_link_20_5 = np.median(loss_link_20_5)
median_loss_link_20_6 = np.median(loss_link_20_6)
median_loss_link_20_7 = np.median(loss_link_20_7)
median_loss_link_20_8 = np.median(loss_link_20_8)
median_loss_link_20_9 = np.median(loss_link_20_9)
median_loss_link_20_10 = np.median(loss_link_20_10)

median_loss_link_20 = np.array([median_loss_link_20_1, median_loss_link_20_2, median_loss_link_20_3, median_loss_link_20_4, median_loss_link_20_5, median_loss_link_20_6, median_loss_link_20_7, median_loss_link_20_8, median_loss_link_20_9, median_loss_link_20_10])

plt.plot(range(1, len(median_loss_link_20) + 1), median_loss_link_20, marker='o', markersize=10, linewidth=3.0, color='salmon')  # 绘制中位数折线图

plt.text(1.0, median_loss_link_20_1+0.3, 'Median: {:.5f}'.format(median_loss_link_20_1), fontproperties='Times New Roman', size=18, ha='center', va='bottom')     # 将箱型图的中位数标记出来
plt.text(2.0, median_loss_link_20_2-2.0, ' {:.5f}'.format(median_loss_link_20_2), fontproperties='Times New Roman', size=18, ha='center', va='bottom')     # 将箱型图的中位数标记出来
plt.text(3.0, median_loss_link_20_3+0.3, ' {:.5f}'.format(median_loss_link_20_3), fontproperties='Times New Roman', size=18, ha='center', va='bottom')     # 将箱型图的中位数标记出来
plt.text(4.0, median_loss_link_20_4-2.0, ' {:.5f}'.format(median_loss_link_20_4), fontproperties='Times New Roman', size=18, ha='center', va='bottom')     # 将箱型图的中位数标记出来
plt.text(5.0, median_loss_link_20_5+0.3, ' {:.5f}'.format(median_loss_link_20_5), fontproperties='Times New Roman', size=18, ha='center', va='bottom')     # 将箱型图的中位数标记出来
plt.text(6.0, median_loss_link_20_6-2.0, ' {:.5f}'.format(median_loss_link_20_6), fontproperties='Times New Roman', size=18, ha='center', va='bottom')     # 将箱型图的中位数标记出来
plt.text(7.0, median_loss_link_20_7+0.3, ' {:.5f}'.format(median_loss_link_20_7), fontproperties='Times New Roman', size=18, ha='center', va='bottom')     # 将箱型图的中位数标记出来
plt.text(8.0, median_loss_link_20_8-2.0, ' {:.5f}'.format(median_loss_link_20_8), fontproperties='Times New Roman', size=18, ha='center', va='bottom')     # 将箱型图的中位数标记出来
plt.text(9.0, median_loss_link_20_9+0.3, ' {:.5f}'.format(median_loss_link_20_9), fontproperties='Times New Roman', size=18, ha='center', va='bottom')     # 将箱型图的中位数标记出来
plt.text(10.0, median_loss_link_20_10-2.0, ' {:.5f}'.format(median_loss_link_20_10), fontproperties='Times New Roman', size=18, ha='center', va='bottom')     # 将箱型图的中位数标记出来

fig = plt.gcf()
fig.set_size_inches(8, 6)  # 设置保存图像的显示大小 宽度为 10英寸，高度为 8英寸
plt.xlabel('(a) 20% monitoring validation set', labelpad=7.5, fontdict={'family': 'Times New Roman', 'size': 18, 'color': 'k'})  # 改变横坐标轴标题字体  labelpad=10表示坐标标题距离坐标轴的垂直距离
plt.ylabel('Absolute Error of link flow', labelpad=7.5, fontdict={'family': 'Times New Roman', 'size': 18, 'color': 'k'})  # 改变纵坐标轴标题字体
plt.xticks(fontproperties='Times New Roman', size=18, rotation=20)
plt.yticks(fontproperties='Times New Roman', size=18)
handles = [Line2D([0], [0], color='salmon', marker='o', lw=2)]
labels = ['Median']   # 自定义图例标签
# plt.legend(handles=handles, labels=labels, loc='upper left')   # 添加图例
plt.legend(handles=handles, labels=labels, loc='upper left', prop={'family': 'Times New Roman', 'size': 18})   # 添加图例
plt.subplots_adjust(right=0.95, bottom=0.2, hspace=0.3, wspace=10.5)
plt.savefig('20% link validation set box.png', dpi=300, format='png')
plt.show()


# 2.2、绘制40%验证集，5次随机的absolute error
# (2.1) draw box plan:
df_link_40_1 = pd.read_csv(r'F:\SEU\MasterThesis\02BTCG_cross_SF64\15BTCG_cross_SF_all_5F_validation_v2_random10\output_link_40_1_link.csv')
df_link_40_2 = pd.read_csv(r'F:\SEU\MasterThesis\02BTCG_cross_SF64\15BTCG_cross_SF_all_5F_validation_v2_random10\output_link_40_2_link.csv')
df_link_40_3 = pd.read_csv(r'F:\SEU\MasterThesis\02BTCG_cross_SF64\15BTCG_cross_SF_all_5F_validation_v2_random10\output_link_40_3_link.csv')
df_link_40_4 = pd.read_csv(r'F:\SEU\MasterThesis\02BTCG_cross_SF64\15BTCG_cross_SF_all_5F_validation_v2_random10\output_link_40_4_link.csv')
df_link_40_5 = pd.read_csv(r'F:\SEU\MasterThesis\02BTCG_cross_SF64\15BTCG_cross_SF_all_5F_validation_v2_random10\output_link_40_5_link.csv')
df_link_40_6 = pd.read_csv(r'F:\SEU\MasterThesis\02BTCG_cross_SF64\15BTCG_cross_SF_all_5F_validation_v2_random10\output_link_40_6_link.csv')
df_link_40_7 = pd.read_csv(r'F:\SEU\MasterThesis\02BTCG_cross_SF64\15BTCG_cross_SF_all_5F_validation_v2_random10\output_link_40_7_link.csv')
df_link_40_8 = pd.read_csv(r'F:\SEU\MasterThesis\02BTCG_cross_SF64\15BTCG_cross_SF_all_5F_validation_v2_random10\output_link_40_8_link.csv')
df_link_40_9 = pd.read_csv(r'F:\SEU\MasterThesis\02BTCG_cross_SF64\15BTCG_cross_SF_all_5F_validation_v2_random10\output_link_40_9_link.csv')
df_link_40_10 = pd.read_csv(r'F:\SEU\MasterThesis\02BTCG_cross_SF64\15BTCG_cross_SF_all_5F_validation_v2_random10\output_link_40_10_link.csv')

estimate_link_count_40_1 = df_link_40_1.iloc[:,3].tolist()
target_count_40_1 = df_link_40_1.iloc[:,4].tolist()
loss_link_40_1 = list()
for i in range(len(target_count_40_1)):
    if target_count_40_1[i] != -1:
        loss_link_40_1.append(list(map(abs,[target_count_40_1[i] - estimate_link_count_40_1[i]])))
loss_link_40_1 = [item for sublist in loss_link_40_1 for item in sublist]   # 将二维列表转换为一维列表

estimate_link_count_40_2 = df_link_40_2.iloc[:,3].tolist()
target_count_40_2 = df_link_40_2.iloc[:,4].tolist()
loss_link_40_2 = list()
for i in range(len(target_count_40_2)):
    if target_count_40_2[i] != -1:
        loss_link_40_2.append(list(map(abs,[target_count_40_2[i] - estimate_link_count_40_2[i]])))
loss_link_40_2 = [item for sublist in loss_link_40_2 for item in sublist]   # 将二维列表转换为一维列表

estimate_link_count_40_3 = df_link_40_3.iloc[:,3].tolist()
target_count_40_3 = df_link_40_3.iloc[:,4].tolist()
loss_link_40_3 = list()
for i in range(len(target_count_40_3)):
    if target_count_40_3[i] != -1:
        loss_link_40_3.append(list(map(abs,[target_count_40_3[i] - estimate_link_count_40_3[i]])))
loss_link_40_3 = [item for sublist in loss_link_40_3 for item in sublist]   # 将二维列表转换为一维列表

estimate_link_count_40_4 = df_link_40_4.iloc[:,3].tolist()
target_count_40_4 = df_link_40_4.iloc[:,4].tolist()
loss_link_40_4 = list()
for i in range(len(target_count_40_4)):
    if target_count_40_4[i] != -1:
        loss_link_40_4.append(list(map(abs,[target_count_40_4[i] - estimate_link_count_40_4[i]])))
loss_link_40_4 = [item for sublist in loss_link_40_4 for item in sublist]   # 将二维列表转换为一维列表

estimate_link_count_40_5 = df_link_40_5.iloc[:,3].tolist()
target_count_40_5 = df_link_40_5.iloc[:,4].tolist()
loss_link_40_5 = list()
for i in range(len(target_count_40_5)):
    if target_count_40_5[i] != -1:
        loss_link_40_5.append(list(map(abs,[target_count_40_5[i] - estimate_link_count_40_5[i]])))
loss_link_40_5 = [item for sublist in loss_link_40_5 for item in sublist]   # 将二维列表转换为一维列表

estimate_link_count_40_6 = df_link_40_6.iloc[:,3].tolist()
target_count_40_6 = df_link_40_6.iloc[:,4].tolist()
loss_link_40_6 = list()
for i in range(len(target_count_40_6)):
    if target_count_40_6[i] != -1:
        loss_link_40_6.append(list(map(abs,[target_count_40_6[i] - estimate_link_count_40_6[i]])))
loss_link_40_6 = [item for sublist in loss_link_40_6 for item in sublist]   # 将二维列表转换为一维列表

estimate_link_count_40_7 = df_link_40_7.iloc[:,3].tolist()
target_count_40_7 = df_link_40_7.iloc[:,4].tolist()
loss_link_40_7 = list()
for i in range(len(target_count_40_7)):
    if target_count_40_7[i] != -1:
        loss_link_40_7.append(list(map(abs,[target_count_40_7[i] - estimate_link_count_40_7[i]])))
loss_link_40_7 = [item for sublist in loss_link_40_7 for item in sublist]   # 将二维列表转换为一维列表

estimate_link_count_40_8 = df_link_40_8.iloc[:,3].tolist()
target_count_40_8 = df_link_40_8.iloc[:,4].tolist()
loss_link_40_8 = list()
for i in range(len(target_count_40_8)):
    if target_count_40_8[i] != -1:
        loss_link_40_8.append(list(map(abs,[target_count_40_8[i] - estimate_link_count_40_8[i]])))
loss_link_40_8 = [item for sublist in loss_link_40_8 for item in sublist]   # 将二维列表转换为一维列表

estimate_link_count_40_9 = df_link_40_9.iloc[:,3].tolist()
target_count_40_9 = df_link_40_9.iloc[:,4].tolist()
loss_link_40_9 = list()
for i in range(len(target_count_40_9)):
    if target_count_40_9[i] != -1:
        loss_link_40_9.append(list(map(abs,[target_count_40_9[i] - estimate_link_count_40_9[i]])))
loss_link_40_9 = [item for sublist in loss_link_40_9 for item in sublist]   # 将二维列表转换为一维列表

estimate_link_count_40_10 = df_link_40_10.iloc[:,3].tolist()
target_count_40_10 = df_link_40_10.iloc[:,4].tolist()
loss_link_40_10 = list()
for i in range(len(target_count_40_10)):
    if target_count_40_10[i] != -1:
        loss_link_40_10.append(list(map(abs,[target_count_40_10[i] - estimate_link_count_40_10[i]])))
loss_link_40_10 = [item for sublist in loss_link_40_10 for item in sublist]   # 将二维列表转换为一维列表

loss_link_40 = (loss_link_40_1, loss_link_40_2, loss_link_40_3, loss_link_40_4, loss_link_40_5, loss_link_40_6, loss_link_40_7, loss_link_40_8, loss_link_40_9, loss_link_40_10)

box_link = plt.boxplot(loss_link_40, patch_artist = True,
                         labels = ['sample 1', 'sample 2', 'sample 3', 'sample 4', 'sample 5', 'sample 6', 'sample 7', 'sample 8', 'sample 9', 'sample 10'],
                        flierprops={'marker': 'o', 'markerfacecolor': 'w', 'markeredgecolor': 'red', 'alpha': 0.5},   # 异常值设置为蓝色
                         medianprops = {'linewidth': 1.5, 'color': 'red'})  # 一定要有patch_artist = True，否则不识别facecolor这个关键词

color = 'red'
for box in box_link['boxes']:
    box.set(facecolor='w', edgecolor='red', linestyle='dashed')

for whisker in box_link['whiskers']:
    whisker.set(color='red', linewidth=1)

for cap in box_link['caps']:
    cap.set(color='red', linewidth=1)

for median in box_link['medians']:
    median.set(color='red', linewidth=1.5)

# （2.2）将5个箱型图的中位线处连接成折线图：
median_loss_link_40_1 = np.median(loss_link_40_1)
median_loss_link_40_2 = np.median(loss_link_40_2)
median_loss_link_40_3 = np.median(loss_link_40_3)
median_loss_link_40_4 = np.median(loss_link_40_4)
median_loss_link_40_5 = np.median(loss_link_40_5)
median_loss_link_40_6 = np.median(loss_link_40_6)
median_loss_link_40_7 = np.median(loss_link_40_7)
median_loss_link_40_8 = np.median(loss_link_40_8)
median_loss_link_40_9 = np.median(loss_link_40_9)
median_loss_link_40_10 = np.median(loss_link_40_10)

median_loss_link_40 = np.array([median_loss_link_40_1, median_loss_link_40_2, median_loss_link_40_3, median_loss_link_40_4, median_loss_link_40_5, median_loss_link_40_6, median_loss_link_40_7, median_loss_link_40_8, median_loss_link_40_9, median_loss_link_40_10])

plt.plot(range(1, len(median_loss_link_40) + 1), median_loss_link_40, marker='o', markersize=10, linewidth=3.0, color='salmon')  # 绘制中位数折线图

plt.text(1.0, median_loss_link_40_1+0.3, 'Median: {:.5f}'.format(median_loss_link_40_1), fontproperties='Times New Roman', size=18, ha='center', va='bottom')     # 将箱型图的中位数标记出来
plt.text(2.0, median_loss_link_40_2-2.0, ' {:.5f}'.format(median_loss_link_40_2), fontproperties='Times New Roman', size=18, ha='center', va='bottom')     # 将箱型图的中位数标记出来
plt.text(3.0, median_loss_link_40_3+0.3, ' {:.5f}'.format(median_loss_link_40_3), fontproperties='Times New Roman', size=18, ha='center', va='bottom')     # 将箱型图的中位数标记出来
plt.text(4.0, median_loss_link_40_4-2.0, ' {:.5f}'.format(median_loss_link_40_4), fontproperties='Times New Roman', size=18, ha='center', va='bottom')     # 将箱型图的中位数标记出来
plt.text(5.0, median_loss_link_40_5+0.3, ' {:.5f}'.format(median_loss_link_40_5), fontproperties='Times New Roman', size=18, ha='center', va='bottom')     # 将箱型图的中位数标记出来
plt.text(6.0, median_loss_link_40_6-2.0, ' {:.5f}'.format(median_loss_link_40_6), fontproperties='Times New Roman', size=18, ha='center', va='bottom')     # 将箱型图的中位数标记出来
plt.text(7.0, median_loss_link_40_7+0.3, ' {:.5f}'.format(median_loss_link_40_7), fontproperties='Times New Roman', size=18, ha='center', va='bottom')     # 将箱型图的中位数标记出来
plt.text(8.0, median_loss_link_40_8-2.0, ' {:.5f}'.format(median_loss_link_40_8), fontproperties='Times New Roman', size=18, ha='center', va='bottom')     # 将箱型图的中位数标记出来
plt.text(9.0, median_loss_link_40_9+0.3, ' {:.5f}'.format(median_loss_link_40_9), fontproperties='Times New Roman', size=18, ha='center', va='bottom')     # 将箱型图的中位数标记出来
plt.text(10.0, median_loss_link_40_10-2.0, ' {:.5f}'.format(median_loss_link_40_10), fontproperties='Times New Roman', size=18, ha='center', va='bottom')     # 将箱型图的中位数标记出来

fig = plt.gcf()
fig.set_size_inches(8, 6)  # 设置保存图像的显示大小 宽度为 10英寸，高度为 8英寸
plt.xlabel('(b) 40% monitoring validation set', labelpad=7.5, fontdict={'family': 'Times New Roman', 'size': 18, 'color': 'k'})  # 改变横坐标轴标题字体  labelpad=10表示坐标标题距离坐标轴的垂直距离
plt.ylabel('Absolute Error of link flow', labelpad=7.5, fontdict={'family': 'Times New Roman', 'size': 18, 'color': 'k'})  # 改变纵坐标轴标题字体
plt.xticks(fontproperties='Times New Roman', size=18, rotation=20)
plt.yticks(fontproperties='Times New Roman', size=18)
handles = [Line2D([0], [0], color='salmon', marker='o', lw=2)]
labels = ['Median']   # 自定义图例标签
# plt.legend(handles=handles, labels=labels, loc='upper left')   # 添加图例
plt.legend(handles=handles, labels=labels, loc='upper left', prop={'family': 'Times New Roman', 'size': 18})   # 添加图例
plt.subplots_adjust(right=0.95, bottom=0.2, hspace=0.3, wspace=10.5)
plt.savefig('40% link validation set box.png', dpi=300, format='png')
plt.show()


# 2.2、绘制60%验证集，5次随机的absolute error
# (2.1) draw box plan:
df_link_60_1 = pd.read_csv(r'F:\SEU\MasterThesis\02BTCG_cross_SF64\15BTCG_cross_SF_all_5F_validation_v2_random10\output_link_60_1_link.csv')
df_link_60_2 = pd.read_csv(r'F:\SEU\MasterThesis\02BTCG_cross_SF64\15BTCG_cross_SF_all_5F_validation_v2_random10\output_link_60_2_link.csv')
df_link_60_3 = pd.read_csv(r'F:\SEU\MasterThesis\02BTCG_cross_SF64\15BTCG_cross_SF_all_5F_validation_v2_random10\output_link_60_3_link.csv')
df_link_60_4 = pd.read_csv(r'F:\SEU\MasterThesis\02BTCG_cross_SF64\15BTCG_cross_SF_all_5F_validation_v2_random10\output_link_60_4_link.csv')
df_link_60_5 = pd.read_csv(r'F:\SEU\MasterThesis\02BTCG_cross_SF64\15BTCG_cross_SF_all_5F_validation_v2_random10\output_link_60_5_link.csv')
df_link_60_6 = pd.read_csv(r'F:\SEU\MasterThesis\02BTCG_cross_SF64\15BTCG_cross_SF_all_5F_validation_v2_random10\output_link_60_6_link.csv')
df_link_60_7 = pd.read_csv(r'F:\SEU\MasterThesis\02BTCG_cross_SF64\15BTCG_cross_SF_all_5F_validation_v2_random10\output_link_60_7_link.csv')
df_link_60_8 = pd.read_csv(r'F:\SEU\MasterThesis\02BTCG_cross_SF64\15BTCG_cross_SF_all_5F_validation_v2_random10\output_link_60_8_link.csv')
df_link_60_9 = pd.read_csv(r'F:\SEU\MasterThesis\02BTCG_cross_SF64\15BTCG_cross_SF_all_5F_validation_v2_random10\output_link_60_9_link.csv')
df_link_60_10 = pd.read_csv(r'F:\SEU\MasterThesis\02BTCG_cross_SF64\15BTCG_cross_SF_all_5F_validation_v2_random10\output_link_60_10_link.csv')

estimate_link_count_60_1 = df_link_60_1.iloc[:,3].tolist()
target_count_60_1 = df_link_60_1.iloc[:,4].tolist()
loss_link_60_1 = list()
for i in range(len(target_count_60_1)):
    if target_count_60_1[i] != -1:
        loss_link_60_1.append(list(map(abs,[target_count_60_1[i] - estimate_link_count_60_1[i]])))
loss_link_60_1 = [item for sublist in loss_link_60_1 for item in sublist]   # 将二维列表转换为一维列表

estimate_link_count_60_2 = df_link_60_2.iloc[:,3].tolist()
target_count_60_2 = df_link_60_2.iloc[:,4].tolist()
loss_link_60_2 = list()
for i in range(len(target_count_60_2)):
    if target_count_60_2[i] != -1:
        loss_link_60_2.append(list(map(abs,[target_count_60_2[i] - estimate_link_count_60_2[i]])))
loss_link_60_2 = [item for sublist in loss_link_60_2 for item in sublist]   # 将二维列表转换为一维列表

estimate_link_count_60_3 = df_link_60_3.iloc[:,3].tolist()
target_count_60_3 = df_link_60_3.iloc[:,4].tolist()
loss_link_60_3 = list()
for i in range(len(target_count_60_3)):
    if target_count_60_3[i] != -1:
        loss_link_60_3.append(list(map(abs,[target_count_60_3[i] - estimate_link_count_60_3[i]])))
loss_link_60_3 = [item for sublist in loss_link_60_3 for item in sublist]   # 将二维列表转换为一维列表

estimate_link_count_60_4 = df_link_60_4.iloc[:,3].tolist()
target_count_60_4 = df_link_60_4.iloc[:,4].tolist()
loss_link_60_4 = list()
for i in range(len(target_count_60_4)):
    if target_count_60_4[i] != -1:
        loss_link_60_4.append(list(map(abs,[target_count_60_4[i] - estimate_link_count_60_4[i]])))
loss_link_60_4 = [item for sublist in loss_link_60_4 for item in sublist]   # 将二维列表转换为一维列表

estimate_link_count_60_5 = df_link_60_5.iloc[:,3].tolist()
target_count_60_5 = df_link_60_5.iloc[:,4].tolist()
loss_link_60_5 = list()
for i in range(len(target_count_60_5)):
    if target_count_60_5[i] != -1:
        loss_link_60_5.append(list(map(abs,[target_count_60_5[i] - estimate_link_count_60_5[i]])))
loss_link_60_5 = [item for sublist in loss_link_60_5 for item in sublist]   # 将二维列表转换为一维列表

estimate_link_count_60_6 = df_link_60_6.iloc[:,3].tolist()
target_count_60_6 = df_link_60_6.iloc[:,4].tolist()
loss_link_60_6 = list()
for i in range(len(target_count_60_6)):
    if target_count_60_6[i] != -1:
        loss_link_60_6.append(list(map(abs,[target_count_60_6[i] - estimate_link_count_60_6[i]])))
loss_link_60_6 = [item for sublist in loss_link_60_6 for item in sublist]   # 将二维列表转换为一维列表

estimate_link_count_60_7 = df_link_60_7.iloc[:,3].tolist()
target_count_60_7 = df_link_60_7.iloc[:,4].tolist()
loss_link_60_7 = list()
for i in range(len(target_count_60_7)):
    if target_count_60_7[i] != -1:
        loss_link_60_7.append(list(map(abs,[target_count_60_7[i] - estimate_link_count_60_7[i]])))
loss_link_60_7 = [item for sublist in loss_link_60_7 for item in sublist]   # 将二维列表转换为一维列表

estimate_link_count_60_8 = df_link_60_8.iloc[:,3].tolist()
target_count_60_8 = df_link_60_8.iloc[:,4].tolist()
loss_link_60_8 = list()
for i in range(len(target_count_60_8)):
    if target_count_60_8[i] != -1:
        loss_link_60_8.append(list(map(abs,[target_count_60_8[i] - estimate_link_count_60_8[i]])))
loss_link_60_8 = [item for sublist in loss_link_60_8 for item in sublist]   # 将二维列表转换为一维列表

estimate_link_count_60_9 = df_link_60_9.iloc[:,3].tolist()
target_count_60_9 = df_link_60_9.iloc[:,4].tolist()
loss_link_60_9 = list()
for i in range(len(target_count_60_9)):
    if target_count_60_9[i] != -1:
        loss_link_60_9.append(list(map(abs,[target_count_60_9[i] - estimate_link_count_60_9[i]])))
loss_link_60_9 = [item for sublist in loss_link_60_9 for item in sublist]   # 将二维列表转换为一维列表

estimate_link_count_60_10 = df_link_60_10.iloc[:,3].tolist()
target_count_60_10 = df_link_60_10.iloc[:,4].tolist()
loss_link_60_10 = list()
for i in range(len(target_count_60_10)):
    if target_count_60_10[i] != -1:
        loss_link_60_10.append(list(map(abs,[target_count_60_10[i] - estimate_link_count_60_10[i]])))
loss_link_60_10 = [item for sublist in loss_link_60_10 for item in sublist]   # 将二维列表转换为一维列表

loss_link_60 = (loss_link_60_1, loss_link_60_2, loss_link_60_3, loss_link_60_4, loss_link_60_5, loss_link_60_6, loss_link_60_7, loss_link_60_8, loss_link_60_9, loss_link_60_10)

box_link = plt.boxplot(loss_link_60, patch_artist = True,
                         labels = ['sample 1', 'sample 2', 'sample 3', 'sample 4', 'sample 5', 'sample 6', 'sample 7', 'sample 8', 'sample 9', 'sample 10'],
                        flierprops={'marker': 'o', 'markerfacecolor': 'w', 'markeredgecolor': 'red', 'alpha': 0.5},   # 异常值设置为蓝色
                         medianprops = {'linewidth': 1.5, 'color': 'red'})  # 一定要有patch_artist = True，否则不识别facecolor这个关键词

color = 'red'
for box in box_link['boxes']:
    box.set(facecolor='w', edgecolor='red', linestyle='dashed')

for whisker in box_link['whiskers']:
    whisker.set(color='red', linewidth=1)

for cap in box_link['caps']:
    cap.set(color='red', linewidth=1)

for median in box_link['medians']:
    median.set(color='red', linewidth=1.5)

# （2.2）将5个箱型图的中位线处连接成折线图：
median_loss_link_60_1 = np.median(loss_link_60_1)
median_loss_link_60_2 = np.median(loss_link_60_2)
median_loss_link_60_3 = np.median(loss_link_60_3)
median_loss_link_60_4 = np.median(loss_link_60_4)
median_loss_link_60_5 = np.median(loss_link_60_5)
median_loss_link_60_6 = np.median(loss_link_60_6)
median_loss_link_60_7 = np.median(loss_link_60_7)
median_loss_link_60_8 = np.median(loss_link_60_8)
median_loss_link_60_9 = np.median(loss_link_60_9)
median_loss_link_60_10 = np.median(loss_link_60_10)

median_loss_link_60 = np.array([median_loss_link_60_1, median_loss_link_60_2, median_loss_link_60_3, median_loss_link_60_4, median_loss_link_60_5, median_loss_link_60_6, median_loss_link_60_7, median_loss_link_60_8, median_loss_link_60_9, median_loss_link_60_10])

plt.plot(range(1, len(median_loss_link_60) + 1), median_loss_link_60, marker='o', markersize=10, linewidth=3.0, color='salmon')  # 绘制中位数折线图

plt.text(1.0, median_loss_link_60_1+0.3, 'Median: {:.5f}'.format(median_loss_link_60_1), fontproperties='Times New Roman', size=18, ha='center', va='bottom')     # 将箱型图的中位数标记出来
plt.text(2.0, median_loss_link_60_2-2.0, ' {:.5f}'.format(median_loss_link_60_2), fontproperties='Times New Roman', size=18, ha='center', va='bottom')     # 将箱型图的中位数标记出来
plt.text(3.0, median_loss_link_60_3+0.3, ' {:.5f}'.format(median_loss_link_60_3), fontproperties='Times New Roman', size=18, ha='center', va='bottom')     # 将箱型图的中位数标记出来
plt.text(4.0, median_loss_link_60_4-2.0, ' {:.5f}'.format(median_loss_link_60_4), fontproperties='Times New Roman', size=18, ha='center', va='bottom')     # 将箱型图的中位数标记出来
plt.text(5.0, median_loss_link_60_5+0.3, ' {:.5f}'.format(median_loss_link_60_5), fontproperties='Times New Roman', size=18, ha='center', va='bottom')     # 将箱型图的中位数标记出来
plt.text(6.0, median_loss_link_60_6-2.0, ' {:.5f}'.format(median_loss_link_60_6), fontproperties='Times New Roman', size=18, ha='center', va='bottom')     # 将箱型图的中位数标记出来
plt.text(7.0, median_loss_link_60_7+0.3, ' {:.5f}'.format(median_loss_link_60_7), fontproperties='Times New Roman', size=18, ha='center', va='bottom')     # 将箱型图的中位数标记出来
plt.text(8.0, median_loss_link_60_8-2.0, ' {:.5f}'.format(median_loss_link_60_8), fontproperties='Times New Roman', size=18, ha='center', va='bottom')     # 将箱型图的中位数标记出来
plt.text(9.0, median_loss_link_60_9+0.3, ' {:.5f}'.format(median_loss_link_60_9), fontproperties='Times New Roman', size=18, ha='center', va='bottom')     # 将箱型图的中位数标记出来
plt.text(10.0, median_loss_link_60_10-2.0, ' {:.5f}'.format(median_loss_link_60_10), fontproperties='Times New Roman', size=18, ha='center', va='bottom')     # 将箱型图的中位数标记出来

fig = plt.gcf()
fig.set_size_inches(8, 6)  # 设置保存图像的显示大小 宽度为 10英寸，高度为 8英寸
plt.xlabel('(c) 60% monitoring validation set', labelpad=7.5, fontdict={'family': 'Times New Roman', 'size': 18, 'color': 'k'})  # 改变横坐标轴标题字体  labelpad=10表示坐标标题距离坐标轴的垂直距离
plt.ylabel('Absolute Error of link flow', labelpad=7.5, fontdict={'family': 'Times New Roman', 'size': 18, 'color': 'k'})  # 改变纵坐标轴标题字体
plt.xticks(fontproperties='Times New Roman', size=18, rotation=20)
plt.yticks(fontproperties='Times New Roman', size=18)
handles = [Line2D([0], [0], color='salmon', marker='o', lw=2)]
labels = ['Median']   # 自定义图例标签
# plt.legend(handles=handles, labels=labels, loc='upper left')   # 添加图例
plt.legend(handles=handles, labels=labels, loc='upper left', prop={'family': 'Times New Roman', 'size': 18})   # 添加图例
plt.subplots_adjust(right=0.95, bottom=0.2, hspace=0.3, wspace=10.5)
plt.savefig('60% link validation set box.png', dpi=300, format='png')
plt.show()


# 2.4、绘制80%验证集，5次随机的absolute error
# (2.1) draw box plan:
df_link_80_1 = pd.read_csv(r'F:\SEU\MasterThesis\02BTCG_cross_SF64\15BTCG_cross_SF_all_5F_validation_v2_random10\output_link_80_1_link.csv')
df_link_80_2 = pd.read_csv(r'F:\SEU\MasterThesis\02BTCG_cross_SF64\15BTCG_cross_SF_all_5F_validation_v2_random10\output_link_80_2_link.csv')
df_link_80_3 = pd.read_csv(r'F:\SEU\MasterThesis\02BTCG_cross_SF64\15BTCG_cross_SF_all_5F_validation_v2_random10\output_link_80_3_link.csv')
df_link_80_4 = pd.read_csv(r'F:\SEU\MasterThesis\02BTCG_cross_SF64\15BTCG_cross_SF_all_5F_validation_v2_random10\output_link_80_4_link.csv')
df_link_80_5 = pd.read_csv(r'F:\SEU\MasterThesis\02BTCG_cross_SF64\15BTCG_cross_SF_all_5F_validation_v2_random10\output_link_80_5_link.csv')
df_link_80_6 = pd.read_csv(r'F:\SEU\MasterThesis\02BTCG_cross_SF64\15BTCG_cross_SF_all_5F_validation_v2_random10\output_link_80_6_link.csv')
df_link_80_7 = pd.read_csv(r'F:\SEU\MasterThesis\02BTCG_cross_SF64\15BTCG_cross_SF_all_5F_validation_v2_random10\output_link_80_7_link.csv')
df_link_80_8 = pd.read_csv(r'F:\SEU\MasterThesis\02BTCG_cross_SF64\15BTCG_cross_SF_all_5F_validation_v2_random10\output_link_80_8_link.csv')
df_link_80_9 = pd.read_csv(r'F:\SEU\MasterThesis\02BTCG_cross_SF64\15BTCG_cross_SF_all_5F_validation_v2_random10\output_link_80_9_link.csv')
df_link_80_10 = pd.read_csv(r'F:\SEU\MasterThesis\02BTCG_cross_SF64\15BTCG_cross_SF_all_5F_validation_v2_random10\output_link_80_10_link.csv')

estimate_link_count_80_1 = df_link_80_1.iloc[:,3].tolist()
target_count_80_1 = df_link_80_1.iloc[:,4].tolist()
loss_link_80_1 = list()
for i in range(len(target_count_80_1)):
    if target_count_80_1[i] != -1:
        loss_link_80_1.append(list(map(abs,[target_count_80_1[i] - estimate_link_count_80_1[i]])))
loss_link_80_1 = [item for sublist in loss_link_80_1 for item in sublist]   # 将二维列表转换为一维列表

estimate_link_count_80_2 = df_link_80_2.iloc[:,3].tolist()
target_count_80_2 = df_link_80_2.iloc[:,4].tolist()
loss_link_80_2 = list()
for i in range(len(target_count_80_2)):
    if target_count_80_2[i] != -1:
        loss_link_80_2.append(list(map(abs,[target_count_80_2[i] - estimate_link_count_80_2[i]])))
loss_link_80_2 = [item for sublist in loss_link_80_2 for item in sublist]   # 将二维列表转换为一维列表

estimate_link_count_80_3 = df_link_80_3.iloc[:,3].tolist()
target_count_80_3 = df_link_80_3.iloc[:,4].tolist()
loss_link_80_3 = list()
for i in range(len(target_count_80_3)):
    if target_count_80_3[i] != -1:
        loss_link_80_3.append(list(map(abs,[target_count_80_3[i] - estimate_link_count_80_3[i]])))
loss_link_80_3 = [item for sublist in loss_link_80_3 for item in sublist]   # 将二维列表转换为一维列表

estimate_link_count_80_4 = df_link_80_4.iloc[:,3].tolist()
target_count_80_4 = df_link_80_4.iloc[:,4].tolist()
loss_link_80_4 = list()
for i in range(len(target_count_80_4)):
    if target_count_80_4[i] != -1:
        loss_link_80_4.append(list(map(abs,[target_count_80_4[i] - estimate_link_count_80_4[i]])))
loss_link_80_4 = [item for sublist in loss_link_80_4 for item in sublist]   # 将二维列表转换为一维列表

estimate_link_count_80_5 = df_link_80_5.iloc[:,3].tolist()
target_count_80_5 = df_link_80_5.iloc[:,4].tolist()
loss_link_80_5 = list()
for i in range(len(target_count_80_5)):
    if target_count_80_5[i] != -1:
        loss_link_80_5.append(list(map(abs,[target_count_80_5[i] - estimate_link_count_80_5[i]])))
loss_link_80_5 = [item for sublist in loss_link_80_5 for item in sublist]   # 将二维列表转换为一维列表

estimate_link_count_80_6 = df_link_80_6.iloc[:,3].tolist()
target_count_80_6 = df_link_80_6.iloc[:,4].tolist()
loss_link_80_6 = list()
for i in range(len(target_count_80_6)):
    if target_count_80_6[i] != -1:
        loss_link_80_6.append(list(map(abs,[target_count_80_6[i] - estimate_link_count_80_6[i]])))
loss_link_80_6 = [item for sublist in loss_link_80_6 for item in sublist]   # 将二维列表转换为一维列表

estimate_link_count_80_7 = df_link_80_7.iloc[:,3].tolist()
target_count_80_7 = df_link_80_7.iloc[:,4].tolist()
loss_link_80_7 = list()
for i in range(len(target_count_80_7)):
    if target_count_80_7[i] != -1:
        loss_link_80_7.append(list(map(abs,[target_count_80_7[i] - estimate_link_count_80_7[i]])))
loss_link_80_7 = [item for sublist in loss_link_80_7 for item in sublist]   # 将二维列表转换为一维列表

estimate_link_count_80_8 = df_link_80_8.iloc[:,3].tolist()
target_count_80_8 = df_link_80_8.iloc[:,4].tolist()
loss_link_80_8 = list()
for i in range(len(target_count_80_8)):
    if target_count_80_8[i] != -1:
        loss_link_80_8.append(list(map(abs,[target_count_80_8[i] - estimate_link_count_80_8[i]])))
loss_link_80_8 = [item for sublist in loss_link_80_8 for item in sublist]   # 将二维列表转换为一维列表

estimate_link_count_80_9 = df_link_80_9.iloc[:,3].tolist()
target_count_80_9 = df_link_80_9.iloc[:,4].tolist()
loss_link_80_9 = list()
for i in range(len(target_count_80_9)):
    if target_count_80_9[i] != -1:
        loss_link_80_9.append(list(map(abs,[target_count_80_9[i] - estimate_link_count_80_9[i]])))
loss_link_80_9 = [item for sublist in loss_link_80_9 for item in sublist]   # 将二维列表转换为一维列表

estimate_link_count_80_10 = df_link_80_10.iloc[:,3].tolist()
target_count_80_10 = df_link_80_10.iloc[:,4].tolist()
loss_link_80_10 = list()
for i in range(len(target_count_80_10)):
    if target_count_80_10[i] != -1:
        loss_link_80_10.append(list(map(abs,[target_count_80_10[i] - estimate_link_count_80_10[i]])))
loss_link_80_10 = [item for sublist in loss_link_80_10 for item in sublist]   # 将二维列表转换为一维列表

loss_link_80 = (loss_link_80_1, loss_link_80_2, loss_link_80_3, loss_link_80_4, loss_link_80_5, loss_link_80_6, loss_link_80_7, loss_link_80_8, loss_link_80_9, loss_link_80_10)

box_link = plt.boxplot(loss_link_80, patch_artist = True,
                         labels = ['sample 1', 'sample 2', 'sample 3', 'sample 4', 'sample 5', 'sample 6', 'sample 7', 'sample 8', 'sample 9', 'sample 10'],
                        flierprops={'marker': 'o', 'markerfacecolor': 'w', 'markeredgecolor': 'red', 'alpha': 0.5},   # 异常值设置为蓝色
                         medianprops = {'linewidth': 1.5, 'color': 'red'})  # 一定要有patch_artist = True，否则不识别facecolor这个关键词

color = 'red'
for box in box_link['boxes']:
    box.set(facecolor='w', edgecolor='red', linestyle='dashed')

for whisker in box_link['whiskers']:
    whisker.set(color='red', linewidth=1)

for cap in box_link['caps']:
    cap.set(color='red', linewidth=1)

for median in box_link['medians']:
    median.set(color='red', linewidth=1.5)

# （2.2）将5个箱型图的中位线处连接成折线图：
median_loss_link_80_1 = np.median(loss_link_80_1)
median_loss_link_80_2 = np.median(loss_link_80_2)
median_loss_link_80_3 = np.median(loss_link_80_3)
median_loss_link_80_4 = np.median(loss_link_80_4)
median_loss_link_80_5 = np.median(loss_link_80_5)
median_loss_link_80_6 = np.median(loss_link_80_6)
median_loss_link_80_7 = np.median(loss_link_80_7)
median_loss_link_80_8 = np.median(loss_link_80_8)
median_loss_link_80_9 = np.median(loss_link_80_9)
median_loss_link_80_10 = np.median(loss_link_80_10)

median_loss_link_80 = np.array([median_loss_link_80_1, median_loss_link_80_2, median_loss_link_80_3, median_loss_link_80_4, median_loss_link_80_5, median_loss_link_80_6, median_loss_link_80_7, median_loss_link_80_8, median_loss_link_80_9, median_loss_link_80_10])

plt.plot(range(1, len(median_loss_link_80) + 1), median_loss_link_80, marker='o', markersize=10, linewidth=3.0, color='salmon')  # 绘制中位数折线图

plt.text(1.0, median_loss_link_80_1+0.3, 'Median: {:.5f}'.format(median_loss_link_80_1), fontproperties='Times New Roman', size=18, ha='center', va='bottom')     # 将箱型图的中位数标记出来
plt.text(2.0, median_loss_link_80_2-2.0, ' {:.5f}'.format(median_loss_link_80_2), fontproperties='Times New Roman', size=18, ha='center', va='bottom')     # 将箱型图的中位数标记出来
plt.text(3.0, median_loss_link_80_3+0.3, ' {:.5f}'.format(median_loss_link_80_3), fontproperties='Times New Roman', size=18, ha='center', va='bottom')     # 将箱型图的中位数标记出来
plt.text(4.0, median_loss_link_80_4-2.0, ' {:.5f}'.format(median_loss_link_80_4), fontproperties='Times New Roman', size=18, ha='center', va='bottom')     # 将箱型图的中位数标记出来
plt.text(5.0, median_loss_link_80_5+0.3, ' {:.5f}'.format(median_loss_link_80_5), fontproperties='Times New Roman', size=18, ha='center', va='bottom')     # 将箱型图的中位数标记出来
plt.text(6.0, median_loss_link_80_6-2.0, ' {:.5f}'.format(median_loss_link_80_6), fontproperties='Times New Roman', size=18, ha='center', va='bottom')     # 将箱型图的中位数标记出来
plt.text(7.0, median_loss_link_80_7+0.3, ' {:.5f}'.format(median_loss_link_80_7), fontproperties='Times New Roman', size=18, ha='center', va='bottom')     # 将箱型图的中位数标记出来
plt.text(8.0, median_loss_link_80_8-2.0, ' {:.5f}'.format(median_loss_link_80_8), fontproperties='Times New Roman', size=18, ha='center', va='bottom')     # 将箱型图的中位数标记出来
plt.text(9.0, median_loss_link_80_9+0.3, ' {:.5f}'.format(median_loss_link_80_9), fontproperties='Times New Roman', size=18, ha='center', va='bottom')     # 将箱型图的中位数标记出来
plt.text(10.0, median_loss_link_80_10-2.0, ' {:.5f}'.format(median_loss_link_80_10), fontproperties='Times New Roman', size=18, ha='center', va='bottom')     # 将箱型图的中位数标记出来

fig = plt.gcf()
fig.set_size_inches(8, 6)  # 设置保存图像的显示大小 宽度为 10英寸，高度为 8英寸
plt.xlabel('(d) 80% monitoring validation set', labelpad=7.5, fontdict={'family': 'Times New Roman', 'size': 18, 'color': 'k'})  # 改变横坐标轴标题字体  labelpad=10表示坐标标题距离坐标轴的垂直距离
plt.ylabel('Absolute Error of link flow', labelpad=7.5, fontdict={'family': 'Times New Roman', 'size': 18, 'color': 'k'})  # 改变纵坐标轴标题字体
plt.xticks(fontproperties='Times New Roman', size=18, rotation=20)
plt.yticks(fontproperties='Times New Roman', size=18)
handles = [Line2D([0], [0], color='salmon', marker='o', lw=2)]
labels = ['Median']   # 自定义图例标签
# plt.legend(handles=handles, labels=labels, loc='upper left')   # 添加图例
plt.legend(handles=handles, labels=labels, loc='upper left', prop={'family': 'Times New Roman', 'size': 18})   # 添加图例
plt.subplots_adjust(right=0.95, bottom=0.2, hspace=0.3, wspace=10.5)
plt.savefig('80% link validation set box.png', dpi=300, format='png')
plt.show()


# 3、计算0%、20%、40%、60%、80%验证集 5次的平均值，分别绘制5个箱型图，每个箱型图是eg.20%的5个平均值，横坐标是0%、20%、40%、60%、80%，纵坐标是AE平均值
# （3.1）calculate the AE average of 0%
df_link_00 = pd.read_csv(r'F:\SEU\MasterThesis\02BTCG_cross_SF64\14BTCG_cross_SF_all_5F\output_link.csv')

estimate_link_count_00 = df_link_00.iloc[:,3].tolist()
target_count_00 = df_link_00.iloc[:,4].tolist()
loss_link_00 = list(map(abs,[target_count_00[i] - estimate_link_count_00[i] for i in range(0,len(target_count_00))]))

ave_loss_link_00 = sum(loss_link_00) / len(loss_link_00)

ave_loss_link_20_1 = sum(loss_link_20_1) / len(loss_link_20_1)
ave_loss_link_20_2 = sum(loss_link_20_2) / len(loss_link_20_2)
ave_loss_link_20_3 = sum(loss_link_20_3) / len(loss_link_20_3)
ave_loss_link_20_4 = sum(loss_link_20_4) / len(loss_link_20_4)
ave_loss_link_20_5 = sum(loss_link_20_5) / len(loss_link_20_5)
ave_loss_link_20_6 = sum(loss_link_20_6) / len(loss_link_20_6)
ave_loss_link_20_7 = sum(loss_link_20_7) / len(loss_link_20_7)
ave_loss_link_20_8 = sum(loss_link_20_8) / len(loss_link_20_8)
ave_loss_link_20_9 = sum(loss_link_20_9) / len(loss_link_20_9)
ave_loss_link_20_10 = sum(loss_link_20_10) / len(loss_link_20_10)
ave_loss_link_20 = [ave_loss_link_20_1, ave_loss_link_20_2, ave_loss_link_20_3, ave_loss_link_20_4, ave_loss_link_20_5, ave_loss_link_20_6, ave_loss_link_20_7, ave_loss_link_20_8, ave_loss_link_20_9, ave_loss_link_20_10]

ave_loss_link_40_1 = sum(loss_link_40_1) / len(loss_link_40_1)
ave_loss_link_40_2 = sum(loss_link_40_2) / len(loss_link_40_2)
ave_loss_link_40_3 = sum(loss_link_40_3) / len(loss_link_40_3)
ave_loss_link_40_4 = sum(loss_link_40_4) / len(loss_link_40_4)
ave_loss_link_40_5 = sum(loss_link_40_5) / len(loss_link_40_5)
ave_loss_link_40_6 = sum(loss_link_40_6) / len(loss_link_40_6)
ave_loss_link_40_7 = sum(loss_link_40_7) / len(loss_link_40_7)
ave_loss_link_40_8 = sum(loss_link_40_8) / len(loss_link_40_8)
ave_loss_link_40_9 = sum(loss_link_40_9) / len(loss_link_40_9)
ave_loss_link_40_10 = sum(loss_link_40_10) / len(loss_link_40_10)
ave_loss_link_40 = [ave_loss_link_40_1, ave_loss_link_40_2, ave_loss_link_40_3, ave_loss_link_40_4, ave_loss_link_40_5, ave_loss_link_40_6, ave_loss_link_40_7, ave_loss_link_40_8, ave_loss_link_40_9, ave_loss_link_40_10]

ave_loss_link_60_1 = sum(loss_link_60_1) / len(loss_link_60_1)
ave_loss_link_60_2 = sum(loss_link_60_2) / len(loss_link_60_2)
ave_loss_link_60_3 = sum(loss_link_60_3) / len(loss_link_60_3)
ave_loss_link_60_4 = sum(loss_link_60_4) / len(loss_link_60_4)
ave_loss_link_60_5 = sum(loss_link_60_5) / len(loss_link_60_5)
ave_loss_link_60_6 = sum(loss_link_60_6) / len(loss_link_60_6)
ave_loss_link_60_7 = sum(loss_link_60_7) / len(loss_link_60_7)
ave_loss_link_60_8 = sum(loss_link_60_8) / len(loss_link_60_8)
ave_loss_link_60_9 = sum(loss_link_60_9) / len(loss_link_60_9)
ave_loss_link_60_10 = sum(loss_link_60_10) / len(loss_link_60_10)
ave_loss_link_60 = [ave_loss_link_60_1, ave_loss_link_60_2, ave_loss_link_60_3, ave_loss_link_60_4, ave_loss_link_60_5, ave_loss_link_60_6, ave_loss_link_60_7, ave_loss_link_60_8, ave_loss_link_60_9, ave_loss_link_60_10]

ave_loss_link_80_1 = sum(loss_link_80_1) / len(loss_link_80_1)
ave_loss_link_80_2 = sum(loss_link_80_2) / len(loss_link_80_2)
ave_loss_link_80_3 = sum(loss_link_80_3) / len(loss_link_80_3)
ave_loss_link_80_4 = sum(loss_link_80_4) / len(loss_link_80_4)
ave_loss_link_80_5 = sum(loss_link_80_5) / len(loss_link_80_5)
ave_loss_link_80_6 = sum(loss_link_80_6) / len(loss_link_80_6)
ave_loss_link_80_7 = sum(loss_link_80_7) / len(loss_link_80_7)
ave_loss_link_80_8 = sum(loss_link_80_8) / len(loss_link_80_8)
ave_loss_link_80_9 = sum(loss_link_80_9) / len(loss_link_80_9)
ave_loss_link_80_10 = sum(loss_link_80_10) / len(loss_link_80_10)
ave_loss_link_80 = [ave_loss_link_80_1, ave_loss_link_80_2, ave_loss_link_80_3, ave_loss_link_80_4, ave_loss_link_80_5, ave_loss_link_80_6, ave_loss_link_80_7, ave_loss_link_80_8, ave_loss_link_80_9, ave_loss_link_80_10]

ave_loss_link = ( ave_loss_link_20, ave_loss_link_40, ave_loss_link_60, ave_loss_link_80)

box_link = plt.boxplot(ave_loss_link, patch_artist = True,
                         labels = ['20%', '40%', '60%', '80%'],
                        flierprops={'marker': 'o', 'markerfacecolor': 'w', 'markeredgecolor': 'red', 'alpha': 0.5},   # 异常值设置为蓝色
                         medianprops = {'linewidth': 1.5, 'color': 'red'})  # 一定要有patch_artist = True，否则不识别facecolor这个关键词

color = 'red'
for box in box_link['boxes']:
    box.set(facecolor='w', edgecolor='red', linestyle='dashed')

for whisker in box_link['whiskers']:
    whisker.set(color='red', linewidth=1)

for cap in box_link['caps']:
    cap.set(color='red', linewidth=1)

for median in box_link['medians']:
    median.set(color='red', linewidth=1.5)

median_loss_link_20 = np.median(ave_loss_link_20)
median_loss_link_40 = np.median(ave_loss_link_40)
median_loss_link_60 = np.median(ave_loss_link_60)
median_loss_link_80 = np.median(ave_loss_link_80)

median_loss_link = np.array([median_loss_link_20, median_loss_link_40, median_loss_link_60, median_loss_link_80])

plt.plot(range(1, len(median_loss_link) + 1), median_loss_link, marker='o', markersize=10, linewidth=3.0, color='salmon')  # 绘制中位数折线图

plt.text(1.0, median_loss_link_20+0.5, 'Median: {:.5f}'.format(median_loss_link_20), color='indianred', fontproperties='Times New Roman', size=18, ha='center', va='bottom')     # 将箱型图的中位数标记出来
plt.text(2.0, median_loss_link_40+0.8, ' {:.5f}'.format(median_loss_link_40), color='indianred', fontproperties='Times New Roman', size=18, ha='center', va='bottom')     # 将箱型图的中位数标记出来
plt.text(3.0, median_loss_link_60+0.4, ' {:.5f}'.format(median_loss_link_60), color='indianred', fontproperties='Times New Roman', size=18, ha='center', va='bottom')     # 将箱型图的中位数标记出来
plt.text(4.0, median_loss_link_80+0.1, ' {:.5f}'.format(median_loss_link_80), color='indianred', fontproperties='Times New Roman', size=18, ha='center', va='bottom')     # 将箱型图的中位数标记出来

# calcualte mean:
mean_20  = np.mean(ave_loss_link_20)
mean_40  = np.mean(ave_loss_link_40)
mean_60  = np.mean(ave_loss_link_60)
mean_80  = np.mean(ave_loss_link_80)
# draw mean -----
plt.axhline(mean_20, color='darkgrey', linestyle='-.', linewidth = 1, label='Mean_20%')
plt.axhline(mean_40, color='darkgrey', linestyle='-.', linewidth = 1, label='Mean_40%')
plt.axhline(mean_60, color='darkgrey', linestyle='-.', linewidth = 1, label='Mean_60%')
plt.axhline(mean_80, color='darkgrey', linestyle='-.', linewidth = 1, label='Mean_80%')

plt.text(1.2, mean_20-0.6, 'Mean: {:.5f}'.format(mean_20), color='grey', fontproperties='Times New Roman', size=18, ha='center', va='bottom')     # 将箱型图的中位数标记出来
plt.text(2.45, mean_40-0.52, ' {:.5f}'.format(mean_40), color='grey', fontproperties='Times New Roman', size=18, ha='center', va='bottom')     # 将箱型图的中位数标记出来
plt.text(3.515, mean_60-0.5, ' {:.5f}'.format(mean_60), color='grey', fontproperties='Times New Roman', size=18, ha='center', va='bottom')     # 将箱型图的中位数标记出来
plt.text(4.2, mean_80-0.5, ' {:.5f}'.format(mean_80), color='grey', fontproperties='Times New Roman', size=18, ha='center', va='bottom')     # 将箱型图的中位数标记出来

fig = plt.gcf()
fig.set_size_inches(8, 6)  # 设置保存图像的显示大小 宽度为 10英寸，高度为 8英寸
plt.xlabel('(b) monitoring validation set', labelpad=5.5, fontdict={'family': 'Times New Roman', 'size': 18, 'color': 'k'})  # 改变横坐标轴标题字体  labelpad=10表示坐标标题距离坐标轴的垂直距离
plt.ylabel('Absolute Error Average of link flow', labelpad=7.5, fontdict={'family': 'Times New Roman', 'size': 18, 'color': 'k'})  # 改变纵坐标轴标题字体
plt.xticks(fontproperties='Times New Roman', size=18)
plt.yticks(fontproperties='Times New Roman', size=18)
# handles = [Line2D([0], [0], color='salmon', marker='o', lw=2), Line2D([0], [0], linestyle='-.', color='indianred', lw=2)]
handles = [Line2D([0], [0], color='salmon', marker='o', lw=2), Line2D([0], [0], linestyle='-.', color='darkgrey', lw=2)]
labels = ['Median', 'Mean']   # 自定义图例标签
# plt.legend(handles=handles, labels=labels, loc='upper left')   # 添加图例
plt.legend(handles=handles, labels=labels, loc='upper left', prop={'family': 'Times New Roman', 'size': 18})   # 添加图例
# plt.subplots_adjust(right=0.95, wspace=10.5)
plt.subplots_adjust(right=0.95, bottom=0.2, hspace=0.3, wspace=10.5)
plt.savefig('link validation set box.png', dpi=300, format='png')
plt.show()


# 对AE_link四舍五入，保留小数点后两位，否则频率分布图中，每个数据都不同，其频率都很小，没有规律
rounded_loss_link_20_1 = [round(num, 2) for num in loss_link_20_1]
rounded_loss_link_20_2 = [round(num, 2) for num in loss_link_20_2]
rounded_loss_link_20_3 = [round(num, 2) for num in loss_link_20_3]
rounded_loss_link_20_4 = [round(num, 2) for num in loss_link_20_4]
rounded_loss_link_20_5 = [round(num, 2) for num in loss_link_20_5]
rounded_loss_link_20_6 = [round(num, 2) for num in loss_link_20_6]
rounded_loss_link_20_7 = [round(num, 2) for num in loss_link_20_7]
rounded_loss_link_20_8 = [round(num, 2) for num in loss_link_20_8]
rounded_loss_link_20_9 = [round(num, 2) for num in loss_link_20_9]
rounded_loss_link_20_10 = [round(num, 2) for num in loss_link_20_10]
rounded_loss_link_40_1 = [round(num, 2) for num in loss_link_40_1]
rounded_loss_link_40_2 = [round(num, 2) for num in loss_link_40_2]
rounded_loss_link_40_3 = [round(num, 2) for num in loss_link_40_3]
rounded_loss_link_40_4 = [round(num, 2) for num in loss_link_40_4]
rounded_loss_link_40_5 = [round(num, 2) for num in loss_link_40_5]
rounded_loss_link_40_6 = [round(num, 2) for num in loss_link_40_6]
rounded_loss_link_40_7 = [round(num, 2) for num in loss_link_40_7]
rounded_loss_link_40_8 = [round(num, 2) for num in loss_link_40_8]
rounded_loss_link_40_9 = [round(num, 2) for num in loss_link_40_9]
rounded_loss_link_40_10 = [round(num, 2) for num in loss_link_40_10]
rounded_loss_link_60_1 = [round(num, 2) for num in loss_link_60_1]
rounded_loss_link_60_2 = [round(num, 2) for num in loss_link_60_2]
rounded_loss_link_60_3 = [round(num, 2) for num in loss_link_60_3]
rounded_loss_link_60_4 = [round(num, 2) for num in loss_link_60_4]
rounded_loss_link_60_5 = [round(num, 2) for num in loss_link_60_5]
rounded_loss_link_60_6 = [round(num, 2) for num in loss_link_60_6]
rounded_loss_link_60_7 = [round(num, 2) for num in loss_link_60_7]
rounded_loss_link_60_8 = [round(num, 2) for num in loss_link_60_8]
rounded_loss_link_60_9 = [round(num, 2) for num in loss_link_60_9]
rounded_loss_link_60_10 = [round(num, 2) for num in loss_link_60_10]
rounded_loss_link_80_1 = [round(num, 2) for num in loss_link_80_1]
rounded_loss_link_80_2 = [round(num, 2) for num in loss_link_80_2]
rounded_loss_link_80_3 = [round(num, 2) for num in loss_link_80_3]
rounded_loss_link_80_4 = [round(num, 2) for num in loss_link_80_4]
rounded_loss_link_80_5 = [round(num, 2) for num in loss_link_80_5]
rounded_loss_link_80_6 = [round(num, 2) for num in loss_link_80_6]
rounded_loss_link_80_7 = [round(num, 2) for num in loss_link_80_7]
rounded_loss_link_80_8 = [round(num, 2) for num in loss_link_80_8]
rounded_loss_link_80_9 = [round(num, 2) for num in loss_link_80_9]
rounded_loss_link_80_10 = [round(num, 2) for num in loss_link_80_10]
# 只绘制频率分布曲线，不绘制分布直方图
plt.rc('font', family='Times New Roman', size = 12)    # 设置全局字体样式，坐标轴的数据在这里会被设置为相应的字体
mu, sigma = norm.fit(rounded_loss_link_20_1)   # 拟合频率分布曲线
x = np.linspace(min(rounded_loss_link_20_1), max(rounded_loss_link_20_1), 100)
y = norm.pdf(x, mu, sigma)
plt.plot(x, y, 'deepskyblue', label='Fitted Curve_20_1')

mu, sigma = norm.fit(rounded_loss_link_20_2)   # 拟合频率分布曲线
x = np.linspace(min(rounded_loss_link_20_2), max(rounded_loss_link_20_2), 100)
y = norm.pdf(x, mu, sigma)
plt.plot(x, y, 'lightskyblue', label='Fitted Curve_20_2')

mu, sigma = norm.fit(rounded_loss_link_20_3)   # 拟合频率分布曲线
x = np.linspace(min(rounded_loss_link_20_3), max(rounded_loss_link_20_3), 100)
y = norm.pdf(x, mu, sigma)
plt.plot(x, y, 'steelblue', label='Fitted Curve_20_3')

mu, sigma = norm.fit(rounded_loss_link_20_4)   # 拟合频率分布曲线
x = np.linspace(min(rounded_loss_link_20_4), max(rounded_loss_link_20_4), 100)
y = norm.pdf(x, mu, sigma)
plt.plot(x, y, 'powderblue', label='Fitted Curve_20_4')

mu, sigma = norm.fit(rounded_loss_link_20_5)   # 拟合频率分布曲线
x = np.linspace(min(rounded_loss_link_20_5), max(rounded_loss_link_20_5), 100)
y = norm.pdf(x, mu, sigma)
plt.plot(x, y, 'dodgerblue', label='Fitted Curve_20_5')

plt.rc('font', family='Times New Roman', size = 12)    # 设置全局字体样式，坐标轴的数据在这里会被设置为相应的字体
mu, sigma = norm.fit(rounded_loss_link_20_6)   # 拟合频率分布曲线
x = np.linspace(min(rounded_loss_link_20_6), max(rounded_loss_link_20_6), 100)
y = norm.pdf(x, mu, sigma)
plt.plot(x, y, 'cadetblue', label='Fitted Curve_20_6')

mu, sigma = norm.fit(rounded_loss_link_20_7)   # 拟合频率分布曲线
x = np.linspace(min(rounded_loss_link_20_7), max(rounded_loss_link_20_7), 100)
y = norm.pdf(x, mu, sigma)
plt.plot(x, y, 'skyblue', label='Fitted Curve_20_7')

mu, sigma = norm.fit(rounded_loss_link_20_8)   # 拟合频率分布曲线
x = np.linspace(min(rounded_loss_link_20_8), max(rounded_loss_link_20_8), 100)
y = norm.pdf(x, mu, sigma)
plt.plot(x, y, 'aliceblue', label='Fitted Curve_20_8')

mu, sigma = norm.fit(rounded_loss_link_20_9)   # 拟合频率分布曲线
x = np.linspace(min(rounded_loss_link_20_9), max(rounded_loss_link_20_9), 100)
y = norm.pdf(x, mu, sigma)
plt.plot(x, y, 'lightsteelblue', label='Fitted Curve_20_9')

mu, sigma = norm.fit(rounded_loss_link_20_10)   # 拟合频率分布曲线
x = np.linspace(min(rounded_loss_link_20_10), max(rounded_loss_link_20_10), 100)
y = norm.pdf(x, mu, sigma)
plt.plot(x, y, 'cornflowerblue', label='Fitted Curve_20_10')

mu, sigma = norm.fit(rounded_loss_link_40_1)   # 拟合频率分布曲线
x = np.linspace(min(rounded_loss_link_40_1), max(rounded_loss_link_40_1), 100)
y = norm.pdf(x, mu, sigma)
plt.plot(x, y, 'thistle', linestyle='-.', label='Fitted Curve_40_1')

mu, sigma = norm.fit(rounded_loss_link_40_2)   # 拟合频率分布曲线
x = np.linspace(min(rounded_loss_link_40_2), max(rounded_loss_link_40_2), 100)
y = norm.pdf(x, mu, sigma)
plt.plot(x, y, 'violet', linestyle='-.', label='Fitted Curve_40_2')

mu, sigma = norm.fit(rounded_loss_link_40_3)   # 拟合频率分布曲线
x = np.linspace(min(rounded_loss_link_40_3), max(rounded_loss_link_40_3), 100)
y = norm.pdf(x, mu, sigma)
plt.plot(x, y, 'm', linestyle='-.', label='Fitted Curve_40_3')

mu, sigma = norm.fit(rounded_loss_link_40_4)   # 拟合频率分布曲线
x = np.linspace(min(rounded_loss_link_40_4), max(rounded_loss_link_40_4), 100)
y = norm.pdf(x, mu, sigma)
plt.plot(x, y, 'fuchsia', linestyle='-.', label='Fitted Curve_40_4')

mu, sigma = norm.fit(rounded_loss_link_40_5)   # 拟合频率分布曲线
x = np.linspace(min(rounded_loss_link_40_5), max(rounded_loss_link_40_5), 100)
y = norm.pdf(x, mu, sigma)
plt.plot(x, y, 'darkviolet', linestyle='-.', label='Fitted Curve_40_5')

mu, sigma = norm.fit(rounded_loss_link_40_6)   # 拟合频率分布曲线
x = np.linspace(min(rounded_loss_link_40_6), max(rounded_loss_link_40_6), 100)
y = norm.pdf(x, mu, sigma)
plt.plot(x, y, 'plum', linestyle='-.', label='Fitted Curve_40_6')

mu, sigma = norm.fit(rounded_loss_link_40_7)   # 拟合频率分布曲线
x = np.linspace(min(rounded_loss_link_40_7), max(rounded_loss_link_40_7), 100)
y = norm.pdf(x, mu, sigma)
plt.plot(x, y, 'mediumorchid', linestyle='-.', label='Fitted Curve_40_7')

mu, sigma = norm.fit(rounded_loss_link_40_8)   # 拟合频率分布曲线
x = np.linspace(min(rounded_loss_link_40_8), max(rounded_loss_link_40_8), 100)
y = norm.pdf(x, mu, sigma)
plt.plot(x, y, 'purple', linestyle='-.', label='Fitted Curve_40_8')

mu, sigma = norm.fit(rounded_loss_link_40_9)   # 拟合频率分布曲线
x = np.linspace(min(rounded_loss_link_40_9), max(rounded_loss_link_40_9), 100)
y = norm.pdf(x, mu, sigma)
plt.plot(x, y, 'orchid', linestyle='-.', label='Fitted Curve_40_9')

mu, sigma = norm.fit(rounded_loss_link_40_10)   # 拟合频率分布曲线
x = np.linspace(min(rounded_loss_link_40_10), max(rounded_loss_link_40_10), 100)
y = norm.pdf(x, mu, sigma)
plt.plot(x, y, 'darkmagenta', linestyle='-.', label='Fitted Curve_40_10')

mu, sigma = norm.fit(rounded_loss_link_60_1)   # 拟合频率分布曲线
x = np.linspace(min(rounded_loss_link_60_1), max(rounded_loss_link_60_1), 100)
y = norm.pdf(x, mu, sigma)
plt.plot(x, y, 'yellow', linestyle='--', label='Fitted Curve_60_1')

mu, sigma = norm.fit(rounded_loss_link_60_2)   # 拟合频率分布曲线
x = np.linspace(min(rounded_loss_link_60_2), max(rounded_loss_link_60_2), 100)
y = norm.pdf(x, mu, sigma)
plt.plot(x, y, 'gold', linestyle='--', label='Fitted Curve_60_2')

mu, sigma = norm.fit(rounded_loss_link_60_3)   # 拟合频率分布曲线
x = np.linspace(min(rounded_loss_link_60_3), max(rounded_loss_link_60_3), 100)
y = norm.pdf(x, mu, sigma)
plt.plot(x, y, 'y', linestyle='--', label='Fitted Curve_60_3')

mu, sigma = norm.fit(rounded_loss_link_60_4)   # 拟合频率分布曲线
x = np.linspace(min(rounded_loss_link_60_4), max(rounded_loss_link_60_4), 100)
y = norm.pdf(x, mu, sigma)
plt.plot(x, y, 'khaki', linestyle='--', label='Fitted Curve_60_4')

mu, sigma = norm.fit(rounded_loss_link_60_5)   # 拟合频率分布曲线
x = np.linspace(min(rounded_loss_link_60_5), max(rounded_loss_link_60_5), 100)
y = norm.pdf(x, mu, sigma)
plt.plot(x, y, 'goldenrod', linestyle='--', label='Fitted Curve_60_5')

mu, sigma = norm.fit(rounded_loss_link_60_6)   # 拟合频率分布曲线
x = np.linspace(min(rounded_loss_link_60_6), max(rounded_loss_link_60_6), 100)
y = norm.pdf(x, mu, sigma)
plt.plot(x, y, 'cornsilk', linestyle='--', label='Fitted Curve_60_6')

mu, sigma = norm.fit(rounded_loss_link_60_7)   # 拟合频率分布曲线
x = np.linspace(min(rounded_loss_link_60_7), max(rounded_loss_link_60_7), 100)
y = norm.pdf(x, mu, sigma)
plt.plot(x, y, 'lemonchiffon', linestyle='--', label='Fitted Curve_60_7')

mu, sigma = norm.fit(rounded_loss_link_60_8)   # 拟合频率分布曲线
x = np.linspace(min(rounded_loss_link_60_8), max(rounded_loss_link_60_8), 100)
y = norm.pdf(x, mu, sigma)
plt.plot(x, y, 'ivory', linestyle='--', label='Fitted Curve_60_8')

mu, sigma = norm.fit(rounded_loss_link_60_9)   # 拟合频率分布曲线
x = np.linspace(min(rounded_loss_link_60_9), max(rounded_loss_link_60_9), 100)
y = norm.pdf(x, mu, sigma)
plt.plot(x, y, 'olive', linestyle='--', label='Fitted Curve_60_9')

mu, sigma = norm.fit(rounded_loss_link_60_10)   # 拟合频率分布曲线
x = np.linspace(min(rounded_loss_link_60_10), max(rounded_loss_link_60_10), 100)
y = norm.pdf(x, mu, sigma)
plt.plot(x, y, 'darkkhaki', linestyle='--', label='Fitted Curve_60_10')

mu, sigma = norm.fit(rounded_loss_link_80_1)   # 拟合频率分布曲线
x = np.linspace(min(rounded_loss_link_80_1), max(rounded_loss_link_80_1), 100)
y = norm.pdf(x, mu, sigma)
plt.plot(x, y, 'lightgreen', linestyle=':', label='Fitted Curve_80_1')

mu, sigma = norm.fit(rounded_loss_link_80_2)   # 拟合频率分布曲线
x = np.linspace(min(rounded_loss_link_80_2), max(rounded_loss_link_80_2), 100)
y = norm.pdf(x, mu, sigma)
plt.plot(x, y, 'g', linestyle=':', label='Fitted Curve_80_2')

mu, sigma = norm.fit(rounded_loss_link_80_3)   # 拟合频率分布曲线
x = np.linspace(min(rounded_loss_link_80_3), max(rounded_loss_link_80_3), 100)
y = norm.pdf(x, mu, sigma)
plt.plot(x, y, 'limegreen', linestyle=':', label='Fitted Curve_80_3')

mu, sigma = norm.fit(rounded_loss_link_80_4)   # 拟合频率分布曲线
x = np.linspace(min(rounded_loss_link_80_4), max(rounded_loss_link_80_4), 100)
y = norm.pdf(x, mu, sigma)
plt.plot(x, y, 'darkgreen', linestyle=':', label='Fitted Curve_80_4')

mu, sigma = norm.fit(rounded_loss_link_80_5)   # 拟合频率分布曲线
x = np.linspace(min(rounded_loss_link_80_5), max(rounded_loss_link_80_5), 100)
y = norm.pdf(x, mu, sigma)
plt.plot(x, y, 'lime', linestyle=':', label='Fitted Curve_80_5')

mu, sigma = norm.fit(rounded_loss_link_80_6)   # 拟合频率分布曲线
x = np.linspace(min(rounded_loss_link_80_6), max(rounded_loss_link_80_6), 100)
y = norm.pdf(x, mu, sigma)
plt.plot(x, y, 'forestgreen', linestyle=':', label='Fitted Curve_80_6')

mu, sigma = norm.fit(rounded_loss_link_80_7)   # 拟合频率分布曲线
x = np.linspace(min(rounded_loss_link_80_7), max(rounded_loss_link_80_7), 100)
y = norm.pdf(x, mu, sigma)
plt.plot(x, y, 'green', linestyle=':', label='Fitted Curve_80_7')

mu, sigma = norm.fit(rounded_loss_link_80_8)   # 拟合频率分布曲线
x = np.linspace(min(rounded_loss_link_80_8), max(rounded_loss_link_80_8), 100)
y = norm.pdf(x, mu, sigma)
plt.plot(x, y, 'seagreen', linestyle=':', label='Fitted Curve_80_8')

mu, sigma = norm.fit(rounded_loss_link_80_9)   # 拟合频率分布曲线
x = np.linspace(min(rounded_loss_link_80_9), max(rounded_loss_link_80_9), 100)
y = norm.pdf(x, mu, sigma)
plt.plot(x, y, 'mediumseagreen', linestyle=':', label='Fitted Curve_80_9')

mu, sigma = norm.fit(rounded_loss_link_80_10)   # 拟合频率分布曲线
x = np.linspace(min(rounded_loss_link_80_10), max(rounded_loss_link_80_10), 100)
y = norm.pdf(x, mu, sigma)
plt.plot(x, y, 'springgreen', linestyle=':', label='Fitted Curve_80_10')

fig = plt.gcf()
fig.set_size_inches(8, 6)  # 设置保存图像的显示大小 宽度为 10英寸，高度为 8英寸
plt.xlabel('Absolute Error Average of link flow', fontdict={'family': 'Times New Roman', 'size': 12, 'color': 'k'})
plt.ylabel('Frequency', fontdict={'family': 'Times New Roman', 'size': 12, 'color': 'k'})
font = fm.FontProperties(family='Times New Roman', style='normal', size=12)
plt.legend(loc='upper right', ncol=2, prop=font)
plt.savefig('link-Frequency Distribution with Fitted Curve.png', dpi=300, format='png')
plt.show()


