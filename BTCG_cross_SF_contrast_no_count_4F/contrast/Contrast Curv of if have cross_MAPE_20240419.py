from mpl_toolkits.mplot3d import Axes3D #绘制3D坐标的函数
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D #绘制3D坐标的函数
plt.rcParams['savefig.dpi']=100
plt.rcParams['figure.dpi']=100
import pandas as pd
import numpy as np
# color:  https://tool.oschina.net/commons?type=3
#(1) draw the contrast curv of 5F & 4F:
df_5F = pd.read_csv(r'F:\SEU\MasterThesis\02BTCG_cross_SF64\14BTCG_cross_SF_all_5F\output_loss.csv')
df_4F = pd.read_csv(r'F:\SEU\MasterThesis\02BTCG_cross_SF64\16BTCG_cross_SF_contrast_no_count_4F\output_loss.csv')

loss_total_5F = []
loss_survey_5F = []
loss_mobile_5F = []
loss_float_5F = []
loss_sensor_5F = []
loss_video_5F = []

loss_total_4F = []
loss_survey_4F = []
loss_mobile_4F = []
loss_float_4F = []
loss_sensor_4F = []

loss_total_5F = df_5F.iloc[:,0].tolist()
loss_total_5F_4 = df_5F.iloc[:,1].tolist()
loss_survey_5F = df_5F.iloc[:,2].tolist()
loss_mobile_5F = df_5F.iloc[:,3].tolist()
loss_float_5F = df_5F.iloc[:,4].tolist()
loss_sensor_5F = df_5F.iloc[:,5].tolist()
loss_video_5F = df_5F.iloc[:,6].tolist()


loss_total_4F = df_4F.iloc[:,0].tolist()
loss_survey_4F = df_4F.iloc[:,1].tolist()
loss_mobile_4F = df_4F.iloc[:,2].tolist()
loss_float_4F = df_4F.iloc[:,3].tolist()
loss_sensor_4F = df_4F.iloc[:,4].tolist()
loss_sensor_4F_legend_column = df_4F.iloc[:,1].tolist()

plt.tight_layout()
plt.plot(loss_survey_4F, 'palegreen', label="F1-(survey)")
plt.plot(loss_mobile_4F, 'moccasin', label="F2-(mobile)")
plt.plot(loss_float_4F, 'cornflowerblue', label="F3-(OBD)")
plt.plot(loss_sensor_4F, 'lightcoral', label="F4-(monitoring_link)")
plt.plot(loss_total_4F, 'lightsteelblue', linestyle='--', linewidth='2.5', label="F-(total)")
plt.plot(loss_sensor_4F_legend_column, 'white', linestyle='--', linewidth='2.5', label=" ")  # 设置图例为两列
plt.plot(loss_sensor_4F_legend_column, 'white', linestyle='--', linewidth='2.5', label=" ")

plt.plot(loss_survey_5F, 'springgreen', label="F1(survey)")
plt.plot(loss_mobile_5F, 'orange', label="F2(mobile)")
plt.plot(loss_float_5F, 'b', label="F3(OBD)")
plt.plot(loss_sensor_5F, 'red', label="F4(monitoring_link)")
plt.plot(loss_total_5F_4, 'slategray', linestyle='--', linewidth='2.5', label="F(total_4)")
plt.plot(loss_video_5F, 'fuchsia', label="F5(monitoring_cross)")
plt.plot(loss_total_5F, 'darkblue', linestyle='--', linewidth='2.5', label="F(total)")



# # Assuming you have already created your plot and set the labels for the legend items
# # For demonstration purposes, let's assume you have a list of labels for all the legend items
# legend_labels = ['F1-(survey)', 'F2-(mobile)', 'F3-(OBD)', 'F4-(monitoring_link)', 'F-(total)', 'F1(survey)', 'F2(mobile)', 'F3(OBD)', 'F4(monitoring_link)', 'F(total_4)', 'F5(monitoring_cross)', 'F(total)']
# # Define the number of items in each column
# num_items_column1 = 5
# num_items_column2 = 7
# # Split legend handles and labels into two lists for each column
# handles_column1 = plt.plot([], [], label="Column 1")[:num_items_column1]
# handles_column2 = plt.plot([], [], label="Column 2")[:num_items_column2]
# labels_column1 = legend_labels[:num_items_column1]
# labels_column2 = legend_labels[num_items_column1:num_items_column1 + num_items_column2]
# # Combine the two lists of handles and labels for plt.legend
# all_handles = handles_column1 + handles_column2
# all_labels = labels_column1 + labels_column2
# # Create the legend with two columns
# # plt.legend(handles=all_handles, labels=all_labels, loc='upper left', ncol=2)






my_font1 = {'family': 'Times New Roman', 'size': 12}
plt.legend(loc="best", fontsize=10, ncol=2 , prop=my_font1)
plt.xlabel('Iteration', labelpad=7.5,
           fontdict={'family': 'Times New Roman', 'size': 12, 'color': 'k'})  # 改变横坐标轴标题字体  labelpad=10表示坐标标题距离坐标轴的垂直距离
plt.ylabel('Loss', labelpad=7.5, fontdict={'family': 'Times New Roman', 'size': 12, 'color': 'k'})  # 改变纵坐标轴标题字体
# plt.axis([-10, 310, -1, 41])  # 改变xy坐标轴范围
plt.axis([-1, 101, -1, 51])  # 改变xy坐标轴范围
plt.xticks(fontproperties='Times New Roman', size=12)
plt.yticks(fontproperties='Times New Roman', size=12)
# plt.savefig('convergence curv.svg', dpi=300, format='svg')
plt.savefig('Contrast Curv of if have cross.png', dpi=1000, format='png')
plt.show()


# (2)draw the contrast curv of survey、mobile、OBD、monitoring:
df_5F_survey = pd.read_csv(r'F:\SEU\MasterThesis\02BTCG_cross_SF64\14BTCG_cross_SF_all_5F\output_ozone.csv')
df_5F_mobile = pd.read_csv(r'F:\SEU\MasterThesis\02BTCG_cross_SF64\14BTCG_cross_SF_all_5F\output_od.csv')
df_5F_OBD = pd.read_csv(r'F:\SEU\MasterThesis\02BTCG_cross_SF64\14BTCG_cross_SF_all_5F\output_path.csv')
df_5F_monitoring = pd.read_csv(r'F:\SEU\MasterThesis\02BTCG_cross_SF64\14BTCG_cross_SF_all_5F\output_link.csv')

df_4F_survey = pd.read_csv(r'F:\SEU\MasterThesis\02BTCG_cross_SF64\16BTCG_cross_SF_contrast_no_count_4F\output_ozone.csv')
df_4F_mobile = pd.read_csv(r'F:\SEU\MasterThesis\02BTCG_cross_SF64\16BTCG_cross_SF_contrast_no_count_4F\output_od.csv')
df_4F_OBD = pd.read_csv(r'F:\SEU\MasterThesis\02BTCG_cross_SF64\16BTCG_cross_SF_contrast_no_count_4F\output_path.csv')
df_4F_monitoring = pd.read_csv(r'F:\SEU\MasterThesis\02BTCG_cross_SF64\16BTCG_cross_SF_contrast_no_count_4F\output_link.csv')

target_generation = df_5F_survey.iloc[:,2].tolist()          # 实际值
estimate_generation_5F = df_5F_survey.iloc[:,1].tolist()     # 预测值_5F
estimate_generation_4F = df_4F_survey.iloc[:,1].tolist()     # 预测值_4F

target_OD_split = df_5F_mobile.iloc[:,4].tolist()
estimate_gamma_5F = df_5F_mobile.iloc[:,2].tolist()
estimate_gamma_4F = df_4F_mobile.iloc[:,2].tolist()

target_proportion = df_5F_OBD.iloc[:,6].tolist()
estimate_path_proportion_5F = df_5F_OBD.iloc[:,4].tolist()
estimate_path_proportion_4F = df_4F_OBD.iloc[:,4].tolist()

target_count = df_5F_monitoring.iloc[:,4].tolist()
estimated_count_5F = df_5F_monitoring.iloc[:,3].tolist()
estimated_count_4F = df_4F_monitoring.iloc[:,3].tolist()

absolute_error_survey_5F = list(map(abs,[(1 - (estimate_generation_5F[i] / target_generation[i]))*100 for i in range(0,len(target_generation))]))
absolute_error_survey_4F = list(map(abs,[(1 - (estimate_generation_4F[i] / target_generation[i]))*100 for i in range(0,len(target_generation))]))

absolute_error_mobile_5F = list(map(abs,[(1 - (estimate_gamma_5F[i] /target_OD_split[i]))*100 for i in range(0,len(target_OD_split))]))
absolute_error_mobile_4F = list(map(abs,[(1 - (estimate_gamma_4F[i] / target_OD_split[i]))*100 for i in range(0,len(target_OD_split))]))

absolute_error_OBD_5F = list(map(abs,[(1 - (estimate_path_proportion_5F[i] / target_proportion[i]))*100 for i in range(0,len(target_proportion))]))
absolute_error_OBD_4F = list(map(abs,[(1 - (estimate_path_proportion_4F[i] / target_proportion[i]))*100 for i in range(0,len(target_proportion))]))

absolute_error_monitoring_5F = list(map(abs,[(1 - (estimated_count_5F[i] / target_count[i]))*100 for i in range(0,len(target_count))]))
absolute_error_monitoring_4F = list(map(abs,[(1 - (estimated_count_4F[i] / target_count[i]))*100 for i in range(0,len(target_count))]))

absolute_error_survey = (absolute_error_survey_5F , absolute_error_survey_4F)
absolute_error_mobile = (absolute_error_mobile_5F , absolute_error_mobile_4F)
absolute_error_OBD = (absolute_error_OBD_5F , absolute_error_OBD_4F)
absolute_error_monitoring = (absolute_error_monitoring_5F , absolute_error_monitoring_4F)

# calcualte mean:
mean_survey1 = round(np.mean(absolute_error_survey_5F),2)
mean_survey2 = round(np.mean(absolute_error_survey_4F),2)
mean_mobile1 = round(np.mean(absolute_error_mobile_5F),2)
mean_mobile2 = round(np.mean(absolute_error_mobile_4F),2)
mean_OBD1 = round(np.mean(absolute_error_OBD_5F),2)
mean_OBD2 = round(np.mean(absolute_error_OBD_4F),2)
mean_monitoring1 = round(np.mean(absolute_error_monitoring_5F),2)
mean_monitoring2 = round(np.mean(absolute_error_monitoring_4F),2)

# how to draw boxplot?  https://blog.csdn.net/Gou_Hailong/article/details/124769916 ; https://blog.csdn.net/qq_37006625/article/details/127908633
# view_survey = plt.boxplot(absolute_error_survey, patch_artist = True, labels = ['5 layer model','4 layer model'], boxprops = {'facecolor':'cyan', 'linewidth':0.8, 'edgecolor':'red'})
# calculate median 中位数
median_survey1 = round(np.median(absolute_error_survey_5F),2)
median_survey2 = round(np.median(absolute_error_survey_4F),2)
median_mobile1 = round(np.median(absolute_error_mobile_5F),2)
median_mobile2 = round(np.median(absolute_error_mobile_4F),2)
median_OBD1 = round(np.median(absolute_error_OBD_5F),2)
median_OBD2 = round(np.median(absolute_error_OBD_4F),2)
median_monitoring1 = round(np.median(absolute_error_monitoring_5F),2)
median_monitoring2 = round(np.median(absolute_error_monitoring_4F),2)
# # 计算统计量：箱型图的最大值、上四分位数、中位数、下四分位数、最小值
# stats_survey_5F = []  # 存储每个数据集的统计量
# for data in absolute_error_survey_5F:
#     # 计算最大值、最小值、中位数、上下四分位数
#     q1_survey_5F, median_survey_5F, q3_survey_5F = np.percentile(data, [25, 50, 75])
#     iqr = q3_survey_5F - q1_survey_5F  # 四分位距
#     lower_bound = q1_survey_5F - 1.5 * iqr  # 最小异常值边界
#     upper_bound = q3_survey_5F + 1.5 * iqr  # 最大异常值边界
#     max_val_survey_5F = np.max(data)
#     min_val_survey_5F = np.min(data)
#     # 如果需要排除异常值，可以使用 lower_bound 和 upper_bound
#     # 但在这里我们只是标记，所以使用 max_val 和 min_val
#     stats_survey_5F.append({'q1_survey_5F': q1_survey_5F, 'median_survey_5F': median_survey_5F, 'q3_survey_5F': q3_survey_5F, 'max_survey_5F': max_val_survey_5F, 'min_survey_5F': min_val_survey_5F})
# stats_survey_4F = []  # 存储每个数据集的统计量
# for data in absolute_error_survey_4F:
#     # 计算最大值、最小值、中位数、上下四分位数
#     q1_survey_4F, median_survey_4F, q3_survey_4F = np.percentile(data, [25, 50, 75])
#     iqr = q3_survey_4F - q1_survey_4F  # 四分位距
#     lower_bound = q1_survey_4F - 1.5 * iqr  # 最小异常值边界
#     upper_bound = q3_survey_4F + 1.5 * iqr  # 最大异常值边界
#     max_val_survey_4F = np.max(data)
#     min_val_survey_4F = np.min(data)
#     # 如果需要排除异常值，可以使用 lower_bound 和 upper_bound
#     # 但在这里我们只是标记，所以使用 max_val 和 min_val
#     stats_survey_4F.append({'q1_survey_5F': q1_survey_4F, 'median_survey_5F': median_survey_4F, 'q3_survey_5F': q3_survey_4F, 'max_survey_5F': max_val_survey_4F, 'min_survey_5F': min_val_survey_4F})
# stats_mobile_5F = []  # 存储每个数据集的统计量
# for data in absolute_error_mobile_5F:
#     # 计算最大值、最小值、中位数、上下四分位数
#     q1_mobile_5F, median_mobile_5F, q3_mobile_5F = np.percentile(data, [25, 50, 75])
#     iqr = q3_mobile_5F - q1_mobile_5F  # 四分位距
#     lower_bound = q1_mobile_5F - 1.5 * iqr  # 最小异常值边界
#     upper_bound = q3_mobile_5F + 1.5 * iqr  # 最大异常值边界
#     max_val_mobile_5F = np.max(data)
#     min_val_mobile_5F = np.min(data)
#     # 如果需要排除异常值，可以使用 lower_bound 和 upper_bound
#     # 但在这里我们只是标记，所以使用 max_val 和 min_val
#     stats_mobile_5F.append({'q1_mobile_5F': q1_mobile_5F, 'median_mobile_5F': median_mobile_5F, 'q3_mobile_5F': q3_mobile_5F, 'max_mobile_5F': max_val_mobile_5F, 'min_mobile_5F': min_val_mobile_5F})
# stats_mobile_4F = []  # 存储每个数据集的统计量
# for data in absolute_error_mobile_4F:
#     # 计算最大值、最小值、中位数、上下四分位数
#     q1_mobile_4F, median_mobile_4F, q3_mobile_4F = np.percentile(data, [25, 50, 75])
#     iqr = q3_mobile_4F - q1_mobile_4F  # 四分位距
#     lower_bound = q1_mobile_4F - 1.5 * iqr  # 最小异常值边界
#     upper_bound = q3_mobile_4F + 1.5 * iqr  # 最大异常值边界
#     max_val_mobile_4F = np.max(data)
#     min_val_mobile_4F = np.min(data)
#     # 如果需要排除异常值，可以使用 lower_bound 和 upper_bound
#     # 但在这里我们只是标记，所以使用 max_val 和 min_val
#     stats_mobile_4F.append({'q1_mobile_5F': q1_mobile_4F, 'median_mobile_5F': median_mobile_4F, 'q3_mobile_5F': q3_mobile_4F, 'max_mobile_5F': max_val_mobile_4F, 'min_mobile_5F': min_val_mobile_4F})
#



#(2.1) draw survey boxplot:
box_survey = plt.boxplot(absolute_error_survey, patch_artist = True,
                         labels = ['5 layer model','4 layer model'],
                         # boxprops = {'facecolor':'cyan', 'linewidth':0.8, 'edgecolor':'red'},
                         medianprops = {'linewidth': 1.5, 'color': 'darkgreen'})  # 一定要有patch_artist = True，否则不识别facecolor这个关键词

color = ['LimeGreen', 'palegreen']
for box, c in zip(box_survey['boxes'] , color):
    box.set(color = 'darkgreen', linewidth = 1)     # 边框
    box.set(facecolor = c)                    # 主体颜色

# draw mean -----
plt.axhline(mean_survey1, color='darkgreen', linestyle='--', linewidth = 1, label='MAPE_5F')
plt.axhline(mean_survey2, color='mediumseagreen', linestyle='--', linewidth = 1, label='MAPE_4F')
plt.legend(loc='upper center', prop={'family': 'Times New Roman', 'size': 18})

# my_font1 = {'family': 'Times New Roman', 'size': 12}
# plt.legend(view_survey['boxes'], fontsize=10, prop=my_font1)
# plt.xlabel('(a) Survey', labelpad=-3.5, fontweight = 'bold', fontdict={'family': 'Times New Roman', 'size': 18, 'color': 'k'})  # 改变横坐标轴标题字体  labelpad=10表示坐标标题距离坐标轴的垂直距离
plt.xlabel('(a) F1_Survey', labelpad=3.5, fontdict={'family': 'Times New Roman', 'size': 18, 'color': 'k'})  # 改变横坐标轴标题字体  labelpad=10表示坐标标题距离坐标轴的垂直距离
# plt.xlabel('Survey', labelpad=7.5, fontdict={'family': 'Times New Roman', 'size': 12, 'color': 'k'})  # 改变横坐标轴标题字体  labelpad=10表示坐标标题距离坐标轴的垂直距离
plt.ylabel('APE of generation flow (%)', labelpad=5.5, fontdict={'family': 'Times New Roman', 'size': 18, 'color': 'k'})  # 改变纵坐标轴标题字体
# plt.axis([-10, 310, -1, 41])  # 改变xy坐标轴范围
# plt.axis([-1, 101, -1, 51])  # 改变xy坐标轴范围
plt.xticks(fontproperties='Times New Roman', size=18)
plt.yticks(fontproperties='Times New Roman', size=18)
# 标记mean值
plt.text(0.6, mean_survey1-1.8, str(mean_survey1)+ '%', ha='left', va='center', color='darkgreen', fontdict={'family': 'Times New Roman', 'size': 18})
plt.text(1.6, mean_survey2+1.4, str(mean_survey2)+ '%', ha='left', va='center', color='mediumseagreen', fontdict={'family': 'Times New Roman', 'size': 18})
# 标记统计量
# y_offset = 2  # 调整文本位置以便清晰可见
# plt.text(1.35, min_val - y_offset, f'Min: {min_val:.2f}%', ha='center', va='top', color='dimgray', fontdict={'family': 'Times New Roman', 'size': 18})
# plt.text(1.35, q1 - y_offset, f'Q1: {q1:.2f}%', ha='center', va='top', color='black', fontdict={'family': 'Times New Roman', 'size': 18})
# plt.text(1.46, median - y_offset, f'Median: {median:.2f}%', ha='center', va='top', color='black', fontdict={'family': 'Times New Roman', 'size': 18, 'weight': 'bold'})
# plt.text(1.35, q3 - y_offset, f'Q3: {q3:.2f}%', ha='center', va='top', color='black', fontdict={'family': 'Times New Roman', 'size': 18})
# plt.text(1.35, max_val - y_offset, f'Max: {max_val:.2f}%', ha='center', va='top', color='black', fontdict={'family': 'Times New Roman', 'size': 18})
# 标记中位数
plt.text(0.99, median_survey1 + 3.0, f'Median: {median_survey1:.2f}%', ha='center', va='top', color='black', fontdict={'family': 'Times New Roman', 'size': 18})
plt.text(1.99, median_survey2 + 3.0, f'Median: {median_survey2:.2f}%', ha='center', va='top', color='black', fontdict={'family': 'Times New Roman', 'size': 18})

plt.subplots_adjust(left=0.15, right=0.95, bottom=0.15, top=0.95, wspace=0.3, hspace=0.3)
# # 使图片边框变大
# fig, ax = plt.subplots()
#     # 在图形上绘制线条
# ax.plot([1, 2, 3, 4], [1, 4, 2, 3])
#     # 设置坐标轴边框大小
# ax.spines['top'].set_linewidth(2)
# ax.spines['right'].set_linewidth(2)
# ax.spines['bottom'].set_linewidth(2)
# ax.spines['left'].set_linewidth(2)
plt.savefig('Contrast box-plot of F1_Survey.png', dpi=300, format='png')
plt.show()

#(2.2) draw mobile boxplot:
box_mobile = plt.boxplot(absolute_error_mobile, patch_artist = True,
                         labels = ['5 layer model','4 layer model'],
                         # boxprops = {'facecolor':'cyan', 'linewidth':0.8, 'edgecolor':'red'},
                         medianprops = {'linewidth': 1.5, 'color': 'chocolate'})  # 一定要有patch_artist = True，否则不识别facecolor这个关键词

color = ['orange', 'moccasin']
for box, c in zip(box_mobile['boxes'] , color):
    box.set(color = 'chocolate', linewidth = 1)     # 边框
    box.set(facecolor = c)                    # 主体颜色

# draw mean -----
plt.axhline(mean_mobile1, color='chocolate', linestyle='--', linewidth = 1, label='MAPE_5F')
plt.axhline(mean_mobile2, color='sandybrown', linestyle='--', linewidth = 1, label='MAPE_4F')
plt.legend(loc='upper center', prop={'family': 'Times New Roman', 'size': 18})

plt.xlabel('(b) F2_Mobile', labelpad=3.5, fontdict={'family': 'Times New Roman', 'size': 18, 'color': 'k'})
plt.ylabel('APE of OD distribution rate (%)', labelpad=5.5, fontdict={'family': 'Times New Roman', 'size': 18, 'color': 'k'})  # 改变纵坐标轴标题字体
plt.xticks(fontproperties='Times New Roman', size=18)
plt.yticks(fontproperties='Times New Roman', size=18)
# 标记mean值
plt.text(0.6, mean_mobile1-5.0, str(mean_mobile1)+ '%', ha='left', va='center', color='chocolate', fontdict={'family': 'Times New Roman', 'size': 18})
plt.text(1.6, mean_mobile2+3.8, str(mean_mobile2)+ '%', ha='left', va='center', color='sandybrown', fontdict={'family': 'Times New Roman', 'size': 18})
# 标记中位数
plt.text(0.99, median_mobile1 + 23.3, f'Median: {median_mobile1:.2f}%', ha='center', va='top', color='black', fontdict={'family': 'Times New Roman', 'size': 18})
plt.text(1.99, median_mobile2 + 23.3, f'Median: {median_mobile2:.2f}%', ha='center', va='top', color='black', fontdict={'family': 'Times New Roman', 'size': 18})

# plt.subplots_adjust(left=0.2, right=0.9, bottom=0.2, top=0.9, wspace=0.3, hspace=0.3)
plt.subplots_adjust(left=0.15, right=0.95, bottom=0.15, top=0.95, wspace=0.3, hspace=0.3)
plt.savefig('Contrast box-plot of F2_Mobile.png', dpi=300, format='png')
plt.show()

#(2.3) draw OBD boxplot:
box_OBD = plt.boxplot(absolute_error_OBD, patch_artist = True,
                         labels = ['5 layer model','4 layer model'],
                         # boxprops = {'facecolor':'cyan', 'linewidth':0.8, 'edgecolor':'red'},
                         medianprops = {'linewidth': 1.5, 'color': '#00008B'})  # 一定要有patch_artist = True，否则不识别facecolor这个关键词
color = ['b', 'cornflowerblue']
for box, c in zip(box_OBD['boxes'] , color):
    box.set(color = 'darkblue', linewidth = 1)     # 边框
    box.set(facecolor = c)                    # 主体颜色

# draw mean -----
plt.axhline(mean_OBD1, color='steelblue', linestyle='--', linewidth = 1, label='MAPE_5F')
plt.axhline(mean_OBD2, color='lightskyblue', linestyle='--', linewidth = 1, label='MAPE_4F')
plt.legend(loc='upper center', prop={'family': 'Times New Roman', 'size': 18})     # 通过修改legend()函数中的参数来改变图例的位置，将loc参数改为'lower center'即可将图例往下挪动。

plt.xlabel('(c) F3_OBD', labelpad=3.5, fontdict={'family': 'Times New Roman', 'size': 18, 'color': 'k'})
plt.ylabel('APE of path selection rate (%)', labelpad=5.5, fontdict={'family': 'Times New Roman', 'size': 18, 'color': 'k'})  # 改变纵坐标轴标题字体
plt.xticks(fontproperties='Times New Roman', size=18)
plt.yticks(fontproperties='Times New Roman', size=18)
# 标记mean值
plt.text(0.65, mean_OBD1-1.2, str(mean_OBD1)+ '%', ha='left', va='center', color='steelblue', fontdict={'family': 'Times New Roman', 'size': 18})
plt.text(1.65, mean_OBD2+1.0, str(mean_OBD2)+ '%', ha='left', va='center', color='lightskyblue', fontdict={'family': 'Times New Roman', 'size': 18})
# 标记中位数
plt.text(0.96, median_OBD1 + 5.3, f'Median: {median_OBD1:.2f}%', ha='center', va='top', color='black', fontdict={'family': 'Times New Roman', 'size': 18})
plt.text(1.96, median_OBD2 + 6.3, f'Median: {median_OBD2:.2f}%', ha='center', va='top', color='black', fontdict={'family': 'Times New Roman', 'size': 18})

# plt.subplots_adjust(left=0.2, right=0.9, bottom=0.2, top=0.9, wspace=0.3, hspace=0.3)
plt.subplots_adjust(left=0.15, right=0.95, bottom=0.15, top=0.95, wspace=0.3, hspace=0.3)
plt.savefig('Contrast box-plot of F3_OBD.png', dpi=300, format='png')
plt.show()


#(2.4) draw monitoring boxplot:
box_monitoring = plt.boxplot(absolute_error_monitoring, patch_artist = True,
                         labels = ['5 layer model','4 layer model'],
                         # boxprops = {'facecolor':'cyan', 'linewidth':0.8, 'edgecolor':'red'},
                         medianprops = {'linewidth': 1.5, 'color': 'darkred'})  # 一定要有patch_artist = True，否则不识别facecolor这个关键词
color = ['IndianRed', 'lightcoral']
for box, c in zip(box_monitoring['boxes'] , color):
    box.set(color = 'darkred', linewidth = 1)     # 边框
    box.set(facecolor = c)                    # 主体颜色

# draw mean -----
plt.axhline(mean_monitoring1, color='darkred', linestyle='--', linewidth = 1, label='MAPE_5F')
plt.axhline(mean_monitoring2, color='salmon', linestyle='--', linewidth = 1, label='MAPE_4F')
plt.legend(loc='upper center', prop={'family': 'Times New Roman', 'size': 18})

plt.xlabel('(d) F4_Monitoring', labelpad=3.5, fontdict={'family': 'Times New Roman', 'size': 18, 'color': 'k'})
plt.ylabel('APE of link flow (%)', labelpad=5.5, fontdict={'family': 'Times New Roman', 'size': 18, 'color': 'k'})  # 改变纵坐标轴标题字体
plt.xticks(fontproperties='Times New Roman', size=18)
plt.yticks(fontproperties='Times New Roman', size=18)
# 标记mean值
plt.text(0.6, mean_monitoring1-5.2, str(mean_monitoring1)+ '%', ha='left', va='center', color='darkred', fontdict={'family': 'Times New Roman', 'size': 18})
plt.text(1.6, mean_monitoring2+4.2, str(mean_monitoring2)+ '%', ha='left', va='center', color='salmon', fontdict={'family': 'Times New Roman', 'size': 18})
# 标记中位数
plt.text(1.39, median_monitoring1 - 10.3, f'Median: {median_monitoring1:.2f}%', ha='center', va='top', color='black', fontdict={'family': 'Times New Roman', 'size': 18})
plt.text(1.99, median_monitoring2 - 10.3, f'Median: {median_monitoring2:.2f}%', ha='center', va='top', color='black', fontdict={'family': 'Times New Roman', 'size': 18})

# plt.subplots_adjust(left=0.2, right=0.9, bottom=0.2, top=0.9, wspace=0.3, hspace=0.3)
plt.subplots_adjust(left=0.15, right=0.95, bottom=0.15, top=0.95, wspace=0.3, hspace=0.3)
plt.savefig('Contrast box-plot of F4_Monitoring.png', dpi=300, format='png')
plt.show()
