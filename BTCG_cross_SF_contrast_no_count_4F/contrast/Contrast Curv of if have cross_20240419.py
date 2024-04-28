import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D #绘制3D坐标的函数
plt.rcParams['savefig.dpi']=100
plt.rcParams['figure.dpi']=100
import pandas as pd
import numpy as np

#(1) draw the contrast curv of 5F & 4F:
df_5F = pd.read_csv(r'F:\SEU\MasterThesis\02BTCG_cross_SF64\14BTCG_cross_SF_all_5F\output_loss.csv')
df_4F = pd.read_csv(r'F:\SEU\MasterThesis\02BTCG_cross_SF64\13BTCG_cross_SF_contrast_no_count_4F\output_loss.csv')

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

df_4F_survey = pd.read_csv(r'F:\SEU\MasterThesis\02BTCG_cross_SF64\13BTCG_cross_SF_contrast_no_count_4F\output_ozone.csv')
df_4F_mobile = pd.read_csv(r'F:\SEU\MasterThesis\02BTCG_cross_SF64\13BTCG_cross_SF_contrast_no_count_4F\output_od.csv')
df_4F_OBD = pd.read_csv(r'F:\SEU\MasterThesis\02BTCG_cross_SF64\13BTCG_cross_SF_contrast_no_count_4F\output_path.csv')
df_4F_monitoring = pd.read_csv(r'F:\SEU\MasterThesis\02BTCG_cross_SF64\13BTCG_cross_SF_contrast_no_count_4F\output_link.csv')

target_generation = df_5F_survey.iloc[:,2].tolist()
estimate_generation_5F = df_5F_survey.iloc[:,1].tolist()
estimate_generation_4F = df_4F_survey.iloc[:,1].tolist()

target_OD_split = df_5F_mobile.iloc[:,4].tolist()
estimate_gamma_5F = df_5F_mobile.iloc[:,2].tolist()
estimate_gamma_4F = df_4F_mobile.iloc[:,2].tolist()

target_proportion = df_5F_OBD.iloc[:,6].tolist()
estimate_path_proportion_5F = df_5F_OBD.iloc[:,4].tolist()
estimate_path_proportion_4F = df_4F_OBD.iloc[:,4].tolist()

target_count = df_5F_monitoring.iloc[:,4].tolist()
estimated_count_5F = df_5F_monitoring.iloc[:,3].tolist()
estimated_count_4F = df_4F_monitoring.iloc[:,3].tolist()

absolute_error_survey_5F = list(map(abs,[target_generation[i] - estimate_generation_5F[i] for i in range(0,len(target_generation))]))
absolute_error_survey_4F = list(map(abs,[target_generation[i] - estimate_generation_4F[i] for i in range(0,len(target_generation))]))

absolute_error_mobile_5F = list(map(abs,[target_OD_split[i] - estimate_gamma_5F[i] for i in range(0,len(target_OD_split))]))
absolute_error_mobile_4F = list(map(abs,[target_OD_split[i] - estimate_gamma_4F[i] for i in range(0,len(target_OD_split))]))

absolute_error_OBD_5F = list(map(abs,[target_proportion[i] - estimate_path_proportion_5F[i] for i in range(0,len(target_proportion))]))
absolute_error_OBD_4F = list(map(abs,[target_proportion[i] - estimate_path_proportion_4F[i] for i in range(0,len(target_proportion))]))

absolute_error_monitoring_5F = list(map(abs,[target_count[i] - estimated_count_5F[i] for i in range(0,len(target_count))]))
absolute_error_monitoring_4F = list(map(abs,[target_count[i] - estimated_count_4F[i] for i in range(0,len(target_count))]))

absolute_error_survey = (absolute_error_survey_5F , absolute_error_survey_4F)
absolute_error_mobile = (absolute_error_mobile_5F , absolute_error_mobile_4F)
absolute_error_OBD = (absolute_error_OBD_5F , absolute_error_OBD_4F)
absolute_error_monitoring = (absolute_error_monitoring_5F , absolute_error_monitoring_4F)

# calcualte mean:
mean_survey1 = np.mean(absolute_error_survey_5F)
mean_survey2 = np.mean(absolute_error_survey_4F)
mean_mobile1 = np.mean(absolute_error_mobile_5F)
mean_mobile2 = np.mean(absolute_error_mobile_4F)
mean_OBD1 = np.mean(absolute_error_OBD_5F)
mean_OBD2 = np.mean(absolute_error_OBD_4F)
mean_monitoring1 = np.mean(absolute_error_monitoring_5F)
mean_monitoring2 = np.mean(absolute_error_monitoring_4F)

# how to draw boxplot?  https://blog.csdn.net/Gou_Hailong/article/details/124769916 ; https://blog.csdn.net/qq_37006625/article/details/127908633
# view_survey = plt.boxplot(absolute_error_survey, patch_artist = True, labels = ['5F model','4F model'], boxprops = {'facecolor':'cyan', 'linewidth':0.8, 'edgecolor':'red'})

#(2.1) draw survey boxplot:
box_survey = plt.boxplot(absolute_error_survey, patch_artist = True,
                         labels = ['5F model','4F model'],
                         # boxprops = {'facecolor':'cyan', 'linewidth':0.8, 'edgecolor':'red'},
                         medianprops = {'linewidth': 1.5, 'color': 'darkgreen'})  # 一定要有patch_artist = True，否则不识别facecolor这个关键词

color = ['springgreen', 'palegreen']
for box, c in zip(box_survey['boxes'] , color):
    box.set(color = 'darkgreen', linewidth = 1)     # 边框
    box.set(facecolor = c)                    # 主体颜色

# draw mean -----
plt.axhline(mean_survey1, color='darkgreen', linestyle='--', linewidth = 1, label='Mean_5F')
plt.axhline(mean_survey2, color='mediumseagreen', linestyle='--', linewidth = 1, label='Mean_4F')
plt.legend(loc='upper center', prop={'family': 'Times New Roman', 'size': 18})

# my_font1 = {'family': 'Times New Roman', 'size': 12}
# plt.legend(view_survey['boxes'], fontsize=10, prop=my_font1)
# plt.xlabel('(a) Survey', labelpad=-3.5, fontweight = 'bold', fontdict={'family': 'Times New Roman', 'size': 18, 'color': 'k'})  # 改变横坐标轴标题字体  labelpad=10表示坐标标题距离坐标轴的垂直距离
plt.xlabel('(a) F1_Survey', labelpad=3.5, fontdict={'family': 'Times New Roman', 'size': 18, 'color': 'k'})  # 改变横坐标轴标题字体  labelpad=10表示坐标标题距离坐标轴的垂直距离
# plt.xlabel('Survey', labelpad=7.5, fontdict={'family': 'Times New Roman', 'size': 12, 'color': 'k'})  # 改变横坐标轴标题字体  labelpad=10表示坐标标题距离坐标轴的垂直距离
plt.ylabel('Absolute Error of ozone_generation', labelpad=5.5, fontdict={'family': 'Times New Roman', 'size': 18, 'color': 'k'})  # 改变纵坐标轴标题字体
# plt.axis([-10, 310, -1, 41])  # 改变xy坐标轴范围
# plt.axis([-1, 101, -1, 51])  # 改变xy坐标轴范围
plt.xticks(fontproperties='Times New Roman', size=18)
plt.yticks(fontproperties='Times New Roman', size=18)
plt.subplots_adjust(left=0.2, right=0.9, bottom=0.2, top=0.9, wspace=0.3, hspace=0.3)
plt.savefig('Contrast box-plot of F1_Survey.png', dpi=300, format='png')
plt.show()

#(2.2) draw mobile boxplot:
box_mobile = plt.boxplot(absolute_error_mobile, patch_artist = True,
                         labels = ['5F model','4F model'],
                         # boxprops = {'facecolor':'cyan', 'linewidth':0.8, 'edgecolor':'red'},
                         medianprops = {'linewidth': 1.5, 'color': 'chocolate'})  # 一定要有patch_artist = True，否则不识别facecolor这个关键词

color = ['orange', 'moccasin']
for box, c in zip(box_mobile['boxes'] , color):
    box.set(color = 'chocolate', linewidth = 1)     # 边框
    box.set(facecolor = c)                    # 主体颜色

# draw mean -----
plt.axhline(mean_mobile1, color='chocolate', linestyle='--', linewidth = 1, label='Mean_5F')
plt.axhline(mean_mobile2, color='sandybrown', linestyle='--', linewidth = 1, label='Mean_4F')
plt.legend(loc='upper center', prop={'family': 'Times New Roman', 'size': 18})

plt.xlabel('(b) F2_Mobile', labelpad=3.5, fontdict={'family': 'Times New Roman', 'size': 18, 'color': 'k'})
plt.ylabel('Absolute Error of OD_flow', labelpad=5.5, fontdict={'family': 'Times New Roman', 'size': 18, 'color': 'k'})  # 改变纵坐标轴标题字体
plt.xticks(fontproperties='Times New Roman', size=18)
plt.yticks(fontproperties='Times New Roman', size=18)
plt.subplots_adjust(left=0.2, right=0.9, bottom=0.2, top=0.9, wspace=0.3, hspace=0.3)
plt.savefig('Contrast box-plot of F2_Mobile.png', dpi=300, format='png')
plt.show()

#(2.3) draw OBD boxplot:
box_OBD = plt.boxplot(absolute_error_OBD, patch_artist = True,
                         labels = ['5F model','4F model'],
                         # boxprops = {'facecolor':'cyan', 'linewidth':0.8, 'edgecolor':'red'},
                         medianprops = {'linewidth': 1.5, 'color': 'darkblue'})  # 一定要有patch_artist = True，否则不识别facecolor这个关键词
color = ['b', 'cornflowerblue']
for box, c in zip(box_OBD['boxes'] , color):
    box.set(color = 'darkblue', linewidth = 1)     # 边框
    box.set(facecolor = c)                    # 主体颜色

# draw mean -----
plt.axhline(mean_OBD1, color='steelblue', linestyle='--', linewidth = 1, label='Mean_5F')
plt.axhline(mean_OBD2, color='lightskyblue', linestyle='--', linewidth = 1, label='Mean_4F')
plt.legend(loc='upper center', prop={'family': 'Times New Roman', 'size': 18})

plt.xlabel('(c) F3_OBD', labelpad=3.5, fontdict={'family': 'Times New Roman', 'size': 18, 'color': 'k'})
plt.ylabel('Absolute Error of path_proportion', labelpad=5.5, fontdict={'family': 'Times New Roman', 'size': 18, 'color': 'k'})  # 改变纵坐标轴标题字体
plt.xticks(fontproperties='Times New Roman', size=18)
plt.yticks(fontproperties='Times New Roman', size=18)
plt.subplots_adjust(left=0.2, right=0.9, bottom=0.2, top=0.9, wspace=0.3, hspace=0.3)
plt.savefig('Contrast box-plot of F3_OBD.png', dpi=300, format='png')
plt.show()


#(2.4) draw monitoring boxplot:
box_monitoring = plt.boxplot(absolute_error_monitoring, patch_artist = True,
                         labels = ['5F model','4F model'],
                         # boxprops = {'facecolor':'cyan', 'linewidth':0.8, 'edgecolor':'red'},
                         medianprops = {'linewidth': 1.5, 'color': 'darkred'})  # 一定要有patch_artist = True，否则不识别facecolor这个关键词
color = ['red', 'lightcoral']
for box, c in zip(box_monitoring['boxes'] , color):
    box.set(color = 'darkred', linewidth = 1)     # 边框
    box.set(facecolor = c)                    # 主体颜色

# draw mean -----
plt.axhline(mean_monitoring1, color='darkred', linestyle='--', linewidth = 1, label='Mean_5F')
plt.axhline(mean_monitoring2, color='salmon', linestyle='--', linewidth = 1, label='Mean_4F')
plt.legend(loc='upper center', prop={'family': 'Times New Roman', 'size': 18})

plt.xlabel('(d) F4_Monitoring', labelpad=3.5, fontdict={'family': 'Times New Roman', 'size': 18, 'color': 'k'})
plt.ylabel('Absolute Error of link_count', labelpad=5.5, fontdict={'family': 'Times New Roman', 'size': 18, 'color': 'k'})  # 改变纵坐标轴标题字体
plt.xticks(fontproperties='Times New Roman', size=18)
plt.yticks(fontproperties='Times New Roman', size=18)
plt.subplots_adjust(left=0.2, right=0.9, bottom=0.2, top=0.9, wspace=0.3, hspace=0.3)
plt.savefig('Contrast box-plot of F4_Monitoring.png', dpi=300, format='png')
plt.show()
