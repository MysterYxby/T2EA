
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

# 设置新罗马字体
# matplotlib.rcParams['font.family'] = 'serif'
# matplotlib.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
# 从Excel文件中读取数据
data = pd.read_excel('Metric/Taylor_Metric/Metirc_Taylor_Layer.xlsx')

# 提取数据
layers = data['Layer']
visible_data = data['Visible']
infrared_data = data['Infrared']

# 创建1*1的子图
fig, axs = plt.subplots(1, 2, figsize=(12, 6))
# fig, axs = plt.subplots(1, 2, figsize=(6, 6))
# 绘制Visible数据
# axs[0].plot(layers, visible_data, marker='o', color='b')
# axs[0].set_title('SSIM',size=18)
# axs[0].set_xlabel('Number of Derivative Network Layers',size=18)
# axs[0].set_ylabel('Values of the Metric',size=18)
# axs[0].text(0.5, -0.15, '(a)', size=18, ha="center", transform=axs.transAxes)
# axs[0].tick_params(axis='both', which='both', bottom=False, top=True, left=True, right=False, labelbottom=True, labeltop=False, labelleft=True, labelright=False)

# # 绘制Infrared数据
# axs[1].plot(layers, infrared_data, marker='s', color='r')
# axs[1].set_title('SSIM',size=18)
# axs[1].set_xlabel('Number of Derivative Network Layers',size=18)
# axs[1].set_ylabel('Values of the Metric',size=18)
# axs[1].text(0.5, -0.15, '(b)', size=18, ha="center", transform=axs[1].transAxes)
# axs[1].tick_params(axis='both', which='both', bottom=False, top=True, left=True, right=False, labelbottom=True, labeltop=False, labelleft=True, labelright=False)

# 绘制Visible数据
axs[0].plot(layers, visible_data, marker='o', color='b')
axs[0].set_title('SSIM',size=18)
axs[0].set_xlabel('微分网络个数',size=18)
axs[0].set_ylabel('指标值',size=18)
# axs[0].text(0.5, -0.15, '(a)', size=18, ha="center", transform=axs[0].transAxes)
axs[0].tick_params(axis='both', which='both', bottom=False, top=True, left=True, right=False, labelbottom=True, labeltop=False, labelleft=True, labelright=False)

# 绘制Infrared数据
axs[1].plot(layers, infrared_data, marker='s', color='r')
axs[1].set_title('SSIM',size=18)
axs[1].set_xlabel('微分网络个数',size=18)
axs[1].set_ylabel('指标值',size=18)
# axs[1].text(0.5, -0.15, '(b)', size=18, ha="center", transform=axs[1].transAxes)
axs[1].tick_params(axis='both', which='both', bottom=False, top=True, left=True, right=False, labelbottom=True, labeltop=False, labelleft=True, labelright=False)

# 调整布局
plt.tight_layout()

# 保存图像为EPS/jpg格式
# fig.savefig('image/中文图/Taylor.eps', format='eps')
# fig.savefig('image/中文图/aylor.pdf', format='pdf')
fig.savefig('image/中文图/Taylor.jpg', format='jpg', dpi=300)
# 保存子图1为图片文件
axs[0].get_figure().savefig('image/中文图/subplot1.jpg',dpi=300)

# 保存子图2为图片文件
axs[1].get_figure().savefig('image/中文图/subplot2.jpg',dpi=300)
# 显示图形
plt.show()


