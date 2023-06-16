import pprint
import numpy as np
from sklearn.cluster import KMeans, k_means
import pandas as pd
from sklearn.metrics import silhouette_score
from matplotlib import pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

path_weather = r'data/第一问数据/预测数据(经过标准化）(风化后）.xlsx'
path_no_weather = r'data/第一问数据/预测数据(经过标准化）.xlsx'
data = pd.read_excel(path_weather, index_col=0, header=0)

ingredients = list(data.columns[:14])

"""
    数据标准化
"""
# 高钾数据标准化
data_kind_1 = data[data['种类'] == '高钾']
data_kind_2 = data[data['种类'] == '铅钡']
new_data_kind_1 = pd.DataFrame(index=data_kind_1.index, columns=data_kind_1.columns)
for index in data_kind_1.index:
    for col_index in range(len(data_kind_1.columns)):
        column = data_kind_1.columns[col_index]
        if 14 <= col_index <= 18:
            new_data_kind_1.loc[index][column] = data_kind_1.loc[index][column]
        else:
            col_sum = sum(data_kind_1.loc[:][column])
            new_data_kind_1.loc[index][column] = data_kind_1.loc[index][column] / col_sum
data_kind_1 = new_data_kind_1

# 铅钡数据标准化
new_data_kind_2 = pd.DataFrame(index=data_kind_2.index, columns=data_kind_2.columns)
for index in data_kind_2.index:
    for col_index in range(len(data_kind_2.columns)):
        column = data_kind_2.columns[col_index]
        if 14 <= col_index <= 18:
            new_data_kind_2.loc[index][column] = data_kind_2.loc[index][column]
        else:
            col_sum = sum(data_kind_2.loc[:][column])
            new_data_kind_2.loc[index][column] = data_kind_2.loc[index][column] / col_sum
data_kind_2 = new_data_kind_2

data_kind_1.to_excel('data/第二问模型/亚类划分结果/高钾（风化后）.xlsx')
data_kind_2.to_excel('data/第二问模型/亚类划分结果/铅钡（风化后）.xlsx')

data_list_kind_1 = []
kind_1_index = []
data_list_kind_2 = []
kind_2_index = []

# 高钾数据
for index in data_kind_1.index:
    # 数据输入
    cur_data_list_kind_1 = []
    for ingredient in ingredients:
        cur_data = data_kind_1.loc[index][ingredient]
        cur_data_list_kind_1.append(cur_data)

    kind_1_index.append(index)
    data_list_kind_1.append(cur_data_list_kind_1)
# 铅钡数据
for index in data_kind_2.index:
    # 数据输入
    cur_data_list_kind_2 = []
    for ingredient in ingredients:
        cur_data = data_kind_2.loc[index][ingredient]
        cur_data_list_kind_2.append(cur_data)

    kind_2_index.append(index)
    data_list_kind_2.append(cur_data_list_kind_2)
# 计算轮廓系数
score_kind_1 = []
score_kind_2 = []
for i in range(2, 7):
    model_1 = k_means(data_list_kind_1, n_clusters=i)
    model_2 = k_means(data_list_kind_2, n_clusters=i)
    score_kind_1.append(silhouette_score(data_list_kind_1, model_1[1]))
    score_kind_2.append(silhouette_score(data_list_kind_2, model_2[1]))




plt.figure()
plt.subplot(1, 2, 1)
plt.plot(range(2, 7), score_kind_1, 'r*-')
plt.xlabel('cluster')
plt.ylabel('轮廓系数')
plt.title('轮廓系数确定的最佳k值(高钾)')

# plt.figure()
plt.subplot(1, 2, 2)
plt.plot(range(2, 7), score_kind_2, 'r*-')
plt.xlabel('cluster')
plt.ylabel('轮廓系数')
plt.title('轮廓系数确定的最佳k值(铅钡)')
plt.subplots_adjust(wspace=0.5)
plt.savefig(r'picture/第二问图像/轮廓系数/最佳k值(高钾与铅钡)（风化后）.png')

# 假如我要构造一个聚类数为3的聚类器
estimator_kind_1 = KMeans(n_clusters=3)  # 构造聚类器
estimator_kind_1.fit(data_list_kind_1)  # 聚类
label_pred = estimator_kind_1.labels_  # 获取聚类标签
centroids = estimator_kind_1.cluster_centers_  # 获取聚类中心
inertia = estimator_kind_1.inertia_  # 获取聚类准则的总和
writer = pd.ExcelWriter('data/第一问数据/亚类划分结果/划分结果（归一化）（风化后）.xlsx')

divide_dict_kind_1 = {}
for num in range(len(kind_1_index)):
    index = kind_1_index[num]
    divide_dict_kind_1[index] = label_pred[num]

divide_dict_df = pd.DataFrame(divide_dict_kind_1.values(), index=divide_dict_kind_1.keys(), columns=['分类'])
divide_dict_df.to_excel(writer, sheet_name='高钾分类')

estimator_kind_2 = KMeans(n_clusters=2)  # 构造聚类器
estimator_kind_2.fit(data_list_kind_2)  # 聚类
label_pred = estimator_kind_2.labels_  # 获取聚类标签
centroids = estimator_kind_2.cluster_centers_  # 获取聚类中心
inertia = estimator_kind_2.inertia_  # 获取聚类准则的总和

divide_dict_kind_2 = {}
for num in range(len(kind_2_index)):
    index = kind_2_index[num]
    divide_dict_kind_2[index] = label_pred[num]

divide_dict_df = pd.DataFrame(divide_dict_kind_2.values(), index=divide_dict_kind_2.keys(), columns=['分类'])
divide_dict_df.to_excel(writer, sheet_name='铅钡分类')
writer.save()


"""
    灵敏度分析
"""
ingredients_list = []
ingredients = list(data.columns[:14])
for num in range(14):
    cur_list = ingredients.copy()
    cur_list.pop(num)
    ingredients_list.append(cur_list)

scores_kind_1 = []
scores_kind_2 = []
for ingredients in ingredients_list:

    data_list_kind_1 = []
    kind_1_index = []
    data_list_kind_2 = []
    kind_2_index = []
    for index in data.index:
        cur_data_list_kind_1 = []
        cur_data_list_kind_2 = []
        data_kind = data.loc[index]['种类']
        if data_kind == '高钾':
            kind_1_index.append(index)
            for ingredient in ingredients:
                cur_data = data.loc[index][ingredient]
                cur_data_list_kind_1.append(cur_data)
            data_list_kind_1.append(cur_data_list_kind_1)
        else:
            kind_2_index.append(index)
            for ingredient in ingredients:
                cur_data = data.loc[index][ingredient]
                cur_data_list_kind_2.append(cur_data)
            data_list_kind_2.append(cur_data_list_kind_2)

    # 计算轮廓系数
    score_kind_1 = []
    score_kind_2 = []
    for i in range(2, 7):
        model_1 = k_means(data_list_kind_1, n_clusters=i)
        model_2 = k_means(data_list_kind_2, n_clusters=i)
        # model = cluster.SpectralClustering(x_1, n_clusters=i + 2)
        score_kind_1.append(silhouette_score(data_list_kind_1, model_1[1]))
        score_kind_2.append(silhouette_score(data_list_kind_2, model_2[1]))

    scores_kind_1.append(max(score_kind_1))
    scores_kind_2.append(max(score_kind_2))

x_label = ['SiO2', 'Na2O', 'K2O', 'CaO', 'MgO', 'Al2O3', 'Fe2O3', 'CuO',
           'PbO', 'BaO', 'P2O5', 'SrO', 'SnO2', 'SO2']
plt.figure()
plt.title('元素与轮廓系数的灵敏度')
plt.xticks(range(len(ingredients) + 1), x_label, rotation=30)
plt.plot(scores_kind_1, label='高钾', marker='*')
plt.plot(scores_kind_2, label='铅钡', marker='*')
plt.legend()
plt.savefig(r'picture/第二问图像/灵敏度分析/分类的灵敏度分析(风化后）.png')