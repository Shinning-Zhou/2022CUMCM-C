import os.path
import pprint

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import style
from pylab import mpl

mpl.rcParams['font.sans-serif'] = ['STZhongsong']  # 指定默认字体：解决plot不能显示中文问题
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
sns.set_style({"font.sans-serif": ['simhei', 'Droid Sans Fallback']})

"""
    数据读取
"""
data_path = r'data/题目原始数据/附件.xlsx'
# 表单1
data_sheet_1 = pd.read_excel(data_path, index_col=0, sheet_name='表单1')
# 表单2
data_sheet_2 = pd.read_excel(data_path, index_col=0, sheet_name='表单2')
# 表单3
data_sheet_3 = pd.read_excel(data_path, index_col=0, sheet_name='表单3')

"""
    数据处理
"""
# 表单1的索引列
sheet_1_index = data_sheet_1.index
# 表单2的索引列
sheet_2_index = data_sheet_2.index
# 表单3的索引列
sheet_3_index = data_sheet_3.index

# 表单1进行数据清洗,空缺项填0
sheet_1_columns = data_sheet_1.columns
for row_index in sheet_1_index:
    for col_index in sheet_1_columns:
        data = data_sheet_1.loc[row_index][col_index]
        if isinstance(data, str):
            continue
        data_sheet_1.loc[row_index][col_index] = 0

# 对表单2进行数据清洗,将空缺的数据补成0
sheet_2_columns = data_sheet_2.columns
for row_index in sheet_2_index:
    for col_index in sheet_2_columns:
        data = data_sheet_2.loc[row_index][col_index]
        if np.isnan(data):
            data_sheet_2.loc[row_index][col_index] = 0

# 表单3进行数据清洗
sheet_3_columns = data_sheet_3.columns
data_sheet_3_copy = data_sheet_3.copy()

sheet_2_data_valid = {}
for row_index in sheet_2_index:
    data = data_sheet_2.loc[row_index]
    data_sum = sum(data)
    if 85 <= data_sum <= 105:
        sheet_2_data_valid[row_index] = 1
    else:
        sheet_2_data_valid[row_index] = 0
sheet_2_data_valid_df = pd.DataFrame(sheet_2_data_valid.values(), index=sheet_2_data_valid.keys(), columns=['是否有效'])
sheet_2_data_valid_df.to_excel('data/第一问数据/表单2数据有效性.xlsx')

"""
    表单1数据统计
"""
no_weather_dict = {}
weathered_dict = {}
weathered_index = []  # 风化的ID
no_weather_index = []  # 未风化的ID
for row_index in sheet_1_index:
    data = data_sheet_1.loc[row_index]
    weather_condition = data['表面风化']
    if weather_condition == '无风化':
        no_weather_index.append(row_index)
        no_weather_dict[row_index] = data
    else:
        weathered_index.append(row_index)
        weathered_dict[row_index] = data

weather_info = {
    '纹饰': {'A': 0, 'B': 0, 'C': 0},
    '类型': {'高钾': 0, '铅钡': 0},
    '颜色': {
        '黑': 0, '蓝绿': 0, '绿': 0, '浅蓝': 0, '浅绿': 0, '深蓝': 0, '深绿': 0, '紫': 0, '无': 0
    }
}
for index in weathered_index:
    data = weathered_dict[index]
    decoration = data[0]
    kind = data[1]
    color = data[2]
    weather_info['纹饰'][decoration] += 1
    weather_info['类型'][kind] += 1
    if color == 0:
        weather_info['颜色']['无'] += 1
    else:
        weather_info['颜色'][color] += 1

no_weather_info = {
    '纹饰': {'A': 0, 'B': 0, 'C': 0},
    '类型': {'高钾': 0, '铅钡': 0},
    '颜色': {
        '黑': 0, '蓝绿': 0, '绿': 0, '浅蓝': 0, '浅绿': 0, '深蓝': 0, '深绿': 0, '紫': 0, '无': 0
    }
}

for index in no_weather_index:
    data = no_weather_dict[index]
    decoration = data[0]
    kind = data[1]
    color = data[2]
    no_weather_info['纹饰'][decoration] += 1
    no_weather_info['类型'][kind] += 1
    if color == 0:
        no_weather_info['颜色']['无'] += 1
    else:
        no_weather_info['颜色'][color] += 1

"""
    表单1堆叠柱状图
"""
# 纹饰图
name_list = no_weather_info['纹饰'].keys()
grid = plt.GridSpec(2, 4)
plt.figure()
plt.subplot(grid[0, 0:2])
plt.bar(range(len(name_list)), no_weather_info['纹饰'].values(), label='未风化', fc='dodgerblue', width=0.4)
plt.bar(range(len(name_list)), weather_info['纹饰'].values(), bottom=list(no_weather_info['纹饰'].values()),
        label='风化', tick_label=list(name_list), fc='xkcd:powder blue', width=0.4)
plt.legend()

# 类型图
name_list = no_weather_info['类型'].keys()
plt.subplot(grid[0, 2:4])
plt.bar(range(len(name_list)), no_weather_info['类型'].values(), label='未风化', fc='dodgerblue', width=0.2)
plt.bar(range(len(name_list)), weather_info['类型'].values(), bottom=list(no_weather_info['类型'].values()),
        label='风化', tick_label=list(name_list), fc='xkcd:powder blue', width=0.2)
plt.legend()

# 颜色图
name_list = no_weather_info['颜色'].keys()
plt.subplot(grid[1, :])
plt.bar(range(len(name_list)), no_weather_info['颜色'].values(), label='未风化', fc='dodgerblue')
plt.bar(range(len(name_list)), weather_info['颜色'].values(), bottom=list(no_weather_info['颜色'].values()),
        label='风化', tick_label=list(name_list), fc='xkcd:powder blue')
plt.legend()
plt.tight_layout()
plt.savefig(r'picture/第一问图像/风化程度与类型、纹饰、颜色关系图/合并图.png')

"""
    结合玻璃类型，分析文物样品表面有无风化化学成分含量的统计规律
    可用数据：
        sheet_2_data_valid： 判断表单二中的数据是否有效
        sheet_2_data_weather: 表单二中每个表项是否风化
        
"""
sheet_2_data_weather = {}
sheet_2_data_kind = {}
sheet_2_data_color = {}
for index in sheet_2_index:
    index_1 = int(index[0:2])
    sheet_2_data_weather[index] = data_sheet_1.loc[index_1]['表面风化']
    sheet_2_data_kind[index] = data_sheet_1.loc[index_1]['类型']
    sheet_2_data_color[index] = data_sheet_1.loc[index_1]['颜色']
"""
    未风化与风化的所有的元素含量
"""
# 未风化与风化的所有的元素含量
no_weather_df = []
no_weather_index_2 = []
weather_df = []
weather_index_2 = []
for index in sheet_2_index:
    data_valid = sheet_2_data_valid[index]
    # 若为无效数据则过滤
    if data_valid == 0:
        continue
    data_weather = sheet_2_data_weather[index]
    if data_weather == '无风化':
        no_weather_index_2.append(index)
        no_weather_df.append(data_sheet_2.loc[index].values)
    else:
        weather_index_2.append(index)
        weather_df.append(data_sheet_2.loc[index].values)

weather_df = pd.DataFrame(weather_df, index=weather_index_2, columns=data_sheet_2.columns)
weather_df.to_excel(r'data/第一问数据/表面有无风化的统计规律/化的所有文物数据.xlsx')
no_weather_df = pd.DataFrame(no_weather_df, index=no_weather_index_2, columns=data_sheet_2.columns)
no_weather_df.to_excel(r'data/第一问数据/表面有无风化的统计规律/未风化的所有文物数据.xlsx')

"""
    表单2所有元素标注，类型和风化或未风化
"""
data_sheet_2_copy = data_sheet_2.copy()
sheet_2_no_weather = []  # 未风化
sheet_2_weather = []  # 风化
sheet_2_severe_weather = []  # 严重风化
sheet_2_color = []
sheet_2_kind = []  # 种类
for index in sheet_2_index:
    data_valid = sheet_2_data_valid[index]
    # 是否风化
    data_weather = sheet_2_data_weather[index]
    # 颜色
    data_color = sheet_2_data_color[index]
    sheet_2_color.append(data_color)
    # 种类
    data_kind = sheet_2_data_kind[index]
    sheet_2_kind.append(data_kind)
    if len(index) > 2:
        judge_word = index[2]
        if judge_word == '未':
            sheet_2_no_weather.append(1)
            sheet_2_weather.append(0)
            sheet_2_severe_weather.append(0)
        elif judge_word == '严':
            sheet_2_no_weather.append(0)
            sheet_2_weather.append(1)
            sheet_2_severe_weather.append(1)
        else:
            sheet_2_no_weather.append(0 if data_weather == '风化' else 1)
            sheet_2_weather.append(1 if data_weather == '风化' else 0)
            sheet_2_severe_weather.append(0)
    else:
        sheet_2_no_weather.append(0 if data_weather == '风化' else 1)
        sheet_2_weather.append(1 if data_weather == '风化' else 0)
        sheet_2_severe_weather.append(0)
# 插入列
data_sheet_2_copy.insert(data_sheet_2_copy.shape[1], '无风化', sheet_2_no_weather)
data_sheet_2_copy.insert(data_sheet_2_copy.shape[1], '风化', sheet_2_weather)
data_sheet_2_copy.insert(data_sheet_2_copy.shape[1], '严重风化', sheet_2_severe_weather)
data_sheet_2_copy.insert(data_sheet_2_copy.shape[1], '颜色', sheet_2_color)
data_sheet_2_copy.insert(data_sheet_2_copy.shape[1], '种类', sheet_2_kind)
# 清洗无用数据
for index in sheet_2_index:
    data_valid = sheet_2_data_valid[index]
    if data_valid == 0:
        data_sheet_2_copy.drop(index, inplace=True)
data_sheet_2_copy.to_excel(r'data/第一问数据/风化前的含量预测/带风化分类与颜色的数据（删去无效数据）.xlsx')

"""
    统计风化的文物中每一个元素的百分比
    新加的列
    data_sheet_2_copy: 是经过处理的数据
    sheet_2_no_weather = []     # 未风化
    sheet_2_weather = []        # 风化
    sheet_2_severe_weather = [] # 严重风化
    sheet_2_color = []          # 颜色
    sheet_2_kind = []           # 种类
"""
sheet_2_pro_path = r'data/第一问数据/风化前的含量预测/带风化分类与颜色的数据（删去无效数据）.xlsx'
sheet_2_pro = pd.read_excel(sheet_2_pro_path, index_col=0, header=0)
# 读取处理过的数据
sheet_2_no_weather = sheet_2_pro.loc[:, '无风化']
sheet_2_weather = sheet_2_pro.loc[:, '风化']
sheet_2_severe_weather = sheet_2_pro.loc[:, '严重风化']
sheet_2_color = sheet_2_pro.loc[:, '颜色']
sheet_2_kind = sheet_2_pro.loc[:, '种类']

sheet_2_pro_index = sheet_2_pro.index
weather_ingredients_dict = {}
no_weather_ingredients_dict = {}
for num in range(len(sheet_2_pro_index)):
    index = sheet_2_pro_index[num]
    # 风化情况
    data_weather = sheet_2_weather[num]
    # 文物种类
    data_kind = sheet_2_kind[num]

    # 无风化情况
    if data_weather == 0:
        for col_index in sheet_2_columns:
            data_col = sheet_2_pro.loc[index][col_index]
            if col_index not in no_weather_ingredients_dict:
                no_weather_ingredients_dict[col_index] = {
                    '高钾': [],
                    '铅钡': []
                }
            no_weather_ingredients_dict[col_index][data_kind].append(data_col)
    else:
        for col_index in sheet_2_columns:
            data_col = sheet_2_pro.loc[index][col_index]
            if col_index not in weather_ingredients_dict:
                weather_ingredients_dict[col_index] = {
                    '高钾': [],
                    '铅钡': []
                }
            weather_ingredients_dict[col_index][data_kind].append(data_col)

"""
    绘制每个元素的折线图与核密度图
"""
for ingredient in weather_ingredients_dict.keys():
    # 有风化高钾数据
    weather_ingredient_kind_1 = weather_ingredients_dict[ingredient]['高钾']
    # 有风化铅钡数据
    weather_ingredient_kind_2 = weather_ingredients_dict[ingredient]['铅钡']
    # 无风化高钾
    no_weather_ingredient_kind_1 = no_weather_ingredients_dict[ingredient]['高钾']
    # 无风化铅钡
    no_weather_ingredient_kind_2 = no_weather_ingredients_dict[ingredient]['铅钡']

    """
        处理数据，最大值，平均值，最小值
    """
    weather_kind_1_dict = {
        'max': max(weather_ingredient_kind_1),
        'mean': np.mean(weather_ingredient_kind_1),
        'min': min(weather_ingredient_kind_1)
    }
    weather_kind_2_dict = {
        'max': max(weather_ingredient_kind_2),
        'mean': np.mean(weather_ingredient_kind_2),
        'min': min(weather_ingredient_kind_2)
    }
    no_weather_kind_1_dict = {
        'max': max(no_weather_ingredient_kind_1),
        'mean': np.mean(no_weather_ingredient_kind_1),
        'min': min(no_weather_ingredient_kind_1)
    }
    no_weather_kind_2_dict = {
        'max': max(no_weather_ingredient_kind_2),
        'mean': np.mean(no_weather_ingredient_kind_2),
        'min': min(no_weather_ingredient_kind_2)
    }
    # 绘制高钾图
    plt.figure()
    sns.set(font='SimHei')
    plt.title('{}含量与风化关系图(高钾）'.format(ingredient))

    sns.kdeplot(weather_ingredient_kind_1, label='有风化高钾')
    sns.kdeplot(no_weather_ingredient_kind_1, label='无风化高钾')
    plt.legend()
    plt.savefig(r'picture/第一问图像/核密度图/高钾/{}含量与风化关系图(高钾）.png'.format(ingredient))

    # 绘制铅钡图
    plt.figure()
    sns.set(font='SimHei')
    plt.title('{}含量与风化关系图(铅钡）'.format(ingredient))
    sns.kdeplot(weather_ingredient_kind_2, label='有风化铅钡')
    sns.kdeplot(no_weather_ingredient_kind_2, label='无风化铅钡')
    plt.legend()
    plt.savefig(r'picture/第一问图像/核密度图/铅钡/{}含量与风化关系图（铅钡）.png'.format(ingredient))
"""
    根据风化点检测数据，预测其风化前的化学成分含量。
"""
# 计算变化率
change_rate_dict = {}
for ingredient in weather_ingredients_dict.keys():
    # 风化后高钾和铅钡的百分比含量
    weather_kind_1_mean = np.mean(weather_ingredients_dict[ingredient]['高钾'])
    weather_kind_2_mean = np.mean(weather_ingredients_dict[ingredient]['铅钡'])
    # 风化前高钾和铅钡的百分比含量
    no_weather_kind_1_mean = np.mean(no_weather_ingredients_dict[ingredient]['高钾'])
    no_weather_kind_2_mean = np.mean(no_weather_ingredients_dict[ingredient]['铅钡'])

    # 风化前方差计算
    weather_kind_1_std = np.std(weather_ingredients_dict[ingredient]['高钾'])
    weather_kind_2_std = np.std(weather_ingredients_dict[ingredient]['铅钡'])
    # 风化后方差计算
    no_weather_kind_1_std = np.std(no_weather_ingredients_dict[ingredient]['高钾'])
    no_weather_kind_2_std = np.std(no_weather_ingredients_dict[ingredient]['铅钡'])

    # 高级方案
    kind_1_change_rate = {
        'before mean': no_weather_kind_1_mean,
        'before std': no_weather_kind_1_std,
        'after mean': weather_kind_1_mean,
        'after std': weather_kind_1_std
    }

    kind_2_change_rate = {
        'before mean': no_weather_kind_2_mean,
        'before std': no_weather_kind_2_std,
        'after mean': weather_kind_2_mean,
        'after std': weather_kind_2_std
    }

    change_rate_dict[ingredient] = {
        '高钾': kind_1_change_rate,
        '铅钡': kind_2_change_rate
    }

change_rate_dict_df = pd.DataFrame(change_rate_dict)

sheet_2_predict_no_weather = pd.DataFrame(index=sheet_2_pro_index, columns=sheet_2_pro.columns)
for num in range(len(sheet_2_pro_index)):
    index = sheet_2_pro_index[num]
    # 风化情况
    data_weather = sheet_2_weather[num]
    # 文物种类
    data_kind = sheet_2_kind[num]

    row_data = sheet_2_predict_no_weather.loc[index]
    for column_index in range(len(sheet_2_pro.columns)):
        column = sheet_2_pro.columns[column_index]
        cur_data = sheet_2_pro.loc[index][column]
        # 风化属性与颜色属性
        if 14 <= column_index <= 18:
            row_data.loc[column] = cur_data
        else:
            # 未风化的情况
            if data_weather == 0:
                row_data.loc[column] = cur_data
            # 风化情况
            else:
                # 高级方案
                before_mean = change_rate_dict[column][data_kind]['before mean']
                before_std = change_rate_dict[column][data_kind]['before std']
                after_mean = change_rate_dict[column][data_kind]['after mean']
                after_std = change_rate_dict[column][data_kind]['after std']

                if after_std == 0:
                    row_data.loc[column] = before_mean
                else:
                    row_data.loc[column] = ((cur_data - after_mean) / after_std) * before_std + before_mean

                if row_data.loc[column] < 0:
                    row_data.loc[column] = 0
sheet_2_predict_no_weather.to_excel(r'data/第二问模型/预测数据（未经标准化）/预测数据（无风化）.xlsx')

"""
    表单2计算所有的风化后的数据
"""

sheet_2_predict_weather = pd.DataFrame(index=sheet_2_pro_index, columns=sheet_2_pro.columns)
for num in range(len(sheet_2_pro_index)):
    index = sheet_2_pro_index[num]
    # 风化情况
    data_weather = sheet_2_weather[num]
    # 文物种类
    data_kind = sheet_2_kind[num]

    row_data = sheet_2_predict_weather.loc[index]
    for column_index in range(len(sheet_2_pro.columns)):
        column = sheet_2_pro.columns[column_index]
        cur_data = sheet_2_pro.loc[index][column]
        # 风化属性与颜色属性
        if 14 <= column_index <= 18:
            row_data.loc[column] = cur_data
        else:
            # 风化的情况
            if data_weather == 1:

                row_data.loc[column] = cur_data
            # 未风化情况
            else:
                # 高级方案
                before_mean = change_rate_dict[column][data_kind]['before mean']
                before_std = change_rate_dict[column][data_kind]['before std']
                after_mean = change_rate_dict[column][data_kind]['after mean']
                after_std = change_rate_dict[column][data_kind]['after std']

                if after_std == 0:
                    row_data.loc[column] = before_mean
                else:
                    if after_std == 0:
                        row_data.loc[column] = before_mean
                    else:
                        row_data.loc[column] = ((cur_data - before_mean) / before_std) * after_std + after_mean

                if row_data.loc[column] < 0:
                    row_data.loc[column] = 0

sheet_2_predict_weather.to_excel(r'data/第二问模型/预测数据（未经标准化）/预测数据（有风化）.xlsx')


"""
    对sheet_2_predict进行归100化
    sheet_2_predict
"""
std_num = 100
for index in sheet_2_predict_no_weather.index:
    column_data = sheet_2_predict_no_weather.loc[index][:14]
    data_sum = sum(column_data)
    if 85 <= data_sum <= 105:
        continue
    else:
        std_rate = data_sum / std_num
        for data_index in range(len(column_data)):
            sheet_2_predict_no_weather.loc[index][data_index] /= std_rate
sheet_2_predict_no_weather.to_excel(r'data/第二问模型/预测数据（经过标准化）/预测数据（无风化）(经过标准化）.xlsx')

"""
    对sheet_2_no_weather进行归100化
"""
for index in sheet_2_predict_weather.index:
    column_data = sheet_2_predict_weather.loc[index][:14]
    data_sum = sum(column_data)
    if 85 <= data_sum <= 105:
        continue
    else:
        std_rate = data_sum / std_num
        for data_index in range(len(column_data)):
            sheet_2_predict_weather.loc[index][data_index] /= std_rate
sheet_2_predict_weather.to_excel(r'data/第二问模型/预测数据（经过标准化）/预测数据(经过标准化）(风化后）.xlsx')

"""
    预测数据归一化
"""
sheet_2_predict_normalize = pd.DataFrame(index=sheet_2_pro_index, columns=sheet_2_pro.columns)
for index in sheet_2_predict_normalize.index:

    for column_index in range(len(sheet_2_pro.columns)):
        column = sheet_2_pro.columns[column_index]
        if 14 <= column_index <= 18:
            sheet_2_predict_normalize.loc[index][column] = sheet_2_predict_no_weather.loc[index][column]
        else:
            # 标准化方案2
            max = np.max(sheet_2_predict_no_weather.loc[:][column])
            min = np.min(sheet_2_predict_no_weather.loc[:][column])
            sheet_2_predict_normalize.loc[index][column] = (sheet_2_predict_no_weather.loc[index][column] - min) / (
                    max - min)

    row_sum = sum(sheet_2_predict_normalize.loc[index][:14])
    if row_sum < 0.85 or row_sum > 1.05:
        sheet_2_predict_normalize.loc[index][:14] /= row_sum

# 列归一化
sheet_2_predict_normalize_copy = sheet_2_predict_normalize.copy()

for index in sheet_2_pro.index:
    for column_index in range(len(sheet_2_pro.columns)):
        column = sheet_2_pro.columns[column_index]
        if column_index < 14:
            col_sum = sum(sheet_2_predict_normalize_copy.loc[:][column])
            if col_sum == 0:
                continue
            sheet_2_predict_normalize.loc[index][column] /= col_sum
sheet_2_predict_normalize.to_excel(
    r'data/第二问模型/预测数据（经过按列归一化）/经过标准化和归一化的预测数据(无风化）.xlsx')

"""
    风化预测数据归一化
"""
sheet_2_predict_weather_normalize = pd.DataFrame(index=sheet_2_pro_index, columns=sheet_2_pro.columns)
for index in sheet_2_predict_normalize.index:
    for column_index in range(len(sheet_2_pro.columns)):
        column = sheet_2_pro.columns[column_index]
        if 14 <= column_index <= 18:
            sheet_2_predict_weather_normalize.loc[index][column] = sheet_2_predict_weather.loc[index][column]
        else:
            # 标准化方案2
            max = np.max(sheet_2_predict_weather.loc[:][column])
            min = np.min(sheet_2_predict_weather.loc[:][column])
            sheet_2_predict_weather_normalize.loc[index][column] = (sheet_2_predict_weather.loc[index][
                                                                        column] - min) / (max - min)
    row_sum = sum(sheet_2_predict_weather_normalize.loc[index][:14])
    if row_sum < 0.85 or row_sum > 1.05:
        sheet_2_predict_weather_normalize.loc[index][:14] /= row_sum

# 列归一化
sheet_2_predict_weather_normalize_copy = sheet_2_predict_weather_normalize.copy()

for index in sheet_2_pro.index:
    for column_index in range(len(sheet_2_pro.columns)):
        column = sheet_2_pro.columns[column_index]
        if column_index < 14:
            col_sum = sum(sheet_2_predict_weather_normalize_copy.loc[:][column])
            if col_sum == 0:
                continue
            sheet_2_predict_weather_normalize.loc[index][column] /= col_sum

sheet_2_predict_weather_normalize.to_excel(
    r'data/第二问模型/预测数据（经过按列归一化）/经过标准化和归一化的预测数据(全部为风化后）.xlsx')

"""
    分类划分（无归一化）
"""
# 风化前的相关元素表
correlation_cofficient_dict = dict.fromkeys(sheet_2_pro.columns[:14])
# 1 代表正相关，0 代表无关，-1 代表负相关
correlation_cofficient_list = [1, 0, 1, 1, 0,
                               1, 1, 1, -1, -1,
                               0, -1, 0, 0]

for index in range(len(correlation_cofficient_dict.keys())):
    key = list(correlation_cofficient_dict.keys())[index]
    correlation_cofficient_dict[key] = correlation_cofficient_list[index]

# 风化后的相关元素表
correlation_cofficient_dict_weather = dict.fromkeys(sheet_2_pro.columns[:14])
correlation_cofficient_list_weather = [1, 0, 1, 1, 0, 0,
                                       0, 0, 1, 1, 1, 1,
                                       0, 0]
for index in range(len(correlation_cofficient_dict_weather.keys())):
    key = list(correlation_cofficient_dict.keys())[index]
    correlation_cofficient_dict_weather[key] = correlation_cofficient_list_weather[index]

"""
    计算分界线相关数据
"""
# 构造绘图数据(风化前）
relevant_ingredient_dict = {}
for index in sheet_2_predict_no_weather.index:
    data_kind = sheet_2_predict_no_weather.loc[index]['种类']

    for ingredient in correlation_cofficient_dict.keys():
        correlation_cofficient = correlation_cofficient_dict[ingredient]
        # 无关
        if correlation_cofficient == 0:
            continue
        # 有关
        else:
            if ingredient not in relevant_ingredient_dict.keys():
                relevant_ingredient_dict[ingredient] = {
                    '高钾': [],
                    '铅钡': []
                }
            ingredient_data = sheet_2_predict_no_weather.loc[index][ingredient]
            relevant_ingredient_dict[ingredient][data_kind].append(ingredient_data)

# 画图数据（风化后）
relevant_ingredient_dict_weather = {}
for index in sheet_2_predict_no_weather.index:
    data_kind = sheet_2_predict_weather.loc[index]['种类']

    for ingredient in correlation_cofficient_dict_weather.keys():
        correlation_cofficient = correlation_cofficient_dict_weather[ingredient]
        # 无关
        if correlation_cofficient == 0:
            continue
        # 有关
        else:
            if ingredient not in relevant_ingredient_dict_weather.keys():
                relevant_ingredient_dict_weather[ingredient] = {
                    '高钾': [],
                    '铅钡': []
                }
            ingredient_data = sheet_2_predict_weather.loc[index][ingredient]
            relevant_ingredient_dict_weather[ingredient][data_kind].append(ingredient_data)

"""
    分类划分，经过归一化数据
"""
# 风化前数据
relevant_ingredient_dict_normalize = {}
for index in sheet_2_predict_no_weather.index:
    data_kind = sheet_2_predict_normalize.loc[index]['种类']

    for ingredient in correlation_cofficient_dict.keys():
        correlation_cofficient = correlation_cofficient_dict[ingredient]
        # 无关
        if correlation_cofficient == 0:
            continue
        # 有关
        else:
            if ingredient not in relevant_ingredient_dict_normalize.keys():
                relevant_ingredient_dict_normalize[ingredient] = {
                    '高钾': [],
                    '铅钡': []
                }
            ingredient_data = sheet_2_predict_normalize.loc[index][ingredient]
            relevant_ingredient_dict_normalize[ingredient][data_kind].append(ingredient_data)

# 风化后数据
relevant_ingredient_dict_weather_normalize = {}
for index in sheet_2_predict_no_weather.index:
    data_kind = sheet_2_predict_normalize.loc[index]['种类']

    for ingredient in correlation_cofficient_dict_weather.keys():
        correlation_cofficient = correlation_cofficient_dict_weather[ingredient]
        # 无关
        if correlation_cofficient == 0:
            continue
        # 有关
        else:
            if ingredient not in relevant_ingredient_dict_weather_normalize.keys():
                relevant_ingredient_dict_weather_normalize[ingredient] = {
                    '高钾': [],
                    '铅钡': []
                }
            ingredient_data = sheet_2_predict_weather_normalize.loc[index][ingredient]
            relevant_ingredient_dict_weather_normalize[ingredient][data_kind].append(ingredient_data)

"""
    计算分界值
"""
# 风化前
best_divide_dict = {}
for ingredient in relevant_ingredient_dict.keys():
    kind_1_data = relevant_ingredient_dict_normalize[ingredient]['高钾']
    kind_2_data = relevant_ingredient_dict_normalize[ingredient]['铅钡']

    # 区间范围, 两个平均值之间, 步长为0.01
    kind_1_mean = np.mean(kind_1_data)
    kind_2_mean = np.mean(kind_2_data)
    if kind_1_mean < kind_2_mean:
        divide_line_list = np.arange(kind_1_mean, kind_2_mean, 0.01)
    else:
        divide_line_list = np.arange(kind_2_mean, kind_1_mean, 0.01)

    # 以平均值的平均值作为最佳分界线初始值
    best_divide = (kind_1_mean + kind_2_mean)
    if kind_1_mean < kind_2_mean:
        kind_1_sel = [sel_data for sel_data in kind_1_data if sel_data > best_divide]
        kind_2_sel = [sel_data for sel_data in kind_2_data if sel_data < best_divide]
    else:
        kind_1_sel = [sel_data for sel_data in kind_1_data if sel_data < best_divide]
        kind_2_sel = [sel_data for sel_data in kind_2_data if sel_data > best_divide]

    # 超过分界线的值到分界线的距离和
    best_distance = sum(abs(data - best_divide) for data in kind_1_sel) \
                    + sum(abs(data - best_divide) for data in kind_2_sel)

    for divide_line in divide_line_list:
        if kind_1_mean < kind_2_mean:
            kind_1_sel = [sel_data for sel_data in kind_1_data if sel_data > divide_line]
            kind_2_sel = [sel_data for sel_data in kind_2_data if sel_data < divide_line]
        else:
            kind_1_sel = [sel_data for sel_data in kind_1_data if sel_data < divide_line]
            kind_2_sel = [sel_data for sel_data in kind_2_data if sel_data > divide_line]

        distance = sum(abs(data - divide_line) for data in kind_1_sel) \
                   + sum(abs(data - divide_line) for data in kind_2_sel)

        # 当找到更好的就更新
        if distance < best_distance:
            best_distance = distance
            best_divide = divide_line

    # 遍历结束，写入结果
    best_divide_dict[ingredient] = {
        'num': best_divide,
        'up': '铅钡' if kind_1_mean < kind_2_mean else '高钾',
        'down': '高钾' if kind_1_mean < kind_2_mean else '铅钡'
    }

    # 画图
    plt.figure()
    plt.title('{}含量分类图'.format(ingredient))
    plt.plot(kind_1_data, label='高钾')
    plt.plot(kind_2_data, label='铅钡')
    plt.hlines(best_divide, -5, 60, linestyles='dashed')
    plt.legend()
    plt.savefig(r'picture/第三问图像/分界线数据/{}(风化前数据）.png'.format(ingredient))

# 输出到excel
best_divide_dict_df = pd.DataFrame(index=best_divide_dict.keys(), columns=['num', 'up', 'down'])
for index in best_divide_dict_df.index:
    for column in best_divide_dict_df.columns:
        best_divide_dict_df.loc[index][column] = best_divide_dict[index][column]
best_divide_dict_df.to_excel(r'data/第三问数据/分界值/分界值（归一化）(风化前）.xlsx')

# 风化后
best_divide_dict_weather = {}
for ingredient in relevant_ingredient_dict_weather.keys():
    kind_1_data = relevant_ingredient_dict_weather_normalize[ingredient]['高钾']
    kind_2_data = relevant_ingredient_dict_weather_normalize[ingredient]['铅钡']

    # 区间范围, 两个平均值之间, 步长为0.01
    kind_1_mean = np.mean(kind_1_data)
    kind_2_mean = np.mean(kind_2_data)
    if kind_1_mean < kind_2_mean:
        divide_line_list = np.arange(kind_1_mean, kind_2_mean, 0.01)
    else:
        divide_line_list = np.arange(kind_2_mean, kind_1_mean, 0.01)

    # 以平均值的平均值作为最佳分界线初始值
    best_divide = (kind_1_mean + kind_2_mean)
    if kind_1_mean < kind_2_mean:
        kind_1_sel = [sel_data for sel_data in kind_1_data if sel_data > best_divide]
        kind_2_sel = [sel_data for sel_data in kind_2_data if sel_data < best_divide]
    else:
        kind_1_sel = [sel_data for sel_data in kind_1_data if sel_data < best_divide]
        kind_2_sel = [sel_data for sel_data in kind_2_data if sel_data > best_divide]

    # 超过分界线的值到分界线的距离和
    best_distance = sum(abs(data - best_divide) for data in kind_1_sel) \
                    + sum(abs(data - best_divide) for data in kind_2_sel)

    for divide_line in divide_line_list:
        if kind_1_mean < kind_2_mean:
            kind_1_sel = [sel_data for sel_data in kind_1_data if sel_data > divide_line]
            kind_2_sel = [sel_data for sel_data in kind_2_data if sel_data < divide_line]
        else:
            kind_1_sel = [sel_data for sel_data in kind_1_data if sel_data < divide_line]
            kind_2_sel = [sel_data for sel_data in kind_2_data if sel_data > divide_line]

        distance = sum(abs(data - divide_line) for data in kind_1_sel) \
                   + sum(abs(data - divide_line) for data in kind_2_sel)

        # 当找到更好的就更新
        if distance < best_distance:
            best_distance = distance
            best_divide = divide_line

    # 遍历结束，写入结果
    best_divide_dict_weather[ingredient] = {
        'num': best_divide,
        'up': '铅钡' if kind_1_mean < kind_2_mean else '高钾',
        'down': '高钾' if kind_1_mean < kind_2_mean else '铅钡'
    }

    # 画图
    plt.figure()
    plt.title('{}含量分类图'.format(ingredient))
    plt.plot(kind_1_data, label='高钾')
    plt.plot(kind_2_data, label='铅钡')
    plt.hlines(best_divide, -5, 60, linestyles='dashed')
    plt.legend()
    plt.savefig(r'picture/第三问图像/分界线数据/{}(风化后数据）.png'.format(ingredient))

# 输出到excel
best_divide_dict_weather_df = pd.DataFrame(index=best_divide_dict_weather.keys(), columns=['num', 'up', 'down'])
for index in best_divide_dict_weather_df.index:
    for column in best_divide_dict_weather_df.columns:
        best_divide_dict_weather_df.loc[index][column] = best_divide_dict_weather[index][column]
best_divide_dict_weather_df.to_excel(r'data/第三问数据/分界值/分界值（归一化）(风化后）.xlsx')

"""
    计算权重
"""

# 计算权重
ingredient_num = [
    -0.693670815, 0.119156192, -0.717030234, -0.344828276, -0.095775661, -0.202264054, -0.272875667,
    -0.054967783, 0.757008765, 0.534965174, 0.285351187, 0.53499528, -0.096593581, 0.121150082
]
# 每种元素的权重(风化前）
ingredient_weight = {}
for ingredient_index in range(len(correlation_cofficient_dict.keys())):
    ingredient = list(correlation_cofficient_dict.keys())[ingredient_index]
    relevant = correlation_cofficient_dict[ingredient]
    if relevant == 0:
        continue
    else:
        ingredient_weight[ingredient] = ingredient_num[ingredient_index]

# 权重归一化
weight_sum = sum(abs(value) for value in ingredient_weight.values())
for ingredient in ingredient_weight.keys():
    value = ingredient_weight[ingredient]
    ingredient_weight[ingredient] = value / weight_sum
ingredient_weight_df = pd.DataFrame(ingredient_weight.values(), index=ingredient_weight.keys(), columns=['权重'])
ingredient_weight_df.to_excel(r'picture/第三问图像/分界线权重/权重（风化前）.xlsx')

# 每种元素的权重(风化后）
ingredient_weight_weather = {}
for ingredient_index in range(len(correlation_cofficient_dict_weather.keys())):
    ingredient = list(correlation_cofficient_dict_weather.keys())[ingredient_index]
    relevant = correlation_cofficient_dict_weather[ingredient]
    if relevant == 0:
        continue
    else:
        ingredient_weight_weather[ingredient] = ingredient_num[ingredient_index]

# 权重归一化
weight_sum = sum(abs(value) for value in ingredient_weight_weather.values())
for ingredient in ingredient_weight_weather.keys():
    value = ingredient_weight_weather[ingredient]
    ingredient_weight_weather[ingredient] = value / weight_sum

ingredient_weight_weather_df = pd.DataFrame(ingredient_weight_weather.values(), index=ingredient_weight_weather.keys(),
                                            columns=['权重'])
ingredient_weight_weather_df.to_excel(r'picture/第三问图像/分界线权重/权重（风化后）.xlsx')

"""
    风化前的规律验证
"""
judge_result = {}
correct_num = 0
for index in sheet_2_predict_no_weather.index:

    kind_1_score = 0  # 高钾得分
    kind_2_score = 0  # 铅钡得分

    for ingredient in ingredient_weight.keys():
        # 分界线
        divide_line = best_divide_dict[ingredient]['num']
        # 对应值
        ingredient_data = sheet_2_predict_normalize.loc[index][ingredient]
        # 上方对应种类
        up_kind = best_divide_dict[ingredient]['up']
        # 下方对应种类
        down_kind = best_divide_dict[ingredient]['down']
        # 权重(绝对值)
        weight = abs(ingredient_weight[ingredient])

        # 确定种类
        if ingredient_data >= divide_line:
            kind = up_kind
        else:
            kind = down_kind

        # 对应加分
        if kind == '高钾':
            kind_1_score += weight * abs(ingredient_data - divide_line)
        else:
            kind_2_score += weight * abs(ingredient_data - divide_line)

    # 遍历完所有元素
    if kind_1_score < kind_2_score:
        predict_kind = '铅钡'
    else:
        predict_kind = '高钾'
    judge_result[index] = {
        '预测种类': predict_kind,
        '高钾得分': kind_1_score,
        '铅钡得分': kind_2_score
    }
    real_kind = sheet_2_predict_normalize.loc[index]['种类']
    if predict_kind == real_kind:
        correct_num += 1

judge_result_df = pd.DataFrame(judge_result.values(), index=judge_result.keys(),
                               columns=['预测种类', '高钾得分', '铅钡得分'])
judge_result_df.to_excel(r'data/第三问数据/模型验证数据/模型检验数据（风化前）.xlsx')

"""
    风化后的规律验证
"""
judge_result_weather = {}
correct_num = 0
for index in sheet_2_predict_no_weather.index:

    kind_1_score = 0  # 高钾得分
    kind_2_score = 0  # 铅钡得分

    for ingredient in ingredient_weight_weather.keys():
        # 分界线
        divide_line = best_divide_dict_weather[ingredient]['num']
        # 对应值
        ingredient_data = sheet_2_predict_weather_normalize.loc[index][ingredient]
        # 上方对应种类
        up_kind = best_divide_dict_weather[ingredient]['up']
        # 下方对应种类
        down_kind = best_divide_dict_weather[ingredient]['down']
        # 权重(绝对值)
        weight = abs(ingredient_weight_weather[ingredient])

        # 确定种类
        if ingredient_data >= divide_line:
            kind = up_kind
        else:
            kind = down_kind

        # 对应加分
        if kind == '高钾':
            kind_1_score += weight * abs(ingredient_data - divide_line)
        else:
            kind_2_score += weight * abs(ingredient_data - divide_line)

    # 遍历完所有元素
    if kind_1_score < kind_2_score:
        predict_kind = '铅钡'
    else:
        predict_kind = '高钾'
    judge_result_weather[index] = {
        '预测种类': predict_kind,
        '高钾得分': kind_1_score,
        '铅钡得分': kind_2_score
    }
    real_kind = sheet_2_predict_normalize.loc[index]['种类']
    if predict_kind == real_kind:
        correct_num += 1

judge_result_df = pd.DataFrame(judge_result_weather.values(), index=judge_result.keys(),
                               columns=['预测种类', '高钾得分', '铅钡得分'])
judge_result_df.to_excel(r'data/第三问数据/模型验证数据/模型检验数据（风化后）.xlsx')

"""
    亚类数据的合理性分析和灵敏度分析(另一份代码里）
"""

"""
    第三问鉴别种类
    可用数据：
        best_divide_dict = {
            'num': best_divide,
            'up': '铅钡' if kind_1_mean < kind_2_mean else '高钾',
            'down': '高钾' if kind_1_mean < kind_2_mean else '铅钡'
        }
        ingredient_weight: 权重
"""
"""
    对表格3进行归一化处理
"""
data_sheet_3_normalize = pd.DataFrame(index=data_sheet_3.index, columns=data_sheet_3.columns)
for index in data_sheet_3_normalize.index:
    for column in data_sheet_3_normalize.columns:
        if column == '表面风化':
            data_sheet_3_normalize.loc[index][column] = data_sheet_3.loc[index][column]
            continue
        mean = np.mean(data_sheet_3.loc[:][column])
        std = np.mean(data_sheet_3.loc[:][column])
        data_sheet_3_normalize.loc[index][column] = (data_sheet_3.loc[index][column] - mean) / std

predict_result_3 = {}
for index in data_sheet_3.index:
    kind_1_score = 0
    kind_2_score = 0

    data_kind = data_sheet_3_normalize.loc[index]['表面风化']
    # 无风化的分界标准
    if data_kind == '无风化':
        for ingredient in ingredient_weight.keys():
            # 分界线
            divide_line = best_divide_dict[ingredient]['num']
            # 上方对应种类
            up_kind = best_divide_dict[ingredient]['up']
            # 下方对应种类
            down_kind = best_divide_dict[ingredient]['down']
            # 对应值
            ingredient_data = data_sheet_3_normalize.loc[index][ingredient]
            # 权重(绝对值)
            weight = abs(ingredient_weight[ingredient])
            # 确定种类
            if ingredient_data >= divide_line:
                kind = up_kind
            else:
                kind = down_kind
            # 对应加分
            if kind == '高钾':
                kind_1_score += weight * abs(ingredient_data - divide_line)
            else:
                kind_2_score += weight * abs(ingredient_data - divide_line)

    # 有风化的分界标准
    else:
        for ingredient in ingredient_weight_weather.keys():
            # 分界线
            divide_line = best_divide_dict_weather[ingredient]['num']
            # 上方对应种类
            up_kind = best_divide_dict_weather[ingredient]['up']
            # 下方对应种类
            down_kind = best_divide_dict_weather[ingredient]['down']
            # 对应值
            ingredient_data = data_sheet_3_normalize.loc[index][ingredient]
            # 权重(绝对值)
            weight = abs(ingredient_weight_weather[ingredient])
            # 确定种类
            if ingredient_data >= divide_line:
                kind = up_kind
            else:
                kind = down_kind
            # 对应加分
            if kind == '高钾':
                kind_1_score += weight * abs(ingredient_data - divide_line)
            else:
                kind_2_score += weight * abs(ingredient_data - divide_line)

    # 遍历完所有元素
    if kind_1_score < kind_2_score:
        predict_kind = '铅钡'
    else:
        predict_kind = '高钾'

    predict_result_3[index] = {
        '预测种类': predict_kind,
        '高钾得分': kind_1_score,
        '铅钡得分': kind_2_score
    }

predict_result_3_df = pd.DataFrame(predict_result_3.values(), index=data_sheet_3.index,
                                   columns=['预测种类', '高钾得分', '铅钡得分'])
predict_result_3_df.to_excel(r'data/第三问数据/预测结果/第三问预测结果(归一化）.xlsx')

"""
    敏感性分析
    分别对每一类数据进行正负百分之100的调整，判断分类结果是否有变化(只对有风化进行含量改变）
"""
result_dict = {}
change_num = 1
for change_ingredient_index in range(len(ingredient_weight_weather.keys()) - change_num + 1):

    # 改动元素列表
    change_ingredient_list = list(ingredient_weight_weather.keys())[
                             change_ingredient_index:(change_ingredient_index + change_num)]

    # 复制数据原始表
    cur_data_sheet_3_normalize = data_sheet_3_normalize.copy()

    # 改动率列表
    change_rate_list = np.arange(0, 2, 0.1)

    cur_result_dict = {}  # 同一元素每个变化率的影响
    # 数据改动
    for change_rate in change_rate_list:

        correct_num = 0
        # 预测分类
        for index in cur_data_sheet_3_normalize.index:

            kind_1_score = 0
            kind_2_score = 0

            data_kind = data_sheet_3_normalize.loc[index]['表面风化']
            # 实际方案
            # # 无风化
            # if data_kind == '无风化':
            #     for ingredient in ingredient_weight.keys():
            #         # 分界线
            #         divide_line = best_divide_dict[ingredient]['num']
            #         # 对应值
            #         if ingredient == change_ingredient:
            #             ingredient_data = change_data.loc[index]
            #         else:
            #             ingredient_data = cur_data_sheet_3_normalize.loc[index][ingredient]
            #         # 上方对应种类
            #         up_kind = best_divide_dict[ingredient]['up']
            #         # 下方对应种类
            #         down_kind = best_divide_dict[ingredient]['down']
            #         # 权重(绝对值)
            #         weight = abs(ingredient_weight[ingredient])
            #
            #         # 确定种类
            #         if ingredient_data >= divide_line:
            #             kind = up_kind
            #         else:
            #             kind = down_kind
            #
            #         # 对应加分
            #         if kind == '高钾':
            #             kind_1_score += weight * abs(ingredient_data - divide_line)
            #         else:
            #             kind_2_score += weight * abs(ingredient_data - divide_line)
            # # 有风化
            # else:
            #     for ingredient in ingredient_weight_weather.keys():
            #         # 分界线
            #         divide_line = best_divide_dict_weather[ingredient]['num']
            #         # 对应值
            #         if ingredient == change_ingredient:
            #             ingredient_data = change_data.loc[index]
            #         else:
            #             ingredient_data = cur_data_sheet_3_normalize.loc[index][ingredient]
            #         # 上方对应种类
            #         up_kind = best_divide_dict_weather[ingredient]['up']
            #         # 下方对应种类
            #         down_kind = best_divide_dict_weather[ingredient]['down']
            #         # 权重(绝对值)
            #         weight = abs(ingredient_weight_weather[ingredient])
            #
            #         # 确定种类
            #         if ingredient_data >= divide_line:
            #             kind = up_kind
            #         else:
            #             kind = down_kind
            #
            #         # 对应加分
            #         if kind == '高钾':
            #             kind_1_score += weight * abs(ingredient_data - divide_line)
            #         else:
            #             kind_2_score += weight * abs(ingredient_data - divide_line)
            # 只用无风化的分界线预测
            for ingredient in ingredient_weight_weather.keys():
                # 分界线
                divide_line = best_divide_dict_weather[ingredient]['num']
                # 对应值
                if ingredient in change_ingredient_list:
                    ingredient_data = cur_data_sheet_3_normalize.loc[index][ingredient] * change_rate
                else:
                    ingredient_data = cur_data_sheet_3_normalize.loc[index][ingredient]
                # 上方对应种类
                up_kind = best_divide_dict_weather[ingredient]['up']
                # 下方对应种类
                down_kind = best_divide_dict_weather[ingredient]['down']
                # 权重(绝对值)
                weight = abs(ingredient_weight_weather[ingredient])

                # 确定种类
                if ingredient_data >= divide_line:
                    kind = up_kind
                else:
                    kind = down_kind

                # 对应加分
                if kind == '高钾':
                    kind_1_score += weight * abs(ingredient_data - divide_line)
                else:
                    kind_2_score += weight * abs(ingredient_data - divide_line)

            # 遍历完所有元素
            if kind_1_score < kind_2_score:
                predict_kind = '铅钡'
                # judge_result[index] = '铅钡'
            else:
                predict_kind = '高钾'

            # 计算正确率
            if predict_kind == predict_result_3[index]['预测种类']:
                correct_num += 1

        # 对于每个元素，关于改动幅度的正确率表
        cur_result_dict[change_rate] = correct_num / len(predict_result_3)

    # 每个元素的结果写入
    result_dict[change_ingredient_index, change_ingredient_index + change_num] = cur_result_dict

result_dict_df = pd.DataFrame(result_dict.values(), index=result_dict.keys())
result_dict_df.to_excel('data/第三问数据/灵敏度分析/灵敏度分析数据(消去{}个元素）.xlsx'.format(change_num))

"""
    第三问灵敏度分析思路二
"""
correct_num_1 = 0  # 无风化分界线的正确数
correct_num_2 = 0  # 有风化分界线的正确数
predict_result_3_2 = {}
# 预测分类
for index in data_sheet_3_normalize.index:

    kind_1_score = 0
    kind_2_score = 0

    data_kind = data_sheet_3_normalize.loc[index]['表面风化']
    # 只用无风化的分界线预测
    for ingredient in ingredient_weight.keys():
        # 分界线
        divide_line = best_divide_dict[ingredient]['num']
        # 对应值
        ingredient_data = data_sheet_3_normalize.loc[index][ingredient]
        # 上方对应种类
        up_kind = best_divide_dict[ingredient]['up']
        # 下方对应种类
        down_kind = best_divide_dict[ingredient]['down']
        # 权重(绝对值)
        weight = abs(ingredient_weight[ingredient])

        # 确定种类
        if ingredient_data >= divide_line:
            kind = up_kind
        else:
            kind = down_kind

        # 对应加分
        if kind == '高钾':
            kind_1_score += weight * abs(ingredient_data - divide_line)
        else:
            kind_2_score += weight * abs(ingredient_data - divide_line)

    # 遍历完所有元素
    if kind_1_score < kind_2_score:
        predict_kind = '铅钡'
    else:
        predict_kind = '高钾'

    # 计算正确率
    if predict_kind == predict_result_3[index]['预测种类']:
        correct_num_1 += 1
    # 有风化的分界线预测
    for ingredient in ingredient_weight_weather.keys():
        # 分界线
        divide_line = best_divide_dict_weather[ingredient]['num']
        # 对应值
        ingredient_data = data_sheet_3_normalize.loc[index][ingredient]
        # 上方对应种类
        up_kind = best_divide_dict_weather[ingredient]['up']
        # 下方对应种类
        down_kind = best_divide_dict_weather[ingredient]['down']
        # 权重(绝对值)
        weight = abs(ingredient_weight_weather[ingredient])

        # 确定种类
        if ingredient_data >= divide_line:
            kind = up_kind
        else:
            kind = down_kind

        # 对应加分
        if kind == '高钾':
            kind_1_score += weight * abs(ingredient_data - divide_line)
        else:
            kind_2_score += weight * abs(ingredient_data - divide_line)

    # 遍历完所有元素
    if kind_1_score < kind_2_score:
        predict_kind = '铅钡'
    else:
        predict_kind = '高钾'

    # 计算正确率
    if predict_kind == predict_result_3[index]['预测种类']:
        correct_num_2 += 1

predict_result_3_2 = {
    '风化前': correct_num_1 / len(predict_result_3),
    '风化后': correct_num_2 / len(predict_result_3),
    '均使用': 1
}
"""
    第三问灵敏度分析绘图
"""

import matplotlib.ticker as ticker

plt.figure()
plt.title('各元素含量对于预测正确率的影响')
for ingredient in result_dict.keys():
    y = list(result_dict[ingredient].values())
    plt.plot(y, marker='*', label=ingredient)

plt.gca().xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
x = np.linspace(0, 2, 21)
x_1 = [float('{:2f}'.format(i)) for i in x]
plt.xticks(range(len(x)), x_1)

plt.xlabel('元素含量')
plt.ylabel('正确率')
plt.legend()
plt.savefig(r'picture/第三问图像/灵敏度分析/灵敏度分析（有风化分界线）（改变{}个元素）.png'.format(change_num))

"""
    第四问
"""
import pingouin as pg
sheet_2_predict_normalize_kind_1 = sheet_2_predict_normalize[sheet_2_predict_normalize['种类'] == '高钾']
sheet_2_predict_normalize_kind_2 = sheet_2_predict_normalize[sheet_2_predict_normalize['种类'] == '铅钡']
sheet_2_predict_normalize_kind_1 = sheet_2_predict_normalize_kind_1.iloc[:, :14]
sheet_2_predict_normalize_kind_2 = sheet_2_predict_normalize_kind_2.iloc[:, :14]
sheet_2_predict_normalize_kind_1 = pd.DataFrame(sheet_2_predict_normalize_kind_1).astype(float)
sheet_2_predict_normalize_kind_2 = pd.DataFrame(sheet_2_predict_normalize_kind_2).astype(float)
corr_kind_1 = sheet_2_predict_normalize_kind_1.corr('spearman')
corr_kind_2 = sheet_2_predict_normalize_kind_2.corr('spearman')

writer = pd.ExcelWriter('data/第四问数据/相关系数.xlsx')

corr_kind_1.to_excel(writer, sheet_name='高钾')
corr_kind_2.to_excel(writer, sheet_name='铅钡')
writer.save()

# 偏相关系数
partial_corr_kind_1 = sheet_2_predict_normalize_kind_1.pcorr()
# print(partial_corr_kind_1)
partial_corr_kind_2 = sheet_2_predict_normalize_kind_2.pcorr()

writer = pd.ExcelWriter('data/第四问数据/偏相关系数.xlsx')
partial_corr_kind_1.to_excel(writer, sheet_name='高钾')
partial_corr_kind_2.to_excel(writer, sheet_name='铅钡')
writer.save()
"""
    热力图绘制
"""


def heat_map(data, title, path):
    # 绘图风格
    style.use('ggplot')
    sns.set_style('whitegrid')
    sns.set_style({"font.sans-serif": ['simhei', 'Droid Sans Fallback']})
    # 设置滑板尺寸
    fig = plt.figure(figsize=(12, 10))

    # 画热力图
    mask = np.zeros_like(data, dtype=bool)
    mask[np.triu_indices_from(mask)] = True

    sns.heatmap(data,
                mask=mask,
                annot=True,
                cmap=sns.diverging_palette(20, 220, n=200),
                center=0)

    plt.title('{}'.format(title), fontsize=15)
    dest = os.path.join(path, title)
    plt.savefig(dest)
    plt.cla()
    plt.close(fig)

heat_map(corr_kind_1, '高钾相关系数', 'picture/第四问热力图')
heat_map(corr_kind_2, '铅钡相关系数', 'picture/第四问热力图')
heat_map(partial_corr_kind_1, '高钾偏相关系数', 'picture/第四问热力图')
heat_map(partial_corr_kind_2, '铅钡偏相关系数', 'picture/第四问热力图')