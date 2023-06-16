import numpy as np
import pandas as pd

"""
    预测数据读取
"""
data_no_weather_path = r'data/第一问数据/预测数据(经过标准化）.xlsx'
data_weather_path = r'data/第一问数据/预测数据(经过标准化）(风化后）.xlsx'

data_no_weather = pd.read_excel(data_no_weather_path, index_col=0)
data_weather = pd.read_excel(data_weather_path, index_col=0)


"""
    分界线读取
"""
divide_line_no_weather_path = r'data/第三问数据/分界值/分界值（归一化）(风化前）.xlsx'
divide_line_weather_path = r'data/第三问数据/分界值/分界值（归一化）(风化后）.xlsx'

divide_line_no_weather = pd.read_excel(divide_line_no_weather_path, index_col=0)
divide_line_weather = pd.read_excel(divide_line_weather_path, index_col=0)
"""
    数据还原
"""
# 无风化分界线还原
new_divide_line_no_weather = pd.DataFrame(index=divide_line_no_weather.index, columns=divide_line_no_weather.columns)
for ingredient in divide_line_no_weather.index:
    # 方案1
    max = np.max(data_no_weather.loc[:][ingredient])
    min = np.max(data_no_weather.loc[:][ingredient])
    new_divide_line_no_weather.loc[ingredient]['num'] = divide_line_no_weather.loc[ingredient]['num'] * (max - min) + min

    new_divide_line_no_weather.loc[ingredient]['up'] = divide_line_no_weather.loc[ingredient]['up']
    new_divide_line_no_weather.loc[ingredient]['down'] = divide_line_no_weather.loc[ingredient]['down']
divide_line_no_weather = new_divide_line_no_weather
# 按行归100化
divide_line_no_weather.to_excel(r'data/第三问数据/分界值/分界值（风化前）.xlsx')

# 有风化分界线还原
new_divide_line_weather = pd.DataFrame(index=divide_line_weather.index, columns=divide_line_weather.columns)
for ingredient in divide_line_weather.index:
    # 方案1
    max = np.max(data_weather.loc[:][ingredient])
    min = np.min(data_weather.loc[:][ingredient])
    new_divide_line_weather.loc[ingredient]['num'] = divide_line_weather.loc[ingredient]['num'] * (max - min) + min

    new_divide_line_weather.loc[ingredient]['up'] = divide_line_weather.loc[ingredient]['up']
    new_divide_line_weather.loc[ingredient]['down'] = divide_line_weather.loc[ingredient]['down']
divide_line_weather = new_divide_line_weather
divide_line_weather.to_excel(r'data/第三问数据/分界值/分界值（风化后）.xlsx')