#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np

def find_global_min_max(folder_path):
    """
    遍历 folder_path 下所有 .npz 文件，找出整个数据集的全局最小值和最大值。
    
    :param folder_path: 存放 npz 文件的文件夹路径
    :return: (global_min, global_max)
    """
    global_min = float('inf')
    global_max = float('-inf')

    # 遍历文件夹下的所有文件
    for filename in os.listdir(folder_path):
        if filename.endswith('.npz'):
            file_path = os.path.join(folder_path, filename)
            # 加载 npz 文件
            data = np.load(file_path)
            
            # npz 文件中可能包含多个数组，通过 data.files 获取所有数组名字
            for array_name in data.files:
                arr = data[array_name]
                arr_min = np.min(arr)
                arr_max = np.max(arr)

                # 更新全局最小值和最大值
                if arr_min < global_min:
                    global_min = arr_min
                if arr_max > global_max:
                    global_max = arr_max

    return global_min, global_max


if __name__ == "__main__":
    # 指定存放 npz 文件的文件夹名称或路径
    folder = "kimberlina_co2_train_label"
    
    # 调用函数获取全局最小值和最大值
    min_val, max_val = find_global_min_max(folder)

    # 打印结果
    print(f"Global min across all NPZ files in '{folder}': {min_val}")
    print(f"Global max across all NPZ files in '{folder}': {max_val}")
