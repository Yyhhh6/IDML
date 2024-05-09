import numpy as np
from torch.utils.data.sampler import Sampler
"""import torch
import torch.nn.functional as F

from tqdm import *"""
from collections import defaultdict
import copy
import random

"""class BalancedSampler(Sampler):
    def __init__(self, data_source, batch_size, images_per_class=3):
        self.data_source = data_source
        self.ys = data_source.ys
        self.num_groups = batch_size // images_per_class
        self.batch_size = batch_size
        self.num_instances = images_per_class
        self.num_samples = len(self.ys)
        self.num_classes = len(set(self.ys))

    def __len__(self):
        return self.num_samples

    def __iter__(self):
        num_batches = len(self.data_source) // self.batch_size
        ret = []
        while num_batches > 0:
            sampled_classes = np.random.choice(self.num_classes, self.num_groups, replace=False)
            for i in range(len(sampled_classes)):
                ith_class_idxs = np.nonzero(np.array(self.ys) == sampled_classes[i])[0]
                class_sel = np.random.choice(ith_class_idxs, size=self.num_instances, replace=True)
                ret.extend(np.random.permutation(class_sel))
            num_batches -= 1
        return iter(ret)"""

class BASampler(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    Args:
    - data_source (list): list of (img_path, pid, camid).
    - images_per_class (int): number of instances per identity in a batch.
    - batch_size (int): number of examples in a batch.
    """

    def __init__(self, data_source, batch_size, images_per_class=3):#此处直接用源码，要改train里面的第三个参数相关内容
        self.data_source = data_source
        self.batch_size = batch_size
        self.images_per_class = images_per_class
        self.num_pids_per_batch = self.batch_size // self.images_per_class
        # self.index_dic = defaultdict(list)
        self.index_dic = {}
        for index, pid in enumerate(self.data_source.ys):
            inner_list = self.index_dic.get(pid, [])
            inner_list.append(index)
            self.index_dic[pid] = inner_list
        self.pids = list(self.index_dic.keys()) # 类别列表，每个类别有唯一一个标签ys

    def __iter__(self):
        # batch_idxs_dict = defaultdict(list)
        batch_idxs_dict = {} # 结构：{F(或pid)：[[batch_idxs], ...], ...}

        for pid in self.pids:
            # idxs = copy.deepcopy(self.index_dic[pid])
            for MIDx, MIDy in self.data_source.connected_MID[pid]: # 此处的pid表示家庭（类别），也就是前面的F
                idxs = self.data_source.get_idx[pid][MIDx] + self.data_source.get_idx[pid][MIDy] # 有血缘关系两个个体的照片的索引合起来
                if len(idxs) < self.images_per_class:
                    idxs = np.random.choice(idxs, size=self.images_per_class, replace=True)
                random.shuffle(idxs)
                batch_idxs = [] # 存储在ys、im_paths或I中的索引
                for idx in idxs:
                    batch_idxs.append(idx)
                    if len(batch_idxs) == self.images_per_class:
                        inner_list = batch_idxs_dict.get(pid, [])
                        inner_list.append(batch_idxs)
                        batch_idxs_dict[pid] = inner_list # 在列表中嵌套列表
                        batch_idxs = []
        # print(f"batch_idxs_dict is {batch_idxs_dict}") 正常
        avai_pids = copy.deepcopy(self.pids)
        final_idxs = [] # 结构：[idxs...]各个batch的dix之间没有间隔，因为外面还套了一个batchsampler。实际上效果就是[[一个batch的idxs], ...]

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            # final_idx = []
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)# TODO：对每batch_size个数据进行打乱，可以定义一个final_idx打乱后才extend进。
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)
            # final_idxs.append(final_idx)

        # print(f"final_idx[0] is {final_idxs[0]}")
        # print(f"num {self.num_pids_per_batch}")
        # exit(1)
        self.length = len(final_idxs)
        return iter(final_idxs)

    def __len__(self):
        return self.length
        