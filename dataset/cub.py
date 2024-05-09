from .base import *
# from collections import defaultdict

# class TreeNode:
#     def __init__(self, data, father = None, mother = None, children = None, sibling = None):
#         self.data = data
#         self.father = father
#         self.mother = mother
#         self.children = children
#         self.sibling = sibling 

#     def add_child(self, child):
#         self.children.append(child)

#     def remove_child(self, child):
#         self.children.remove(child)

# class Tree:
#     def __init__(self, root_data):
#         self.root = TreeNode(root_data)

#     # 根据类别判断如何添加这两个节点
#     def add_node(self, node1, node2, relation):
#         exist1 = self.find_node(self.root, node1)
#         exist2 = self.find_node(self.root, node2)
#         # 如果两者都存在，说明先前已经处理过或者能够从其他途径判断二者关系
#         if exist1 and exist2:
#             return
#         # 如果只有一个存在
#         if exist1 or exist2:
#             exist, nonexist = None, None
#             reverse = False # relation是否反过来
#             if exist1:
#                 exist, nonexist = exist1, exist2
#             else:
#                 exist, nonexist = exist2, exist1
#                 reverse = True
#             if relation == "fd" or relation == "fs":
#                 if reverse:
#                     pexist_m, pexist_f = exist.mother, exist.father
#                     if pexist_m == self.root and pexist_f == self.root: # 如果父母节点都是虚根节点
#                         new_node = TreeNode(nonexist.data, self.root, self.root, self.root.children)
#                         self.root.children
#                         pexist_f.children = new_node
#                         for child in new_node.children:
#                             child.parent = new_node
#                     else:
#                         ppexist = pexist.parent
#                         new_node = TreeNode(nonexist.data, ppexist, )
#                         ppexist.children.append()
#                         return new_node
#                 new_node = TreeNode(nonexist.data)


#         parent_node = self.find_node(self.root, parent_data)
#         if parent_node:
#             new_node = TreeNode(node_data)
#             parent_node.add_child(new_node)
#         else:
#             print("Parent not found")

#     def remove_node(self, node_data):
#         node_to_remove = self.find_node(self.root, node_data)
#         if node_to_remove:
#             parent_node = self.find_parent_node(self.root, node_data)
#             parent_node.remove_child(node_to_remove)
#         else:
#             print("Node not found")

#     def find_node(self, current_node, node_data):
#         if current_node.data == node_data:
#             return current_node
#         for child in current_node.children:
#             found_node = self.find_node(child, node_data)
#             if found_node:
#                 return found_node
#         return None
    
#     def find_parent_node(self, current_node, node_data):
#         for child in current_node.children:
#             if child.data == node_data:
#                 return current_node
#             parent_node = self.find_parent_node(child, node_data)
#             if parent_node:
#                 return parent_node
#         return None



class CUBirds(BaseDataset):
    def __init__(self, root, mode, transform = None):
        #self.root = root 
        self.mode = mode
        self.transform = transform
        self.get_idx = {} # 结构：{F:{MID:[idx...], ...}, ...}。获得每个人的所有照片的索引
        self.connected_MID = {} # 结构：{ys:[(MIDx,MIDy), ...], ...}。由pairs获得每个家庭中有直接血缘关系的人
        if self.mode == 'train':
            self.sample_path = root + '/sample0/train_sort_A2_m.txt'
            self.root = root + '/Train_A/train-faces'
            self.classes = range(len(os.listdir(self.root)))
        elif self.mode == 'eval':
            self.sample_path = root + '/sample0/val_A.txt'#两种val的文件有什么区别？
            self.root = root + '/Validation_A/val-faces'
            self.classes = range(len(os.listdir(self.root)))

        self.sample_dict, connected = self.load_sample()
        BaseDataset.__init__(self, self.root, self.mode, self.transform)
        index = 0
        for idx, (F, MID_and_path) in enumerate(self.sample_dict.items()):
            self.connected_MID[idx] = connected[F]
            for MID, paths in MID_and_path.items():
                for path in paths:
                    self.ys.append(idx) # 每个家庭标签，有多少张图片就有多少项
                    self.im_paths.append(os.path.join(self.root, F, MID, path))
                    self.get_idx[idx] = self.get_idx.get(idx, {})
                    inner_list = self.get_idx[idx].get(MID, [])
                    inner_list.append(index)
                    self.get_idx[idx][MID] = inner_list
                    
                    self.I[index] = idx # index是索引，idx是对应的类别标签，可以由此用索引获得类别
                    index += 1
        # print(f"get_index is {self.get_idx}")
    # 从直接读取照片来确定分类，到读取pairs确定分类 
    """
    以F、MID作为索引，用列表存储照片的位置
    # 用树来表示结构关系，从叶节点回溯获得有血缘关系的多组索引（此处注意到数据集有特征：标签和顺序相对应）
    直接通过pairs的数据获得血亲的多组索引
    再在sampler采样时若抽到对应家庭，随机选取一组索引，再在其中对照片进行平衡采样
    """
    def load_sample(self):
        # sample_dict = defaultdict(list)
       
        sample_dict = {} # 结构：{F:{MID:[path...], ...}, ...}
        connected_dict = {} # 结构：{F:[(MIDx,MIDy), ...], ...}
        # family_tree = {} # 结构：{F:root, ...}
        sample_path = self.sample_path
        f = open(sample_path, "r+", encoding='utf-8') 
        # i = 0
        while True:
            # i += 1
            line = f.readline().replace('\n', '')
            if not line: # 最后一行读取时为空
                # print(f"not line i is:{i}")
                break
            else:
                tmp = line.split()

                path1 = tmp[1].split('/')
                inner_dict1 = sample_dict.get(path1[2], {})
                inner_list1 = inner_dict1.get(path1[3], [])
                inner_list1.append(path1[4])
                sample_dict[path1[2]] = inner_dict1
                sample_dict[path1[2]][path1[3]] = inner_list1
                path2 = tmp[2].split('/')
                inner_dict2 = sample_dict.get(path2[2], {})
                inner_list2 = inner_dict2.get(path2[3], [])
                inner_list2.append(path2[4])
                sample_dict[path2[2]] = inner_dict2
                sample_dict[path2[2]][path2[3]] = inner_list2

                # path1 = tmp[1].split('/')
                # sample_dict[path1[2]] = sample_dict.get(path1[2], {}) # 获得内层字典
                # sample_dict[path1[2]][path1[3]] = sample_dict[path1[2]].get(path1[3], []).append(path1[4]) # 把照片路径添加进入列表
                # path2 = tmp[2].split('/')
                # sample_dict[path2[2]] = sample_dict.get(path2[2], {})
                # sample_dict[path2[2]][path2[3]] = sample_dict[path2[2]].get(path2[3], []).append(path2[4])
                inner_list = connected_dict.get(path1[2], [])
                inner_list.append((path1[3], path2[3]))
                connected_dict[path1[2]] = inner_list
                # for i in (1, 2):
                #     path = tmp[i].split('/')
                #     sample_path[path[2]] = sample_path.get(path[2], {}) 
                #     sample_path[path[2]][path[3]] = sample_path[path[2]].get(path[3],[]).append(path[4])
                # family = path1[2] # 获得家庭类别
                try:
                    assert path1[2] == path2[2] # 二者必属于同一个家庭
                except:
                    # 极个别情况出错，不考虑
                    # 如：50287 Validation_A/val-faces/F0633/MID3/P06648_face1.jpg Validation_A/val-faces/F0990/MID9/P10429_face4.jpg sibs 0
                    # print(f"i is {i}, path1 is {path1}, path2 is {path2}")
                    # exit(1)
                    pass
                # node1, node2 = path1[3], path2[3]
                # relationship = tmp[3]
                # family_tree[family] = family_tree.get(family, Tree()).add_node(node1, node2, relationship) # family_tree中每一个value是一个Tree,它的root是一个虚拟根节点
                # sample_dict[path1[2]].append(os.path.join(path1[3], path1[4]))
                # sample_dict[path2[2]].append(os.path.join(path2[3], path2[4]))
        
        f.close() 
        # 去重
        # print(f"sample_dict[F0001] is {sample_dict['F0001']}")
        # print(f"connected_dict[F0001] is {connected_dict['F0001']}")
        for idx1, values1 in sample_dict.items():
            for idx2, values2 in values1.items():
                sample_dict[idx1][idx2] = list(set(values2))
        for key, connected_MID in connected_dict.items():
            for i in connected_MID: # 使得内部顺序无影响
                i = set(i)
            connected_MID = list(set(connected_MID))
            for i in connected_MID:
                i = list(i) # TODO：多此一举？
            connected_dict[key] = connected_MID
        # print("after")
        # print(f"sample_dict[F0001] is {sample_dict['F0001']}")
        # print(f"connected_dict[F0001] is {connected_dict['F0001']}")
        return sample_dict, connected_dict

        
        """index = 0
        for i in torchvision.datasets.ImageFolder(root = 
                os.path.join(self.root, 'images')).imgs:
            # i[1]: label, i[0]: root
            y = i[1]
            # fn needed for removing non-images starting with `._`
            fn = os.path.split(i[0])[1]
            if y in self.classes and fn[:2] != '._':
                self.ys += [y]
                self.I += [index]
                self.im_paths.append(os.path.join(self.root, i[0]))
                index += 1"""
    # def __getitem__(self, indexs):

    #     def img_load(index):
    #         im = PIL.Image.open(self.im_paths[index])
    #         # convert gray to rgb
    #         if len(list(im.split())) == 1 : im = im.convert('RGB') 
    #         if self.transform is not None:
    #             im = self.transform(im)
    #         return im
        
    #     ims = []
    #     targets = []

    #     for index in indexs:
    #         im = img_load(index)
    #         target = self.ys[index]
    #         ims.append(im)
    #         targets.append(target)
    #     #ims : 150x3x224x224
    #     # print(torch.stack(ims))
    #     # print(torch.stack(ims).shape)
    #     # print(torch.tensor(targets))
    #     # exit(1)
    #     return torch.stack(ims), torch.tensor(targets)