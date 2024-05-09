import networkx as nx
import matplotlib.pyplot as plt

# 创建一个空的有向图
G = nx.DiGraph()

# 添加家庭成员节点
family_members = ['Grandpa', 'Grandma', 'Dad', 'Mom', 'Child1', 'Child2']
G.add_nodes_from(family_members)

# 添加家庭关系边
family_relationships = [('Grandpa', 'Dad'), ('Grandma', 'Dad'), ('Grandpa', 'Mom'),
                        ('Grandma', 'Mom'), ('Dad', 'Child1'), ('Mom', 'Child1'),
                        ('Dad', 'Child2'), ('Mom', 'Child2')]
G.add_edges_from(family_relationships)

# 绘制图形
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_size=2000, node_color='lightblue', font_size=10, font_weight='bold', arrowsize=20)
plt.show()