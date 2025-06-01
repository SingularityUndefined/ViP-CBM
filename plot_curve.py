import numpy as np
import matplotlib.pyplot as plt

model_name = ['ViP-CBM', 'ViP-CBM-LP', 'ViP-CBM-margin', 'scalar-CBM', 'CEM', 'ProbCBM']
metric = ['Concept_Hamming_Score', 'Concept EMR', 'min_Group_EMR', 'Class_Accuracy']
concept_number = np.array([32, 59, 86, 112]) * 100 / 112
concept_number = [25, 50, 75, 100]

model_25 = np.array([
    [93.5833, 48.9990, 72.6441, 62.8754],
    [94.6863, 52.1401, 76.4757, 70.0725],
    [94.7505, 55.7473, 77.6666, 68.9506],
    [94.9441, 49.4995, 76.8036, 71.1771],
    [94.6771, 48.8954, 75.9406, 71.8157],
    [94.7435, 48.8091, 75.8889, 71.5913]
])
model_50 = np.array([
    [94.8307, 48.9299, 75.8889, 67.5181],
    [94.8799, 41.5948, 75.6645, 69.8481],
    [94.8281, 42.6130, 75.9924, 68.8298],
    [95.4313, 49.1543, 77.8391, 72.3680],
    [95.0849, 42.6476, 76.0442, 72.7649],
    [95.2326, 45.0639, 75.9751, 72.7132]
])

model_75 = np.array([
    [95.1209, 39.5927, 77.7701, 72.6096],
    [94.5784, 35.3297, 73.4380, 68.8816],
    [94.6438, 47.1004, 75.8371, 67.2247],
    [95.4090, 43.9075, 77.9427, 71.9710],
    [94.9428, 32.6200, 74.0421, 73.5589],
    [95.2658, 41.4739, 76.8208, 73.2482]
])

model_100 = np.array([
    [94.8450, 35.7957, 76.0269, 71.2289],
    [93.6936, 33.4139, 70.1760, 63.6175],
    [95.0226, 40.6800, 76.9244, 71.6603],
    [95.2778, 41.6465, 77.8219, 72.0400],
    [95.1028, 35.6231, 77.3386, 73.2137],
    [95.1602, 39.4546, 77.8391, 72.0573]
])

model_perform_stack = np.stack((model_25, model_50, model_75, model_100), axis=0)
print(model_perform_stack.shape)

# plt.figure(figsize=(10, 6))
for i, m in enumerate(metric):
    plt.figure()
    plt.rcParams.update({
        'axes.labelsize': 'xx-large',     # 坐标轴标签字号
        'xtick.labelsize': 'large',    # X轴刻度字号
        'ytick.labelsize': 'large',    # Y轴刻度字号
        'legend.fontsize': 'large',    # 图例条目字号
        'legend.title_fontsize':'large', # 图例标题字号
        'figure.titlesize': 'xx-large'    # 标题字号
    })
    for j, model in enumerate(model_name):
        plt.plot(concept_number, model_perform_stack[:, j, i], marker='o', label=model)
    plt.subplots_adjust(left=0.15, bottom=0.15) 
    # plt.title(f'{m}')
    plt.xlabel('Know Concept Group Ratio (%)')
    plt.ylabel(m.replace('_', ' '))
    plt.xticks(concept_number)
    plt.grid(True)
    plt.legend()
    plt.savefig(f'performance_{m}.pdf')
    # plt.show()