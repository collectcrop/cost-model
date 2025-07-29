import matplotlib.pyplot as plt
import pandas as pd
# 数据
epsilon = [2, 4, 6, 7, 8, 9, 10, 11, 16, 32]

query_time_ns = [4607, 4150, 4215, 4094, 4054, 3977, 4077, 4069, 4155, 4486]

cache_hit_ratio = [1.2, 13.7, 17.9, 19.1, 20.0, 20.7, 21.3, 21.7, 23.0, 24.7]
# df = pd.read_csv('benchmark_results.csv')
# deviation_threshold = 0.3
# def remove_outliers_by_mean_deviation(group, column, threshold=0.3):
#     mean = group[column].mean()
#     lower = mean * (1 - threshold)
#     upper = mean * (1 + threshold)
#     return group[(group[column] >= lower) & (group[column] <= upper)]
# cleaned = df.groupby('epsilon').apply(lambda g: remove_outliers_by_mean_deviation(g, 'avg_query_time_ns', deviation_threshold))
# cleaned = cleaned.reset_index(drop=True)

# grouped = cleaned.groupby('epsilon').agg({
#     'avg_query_time_ns': 'mean',
#     'avg_cache_hit_ratio': 'mean'
# }).reset_index()

# print(grouped)
# epsilon = grouped['epsilon'].tolist()[:-1]
# query_time_ns = grouped['avg_query_time_ns'].tolist()[:-1]
# cache_hit_ratio = grouped['avg_cache_hit_ratio'].tolist()[:-1]
# cache_hit_ratio = [0.644,0.755,0.773,0.775,0.777,0.779,0.780,0.781,0.782,0.783,0.783,0.783,0.784,0.774,0.749]
# 创建图形和坐标轴
fig, ax1 = plt.subplots()

# 绘制 query time 曲线
color = 'tab:blue'
ax1.set_xlabel('ε (epsilon)')
ax1.set_ylabel('Query Time (ns)', color=color)
ax1.plot(epsilon, query_time_ns, marker='o', color=color, label='Query Time (ns)')
ax1.tick_params(axis='y', labelcolor=color)
ax1.grid(True, linestyle='--', alpha=0.5)
ax1.axvline(x=9, color='gray', linestyle='--', linewidth=1)
ax1.text(9, 4000, 'optimal ε=9', color='gray', ha='center', va='bottom')

# 创建第二个坐标轴，共享 x 轴
ax2 = ax1.twinx()

# 绘制 cache hit ratio 曲线
color = 'tab:red'
ax2.set_ylabel('Cache Hit Ratio (%)', color=color)
ax2.plot(epsilon, cache_hit_ratio, marker='s', color=color, label='Cache Hit Ratio (%)')
ax2.tick_params(axis='y', labelcolor=color)

# 图例合并显示
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
plt.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

plt.title('PGM Index Performance vs Epsilon')
plt.tight_layout()
plt.show()