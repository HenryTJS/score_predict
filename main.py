import random
import numpy as np
from collections import defaultdict
import copy
import matplotlib.pyplot as plt
import matplotlib
import os
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 支持中文显示
matplotlib.rcParams['axes.unicode_minus'] = False  # 支持负号显示

TIMES = 10000  # 蒙特卡洛模拟次数
MAXSCORE = 8  # 单场最大得分（熔断球数）
WIN = 8  # 晋级球队数
VICTORY = 3  # 胜得分
DRAW = 1  # 平得分
LOSS = 0  # 负得分
PRE = "xiangchao"  # 预测的比赛

with open(f"teams/{PRE}_teams.txt", "r", encoding="utf-8") as file:
    content = [line.strip() for line in file.readlines()]  # 球队名称

N = len(content)  # 参赛球队数
relation = defaultdict(int)  # 球队名到编号的映射
win = np.zeros((N, N), dtype=int)  # 改为NumPy数组
for x, y in zip(content, [i for i in range(1, N + 1)]):
    relation[x] = y

def already_score(file_name, rounds=None):
    with open(file_name, "r", encoding="utf-8") as file:
        all_games = [line.strip().split() for line in file.readlines()]
    
    # 如果指定了轮数，只读取前rounds轮的数据
    if rounds is not None:
        games_per_round = N // 2
        max_games = rounds * games_per_round
        score_alreadys = all_games[:max_games]
    else:
        score_alreadys = all_games
    
    already = np.zeros((N, N, 3), dtype=int)  # 改为NumPy数组
    for i, j, k, l in score_alreadys:
        j = int(j)
        k = int(k)
        already[relation[i] - 1][relation[l] - 1][2] = j
        already[relation[l] - 1][relation[i] - 1][2] = k
        already[relation[i] - 1][relation[l] - 1][1] = j - k
        already[relation[l] - 1][relation[i] - 1][1] = k - j
        if j == k:
            already[relation[i] - 1][relation[l] - 1][0] = DRAW
            already[relation[l] - 1][relation[i] - 1][0] = DRAW
        elif j > k:
            already[relation[i] - 1][relation[l] - 1][0] = VICTORY
            already[relation[l] - 1][relation[i] - 1][0] = LOSS
        else:
            already[relation[i] - 1][relation[l] - 1][0] = LOSS
            already[relation[l] - 1][relation[i] - 1][0] = VICTORY
    return already

def second_sort(score, graph, i, j):
    if j - i == 1:
        return score
    arr = np.array([score[k][0] for k in range(i, j)]) - 1  # 转换为0-based索引
    n = len(arr)
    submatrix = graph[np.ix_(arr, arr)]  # NumPy高级索引提取子矩阵
    
    newgraph = np.zeros((n, 4), dtype=int)
    newgraph[:, 0] = arr + 1  # 恢复1-based ID
    newgraph[:, 1] = np.sum(submatrix[..., 0], axis=1)  # 总积分
    newgraph[:, 2] = np.sum(submatrix[..., 1], axis=1)  # 净胜球
    newgraph[:, 3] = np.sum(submatrix[..., 2], axis=1)  # 总进球
    
    # 保持原排序逻辑：积分->净胜球->进球数（降序）
    sort_idx = np.lexsort((-newgraph[:, 3], -newgraph[:, 2], -newgraph[:, 1]))
    ordered_ids = newgraph[sort_idx, 0]
    
    # 重建score切片（保持原列表结构）
    id_to_row = {row[0]: row for row in score[i:j]}
    score[i:j] = [id_to_row[team_id] for team_id in ordered_ids]
    return score

def random_score(already):
    graph = copy.deepcopy(already)
    for i in range(N):
        for j in range(i):
            if graph[i][j][0] != 0 or graph[i][j][1] != 0 or graph[i][j][2] != 0:
                continue
            graph[i][j][2] = random.randint(0, MAXSCORE)
            graph[j][i][2] = random.randint(0, MAXSCORE)
            while graph[i][j][2] == graph[j][i][2] == MAXSCORE:
                graph[i][j][2] = random.randint(0, MAXSCORE)
                graph[j][i][2] = random.randint(0, MAXSCORE)
            if graph[i][j][2] > graph[j][i][2]:
                graph[i][j][0] = 3
                graph[j][i][0] = 0
                graph[i][j][1] = graph[i][j][2] - graph[j][i][2]
                graph[j][i][1] = -graph[i][j][1]
            elif graph[i][j][2] < graph[j][i][2]:
                graph[i][j][0] = 0
                graph[j][i][0] = 3
                graph[i][j][1] = graph[i][j][2] - graph[j][i][2]
                graph[j][i][1] = -graph[i][j][1]
            else:
                graph[i][j][0] = 1
                graph[j][i][0] = 1
                graph[i][j][1] = 0
                graph[j][i][1] = 0
    return graph

def simulate_round(rounds):
    already = already_score(f"scores/{PRE}_scores.txt", rounds)
    win = np.zeros((N, N), dtype=int)
    
    for _ in range(TIMES):
        graph = copy.deepcopy(random_score(already))
        score = np.zeros((N, 4), dtype=int)
        score[:, 0] = np.arange(1, N + 1)
        
        # 计算积分/净胜球/进球
        for i in range(N):
            score[i, 1] = np.sum(graph[i, :, 0])
            score[i, 2] = np.sum(graph[i, :, 1])
            score[i, 3] = np.sum(graph[i, :, 2])
        
        # 排序
        sort_idx = np.lexsort((-score[:, 3], -score[:, 2], -score[:, 1]))
        score = score[sort_idx]
        score = [list(row) for row in score]
        
        # 处理同分球队
        i = 0
        while i < N:
            j = i + 1
            while j < N and score[j][1] == score[i][1]:
                j += 1
            score = second_sort(score, graph, i, j)
            i = j
        
        # 统计结果
        for k in range(N):
            team_id = score[k][0] - 1
            win[team_id][k] += 1
    
    # 计算出线概率
    probabilities = [sum(w[:WIN]) / TIMES for w in win]
    return probabilities

def main():
    print("开始分轮次模拟...")
    
    # 读取实际数据行数
    with open(f"scores/{PRE}_scores.txt", "r", encoding="utf-8") as file:
        actual_games = len([line for line in file.readlines() if line.strip()])
    
    # 计算实际轮数
    games_per_round = N // 2
    total_rounds = actual_games // games_per_round
    
    print(f"球队数: {N}")
    print(f"每轮比赛数: {games_per_round}")
    print(f"实际比赛数: {actual_games}")
    print(f"实际轮数: {total_rounds}")
    
    if total_rounds == 0:
        print("错误：没有足够的比赛数据")
        return
    
    # 存储每轮的结果
    round_results = {}
    
    # 添加第0轮：所有球队出线概率相等
    initial_prob = WIN / N
    round_results[0] = [initial_prob] * N
    print(f"第0轮：所有球队出线概率均为 {initial_prob:.4f}")
    
    for round_num in range(1, total_rounds + 1):
        print(f"正在模拟第{round_num}轮...")
        probabilities = simulate_round(round_num)
        round_results[round_num] = probabilities
    
    # 创建折线图
    plt.figure(figsize=(12, 8))
    
    # 为每支球队绘制折线
    for i, team in enumerate(content):
        rounds = list(round_results.keys())
        probs = [round_results[r][i] for r in rounds]
        plt.plot(rounds, probs, marker='o', label=team, linewidth=2, markersize=4)
    
    plt.xlabel('轮次', fontsize=12)
    plt.ylabel('出线概率', fontsize=12)
    plt.title('各队出线概率随轮数变化', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    # 创建graphs目录（如果不存在）
    os.makedirs('graphs', exist_ok=True)
    
    # 保存图片
    plt.savefig(f'graphs/{PRE}_graphs.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 输出最终结果
    print("\n最终出线概率:")
    final_probs = round_results[total_rounds]
    for team, prob in zip(content, final_probs):
        print(f"{team}: {prob:.4f}")

if __name__ == '__main__':
    main()