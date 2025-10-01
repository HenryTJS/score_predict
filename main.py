import random
from collections import defaultdict
import copy

with open("teams.txt", "r", encoding="utf-8") as file:
    content = [line.strip() for line in file.readlines()] # 球队名称

N = len(content) # 参赛球队数
TIMES = 10000 # 蒙特卡洛模拟次数
MAXSCORE = 8 # 单场最大得分（熔断球数）
WIN = 8 # 晋级球队数
relation = defaultdict(int) # 球队名到编号的映射
win = [[0] * N for _ in range(N)] # 获胜次数矩阵
for x, y in zip(content, [i for i in range(1, N + 1)]):
    relation[x] = y

def already_score(file_name):
    with open(file_name, "r", encoding="utf-8") as file:
        score_alreadys = [line.strip().split() for line in file.readlines()] # 已赛战绩
    already = [[[0, 0, 0] for _ in range(N)] for _ in range(N)] # 已赛战绩矩阵
    for i, j, k, l in score_alreadys:
        j = int(j)
        k = int(k)
        already[relation[i] - 1][relation[l] - 1][2] = j
        already[relation[l] - 1][relation[i] - 1][2] = k
        already[relation[i] - 1][relation[l] - 1][1] = j - k
        already[relation[l] - 1][relation[i] - 1][1] = k - j
        if j == k:
            already[relation[i] - 1][relation[l] - 1][0] = 1
            already[relation[l] - 1][relation[i] - 1][0] = 1
        elif j > k:
            already[relation[i] - 1][relation[l] - 1][0] = 3
            already[relation[l] - 1][relation[i] - 1][0] = 0
        else:
            already[relation[i] - 1][relation[l] - 1][0] = 0
            already[relation[l] - 1][relation[i] - 1][0] = 3

def second_sort(score, graph, i, j):
    if j - i == 1:
        return score
    arr = []
    for k in range(i, j):
        arr.append(score[k][0])
    n = len(arr)
    submatrix = []
    for row_idx in arr:
        row = []
        for col_idx in arr:
            row.append(graph[row_idx - 1][col_idx - 1])
        submatrix.append(row)
    newgraph = [[0, 0, 0, 0] for _ in range(n)]
    for r in range(n):
        newgraph[r][0] = arr[r]
        for c in range(n):
            newgraph[r][1] += submatrix[r][c][0]
            newgraph[r][2] += submatrix[r][c][1]
            newgraph[r][3] += submatrix[r][c][2]
    newgraph.sort(key=lambda x: (x[1], x[2], x[3]), reverse=True) # 积分相同队伍相互得分、净胜球数、进球数
    id_to_row = {row[0]: row for row in score[i:j]}
    ordered_ids = [row[0] for row in newgraph]
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

def main():
    already = already_score("scores.txt")
    for _ in range(TIMES):
        graph = copy.deepcopy(random_score(already))
        score = [[0, 0, 0, 0] for _ in range(N)]
        for i in range(N):
            score[i][0] = i + 1
            for j in range(N):
                score[i][1] += graph[i][j][0]
                score[i][2] += graph[i][j][1]
                score[i][3] += graph[i][j][2]

        score.sort(key=lambda x: (x[1], x[2], x[3]), reverse=True) # 全局得分、净胜球数、进球数
        i = 0
        while i < N:
            j = i + 1
            while j < N and score[j][1] == score[i][1]:
                j += 1
            score = second_sort(score, graph, i, j)
            i = j
        for k in range(N):
            win[score[k][0] - 1][k] += 1
    print([sum(w[:WIN]) / TIMES for w in win]) # 出线概率

if __name__ == '__main__':
    main()