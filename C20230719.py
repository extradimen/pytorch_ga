# 第一代代码
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import datetime
from sklearn.metrics import pairwise_distances

# 设置 GPU 设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Problem:
    def __init__(self, NOBJ, K, BOUND_LOW, BOUND_UP):
        self.NOBJ = NOBJ
        self.K = K
        self.NDIM = NOBJ + K - 1
        self.BOUND_LOW = BOUND_LOW
        self.BOUND_UP = BOUND_UP

    def evaluate(self, individual):
        raise NotImplementedError("evaluate() method is not implemented.")

    def calculate_pf(self, population):
        non_dominated_pop = []
        for i, ind in enumerate(population):
            dominated = False
            for j, other_ind in enumerate(population):
                if i != j and torch.all(ind <= other_ind):
                    dominated = True
                    break
            if not dominated:
                non_dominated_pop.append(ind)
        return non_dominated_pop

    def calculate_igd(self, pf, ref_points):
        distances = pairwise_distances(pf, ref_points, metric='euclidean')
        min_distances = torch.min(distances, dim=0).values
        igd = torch.mean(min_distances)
        return igd

class DTLZ(Problem):
    def __init__(self, NOBJ, K, BOUND_LOW, BOUND_UP, problem_id):
        super().__init__(NOBJ, K, BOUND_LOW, BOUND_UP)
        self.problem_id = problem_id

    def evaluate(self, individual):
        g = torch.sum(torch.square(individual[self.NOBJ-1:] - 0.5))
        f = 0.5 * torch.prod(individual[:self.NOBJ]) * (1 + g)
        return f

class CDTLZ(Problem):
    def __init__(self, NOBJ, K, BOUND_LOW, BOUND_UP, problem_id):
        super().__init__(NOBJ, K, BOUND_LOW, BOUND_UP)
        self.problem_id = problem_id

    def evaluate(self, individual):
        g = torch.sum(torch.square(individual[self.NOBJ-1:] - 0.5) - torch.cos(20 * torch.tensor(np.pi, device=device) * (individual[self.NOBJ-1:] - 0.5)))
        f = 0.5 * torch.prod(individual[:self.NOBJ]) * (1 + g)
        return f

# 定义 DTLZ1, DTLZ2, DTLZ3, DTLZ4 和 C1DTLZ1, C1DTLZ2, C1DTLZ3, C1DTLZ4 对象

class DTLZ1(DTLZ):
    def __init__(self, NOBJ, K):
        super().__init__(NOBJ, K, 0, 1, problem_id=1)

class DTLZ2(DTLZ):
    def __init__(self, NOBJ, K):
        super().__init__(NOBJ, K, 0, 1, problem_id=2)

class DTLZ3(DTLZ):
    def __init__(self, NOBJ, K):
        super().__init__(NOBJ, K, 0, 1, problem_id=3)

class DTLZ4(DTLZ):
    def __init__(self, NOBJ, K):
        super().__init__(NOBJ, K, 0, 1, problem_id=4)

class C1DTLZ1(CDTLZ):
    def __init__(self, NOBJ, K):
        super().__init__(NOBJ, K, 0, 1, problem_id=1)

    def evaluate(self, individual):
        g = torch.sum(torch.square(individual[self.NOBJ-1:]) - torch.cos(20 * torch.tensor(np.pi, device=device) * (individual[self.NOBJ-1:] - 0.5)))
        h = torch.sum(individual[:self.NOBJ])
        f = torch.zeros(self.NOBJ)
        for i in range(self.NOBJ):
            f[i] = 0.5 * h * (1 + g)
            if i > 0:
                f[i] *= torch.prod(torch.cos(individual[:self.NOBJ-i] * torch.tensor(np.pi, device=device) / 2))
        return f


class C1DTLZ2(CDTLZ):
    def __init__(self, NOBJ, K):
        super().__init__(NOBJ, K, 0, 1, problem_id=2)

class C1DTLZ3(CDTLZ):
    def __init__(self, NOBJ, K):
        super().__init__(NOBJ, K, 0, 1, problem_id=3)

class C1DTLZ4(CDTLZ):
    def __init__(self, NOBJ, K):
        super().__init__(NOBJ, K, 0, 1, problem_id=4)

def dominates(fitness_a, fitness_b):
    lesser_equal = torch.all(fitness_a <= fitness_b)
    lesser = torch.any(fitness_a < fitness_b)

    return lesser_equal and lesser
        
def non_dominated_sort(fitness_values):
    population_size = fitness_values.size(0)
    dominated_counts = torch.zeros(population_size, dtype=torch.int32)
    dominated_list = [[] for _ in range(population_size)]
    ranks = torch.zeros(population_size, dtype=torch.int32)
    fronts = [[]]

    for i in range(population_size):
        for j in range(i + 1, population_size):
            if dominates(fitness_values[i], fitness_values[j]):
                dominated_counts[j] += 1
                dominated_list[i].append(j)
            elif dominates(fitness_values[j], fitness_values[i]):
                dominated_counts[i] += 1
                dominated_list[j].append(i)

        if dominated_counts[i] == 0:
            ranks[i] = 0
            fronts[0].append(i)

    front_index = 0
    while fronts[front_index]:
        next_front = []
        for i in fronts[front_index]:
            for j in dominated_list[i]:
                dominated_counts[j] -= 1
                if dominated_counts[j] == 0:
                    ranks[j] = front_index + 1
                    next_front.append(j)
        front_index += 1
        fronts.append(next_front)

    return fronts[:-1]


NOBJ = 100
K = 20
BOUND_LOW = 0
BOUND_UP = 1
P = [2, 1]
SCALES = [1, 0.5]
CXPB = 1.0
MUTPB = 1.0
eta = 35
iteration = 200
pop = 100

# 定义问题对象
problem = C1DTLZ1(NOBJ, K)

# 计算参考点
ref_points = []
for p, s in zip(P, SCALES):
    ref_points.append(torch.from_numpy(np.random.uniform(size=(p, problem.NOBJ))) * s)
ref_points = torch.cat(ref_points, dim=0).to(device)

# 初始化种群并将其移动到指定设备device上面
population = torch.rand(pop, problem.NDIM, dtype=torch.float32, device=device)

start_time = datetime.datetime.now()
# NSGA-III算法迭代 # 捉对交叉
for gen in range(iteration):
    # 由parent生成offspring
    parent = population
    # 选择和交叉
    mating_pool = torch.zeros((pop, problem.NDIM), dtype=torch.float32, device=device)
    pair_count = pop // 2  # 需要进行交叉的组数

    for i in range(pair_count):
        a = parent[i * 2]
        b = parent[i * 2 + 1]
    
        # 进行交叉操作，得到两个子代
        child1 = torch.cat([a[:problem.NOBJ], b[problem.NOBJ:]], dim=0)
        child2 = torch.cat([b[:problem.NOBJ], a[problem.NOBJ:]], dim=0)
    
        # 将子代放入交叉池
        mating_pool[i * 2] = child1
        mating_pool[i * 2 + 1] = child2

    # 变异
    mutated_pop = torch.zeros((pop, problem.NDIM), dtype=torch.float32, device=device)
    for i in range(pop):
        if random.random() < MUTPB:
            mutant = torch.clone(mating_pool[i])
            for j in range(problem.NDIM):
                if random.random() < 1.0 / problem.NDIM:
                    lower = max(problem.BOUND_LOW, mutant[j] - 0.1)
                    upper = min(problem.BOUND_UP, mutant[j] + 0.1)
                    mutant[j] = random.uniform(lower, upper)
            mutated_pop[i] = mutant
        else:
            mutated_pop[i] = mating_pool[i]
    offspring = mutated_pop
    
    population = torch.cat([parent, offspring], dim=0)
    
    
    
    # 计算适应度值并将其移动到GPU上
    fitness_values = torch.stack([problem.evaluate(individual) for individual in population])
    fitness_values = fitness_values.to(device)

    # 计算非支配排序和拥挤度距离
    ranked_indices = non_dominated_sort(fitness_values)
    fitness_ranks = torch.zeros(pop*2, dtype=torch.int32, device=device)
    fitness_crowding_distances = torch.zeros(pop*2, dtype=torch.float32, device=device)

    # 遍历每个等级
    for rank, indices in enumerate(ranked_indices):
        # 获取当前等级的个体适应度值和种群
        ranked_fitness_values = fitness_values[indices]
        ranked_population = population[indices]

        # 计算拥挤度距离
        num_individuals = ranked_fitness_values.shape[0]
        crowding_distances = torch.zeros(num_individuals, dtype=torch.float32, device=device)

        # 对每个目标函数进行排序
        for obj in range(problem.NOBJ):
            sorted_indices = torch.argsort(ranked_fitness_values[:, obj])
            sorted_fitness_values = ranked_fitness_values[sorted_indices, obj]

            # 设置边界个体的拥挤度距离为无穷大
            crowding_distances[sorted_indices[0]] = float('inf')
            crowding_distances[sorted_indices[-1]] = float('inf')

            # 计算中间个体的拥挤度距离
            if num_individuals > 2:
                min_fitness = sorted_fitness_values[0]
                max_fitness = sorted_fitness_values[-1]
                normalized_fitness = (sorted_fitness_values - min_fitness) / (max_fitness - min_fitness + 1e-10)
                crowding_distances[sorted_indices[1:-1]] += normalized_fitness[2:] - normalized_fitness[:-2]

        # 更新适应度排名和拥挤度距离
        fitness_ranks[indices] = rank + 1
        fitness_crowding_distances[indices] = crowding_distances

    # 归一化拥挤度距离
    fitness_crowding_distances /= (pop*2 - 1)

    # 最终的完整排序
    final_rankings = torch.argsort(fitness_ranks)
    
    ranked_pop = population[final_rankings][:pop]
    population = ranked_pop
    
    print('min is: ',fitness_values[0],'mean is: ',fitness_values.mean(axis=0))

finish_time = datetime.datetime.now()
print('time spending is: ',finish_time-start_time)

