import torch
import torch.nn as nn
import torch.optim as optim
import random
import math
import numpy as np
import datetime
import autograd.numpy as anp
import torch.nn.functional as F
import numpy as np

# 设置 GPU 设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def factorial(x):
    return math.factorial(x)

class Problem:
    def __init__(self, n_var=-1, n_obj=-1, n_constr=0, xl=None, xu=None, type_var=torch.float32):
        self.n_var = n_var
        self.n_obj = n_obj
        self.n_constr = n_constr
        self.xl = xl
        self.xu = xu
        self.type_var = type_var

    def evaluate(self, individual):
        raise NotImplementedError("evaluate() method is not implemented.")


    def calculate_igd(self, pf, ref_points):
        distances = torch.cdist(pf, ref_points, p=2)
        min_distances = torch.min(distances, dim=0).values
        igd = torch.mean(min_distances)
        return igd

class DTLZ(Problem):
    def __init__(self, n_var, n_obj, k=None):
        if n_var:
            self.k = n_var - n_obj + 1
        elif k:
            self.k = k
            n_var = k + n_obj - 1
        else:
            raise Exception("Either provide number of variables or k!")

        super().__init__(n_var=n_var, n_obj=n_obj, n_constr=0, xl=0, xu=1, type_var=torch.float32)

    def g1(self, X_M):
        return 100 * (self.k + torch.sum(torch.square(X_M - 0.5) - torch.cos(20 * np.pi * (X_M - 0.5)), dim=1))

    def g2(self, X_M):
        return torch.sum(torch.square(X_M - 0.5), dim=1)

    def obj_func(self, X_, g, alpha=1):
        f = []

        for i in range(0, self.n_obj):
            _f = (1 + g)
            _f *= torch.prod(torch.cos(torch.pow(X_[:, :X_.shape[1] - i], alpha) * np.pi / 2.0), dim=1)
            if i > 0:
                _f *= torch.sin(torch.pow(X_[:, X_.shape[1] - i], alpha) * np.pi / 2.0)

            f.append(_f)

        f = torch.column_stack(f)
        return f

class DTLZ1(DTLZ):
    def __init__(self, n_var=7, n_obj=3, **kwargs):
        super().__init__(n_var, n_obj, **kwargs)

    def _calc_pareto_front(self, ref_dirs=None):
        return 0.5 * ref_dirs

    def evaluate(self, x, *args, **kwargs):
        X_, X_M = x[:, :self.n_obj - 1], x[:, self.n_obj - 1:]
        g = self.g1(X_M)

        f = []
        for i in range(0, self.n_obj):
            _f = 0.5 * (1 + g)
            _f *= torch.prod(X_[:, :X_.shape[1] - i], dim=1)
            if i > 0:
                _f *= 1 - X_[:, X_.shape[1] - i]
            f.append(_f)
        F = torch.column_stack(f)
        G = constraint_c1_linear(F)
        V = calc_constraint_violation(G)
        return F+V

def constraint_c1_linear(f):
    g = - (1 - f[:, -1] / 0.6 - torch.sum(f[:, :-1] / 0.5, dim=1))
    gg = torch.unsqueeze(g, dim=1)
    return gg

def calc_constraint_violation(G):
    if G is None:
        return None
    elif G.shape[1] == 0:
        return torch.zeros(G.shape[0], 1)
    else:
        return torch.sum(G * (G > 0).type(torch.float32), dim=1, keepdim=True)


def dominates(fitness_a, fitness_b):
    # 判断个体 a 是否支配个体 b
    # 如果个体 a 在所有目标函数上都不大于个体 b，且在至少一个目标函数上小于个体 b，则 a 支配 b

    # 注意：假设 fitness_a 和 fitness_b 是大小相同的张量或数组

    # 检查是否所有目标函数值 a_i 都小于等于 b_i
    lesser_equal = torch.all(fitness_a <= fitness_b)

    # 检查是否至少存在一个目标函数值 a_i 小于 b_i
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


def uniform_reference_points(nobj, p=4, scaling=None):
    import numpy
    """Generate reference points uniformly on the hyperplane intersecting
    each axis at 1. The scaling factor is used to combine multiple layers of
    reference points.
    """
    def gen_refs_recursive(ref, nobj, left, total, depth):
        points = []
        if depth == nobj - 1:
            ref[depth] = left / total
            points.append(ref)
        else:
            for i in range(left + 1):
                ref[depth] = i / total
                points.extend(gen_refs_recursive(ref.copy(), nobj, left - i, total, depth + 1))
        return points

    ref_points = numpy.array(gen_refs_recursive(numpy.zeros(nobj), nobj, p, p, 0))
    if scaling is not None:
        ref_points *= scaling
        ref_points += (1 - scaling) / nobj

    return ref_points


def calculate_igd(approx_pareto, true_pareto):
    # 将输入张量的数据类型转换为torch.float
    approx_pareto = approx_pareto.float()
    true_pareto = true_pareto.float()

    # 计算距离矩阵
    distances = torch.cdist(approx_pareto, true_pareto, p=2)

    # 计算每个参考点的最小距离
    min_distances = torch.min(distances, dim=1).values

    # 计算IGD值
    igd = torch.mean(min_distances)
    return igd

# 设置参数
NOBJ = 20
K = 10
BOUND_LOW = 0
BOUND_UP = 1
P = [2, 1]
SCALES = [1, 0.5]
CXPB = 1.0
MUTPB = 1.0
eta = 35
iteration = 2000

H = factorial(NOBJ + P[0] - 1) / (factorial(P[0]) * factorial(NOBJ - 1))+factorial(NOBJ + P[1] - 1) / (factorial(P[1]) * factorial(NOBJ - 1))


ref_points = [uniform_reference_points(NOBJ, p, s) for p, s in zip(P, SCALES)]
ref_points = np.concatenate(ref_points, axis=0)
_, uniques = np.unique(ref_points, axis=0, return_index=True)
ref_points = ref_points[uniques]
ref_points = torch.tensor(ref_points * 1)



pop_size = int(H)
num_populations = 20  # 新增参数：种群数量

# 定义问题对象
problem = DTLZ1(n_var=NOBJ+K-1, n_obj=NOBJ)

# 初始化种群并将其移动到指定设备device上面
populations = []
for _ in range(num_populations):
    population = torch.rand(pop_size, problem.n_var, dtype=torch.float32, device=device)
    populations.append(population)

#ref_dirs = uniform_reference_points(nobj=3)
pf = problem._calc_pareto_front(ref_points)
pf = pf.to(device)
start_time = datetime.datetime.now()
# NSGA-III算法迭代 # 随机选取进行交叉
for gen in range(iteration):
    # 存储每个种群的igd
    igd_all = []

    # 遍历每个种群
    for i in range(num_populations):
        population = populations[i]

        # 计算适应度值并将其移动到GPU上
        fitness_values = problem.evaluate(population)
        fitness_values = fitness_values.to(device)

        # 计算非支配排序和拥挤度距离
        ranked_indices = non_dominated_sort(fitness_values)
        fitness_ranks = torch.zeros(pop_size, dtype=torch.int32, device=device)
        fitness_crowding_distances = torch.zeros(pop_size, dtype=torch.float32, device=device)

        # 遍历每个等级
        for rank, indices in enumerate(ranked_indices):
            # 获取当前等级的个体适应度值和种群
            ranked_fitness_values = fitness_values[indices]
            ranked_population = population[indices]

            # 计算拥挤度距离
            num_individuals = ranked_fitness_values.shape[0]
            crowding_distances = torch.zeros(num_individuals, dtype=torch.float32, device=device)

            # 对每个目标函数进行排序
            for obj in range(problem.n_obj):
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
        fitness_crowding_distances /= (pop_size - 1)

        # 最终的完整排序
        final_rankings = torch.argsort(fitness_ranks)
        ranked_pop = population[final_rankings]

        # 选择和交叉
        mating_pool = torch.zeros((pop_size, problem.n_var), dtype=torch.float32, device=device)
        for j in range(pop_size):
            selected = torch.multinomial(torch.ones(pop_size), 2, replacement=False)
            a, b = ranked_pop[selected[0]], ranked_pop[selected[1]]
            child = torch.cat([a[:problem.n_obj], b[problem.n_obj:]], dim=0)
            mating_pool[j] = child

        # 变异
        mutated_pop = torch.zeros((pop_size, problem.n_var), dtype=torch.float32, device=device)
        for j in range(pop_size):
            if random.random() < MUTPB:
                mutant = torch.clone(mating_pool[j])
                for k in range(problem.n_var):
                    if random.random() < 1.0 / problem.n_var:
                        lower = max(problem.xl, mutant[k] - 0.1)
                        upper = min(problem.xu, mutant[k] + 0.1)
                        mutant[k] = random.uniform(lower, upper)
                mutated_pop[j] = mutant
            else:
                mutated_pop[j] = mating_pool[j]

        # 更新种群
        population = torch.cat([ranked_pop[:pop_size // 2], mutated_pop[pop_size // 2:]], dim=0)
        populations[i] = population

        # 存储适应度值
        igd_all.append(calculate_igd(fitness_values, pf))

    # 输出IGD值
    print("Generation:", gen, "IGD:", torch.mean(torch.tensor(igd_all)))

    # 在每间隔10代的时候进行替换操作
    if (gen + 1) % 10 == 0:
        # 选取前10%的个体放到公共种群
        num_top_individuals = int(0.1 * pop_size)
        top_individuals = torch.empty((0,))
        for population in populations:
            top_individuals=torch.cat((top_individuals, population[:num_top_individuals]), dim=0)
            
        # 随机选择个体替换每个种群中的个体
        for i in range(num_populations):
            population = populations[i]
            population = population[:pop_size - num_top_individuals]
            random_indices = random.sample(range(top_individuals.size(0)), num_top_individuals)
            new_tensor = top_individuals[random_indices, :]
            population = torch.cat((population, new_tensor), dim=0)
        # 更新种群
            populations[i] = population


finish_time = datetime.datetime.now()
print('time spending is: ', finish_time - start_time)
