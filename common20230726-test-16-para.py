#!/usr/bin/env python
# coding: utf-8

# 先进行变异生成pop_size个offspring,与原先parent一起进行排序选择，之前是在offspring和parent中各排序然后选择一半
# 增加了mutation （mutPolynomialBounded）
# 使用DEAP的nsga3算法
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import math
import numpy as np
import datetime
import autograd.numpy as anp
import torch.nn.functional as F
import numpy as np
from deap import tools, creator, base, algorithms
import pandas as pd
import random

# 设置 GPU 设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def factorial(x):
    return math.factorial(x)

def uniform(low, up, size=None):
    try:
        return [random.uniform(a, b) for a, b in zip(low, up)]
    except TypeError:
        return [random.uniform(a, b) for a, b in zip([low] * size, [up] * size)]

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
        return F

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

def cxSimulatedBinaryBounded(ind1, ind2, eta, low, up):
    """Executes a simulated binary crossover that modify in-place the input
    individuals. The simulated binary crossover expects :term:`sequence`
    individuals of floating point numbers.

    :param ind1: The first individual participating in the crossover.
    :param ind2: The second individual participating in the crossover.
    :param eta: Crowding degree of the crossover. A high eta will produce
                children resembling to their parents, while a small eta will
                produce solutions much more different.
    :param low: A value or a :term:`python:sequence` of values that is the lower
                bound of the search space.
    :param up: A value or a :term:`python:sequence` of values that is the upper
               bound of the search space.
    :returns: A tuple of two individuals.

    This function uses the :func:`~random.random` function from the python base
    :mod:`random` module.

    .. note::
       This implementation is similar to the one implemented in the
       original NSGA-II C code presented by Deb.
    """
    size = min(len(ind1), len(ind2))

    for i in range(size):
        xl = low
        xu = up
        if random.random() <= 0.5:
            # This epsilon should probably be changed for 0 since
            # floating point arithmetic in Python is safer
            if bool(abs(ind1[i] - ind2[i]) > 1e-14):
                x1 = min(ind1[i], ind2[i])
                x2 = max(ind1[i], ind2[i])
                rand = random.random()

                beta = 1.0 + (2.0 * (x1 - xl) / (x2 - x1))
                alpha = 2.0 - beta ** -(eta + 1)
                if rand <= 1.0 / alpha:
                    beta_q = (rand * alpha) ** (1.0 / (eta + 1))
                else:
                    beta_q = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1))

                c1 = 0.5 * (x1 + x2 - beta_q * (x2 - x1))

                beta = 1.0 + (2.0 * (xu - x2) / (x2 - x1))
                alpha = 2.0 - beta ** -(eta + 1)
                if rand <= 1.0 / alpha:
                    beta_q = (rand * alpha) ** (1.0 / (eta + 1))
                else:
                    beta_q = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1))
                c2 = 0.5 * (x1 + x2 + beta_q * (x2 - x1))

                c1 = min(max(c1, xl), xu)
                c2 = min(max(c2, xl), xu)

                if random.random() <= 0.5:
                    ind1[i] = c2
                    ind2[i] = c1
                else:
                    ind1[i] = c1
                    ind2[i] = c2

    return ind1, ind2

def mutPolynomialBounded(individual, eta, low, up, indpb):
    """Polynomial mutation as implemented in original NSGA-II algorithm in
    C by Deb.

    :param individual: :term:`Sequence <sequence>` individual to be mutated.
    :param eta: Crowding degree of the mutation. A high eta will produce
                a mutant resembling its parent, while a small eta will
                produce a solution much more different.
    :param low: A value or a :term:`python:sequence` of values that
                is the lower bound of the search space.
    :param up: A value or a :term:`python:sequence` of values that
               is the upper bound of the search space.
    :returns: A tuple of one individual.
    """
    size = len(individual)
    xl = low
    xu = up
    
    for i in range(size):
         
        if random.random() <= indpb:
            x = individual[i]
            delta_1 = (x - xl) / (xu - xl)
            delta_2 = (xu - x) / (xu - xl)
            rand = random.random()
            mut_pow = 1.0 / (eta + 1.)

            if rand < 0.5:
                xy = 1.0 - delta_1
                val = 2.0 * rand + (1.0 - 2.0 * rand) * xy ** (eta + 1)
                delta_q = val ** mut_pow - 1.0
            else:
                xy = 1.0 - delta_2
                val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * xy ** (eta + 1)
                delta_q = 1.0 - val ** mut_pow

            x = x + delta_q * (xu - xl)
            x = min(max(x, xl), xu)
            individual[i] = x
    return individual,

# insert_start

# 遍历每个种群
def para_evolve(popu):
    offspring = copy.deepcopy(popu)
    parent = copy.deepcopy(popu)
    #print('offspring_init',offspring)
    # 对 offspring 进行交叉
    for ii in range(1, len(offspring), 2):
        if random.random() < cxpb:
            offspring[ii-1],offspring[ii] = cxSimulatedBinaryBounded(offspring[ii-1], offspring[ii], 30, low, up)
    #print('offspring_cross',offspring)
    # 对 offspring 进行变异
    for iii in range(len(offspring)):
        if random.random() < mutpb:
            offspring[iii], = mutPolynomialBounded(offspring[iii], 20, low, up, 1.0/NDIM)
    #result_offspring.append(pd.DataFrame(offspring))
    #print('offspring_mutate',offspring)
    # 更新种群
    population = torch.cat([parent,offspring], dim=0)
        
    # 计算适应度值并将其移动到GPU上
    fitness_values = problem.evaluate(population)
    fitness_values = fitness_values.to(device)
        
    # 生成DEAP中的population对象，并用前面一步得到的population进行替换
    pop_deap = toolbox.population(n=pop_size*2)
    for row_ in range(len(pop_deap)):
        for column_ in range(len(pop_deap[0])):
            pop_deap[row_][column_] = population[row_][column_].item()
        
    # 计算DEAP的population对象的fitness
    for row__, f_v in zip(range(len(pop_deap)),fitness_values):
        pop_deap[row__].fitness.values = tuple(f_v.cpu().numpy())
    #print(pop_deap)
    #result_pop_offspring.append(pop_deap)
    #result_f_pop_offspring.append([pop_deap[0].fitness.values,pop_deap[1].fitness.values])
    #result_selection_deap.append(toolbox.select(pop_deap,pop_size))
    popu = torch.tensor(toolbox.select(pop_deap,pop_size))
    #result_selection_torch.append(pd.DataFrame(popu))
    # 计算适应度值并将其移动到GPU上
    fitness_values_ = problem.evaluate(popu)
    fitness_values_ = fitness_values_.to(device)
    # 存储适应度值
    return popu,calculate_igd(pf,fitness_values_)

# insert_finish

# 设置参数
NOBJ = 30
K = 20
NDIM = NOBJ + K - 1
BOUND_LOW = low = 0
BOUND_UP = up = 1
P = [2, 1]
SCALES = [1, 0.5]
CXPB = cxpb = 1.0
MUTPB = mutpb = 1.0

iteration = 1500

H = factorial(NOBJ + P[0] - 1) / (factorial(P[0]) * factorial(NOBJ - 1))+factorial(NOBJ + P[1] - 1) / (factorial(P[1]) * factorial(NOBJ - 1))


ref_points = [uniform_reference_points(NOBJ, p, s) for p, s in zip(P, SCALES)]
ref_points = np.concatenate(ref_points, axis=0)
_, uniques = np.unique(ref_points, axis=0, return_index=True)
ref_points = ref_points[uniques]
ref_points = torch.tensor(ref_points * 1)



pop_size = int(H + (4 - H % 4))
num_populations = 4  # 新增参数：种群数量

# 定义问题对象
problem = DTLZ1(n_var=NOBJ+K-1, n_obj=NOBJ)

toolbox = base.Toolbox()

creator.create("FitnessMin", base.Fitness, weights=(-1.0,) * NOBJ)
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox.register("attr_float", uniform, BOUND_LOW, BOUND_UP, NDIM)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_float)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("select", tools.selNSGA3, ref_points=ref_points.numpy())

import copy

result_offspring = []
result_pop_offspring = []
result_f_pop_offspring = []
result_selection_torch = []
result_selection_deap = []
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

    # insert start
    mp.set_start_method('spawn') 
    pool = mp.Pool(processes=num_populations)    
    for i in range(num_populations):
        populations[i],igd_= para_evolve(populations[i])
        igd_all.append(igd_)
    # insert finish

    # 输出IGD值
    print("Generation:", gen, "IGD:", torch.mean(torch.tensor(igd_all)))

    # 在每间隔10代的时候进行替换操作
    if (gen + 1) % 10 == 0:
        # 选取前10%的个体放到公共种群
        num_top_individuals = int(0.1 * pop_size)
        top_individuals = torch.empty((0,)).to(device) 
        for population in populations:
            top_individuals=torch.cat((top_individuals, population[:num_top_individuals].to(device)), dim=0)
           
        # 随机选择个体替换每个种群中的个体
        for i in range(num_populations):
            population = populations[i]
            population = population[:pop_size - num_top_individuals]
            random_indices = random.sample(range(top_individuals.size(0)), num_top_individuals)
            new_tensor = top_individuals[random_indices, :]
            population = torch.cat((population.to(device), new_tensor), dim=0)
        # 更新种群
            populations[i] = population


finish_time = datetime.datetime.now()
print('time spending is: ', finish_time - start_time)
