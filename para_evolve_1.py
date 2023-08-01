import torch
# 遍历每个种群
def para_evolve1(popu):
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
