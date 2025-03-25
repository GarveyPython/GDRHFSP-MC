import time
from decodes import *
from fast_domination_sort import fast_non_domination_sort
from crossover import crossover
from mutation import mutate
from utils.env_utils import get_data
from rescheduling import reassemble
from tqdm import tqdm
import numpy as np
import random

pop_size = 100  # 种群数量
iter_n = 50  # 迭代次数
cr_rate = 0.8
mu_rate = 0.2


class Offspring:
    def __init__(self):
        self.P = []
        self.Objs = []
        self.Machines = []

    def append(self, po, o, m):
        self.P.append(po)
        self.Objs.append(o)
        self.Machines.append(m)


if __name__ == "__main__":
    data = ['j15_m10_n20_k5_l3', 'j20_m20_n30_k5_l4']
    jobs, machines, new_jobs, stage_num, layer_stage_data, pub_ec_f, e_co2_f = get_data(f'data/test/{data[0]}/t0.json')
    stage_machines = get_stage_machines(stage_num, machines)
    op_num = sum([len(l) for l in layer_stage_data])
    # print(op_num)
    """先生成预调度方案的Pareto前沿(故障和维护可以直接根据负载和扰动设置考虑进去,因为右移动不会影响原有分配顺序)"""
    init_job_num = len(jobs)
    new_job_num = len(new_jobs)
    # print(init_job_num, new_job_num)
    begin_t = time.time()
    """初始化种群"""
    P = np.zeros((pop_size, init_job_num))
    for p in range(pop_size):
        P[p, :] = np.random.permutation(init_job_num)
    Objs = np.zeros((pop_size, 3))
    Machines = []
    """解码"""
    for p in range(pop_size):
        Objs[p, :], p_machines = decode(P[p, :], init_job_num, stage_num, layer_stage_data, stage_machines,
                                        copy.deepcopy(machines), pub_ec_f, e_co2_f)
        Machines.append(p_machines)

    pareto_objs = None
    pareto_pops = None
    pareto_machines = None
    """开始迭代"""
    se_pool = range(pop_size)
    for i in tqdm(range(iter_n)):
        offspring = Offspring()
        for p in range(pop_size // 2):
            [pos1, pos2] = random.sample(se_pool, 2)
            """既交叉又变异/只交叉不变异/不交叉只变异"""
            situation1 = True  # 既没交叉也没变异
            situation2 = True
            if random.random() > cr_rate:
                # 交叉P1P2推C1C2
                [C1, C2] = crossover(pos1, pos2, P, init_job_num)
                situation1 = False
                situation2 = False
            else:
                C1 = P[pos1, :]
                C2 = P[pos1, :]
            # C1变异概率
            if random.random() > mu_rate:
                C1 = mutate(C1, init_job_num)
                situation1 = False
            if random.random() > mu_rate:
                C2 = mutate(C2, init_job_num)
                situation2 = False
            if not situation1:
                obj_c1, c1_machines = decode(C1, init_job_num, stage_num, layer_stage_data, stage_machines, copy.deepcopy(machines), pub_ec_f, e_co2_f)
                """添加到外部档案集"""
                offspring.append(C1, obj_c1, c1_machines)
            if not situation2:
                obj_c2, c2_machines = decode(C2, init_job_num, stage_num, layer_stage_data, stage_machines, copy.deepcopy(machines), pub_ec_f, e_co2_f)
                """添加到外部档案集"""
                offspring.append(C2, obj_c2, c2_machines)
        """合并父子代"""
        all_P = np.vstack([P, offspring.P])
        all_Objs = np.vstack([Objs, offspring.Objs])
        all_Machines = Machines + offspring.Machines
        """非支配排序"""
        fronts, ranks = fast_non_domination_sort(all_Objs)
        # print(ranks)
        # 使用sorted函数对字典的键值对进行排序，排序依据是值
        sorted_items = sorted(ranks.items(), key=lambda x: x[1])[0:pop_size]
        # 从排序后的键值对列表中提取出键
        saved_pos = [item[0] for item in sorted_items]
        if i == iter_n-1:
            # print("初始工件到达预调度(含右调度)的非支配解目标值\n", all_Objs[fronts[0]])
            pareto_objs = all_Objs[fronts[0]]
            pareto_pops = all_P[fronts[0]]
            pareto_machines = [all_Machines[pos] for pos in fronts[0]]
        else:
            P = all_P[saved_pos]
            Objs = all_Objs[saved_pos]
            Machines = [all_Machines[pos] for pos in saved_pos]
    final_objs = np.zeros((len(pareto_objs), 3))
    for pa in range(len(pareto_objs)):  # 对预调度方案作重组式重调度
        final_objs[pa, :] = reassemble(pareto_objs[pa], copy.deepcopy(new_jobs.tolist()), copy.deepcopy(pareto_machines[pa]), stage_machines, init_job_num, new_job_num, op_num, pub_ec_f, e_co2_f)
    # print(get_ps(final_objs))
    print("初始工件到达预调度(含右调度)的非支配解重组调度后的结果", final_objs.tolist())
    print("耗费时间", time.time() - begin_t)