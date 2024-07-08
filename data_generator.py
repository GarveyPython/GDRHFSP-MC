"""
多约束可重入混合流水车间绿色动态调度问题(GDRHFSP-MC)基准算例数据生成器
# 论文数学符号及模型的描述中，索引从1开始；代码的数据中，索引从0开始,论文绘图时,索引需+1
"""
import random
import numpy as np
import os
import uuid
import json
import argparse

# 固定随机种子
random.seed(10)
np.random.seed(10)

parser = argparse.ArgumentParser()
parser.add_argument('--stage_num', type=int, default=0, help='processing stage (workstation) num in the system')
parser.add_argument('--layer_num', type=int, default=0, help='layer num of jobs in the system')
parser.add_argument('--layer_stage_matrix', type=list, default=[], help='layer_stage_matrix of jobs in the system')
parser.add_argument('--init_job_num', type=int, default=0, help='initial job num in the system')
parser.add_argument('--machine_num', type=int, default=0, help='machine num in the system')
parser.add_argument('--new_job_num', type=int, default=0, help='new job num in the system')
parser.add_argument('--e_co2_f', type=float, default=0.6747, help='electricity CO2 emission factor(kgCO2·(kW·h)-1)')
parser.add_argument('--process_time', type=list, default=[20, 60], help='process time interval(min)')
parser.add_argument('--pub_ec_f', type=int, default=3, help='public unit energy consumption(kW·h)')
parser.add_argument('--proc_ec_f', type=list, default=[30, 50], help='process unit energy consumption(kW·h)')
parser.add_argument('--idle_ec_f', type=list, default=[10, 20], help='idle unit energy consumption(kW·h)')
parser.add_argument('--transport_ec_f', type=int, default=2, help='AGV transport unit energy consumption(kW·h)')
parser.add_argument('--wait_ec_f', type=int, default=0.5, help='AGV wait unit energy consumption(kW·h)')
parser.add_argument('--buffer_size', type=list, default=[3, 5], help='the buffer size range of machines')
parser.add_argument('--s_transport_time', type=list, default=[1, 15],
                    help='the transport time between machines in the same stage')
parser.add_argument('--c_transport_time', type=list, default=[5, 30],
                    help='the transport time between machines crossing the different stage')
parser.add_argument('-pre_maintenance_interval', type=list, default=[5, 8],
                    help='the interval of two adjacent pre-maintenance of a machine(h)')
parser.add_argument('-pre_maintenance_time', type=list, default=[15, 30],
                    help='the basic duration of a pre-maintenance of a machine(min)')
# mean_at=20-60(min)
parser.add_argument('-interval_arrival', type=list, default=[20, 60],
                    help='the mean arrival time(min) interval of new jobs')
parser.add_argument('-max_pre_fac', type=float, default=None,
                    help='max_pre_maintenance_factor')
parser.add_argument('-min_pre_fac', type=float, default=None,
                    help='min_pre_maintenance_factor')

# 工件平均到达间隔mean_at在generator_insert_job_info设置
# 故障率break_p和修复时间repair_range参数在动态调度环境中设置,在调度过程中随机发生加工故障
"""
# env.py中的故障扰动函数
def get_break(m_obj, repair_range=[10, 30], break_p=0.1): # 故障时间和故障率
    break_duration = 0
    break_down = random.random()
    uk_bp = cal_all_machines_interval_workload_percentile()
    # 预维护间隔内已负载比例位于top 10%的机器机器更容易故障
    if m_obj.cal_interval_workload() / m_obj.pre_maintenance_interval >= uk_bp:
        break_down = min(break_down, random.random())
    if break_down < break_p:
        break_duration = random.randint(repair_range[0], repair_range[1])
    return break_duration
"""


# 机器预维护因子设置方法,预维护间隔大的机器(机器性能较优),因子倾向于小,但也有一定的随机扰动
def set_pre_maintenance_factor(args, pre_maintenance_interval):
    lb = args.pre_maintenance_interval[0]
    ub = args.pre_maintenance_interval[1]
    pre_maintenance_factor = round((lb + ub - pre_maintenance_interval) / 200 + random.randint(0, 3) / 1000, 3)
    # 监测生成的因子范围
    # if args.max_pre_fac is None or pre_maintenance_factor > args.max_pre_fac: # 0.043
    #     args.max_pre_fac = pre_maintenance_factor
    # if args.min_pre_fac is None or pre_maintenance_factor < args.min_pre_fac: # 0.025
    #     args.min_pre_fac = pre_maintenance_factor
    return pre_maintenance_factor


# 生成道次阶段匹配表
def generate_layer_stage_matrix(ls_matrix, stage_num, layer_num):
    a1, b1 = 1, round(stage_num / 2)
    a2, b2 = round(stage_num / 2), round(stage_num * 2 / 3)
    # print(layer_num)
    for i in range(layer_num):
        if i == 0:  # 默认第0道次的开始阶段为0
            layer_start_stage = 0
            layer_completion_stage = np.random.randint(a2, b2) if a2 != b2 else a2

        elif i == layer_num - 1:  # 默认最后道次的结束阶段为最后一个阶段
            layer_start_stage = np.random.randint(a1, b1) if a1 != b1 else a1
            layer_completion_stage = stage_num - 1
        else:
            layer_start_stage = np.random.randint(a1, b1) if a1 != b1 else a1
            layer_completion_stage = np.random.randint(a2, b2) if a2 != b2 else a2
        ls_matrix.append([])
        ls_matrix[i] = [j for j in range(layer_start_stage, layer_completion_stage + 1)]
    return ls_matrix


class NoIndent(object):
    def __init__(self, value):
        self.value = value


class NoIndentEncoder(json.JSONEncoder):
    def __init__(self, *args, **kwargs):
        super(NoIndentEncoder, self).__init__(*args, **kwargs)
        self.kwargs = dict(kwargs)
        del self.kwargs['indent']
        self._replacement_map = {}

    def default(self, o):
        if isinstance(o, NoIndent):
            key = uuid.uuid4().hex
            self._replacement_map[key] = json.dumps(o.value, **self.kwargs)
            # print(self._replacement_map[key])
            # print("@@%s@@" % (key,))
            return "@@%s@@" % (key,)
        else:
            return super(NoIndentEncoder, self).default(o)

    def encode(self, o):
        # print("-----------------------------------------------")
        # print("encode被调用")
        result = super(NoIndentEncoder, self).encode(o)
        for k, v in iter(self._replacement_map.items()):
            result = result.replace('"@@%s@@"' % (k,), v)
            # print(result)
        return result


def generator_job_info(data, args, job_key, job_format, arrival_t, flag=False):
    if not flag:
        job_num = args.init_job_num
    else:
        job_num = args.new_job_num
    for j in range(job_num):
        name = job_format.format(j)
        data[job_key][name] = {}
        data[job_key][name]["arrival_t"] = arrival_t[j]


# 生成插入工件的信息:包含基本工件信息和插入时间
def generator_insert_job_info(data, args, new_job_key, new_job_format):
    mean_at = np.random.randint(low=args.interval_arrival[0], high=args.interval_arrival[1])
    # 工件的到达时间服从指数分布
    # 生成到达时间
    # 第一个参数scale代表1/lamda 即1/lamda=mean_at
    arrival_t = np.cumsum(np.round(np.random.exponential(mean_at, size=args.new_job_num))).astype(int).tolist()
    if arrival_t[0] == 0:
        raise Exception("the first new job's arrival time cannot be zero")
    generator_job_info(data, args, new_job_key, new_job_format, arrival_t, flag=True)


# 生成运输信息
def generate_transport_info(data, args, machine_key, machine_format):
    mach_num = args.machine_num
    transport_matrix = np.zeros((mach_num, mach_num))
    for i in range(mach_num):
        m_name = machine_format.format(i)
        m_belong_stage = data[machine_key][m_name]
        for j in range(mach_num):
            if i != j:  # 设定同一个机器间不存在运输
                j_name = machine_format.format(j)
                j_belong_stage = data[machine_key][j_name]
                if m_belong_stage == j_belong_stage:  # 同一个工站
                    # random.randint是左闭右闭
                    if transport_matrix[j][i] != 0:
                        # 从机器j到机器i的运输时长
                        transport_matrix[i][j] = transport_matrix[j][i]
                    else:
                        transport_matrix[i][j] = random.randint(args.s_transport_time[0], args.s_transport_time[1])
                else:  # 跨阶段
                    if transport_matrix[j][i] != 0:
                        transport_matrix[i][j] = transport_matrix[j][i]
                    else:
                        transport_matrix[i][j] = random.randint(args.c_transport_time[0], args.c_transport_time[1])
    # print(transport_matrix.astype('int').tolist())
    data[machine_key]['transport_matrix'] = NoIndent(transport_matrix.astype('int').tolist())
    data[machine_key]['transport_ec_f'] = args.transport_ec_f  # AGV运输功率
    data[machine_key]['wait_ec_f'] = args.wait_ec_f  # AGV等待功率


# 生成工序的加工时间序列
def generate_processing_times(proc_E, num_pieces, args):
    min_time = args.process_time[0]
    max_time = args.process_time[1]
    assert args.proc_ec_f[0] <= proc_E <= args.proc_ec_f[
        1], "proc_E must fitting requirement"

    # 加工时间与能耗之间有一定反向关系，即加工时间小的,能耗消耗也倾于大,但也有一定的随机扰动,如受工件型号与机器适配度、工件所处道次因素的影响
    # 确定一个基础的加工时间
    base_time = max_time - (
            (max_time - min_time) * (proc_E - args.proc_ec_f[0]) / (args.proc_ec_f[1] - args.proc_ec_f[0]))  # 映射
    base_time = max(min_time, base_time)  # 确保基础加工时间不小于min_time

    # 生成5min内即[-2.5, 2.5]之间的随机浮点数扰动,因为加工时间与能耗不完全是反比,加工时间还有工件\道次适配度的问题
    perturbations = np.random.uniform(low=-2.5, high=2.5, size=num_pieces)

    # 将扰动加到基础时间上，并确保结果不超过最大和最小时间
    perturbed_times = base_time + perturbations
    final_times = np.clip(perturbed_times, min_time, max_time)

    # 加工时间整数化,四舍五入到最接近的整数
    final_times = np.round(final_times).astype(int)

    return final_times


# 生成机器具体信息
def machine_details_info(data, m, args, machine_key, machine_format, layer_key, layer_format, job_key,
                         new_job_key, belong_stage=None):
    name = machine_format.format(m)
    data[machine_key][name] = {}
    # np.random.randint是左闭右开
    data[machine_key][name]['buffer_size'] = np.random.randint(args.buffer_size[0], args.buffer_size[1] + 1)
    data[machine_key][name]["belong_stage"] = np.random.randint(0, args.stage_num) if belong_stage is None else belong_stage
    # 生成范围内随机小数,并保留两位小数
    data[machine_key][name]['proc_ec_f'] = round(random.uniform(args.proc_ec_f[0], args.proc_ec_f[1]), 2)
    data[machine_key][name]["idle_ec_f"] = round(random.uniform(args.idle_ec_f[0], args.idle_ec_f[1]), 2)
    set_maintenance_interval = np.random.randint(args.pre_maintenance_interval[0], args.pre_maintenance_interval[1] + 1)
    data[machine_key][name]['pre_maintenance_interval'] = set_maintenance_interval
    data[machine_key][name]['pre_maintenance_time'] = np.random.randint(args.pre_maintenance_time[0],
                                                                        args.pre_maintenance_time[1] + 1)
    data[machine_key][name]['pre_maintenance_factor'] = set_pre_maintenance_factor(args, set_maintenance_interval)

    layer_stage_matrix = data[layer_key]["layer_stage_matrix"]
    layer_num = data[layer_key]["layer_num"]
    init_job_num = data[job_key]["init_job_num"]
    new_job_num = data[new_job_key]["new_job_num"]

    for l in range(layer_num):
        # 如果该机器所属的的阶段属于该道次
        if data[machine_key][name]["belong_stage"] in layer_stage_matrix[l]:
            l_name = layer_format.format(l)  # 标记好道次
            data[machine_key][name][l_name] = {}
            init_p_t = generate_processing_times(data[machine_key][name]['proc_ec_f'], init_job_num, args)
            # 为初始到达的工件设定加工时间
            data[machine_key][name][l_name]["init_p_ts"] = NoIndent(init_p_t.tolist())
            # 为动态到达的工件设定加工时间
            new_p_t = generate_processing_times(data[machine_key][name]['proc_ec_f'], new_job_num, args)
            data[machine_key][name][l_name]["new_p_ts"] = NoIndent(new_p_t.tolist())


def rename_key(dictionary, old_key, new_key):
    value = dictionary.pop(old_key)
    dictionary.setdefault(new_key, value)


def generator_machine_info(data, args, machine_key, machine_format, layer_key, layer_format, job_key, new_job_key):
    stage_num = args.stage_num
    machine_num = args.machine_num
    # 先确保每个阶段至少有两台并行机
    if stage_num * 2 > machine_num:
        return
    for i in range(stage_num):
        machine_details_info(data, i, args, machine_key, machine_format, layer_key, layer_format, job_key, new_job_key,
                             i)
    for i in range(stage_num, stage_num * 2):
        machine_details_info(data, i, args, machine_key, machine_format, layer_key, layer_format, job_key, new_job_key,
                             i - stage_num)
    # 随机归置剩余的机器
    for i in range(stage_num * 2, machine_num):
        machine_details_info(data, i, args, machine_key, machine_format, layer_key, layer_format, job_key, new_job_key,
                             belong_stage=None)


def generate_instance(args):
    stage_key, stage_format = "stage", "stage{}"
    layer_key, layer_format = "layer", "layer{}"
    machine_key, machine_format = "machine", "machine{}"
    job_key, job_format = "job", "job{}"
    new_job_key, new_job_format = "new_job", "new_job{}"
    data = {stage_key: {}, layer_key: {}, machine_key: {}, job_key: {}, new_job_key: {}}
    data[stage_key]["stage_num"] = args.stage_num
    data[layer_key]["layer_num"] = args.layer_num
    data[layer_key]["layer_stage_matrix"] = generate_layer_stage_matrix([], args.stage_num, args.layer_num)
    data[machine_key]["machine_num"] = args.machine_num
    data[job_key]["init_job_num"] = args.init_job_num
    data[new_job_key]["new_job_num"] = args.new_job_num
    arrival_t0 = np.zeros(args.init_job_num).astype(int).tolist()
    generator_job_info(data, args, job_key, job_format, arrival_t0, flag=False)
    generator_insert_job_info(data, args, new_job_key, new_job_format)
    generator_machine_info(data, args, machine_key, machine_format, layer_key, layer_format, job_key, new_job_key)
    # 最后，将道次阶段矩阵也编码
    data[layer_key]["layer_stage_matrix"] = NoIndent(data[layer_key]["layer_stage_matrix"])

    generate_transport_info(data, args, machine_key, machine_format)
    data[machine_key]['pub_ec_f'] = args.pub_ec_f  # 公共能耗因子
    data[machine_key]['e_co2_f'] = args.e_co2_f  # 碳排放因子
    return data


# 生成训练数据
def generate_train_data():
    args = parser.parse_args()
    path = '../data/train'
    # 使得训练数据的组合异于测试,以体现泛化性
    jn = [30]  # 初始工件数量
    mn = [30]  # 加工机器数量
    nn = [40]  # 新工件数量
    sn = [5]  # 阶段数量
    ln = [6]  # 道次数量
    times = 30
    for k in range(len(jn)):
        j = jn[k]
        m = mn[k]
        n = nn[k]
        s = sn[k]
        l = ln[k]
        dire = 'j{}_m{}_n{}_k{}_l{}'.format(j, m, n, s, l)
        data_path = '/'.join([path, dire])
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        for t in range(times):
            args.stage_num = s
            args.layer_num = l
            args.init_job_num = j
            args.machine_num = m
            args.new_job_num = n

            data = generate_instance(args)

            name = "t{}.json".format(t)
            file = "/".join([data_path, name])
            with open(file, 'w') as f:
                obj = json.dumps(data, indent=3, cls=NoIndentEncoder)
                f.write(obj)
    # 查看构建的预维护因子范围
    # print('pre_maintenance_factor: ', args.max_pre_fac, args.min_pre_fac)
    print('finish generating train data!!!')


# 生成测试数据
def generator_test_data():
    args = parser.parse_args()
    path = "../data/test"

    mn = [10, 20, 20, 30, 30, 40, 50, 50, 60]
    jn = [15, 20, 20, 25, 30, 35, 40, 45, 50]
    nn = [20, 30, 40, 50, 60, 70, 80, 90, 100]
    sn = [5, 5, 6, 6, 6, 7, 7, 7, 8]
    ln = [3, 4, 4, 5, 6, 6, 7, 8, 10]
    inst = []
    for i in range(len(jn)):
        inst.append([jn[i], mn[i], nn[i], sn[i], ln[i]])
    times = 1
    for i in range(len(inst)):
        j, m, n, s, l = inst[i]
        dire = 'j{}_m{}_n{}_k{}_l{}'.format(j, m, n, s, l)
        data_path = '/'.join([path, dire])
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        for t in range(times):
            args.stage_num = s
            args.layer_num = l
            args.init_job_num = j
            args.machine_num = m
            args.new_job_num = n

            data = generate_instance(args)

            name = "t{}.json".format(t)
            file = "/".join([data_path, name])
            with open(file, 'w') as f:
                obj = json.dumps(data, indent=3, cls=NoIndentEncoder)
                f.write(obj)
    # 查看构建的预维护因子范围
    # print('pre_maintenance_factor: ', args.max_pre_fac, args.min_pre_fac)
    print('finish generating test data!!!')


if __name__ == "__main__":
    # 训练和测试算例生成时,两个函数分开调用,分两次运行
    # generate_train_data()
    generator_test_data()
