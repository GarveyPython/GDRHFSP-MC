import numpy as np
import os
import shutil
from utils.agent import Sequence_Agent, Route_Agent
from utils.env import PPO_ENV
from utils.config import config
from utils.uniform_weight import cweight
from utils.utils import get_data, get_weight


def param_experiment_train():
    args = config()
    Epoch_Batch = [
        [15, 128]]
    if not os.path.exists(args.param_comb_ckpt):
        os.makedirs(args.param_comb_ckpt)

    for param_comb in Epoch_Batch:
        args.epochs = param_comb[0]
        args.batch_size = param_comb[1]
        args.a_update_step = 10
        args.c_update_step = 10
        # **********************
        # 设置采用的Clip方法
        args.clip = 'EMD'  # PPO or EMD
        args.network = 'Attention'
        args.gamma = 0.99
        args.gae_lambda = 0.95
        name = "E{}_B{}_S{}_{}_{}".format(param_comb[0], param_comb[1], args.a_update_step, args.clip, args.network)
        dirname = '/'.join([args.param_comb_ckpt, name])  # 参数组合ckpt保存文件夹
        if os.path.exists(dirname):
            shutil.rmtree(dirname)
            os.makedirs(dirname)
        args.sa_ckpt_path = '/'.join([dirname, "sa"])  # sa ckpt文件路径
        args.ra_ckpt_path = '/'.join([dirname, "ra"])  # ra ckpt文件路径
        train(args)
    print('train end')


def train(args):
    global job, sa_reward, done1
    if not os.path.exists(args.sa_ckpt_path):
        os.makedirs(args.sa_ckpt_path)
    if not os.path.exists(args.ra_ckpt_path):
        os.makedirs(args.ra_ckpt_path)
    train_data = get_data(args.train_data)
    train_data_size = len(train_data)
    weight, size = get_weight(args)

    # sa_reward_episodic = []
    # ra_reward_episodic = []
    sa_reward_doc = []
    ra_reward_doc = []
    MEAN_RA_REWARD = []
    MEAN_SA_REWARD = []
    for i in range(size):
        sa_reward_doc.append([])
        ra_reward_doc.append([])
        # sa_reward_episodic.append([])
        # ra_reward_episodic.append([])
        # 考虑到裁剪机制的衰减影响, 训练新的一组权重时, 重新初始化智能体
        sa = Sequence_Agent(args)
        ra = Route_Agent(args)
        w = weight[size - 1 - i].reshape(1, -1)
        for epoch in range(args.epochs): # 这里的epoch注意是episode
            sa_reward_doc[i].append([])
            ra_reward_doc[i].append([])
            done2 = False  # 所有工件是否已分配完成
            inst = train_data[(i * args.epochs + epoch) % train_data_size]
            env = PPO_ENV(inst, args)
            cweight(w, env, sa, ra)  # 对env中w权重向量的赋值在该函数里
            sa_state, mach_index1, t = env.reset(ra, ra_reward_doc=ra_reward_doc[i][epoch])  # 初始工件到达,RA分配

            while True:
                # print(t)
                if env.check_njob_arrival(t):
                    job = env.njob_insert()  # 新工件插入事件
                    env.njob_route(job, t, ra, ra_reward_doc=ra_reward_doc[i][epoch])

                sa_action = sa.choose_action(sa_state)  # 在mach的候选buffer中选择1个工件进行加工
                # SA_STEP事件
                # if not env.machines[mach_index1].block_state:  # 机器没被阻塞才能继续加工(已改为阻塞AGV)
                if len(env.machines[mach_index1].buffer_op) > 0:
                    job, sa_reward, done1, end = env.sa_step(mach_index1, sa_action, t)
                    if len(sa_reward_doc[i][epoch]) == 0:
                        sa_reward_doc[i][epoch].append(sa_reward)
                    else:
                        sa_reward_doc[i][epoch].append(sa_reward_doc[i][epoch][-1] + sa_reward)

                if not done2 and job.not_finish() and not job.block_state:
                    ra_state = env.get_ra_state()
                    ra_action = ra.choose_action(ra_state)
                    # RA_STEP事件(不含初始工件和新工件的第一次分配)
                    ra_reward, done2 = env.ra_step(job, ra_action, t)
                    ra.store(ra_state, ra_action, ra_reward, done2)
                    if len(ra_reward_doc[i][epoch]) == 0:
                        ra_reward_doc[i][epoch].append(ra_reward)
                    else:
                        ra_reward_doc[i][epoch].append(ra_reward_doc[i][epoch][-1] + ra_reward)
                    if ra.buffer.cnt == args.batch_size:
                        ra.learn(ra_state, done2)
                    if done2 and ra.buffer.cnt != 0:
                        ra.learn(ra_state, done2)
                sa_state_, mach_index1, t = env.step(ra, t)
                # print(env.machines[mach_index1].block_state)
                # print(sa_state)
                sa.buffer.store(sa_state, sa_action, sa_reward, sa_state_, done1)
                if sa.buffer.cnt == args.batch_size:
                    sa.learn(sa_state_, done1)
                if done1 and sa.buffer.cnt != 0:
                    sa.learn(sa_state_, done1)
                sa_state = sa_state_
                if done1:  # 所有工件已加工完成
                    break
            # print("sa_action = ", a1)
            # print("ra_action = ", a2)
            obj = env.cal_objective()
            inst = inst.split('/')[-1]
            weight_print = np.around(w.reshape(-1, ), 2).tolist()
            # print(env.sys_state())
            print(f'inst {inst}, weight{weight_print}, epoch{epoch} | obj1 = {obj[0]}, obj2 = {obj[1]}, obj3 = {obj[2]}', 'ra_reward', ra_reward_doc[i][epoch][-1], 'sa_reward', sa_reward_doc[i][epoch][-1])
            ra.record = None  # 回合转移,避免RA震荡
            # print(ra_reward_doc)
            # print(sa_reward_doc)
        # print("RA_REWARD", ra_reward_doc)
        # print("SA_REWARD", sa_reward_doc)
        ra_reward_w = ra_reward_doc[i]
        sa_reward_w = sa_reward_doc[i]
        MEAN_RA_REWARD.append(np.mean(np.array(ra_reward_w), axis=0).tolist())
        MEAN_SA_REWARD.append(np.mean(np.array(sa_reward_w), axis=0).tolist())
        print(MEAN_RA_REWARD[i])
        print(MEAN_SA_REWARD[i])
        sa.save(weight=w)  # 传入SA采样权重
        ra.save(weight=w)  # 传入RA采样权重
    Final_RA_MEAN_REWARD = np.mean(np.array(MEAN_RA_REWARD), axis=0).tolist()
    print('RA', Final_RA_MEAN_REWARD)
    Final_SA_MEAN_REWARD = np.mean(np.array(MEAN_SA_REWARD), axis=0).tolist()
    print('SA', Final_SA_MEAN_REWARD)
    print('--------end-------')


if __name__ == "__main__":
    # 在训练之前设置好训练参数
    param_experiment_train()
