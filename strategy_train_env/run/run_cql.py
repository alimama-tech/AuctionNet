import numpy as np
import torch
import logging
from bidding_train_env.common.utils import normalize_state, normalize_reward, save_normalize_dict
from bidding_train_env.baseline.iql.replay_buffer import ReplayBuffer
from bidding_train_env.baseline.cql.cql import CQL
import sys
import pandas as pd
import ast
import pickle
from torch.utils.tensorboard import SummaryWriter
# Configure logging
logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s] [%(name)s] [%(filename)s(%(lineno)d)] [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

STATE_DIM = 16

def train_cql_model(seed=1):
    """
    Train the CQL model.
    """
    train_data_path = "./data/traffic/training_data_rlData_folder/training_data_all-rlData.csv"
    training_data = pd.read_csv(train_data_path)

    def safe_literal_eval(val):
        if pd.isna(val):
            return val  # 如果是NaN，返回NaN
        try:
            return ast.literal_eval(val)
        except (ValueError, SyntaxError):
            print(ValueError)
            return val  # 如果解析出错，返回原值

    # 使用apply方法应用上述函数
    training_data["state"] = training_data["state"].apply(safe_literal_eval)
    training_data["next_state"] = training_data["next_state"].apply(safe_literal_eval)
    STATE_DIM = len(training_data['state'].iloc[0])

    is_normalize = True
    if is_normalize:
        normalize_dic = normalize_state(training_data, STATE_DIM, normalize_indices=[13, 14, 15])
        training_data['reward'] = normalize_reward(training_data, "reward_continuous")
        save_normalize_dict(normalize_dic, "saved_model/CQLtest")

    # Build replay buffer
    replay_buffer = ReplayBuffer()
    add_to_replay_buffer(replay_buffer, training_data, is_normalize)

    # Train model
    model = CQL(dim_obs=STATE_DIM)
    train_model_steps(model, replay_buffer, seed=seed)

    # Save model
    # model.save_net("saved_model/CQLtest")
    #model.save_jit("saved_model/CQLtest", seed=seed, step=i)

    # Test trained model
    test_trained_model(model, replay_buffer)

def add_to_replay_buffer(replay_buffer, training_data, is_normalize):
    for row in training_data.itertuples():
        state, action, reward, next_state, done = row.state if not is_normalize else row.normalize_state, row.action, row.reward if not is_normalize else row.normalize_reward, row.next_state if not is_normalize else row.normalize_nextstate, row.done
        # ! 去掉了所有的done==1的数据
        if done != 1:
            replay_buffer.push(np.array(state), np.array([action]), np.array([reward]), np.array(next_state),
                               np.array([done]))
        else:
            replay_buffer.push(np.array(state), np.array([action]), np.array([reward]), np.zeros_like(state),
                               np.array([done]))

def train_model_steps(model, replay_buffer, step_num=100000, batch_size=100, seed=1):
    for i in range(step_num):
        states, actions, rewards, next_states, terminals = replay_buffer.sample(batch_size)
        q1_loss, q2_loss, policy_loss = model.step(states, actions, rewards, next_states, terminals, i)
        if i == 0:
            writer = SummaryWriter(log_dir="tensorboard/CQL/" + str(seed))
        writer.add_scalar('Loss/q1_loss', q1_loss, i)
        writer.add_scalar('Loss/q2_loss', q2_loss, i)
        writer.add_scalar('Loss/policy_loss', policy_loss, i)
        if i == step_num - 1:
            writer.close()
        if i+1 % 10000 == 0:
            model.save_jit("saved_model/CQLtest", seed=seed, step=i+1)
        #logger.info(f'Step: {i} Q_loss: {q_loss} V_loss: {v_loss} A_loss: {a_loss}')

def test_trained_model(model, replay_buffer):
    for i in range(100):
        states, actions, rewards, next_states, terminals = replay_buffer.sample(1)
        pred_actions = model.take_actions(torch.tensor(states, dtype=torch.float))
        actions = actions.cpu().detach().numpy()
        tem = np.concatenate((actions, pred_actions), axis=1)
        print("concate:",tem)

def run_cql(seed=1):
    """
    Run CQL model training and evaluation.
    """
    train_cql_model(seed=seed)

if __name__ == '__main__':
    run_cql()
