import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from stable_baselines3 import PPO
import os
# import torch
# import tensorboard
# try:
#     import torch
# except ImportError:
#     raise ModuleNotFoundError(
#         f"Optional dependency 'torch' not found. To run training of reinforcement learning agent"
#         f"please install dependency via 'pip install torch'."
#     )


def train_agent(actor=None, training_steps=2048, clear_memory=True):
    """
    Trains and updates RL model by calling PPO internal function `learn()` of the RL model.
    Model is trained for one batch training interval of length training_steps.
    @param actor: RL agent object
    @param training_steps: number of training steps of the batch training after which model is updated and saved
    @param clear_memory: optionally the instances of training attributes in rl_environment can be reset
    """

    # model directory to save updated model in
    algorithm = "new_start/24/1.4/norm_bank_reward_04-03-21-4"
    models_dir = f"rl-models/{algorithm}"

    # train the rl model
    actor.rl_model.learn(
        total_timesteps=training_steps,
        reset_num_timesteps=False,
        tb_log_name=f'{actor.id}_{algorithm}',
        progress_bar=True,
    )

    # save model
    actor.rl_model.save(f"{models_dir}/{actor.rl_model.num_timesteps}")

    # save performance of current training interval, i.e rewards, banks and battery socs
    interval, _ = divmod(actor.rl_environment.t_step_simply, actor.rl_environment.training_interval)
    REWARDS = pd.DataFrame.from_dict(actor.rl_environment.rewards_simply, orient="index", columns=[f"Reward Interval {interval}"])
    BANKS = pd.DataFrame.from_dict(actor.rl_environment.banks_simply, orient="index", columns=[f"Bank Interval {interval}"])
    SOCS = pd.DataFrame.from_dict(actor.rl_environment.socs_simply, orient="index", columns=[f"Battery SoC Interval {interval}"])
    RESULTS = pd.concat([REWARDS, BANKS, SOCS], axis=1).reset_index()
    RESULTS.rename(columns={"index": "t_step"}, inplace=True)
    if not os.path.exists(f"{models_dir}/benchmarks"):
        os.makedirs(f"{models_dir}/benchmarks")
    RESULTS.to_csv(f"{models_dir}/benchmarks/Results_interval_{interval}.csv")

    # reset environment variables to clear memory (optional)
    if clear_memory:
        actor.rl_environment.reset_train_env()


def predict_agent(actor=None):
    """
    Determines a predicted action and corresponding schedule value for a given RL agent.
    @param actor: RL agent
    @return: schedule prediction value
    """

    env = actor.rl_environment
    model = actor.rl_model

    observation, _ = env.reset(options="prediction")

    actor.action = model.predict(observation)[0]
    next_action = np.round(env.action_energy_values[actor.action], 3)

    return next_action


def load_model(actor=None, model_path=None, pretrained=True):
    """
    Imports a RL model for a given RL agent.
    @param actor: RL agent
    @param model_path: path where the model is stored in
    @param pretrained: name of the pretrained model. if none is given a new model is created
    @return: RL model
    """

    # import model
    # TODO: make import of model variable to each actor and its identifier
    env = actor.rl_environment
    if pretrained:
        model = PPO.load(model_path, env=env)
    else:
        model = PPO("MlpPolicy", env=env, verbose=1, device='cpu')
    return model


def visualise_performance(results_dir, intervals):
    results_list = [pd.read_csv(f"{results_dir}/Results_interval_{interval}.csv", index_col=0).drop(columns=["t_step"])
                    for interval in range(1, intervals + 1)]
    results_df = pd.concat(results_list, axis=1)
    reward_df = results_df.filter(like="Reward", axis=1)
    bank_df = results_df.filter(like="Bank", axis=1)
    profits_df = bank_df - bank_df.loc[0]
    soc_df = results_df.filter(like="SoC", axis=1)

    # Create a figure and axis object
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 9))

    # Create the boxplot
    sns.boxplot(data=reward_df, ax=ax1)

    # create linechart for Bank and Battery SoC after interval
    profits_df.plot(ax=ax2)
    soc_df.plot(ax=ax3)

    # Label the y-axis
    ax1.set_ylabel('Rewards in training interval 1')
    ax2.set_ylabel('Bank profits in training interval 1')
    ax3.set_ylabel('Battery SoC in training interval 1')

    # Set x-axis label and ticks
    ax2.set_xlabel("timesteps")
    ax3.set_xlabel("timesteps")
    ax2.xaxis.set_ticks(np.arange(0, len(bank_df) + 1, 5))
    ax3.xaxis.set_ticks(np.arange(0, len(bank_df) + 1, 5))

    # Rotate the xticklabels by 45 degrees
    xtick_labels = ax1.get_xticklabels()
    ax1.set_xticklabels(xtick_labels, rotation=45)

    # Add grid lines
    ax1.yaxis.grid(True)
    ax2.yaxis.grid(True)
    ax3.yaxis.grid(True)

    # show legend for linecharts
    ax2.legend(loc="upper center")
    ax3.legend(loc="upper center")

    # Show the plot
    fig.suptitle('Comparison of agents performance in different intervals')
    fig.tight_layout()
    fig.savefig(f'{results_dir}/performance_test.png', dpi=300)
    plt.show()
