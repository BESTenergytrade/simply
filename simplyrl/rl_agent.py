import numpy as np
from stable_baselines3 import PPO
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
