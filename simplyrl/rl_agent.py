from stable_baselines3 import PPO
import os
import datetime
import numpy as np


def train_agent(actor: object,
                algorithm_name: str,
                pretrained_model: str = None,
                restart: bool = False,
                version: int = 0,
                cfg=None) -> tuple[str, PPO]:

    # Set up training variables
    time_now = datetime.datetime.now()
    time_now = time_now.strftime("%m-%d-%H-%M")

    algorithm_name = f'{algorithm_name}_{time_now}'
    TIMESTEPS = 10000
    models_dir = f"rl_models/{algorithm_name}"
    logdir = "logs"

    env = actor.rl_environment

    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    # Restart training from a previous model
    if pretrained_model:
        models_dir = f"rl-models/{algorithm_name}"
        pretrained_model_path = f"rl-models/{pretrained_model}/{str(version)}"
        # algorithm_name = pretrained_model
        model = PPO.load(pretrained_model_path, reset_num_timesteps=restart, tensorboard_log=logdir, verbose=1, env=env)
    else:
        #         model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=logdir, device='cpu')
        model = PPO("MlpPolicy", env, verbose=1, device='cpu')

    # Train the agent
    for i in range(1, 100000):
        print(i)
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=restart, tb_log_name=f'{algorithm_name}')
        if i % 2 == 0:
            model.save(f"{models_dir}/{TIMESTEPS * i + version}")

    return algorithm_name, model


def predict_agent(actor=None, algorithm=None, best_timestep=None, horizon=24):

    env = actor.rl_environment
    # load RL model to create market schedule prediction
    models_dir = f"rl-models/{algorithm}"
    model_path = f"{models_dir}/{best_timestep}"

    model = PPO.load(model_path, env=env)

    observation = env.reset()

    action = model.predict(observation)
    next_action = np.round(env.action_energy_values[action[0]], 3)

    return next_action
