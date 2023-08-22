from stable_baselines3 import PPO
import os
import datetime



def train_agent(algorithm_name: str,
                initial_soc: float = None,
                horizon: int = 96,
                pretrained_model: str = None,
                restart: bool = False,
                version: int = 0,
                energy_unit: float = 0.001,
                stored_value: float = 0.7,
                cfg=None) -> tuple[str, PPO]:

    # Set up training variables
    time_now = datetime.datetime.now()
    time_now = time_now.strftime("%m-%d-%H-%M")

    algorithm_name = f'{algorithm_name}_{time_now}'
    TIMESTEPS = 10000
    models_dir = f"rl_models/{algorithm_name}"
    logdir = "logs"

    env = EnergyEnv(horizon=horizon, initial_soc=initial_soc, energy_unit=energy_unit, stored_value=stored_value,
                    training=True)

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


def predict_agent(algorithm, best_timestep, initial_soc, horizon, gap=False, actor_df=None):

    horizon = horizon

    env = EnergyEnv(horizon=horizon, initial_soc=initial_soc, energy_unit=0.001, stored_value=0.7, actor_df=None,
                    gap=False, training=False, market=None, scenario=None, actor=None)

    # load RL model to create market schedule prediction
    models_dir = f"rl-models/{algorithm}"
    model_path = f"{models_dir}/{best_timestep}"

    model = PPO.load(model_path, env=env)

    observation = env.reset()

    banks = []
    banks_with_battery = []
    socs = []

    for i in range(horizon):
        action, _ = model.predict(observation)
        observation, reward, done, _ = env.step(action)
        if done:
            print(f'{algorithm} agent failed')
            break

        # Update evaluation logs
        banks.append(env.actor.bank)
        banks_with_battery.append(env.actor.bank + (env.actor.battery.energy() * actor_df.iloc[:24, 2].mean()))
        socs.append(env.actor.battery.soc)

    if not done:
        return banks, banks_with_battery, socs
    else:
        return None, _, _
