import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from argparse import ArgumentParser
from datetime import datetime
from match_market import main
from simply.config import Config
from simply.actor import Actor
from simply.market_maker import MarketMaker
from tqdm import tqdm
import os
import time


def run_testing(cfg, nb_ts=2688):
    """
    runs tests on common dataset to provide performance measure for RL training progress
    @param cfg: config file for
    @return:
    """
    pretrained_model = cfg.pretrained_model
    intervals, _ = divmod(cfg.nb_ts, cfg.training_interval)
    if pretrained_model == "":
        pretrained_model = "2048"

    # if testing number of timesteps smaller than simulation timesteps
    if nb_ts < cfg.nb_ts:
        cfg.nb_ts = nb_ts

    models = list(range(int(pretrained_model), int(pretrained_model) + 2048 * intervals, 2048))
    cfg.train_rl = False
    REWARDS = {}
    ENERGY = {}
    PVS = {}
    LOADS = {}
    SOCS = {}
    BANKS = {}
    for version in tqdm(models):
        cfg.pretrained_model = str(version)
        sc = main(cfg)
        print(f"-----------Test run model {version} is completed. Logging data.")
        # find participant that is RL agent, only works with ONE Rl agent
        for participant in sc.market_participants:
            if isinstance(participant, Actor):
                if participant.strategy == 4:
                    REWARDS[str(version)] = participant.rewards
                    ENERGY[str(version)] = participant.matched_energies
                    PVS[str(version)] = participant.data.pv
                    LOADS[str(version)] = participant.data.load
                    SOCS[str(version)] = participant.socs
                    BANKS[str(version)] = participant.rl_bank
            elif isinstance(participant, MarketMaker):
                MARKET_PRICES = participant.all_buy_prices

    # turn results of test runs into
    # buying energy as well as pv have positive sign
    # selling energy has negative sign
    rewards_df = pd.DataFrame.from_dict(REWARDS)
    energy_df = pd.DataFrame.from_dict(ENERGY)
    market_prices_df = pd.DataFrame(MARKET_PRICES, columns=["market_price"])
    pvs_df = pd.DataFrame.from_dict(PVS)
    loads_df = pd.DataFrame.from_dict(LOADS)
    socs_df = pd.DataFrame.from_dict(SOCS)
    banks_df = pd.DataFrame.from_dict(BANKS, orient="index", columns=["profit"])

    return rewards_df, energy_df, market_prices_df, pvs_df, loads_df, socs_df, banks_df


# %%
def visualise_performance(rewards_df, socs_df, banks_df, intervals, show_plot=True):
    # Create a figure and axis object
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 9))

    # Create the boxplot
    sns.boxplot(data=rewards_df, ax=ax1, showfliers=False)

    # create linechart for Bank and Battery SoC after interval
    banks_df.plot(ax=ax2, marker="x")
    socs_df.plot(ax=ax3)

    # Label the y-axis
    ax1.set_ylabel('Rewards in evaluation period')
    ax2.set_ylabel('Bank profits in evaluation period')
    ax3.set_ylabel('Battery SoC in evaluation period')

    # Set x-axis label and ticks
    ax3.set_xlabel("timesteps")
    ax3.xaxis.set_ticks(np.arange(0, len(socs_df) + 1, len(socs_df) / 10))

    # Rotate the xticklabels by 90 degrees
    xtick_labels = [f"Model {mod}" for mod in rewards_df.columns]
    ax1.set_xticks(list(range(intervals)), xtick_labels, rotation=45, ha="right")
    xtick_labels = [f"Model {mod}" for mod in banks_df.index]
    ax2.set_xticks(list(range(intervals)), xtick_labels, rotation=45, ha="right")

    # Add grid lines
    ax1.yaxis.grid(True)
    ax2.yaxis.grid(True)
    ax3.yaxis.grid(True)

    # show legend for linecharts
    # ax2.legend(loc="upper center")
    ax3.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)

    # Show the plot
    fig.suptitle('Comparison of agents performance in different intervals')
    fig.tight_layout()
    # datetime object containing current date and time
    now = datetime.now()

    # dd/mm/YY H:M:S
    dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")
    fig.savefig(f'{results_dir}/performance_test_{dt_string}.png', dpi=300)
    if show_plot:
        plt.show()


# %%
def visualise_trading_behaviour(energy_df, market_prices_df, pvs_df, loads_df, socs_df, n_ts, seed, interval, show_plot=True):
    # choose random interval of length n_ts for plotting behaviour
    np.random.seed(seed)
    t_start = np.random.choice(list(range(len(market_prices_df.iloc[:, 0]) - n_ts)))
    # select data
    energy = energy_df.iloc[t_start:t_start + n_ts, interval - 1]
    market_prices = market_prices_df.iloc[t_start:t_start + n_ts, 0]
    pv = pvs_df.iloc[t_start:t_start + n_ts, interval - 1]
    battery = socs_df.iloc[t_start:t_start + n_ts, interval - 1] * 13.5
    load = loads_df.iloc[t_start:t_start + n_ts, interval - 1]

    # Create a figure and axis object
    fig, ax = plt.subplots(figsize=(16, 9))

    ax2 = ax.twinx()

    # create linechart for energy, market prices and actor pv and load
    energy.plot(ax=ax, linestyle="solid", label="energy", color="#2c777a")
    market_prices.plot(ax=ax2, linestyle="dashed", label="market_price", color="#800020")
    pv.plot(ax=ax, linestyle="solid", label="pv", color="#fff16a")
    battery.plot(ax=ax, linestyle="solid", label="battery", color="#47dc7a", alpha=0.3)
    load.plot(ax=ax, linestyle="dashdot", label="load", color="#2e313f")

    # added these three lines
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()

    # Label the y-axis
    ax.set_ylabel("Energy in kWh")
    ax2.set_ylabel("Price in €/kWh")
    market_prices_range = market_prices.max() * 1.2 - market_prices.min()
    ax2.set_ylim(market_prices.min() - market_prices_range, market_prices.min() + market_prices_range)

    # Set x-axis label and ticks
    ax.set_xlabel("timesteps")
    ax.xaxis.set_ticks(np.arange(t_start, t_start + n_ts + 1, n_ts / 12))

    # Add grid lines
    ax.yaxis.grid(True)

    # show legend for linecharts
    ax.legend(lines + lines2, labels + labels2, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)

    # Show the plot
    fig.suptitle(f'Trading behaviour of rl model {pvs_df.columns[interval - 1]}')
    fig.tight_layout()
    # datetime object containing current date and time
    now = datetime.now()

    # dd/mm/YY H:M:S
    dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")
    fig.savefig(f'{results_dir}/trading_behaviour_test_model_{pvs_df.columns[interval - 1]}_{dt_string}.png', dpi=300)
    if show_plot:
        plt.show()


if __name__ == "__main__":
    # to track run time
    start_time = time.time()

    algorithm = "new_start/24/1.4/norm_bank_reward_04-03-21-4"
    # algorithm = "new_start/24/1.4/norm_bank_reward_new"
    results_dir = f"rl-models/{algorithm}/benchmarks"

    parser = ArgumentParser(description='Entry point for market simulation')
    # parser.add_argument('config', nargs='?', default="", help='configuration file')
    # Replaced the above line to take in the project directory (which will contain the config file)
    # instead of putting in the config file
    # also made it mandatory
    parser.add_argument('project_dir', nargs='?', default=None, help='project directory path')
    args = parser.parse_args()
    # Raise error if project directory not specified
    if args.project_dir is None:
        raise (
            FileNotFoundError(
                "Project directory path must be specified. Please provide the path as a "
                "command-line argument."))

    project_dir = args.project_dir

    # This means that the config file must always be in the project directory
    config_file = os.path.join(project_dir, "config.cfg")
    # Raise error if config.(cfg|txt) file not found in project directory
    if not os.path.isfile(config_file):
        config_file = os.path.join(project_dir, "config.txt")
        if not os.path.isfile(config_file):
            raise (FileNotFoundError(
                f"Config file not found in project directory: {project_dir}"))
    cfg = Config(config_file, project_dir)
    intervals, _ = divmod(cfg.nb_ts, cfg.training_interval)

    # run testing and store results of test runs
    rewards_df, energy_df, market_prices_df, pvs_df, loads_df, socs_df, banks_df = run_testing(cfg)
    # create directory for test run
    dt_string = datetime.now().strftime("%d-%m-%Y_%H-%M")  # dd/mm/YY H:M
    if not os.path.exists(f'{results_dir}/testing_run_{dt_string}'):
        os.mkdir(f'{results_dir}/testing_run_{dt_string}')
    # save files
    rewards_df.to_csv(f'{results_dir}/testing_run_{dt_string}/rewards_df.csv')
    energy_df.to_csv(f'{results_dir}/testing_run_{dt_string}/energy_df.csv')
    market_prices_df.to_csv(f'{results_dir}/testing_run_{dt_string}/market_prices_df.csv')
    pvs_df.to_csv(f'{results_dir}/testing_run_{dt_string}/pvs_df.csv')
    loads_df.to_csv(f'{results_dir}/testing_run_{dt_string}/loads_df.csv')
    socs_df.to_csv(f'{results_dir}/testing_run_{dt_string}/socs_df.csv')
    banks_df.to_csv(f'{results_dir}/testing_run_{dt_string}/banks_df.csv')

    # visualise the trading behaviour for randomly selected timeframe of length n_ts
    visualise_trading_behaviour(energy_df, market_prices_df, pvs_df, loads_df, socs_df, n_ts=24, seed=1234, interval=1, show_plot=True)
    # visualise the performance of the model training
    visualise_performance(rewards_df=rewards_df, socs_df=socs_df, banks_df=banks_df, intervals=intervals, show_plot=True)

    sim_time = (time.time() - start_time)
    if sim_time > 120:
        mins, secs = divmod(sim_time, 60)
        print(f'It took {int(mins)} minutes and {int(secs)} seconds to execute simulation.')
    else:
        print(f'It took {sim_time:.3f} seconds to execute simulation.')