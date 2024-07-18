import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import rc
from matplotlib.dates import DateFormatter, AutoDateLocator
from pathlib import Path
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from matplotlib.lines import Line2D

# CONFIGURATION
scenarios = ["2021-2023", "2035_noChange", "2035_dynPrice", "2035_AU", "2035_nodal", "2035_AU_pricing", "2035_AU_pricing0-001", "2035_dynPrice_2023", "2035_dynPrice_strat4"]
selected_scenario = scenarios[3]
default_path = "projects/example_projects/" + selected_scenario
input_folder = Path(default_path) / "market_results/"
output_folder = ""

# Set the household ID
household_id = 2277736

# Font configuration
font = {'weight': 'bold', 'size': 12}
rc('font', **font)

def aggregate_data(input_folder):
    aggregated_data = pd.DataFrame()
    for file_path in input_folder.glob("actor*.csv"):
        df = pd.read_csv(file_path)
        if 'ev_soe' not in df.columns:
            df['ev_soe'] = 0
        if aggregated_data.empty:
            aggregated_data = df[['schedule', 'bat_soe', 'ev_soe', 'traded_energy']].copy()
        else:
            aggregated_data = aggregated_data.add(df[['schedule', 'bat_soe', 'ev_soe', 'traded_energy']], fill_value=0)
    return aggregated_data

def format_timestamps(index):
    """Format timestamps to 'Day X-HH:MM' using actual date and time."""
    days = (index - index[0]).days + 1  # Day starts from 1
    return [f"Day {day}-{ts.strftime('%H:%M')}" for day, ts in zip(days, index)]

def plot_actor(data, hh_id, output_folder, selected_scenario, data_sc=None):
    ev_soe = data.get('ev_soe')
    rli_colors = {
        'dblue': (0, 46 / 255, 80 / 255),
        'lblue': (34 / 255, 116 / 255, 165 / 255),
        'dgrey': (51 / 255, 88 / 255, 115 / 255),
        'green': (68 / 255, 175 / 255, 105 / 255),
        'lgrey': (51*2 / 255, 88*2 / 255, 115*2 / 255),
        'red': 'red'  # Add 'red' color
    }

    fig, (ax2, ax) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    
    if 'ev_demand' not in data.columns:
        data['ev_demand'] = 0

    lns1_0 = ax.step(data.index, data['traded_energy'], '-.', color="orange", alpha=1.0, where='post', label='Traded Energy')
    lns1_1 = ax.step(data.index, data[['schedule', 'ev_demand']].sum(axis=1), '-', color=rli_colors['lblue'], alpha=1.0, where='post', label='Scheduled Energy', linewidth=2)
    ax.fill_between(x=data.index, y1=0, y2=data['bat_soe'].diff(), edgecolor=rli_colors['green'], facecolor="lightgreen", alpha=0.5, interpolate=False, step='post', label='Battery Charge')

    if ev_soe is not None:
        ax.fill_between(x=data.index, y1=data['bat_soe'].diff(), y2=data['bat_soe'].diff() + data['ev_soe'].diff(), edgecolor=rli_colors['dblue'], facecolor="darkgreen", interpolate=False, step='post', label='EV Charge')

    lns1_2_proxy = [Line2D([0], [0], linestyle='-', color='lightgreen', linewidth=10, label='Battery Charge')]
    lns1_3_proxy = [Line2D([0], [0], linestyle='-', color='darkgreen', linewidth=10, label='EV Charge')] if ev_soe is not None else []
    lns = lns1_0 + lns1_1 + lns1_2_proxy + lns1_3_proxy
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc='upper right')  # Legend konumunu sabitledik

    lns2_1 = ax2.step(data.index, data['mm_buy_prices'], '--', color=rli_colors['green'], where='post', label='Guaranteed Selling Price')
    lns2_2 = ax2.step(data.index, data['mm_sell_prices'], '.--', color=rli_colors['red'], where='post', label='Guaranteed Buying Price')
    data["trade_cost"] = data['bank'].diff()
    data["trade_price"] = - data["trade_cost"] / data['traded_energy']

    lns = lns2_1 + lns2_2
    if selected_scenario == "2035_AU":
        freq = pd.infer_freq(data.index)
        if freq is not None:
            freq_unit = freq[-1]
            freq_value = freq[:-1] if freq[:-1] is None else 0
            time_shift = pd.to_timedelta(freq_value, unit=freq_unit)
            lns2_4 = ax2.step(data.index + time_shift / 2, data["trade_price"], ' ', color="orange", marker="*", markersize=10, label='Trading Price')
        else:
            lns2_4 = ax2.step(data.index, data["trade_price"], ' ', color="orange", marker="*", markersize=10, label='Trading Price')
        lns += lns2_4

    labs = [l.get_label() for l in lns]
    ax2.legend(lns, labs, loc='upper right')  # Legend konumunu sabitledik

    ax2.set_title(f"Actor {hh_id}")
    ax2.set_ylabel(r"EUR/kWh")
    ax.set_ylabel(r"kWh")
    ax.set_xlabel("Timestep")

    # Custom x-axis ticks
    formatted_labels = format_timestamps(data.index)
    ax.set_xticks(data.index[::len(data.index)//10])  # Adjust the number of ticks as needed
    ax.set_xticklabels(formatted_labels[::len(data.index)//10], rotation=45, ha="right")
    
    plt.subplots_adjust(wspace=0, hspace=0.02)
    
    if output_folder:
        plt.savefig(output_folder + f'{selected_scenario}_order_and_price_actor_building_{hh_id}.svg', bbox_inches='tight')
    plt.show()

def plot_timeseries_price_and_orders(input_folder, hh_id, output_folder):
    Tk().withdraw()
    filename = askopenfilename(title="Which file should be plotted?")
    input_file = Path(filename) if filename else input_folder / f'actor_building_{str(hh_id)}.csv'
    data = pd.read_csv(input_file, index_col=0, parse_dates=True)
    plot_actor(data, hh_id, output_folder, selected_scenario)

if __name__ == "__main__":
    plot_timeseries_price_and_orders(input_folder, household_id, output_folder)
