import argparse
import os

from simply.scenario_helper import main

"""
This script creates a simulation scenario from a config JSON file, network JSON file,
configuration text file, a loads assignment CSV file, and a data directory.
"""


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Entry point for market simulation')
    parser.add_argument('project_dir', help='project directory path')
    parser.add_argument('--data_dir', default='', help='data directory')
    args = parser.parse_args()

    if args.project_dir is None:
        raise FileNotFoundError(
            "Project directory path must be specified. Please provide the path as a command-line "
            "argument.")
    data_dir = args.data_dir if args.data_dir is not None else os.path.join(args.project_dir,
                                                                            "scenario_inputs")
    if args.data_dir is None:
        print(f"Using data directory: {data_dir}")

    # Call the main function with the specified scenario directory and data directory
    main(args.project_dir, args.data_dir)
