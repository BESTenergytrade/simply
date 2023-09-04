from pathlib import Path

def main(project_dir, data_dir):
    print('project_dir:',project_dir)
    # File path for the matches csv
    project_dir = Path(project_dir)
    matches_fp = project_dir / 'market_results' / 'matches.csv'
    print(f'matches_fp: {matches_fp}')
