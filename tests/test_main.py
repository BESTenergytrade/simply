import subprocess
from argparse import ArgumentParser
from simply.config import Config
from main import main
import sys


def test_main(tmp_path):
    parser = ArgumentParser(description='Entry point for market simulation')
    parser.add_argument('config', nargs='?', default="", help='configuration file')
    args = parser.parse_args()

    cfg = Config(args.config)
    cfg_path = (tmp_path / "config.cfg")
    with open(cfg_path, "w") as f:
        for key, value in cfg.__dict__.items():
            if key == "path":
                f.write("path = " + str(tmp_path))
            else:
                f.write(key + " = " + str(value) + "\n")

    sys.argv = ["foo", str(cfg_path).replace("\\", "/")]
    main()
