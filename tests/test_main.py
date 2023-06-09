from simply.config import Config
from main import main
import sys


def test_main(tmp_path):
    cfg = Config("")
    cfg_path = (tmp_path / "config.cfg")
    with open(cfg_path, "w") as f:
        for key, value in cfg.__dict__.items():
            if key == "path":
                f.write("path = " + str(tmp_path/"output"))
            else:
                f.write(key + " = " + str(value))
            f.write("\n")

    sys.argv = ["foo", str(cfg_path).replace("\\", "/")]
    main()
