============================= test session starts ==============================
platform linux -- Python 3.8.10, pytest-7.4.0, pluggy-1.2.0
rootdir: /home/friederike/Code/simply-stuff/simply/simply
collected 3 items

tests/test_main.py FFF                                                   [100%]

=================================== FAILURES ===================================
______________________________ TestMain.test_main ______________________________

self = <tests.test_main.TestMain object at 0x7f4206ebe7f0>
tmp_path = PosixPath('/tmp/pytest-of-friederike/pytest-21/test_main0')

    def test_main(self, tmp_path):
        cfg = Config("", "")
        cfg.path = Path((tmp_path / "output"))
        current_path = Path.cwd()
        print("Hier bin ich:", current_path)
>       main(cfg)

tests/test_main.py:13: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

cfg = <simply.config.Config object at 0x7f420696c9d0>

    def main(cfg: Config):
    
        print(cfg)
        # Checks if actor files with the correct format exist in the cfg.scenario_path
    
        current_path = Path.cwd()
        print("Hier bin ich, jetzt:", current_path)
    
    
        # --------------------------------
        #def list_files_in_path(current_path, pattern='*'):
        #    file_list = glob.glob(os.path.join(path, pattern))
        #    return file_list
    
        # Replace 'your_path_here' with the path you want to list files from
        path_to_list = current_path
>       files_in_path = list_files_in_path(path_to_list)
E       NameError: name 'list_files_in_path' is not defined

match_market.py:39: NameError
----------------------------- Captured stdout call -----------------------------
Hier bin ich: /home/friederike/Code/simply-stuff/simply/simply
<simply.config.Config object at 0x7f420696c9d0>
Hier bin ich, jetzt: /home/friederike/Code/simply-stuff/simply/simply
_______________________ TestMain.test_load_scenario_csv ________________________

self = <tests.test_main.TestMain object at 0x7f420696c2e0>
tmp_path = PosixPath('/tmp/pytest-of-friederike/pytest-21/test_load_scenario_csv0')

    def test_load_scenario_csv(self, tmp_path):
        cfg = Config("","")
        # cfg.save_csv = True is the default value. Therefore, we don't set it
        cfg.data_format = "csv"
        cfg.path = Path((tmp_path / "output"))
>       main(cfg)

tests/test_main.py:20: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

cfg = <simply.config.Config object at 0x7f420691e8e0>

    def main(cfg: Config):
    
        print(cfg)
        # Checks if actor files with the correct format exist in the cfg.scenario_path
    
        current_path = Path.cwd()
        print("Hier bin ich, jetzt:", current_path)
    
    
        # --------------------------------
        #def list_files_in_path(current_path, pattern='*'):
        #    file_list = glob.glob(os.path.join(path, pattern))
        #    return file_list
    
        # Replace 'your_path_here' with the path you want to list files from
        path_to_list = current_path
>       files_in_path = list_files_in_path(path_to_list)
E       NameError: name 'list_files_in_path' is not defined

match_market.py:39: NameError
----------------------------- Captured stdout call -----------------------------
<simply.config.Config object at 0x7f420691e8e0>
Hier bin ich, jetzt: /home/friederike/Code/simply-stuff/simply/simply
_______________________ TestMain.test_load_scenario_json _______________________

self = <tests.test_main.TestMain object at 0x7f420696c430>
tmp_path = PosixPath('/tmp/pytest-of-friederike/pytest-21/test_load_scenario_json0')

    def test_load_scenario_json(self, tmp_path):
        cfg = Config("","")
        cfg.save_csv = True
        cfg.data_format = "json"
        cfg.path = Path((tmp_path / "output"))
>       main(cfg)

tests/test_main.py:30: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

cfg = <simply.config.Config object at 0x7f420696cb20>

    def main(cfg: Config):
    
        print(cfg)
        # Checks if actor files with the correct format exist in the cfg.scenario_path
    
        current_path = Path.cwd()
        print("Hier bin ich, jetzt:", current_path)
    
    
        # --------------------------------
        #def list_files_in_path(current_path, pattern='*'):
        #    file_list = glob.glob(os.path.join(path, pattern))
        #    return file_list
    
        # Replace 'your_path_here' with the path you want to list files from
        path_to_list = current_path
>       files_in_path = list_files_in_path(path_to_list)
E       NameError: name 'list_files_in_path' is not defined

match_market.py:39: NameError
----------------------------- Captured stdout call -----------------------------
<simply.config.Config object at 0x7f420696cb20>
Hier bin ich, jetzt: /home/friederike/Code/simply-stuff/simply/simply
=============================== warnings summary ===============================
simply/config.py:53
tests/test_main.py::TestMain::test_main
tests/test_main.py::TestMain::test_load_scenario_csv
tests/test_main.py::TestMain::test_load_scenario_json
  /home/friederike/Code/simply-stuff/simply/simply/simply/config.py:53: UserWarning: No Configuration file path was provided. Default values will be used.
    warnings.warn("No Configuration file path was provided. Default values will be used.")

simply/config.py:59
tests/test_main.py::TestMain::test_main
tests/test_main.py::TestMain::test_load_scenario_csv
tests/test_main.py::TestMain::test_load_scenario_json
  /home/friederike/Code/simply-stuff/simply/simply/simply/config.py:59: UserWarning: No project_dir was provided. Default project_dir ./projects/example_projects/example_project is used
    warnings.warn("No project_dir was provided. Default project_dir ./projects/example_projects/example_project is used")

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
=========================== short test summary info ============================
FAILED tests/test_main.py::TestMain::test_main - NameError: name 'list_files_...
FAILED tests/test_main.py::TestMain::test_load_scenario_csv - NameError: name...
FAILED tests/test_main.py::TestMain::test_load_scenario_json - NameError: nam...
======================== 3 failed, 8 warnings in 0.85s =========================
