from setuptools import find_packages, setup

try:
    with open("requirements.txt") as simply_req:
        REQUIREMENTS = simply_req.read()

except OSError:
    # Shouldn't happen
    REQUIREMENTS = []


with open("README.md", "r", encoding="utf-8") as readme:
    README = readme.read()

VERSION = 0.1

setup(
    name="simply",
    description="Simulation of Electricity Markets in Python",
    long_description=README,
    url="https://github.com/BESTenergytrade/simply.git",
    version=VERSION,
    packages=find_packages(where=".", exclude=["tests"]),
    package_dir={"simply": "simply"},
    package_data={},
    install_requires=REQUIREMENTS,
    zip_safe=False,
)
