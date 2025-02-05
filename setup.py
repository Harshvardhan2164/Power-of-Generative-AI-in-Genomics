from setuptools import find_packages, setup
from typing import List

HYPHEN_e_DOT = '-e .'

def get_requirements(file_path:str) -> List[str]:
    requirements = []
    with open(file_path) as file:
        requirements = file.readlines()
        requirements = [req.replace("\n", "") for req in requirements]

        if HYPHEN_e_DOT in requirements:
            requirements.remove(HYPHEN_e_DOT)

    return requirements

setup(
    name="Power of Generative AI in Genomics",
    version='0.0.1',
    author='Avani Gajallewar, Harshvardhan Sharma, Shantanu Gupta',
    packages=find_packages(),
    install_packages=get_requirements('requirements.txt')
)