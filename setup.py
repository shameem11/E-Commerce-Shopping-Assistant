from setuptools import setup, find_packages
from typing import List


def get_requirements() -> List[str]:

    requirement_list : list[str] = []

    return requirement_list


setup(
    name='E-commerce',
    author='Your Name',
    author_email='shameemmon.mk@gmail.com',
    version='0.0.1',
    packages=find_packages(),
    install_requires=get_requirements()
)
