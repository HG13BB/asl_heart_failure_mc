from setuptools import find_packages
from setuptools import setup

setup(
    name='heart_failure_simple_dnn_trainer',
    version='0.1',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'fire',
    ],
    description='Heart failure simple dnn model training application.'
)
