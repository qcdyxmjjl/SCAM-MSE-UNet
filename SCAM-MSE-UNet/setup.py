from setuptools import setup, find_packages

setup(
    name="pytorch-unet",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'torch',
        'torchvision',
        'numpy',
        'matplotlib',
        'tqdm',
        'wandb',
    ],
) 