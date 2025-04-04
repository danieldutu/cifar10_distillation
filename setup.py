from setuptools import setup, find_packages

setup(
    name="distil_pytorch",
    version="0.1.0",
    description="Knowledge Distillation with PyTorch on CIFAR-10",
    author="Daniel",
    packages=find_packages(),
    install_requires=[
        "torch>=1.9.0",
        "torchvision>=0.10.0",
        "matplotlib>=3.4.0",
        "numpy>=1.20.0",
        "tabulate>=0.8.9",
        "jupyter>=1.0.0",
        "jupytext>=1.13.0",
        "ipykernel>=6.0.0",
        "tqdm>=4.62.0",  # For progress bars
    ],
    python_requires=">=3.9",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
