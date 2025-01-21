from setuptools import setup, find_packages

setup(
    name="dinov2_mod",
    version="0.1",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "torch>=1.9.0",
        "torchvision>=0.10.0",
        "scikit-learn>=0.24.0",
        "numpy",
        "Pillow",
        "tqdm",
        "joblib",
        "matplotlib",
    ],
    author="Your Name",
    description="A custom pipeline for feature extraction and prediction using DINOv2.",
    url="https://github.com/yourusername/dinov2_mod",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
