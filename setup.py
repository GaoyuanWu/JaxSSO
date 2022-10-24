import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="JaxSSO",
    version="0.0.1",
    author="Gaoyuan Wu",
    author_email="gaoyuanw@princeton.edu",
    description="A framework for structural shape optimization based on automatic differentiation (AD) and the adjoint method, enabled by JAX",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/GaoyuanWu/JaxSSO",
    packages=setuptools.find_packages(include=['JaxSSO']),
    keywords=["jax", "automatic-differentiation", "shape optimization", "form-finding", "structural optimization"],
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    install_requires=[
        'numpy',
        'matplotlib',
        'jax'
    ],
    extras_require = {
        'FEA Solver': ['PyNiteFEA']
    },
    include_package_data = True,
    python_requires=">=3.7",
)
