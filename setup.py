from setuptools import setup, find_packages
setup(
    name='easyopt',
    version='0.0.0',
    description='easy trajectory optimization framework',
    license=license,
    install_requires=[
        'numpy',
        'scipy',
        'tinyfk>=0.2.3'
        ],
    packages=find_packages(exclude=('tests', 'docs'))
)
