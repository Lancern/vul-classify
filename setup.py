from setuptools import setup, find_packages


# packages = find_packages(exclude=('tests',)) + find_packages('./asm2vec', exclude=('tests', 'examples'))
# setup(name='vulcls', version='1.0', packages=packages)

setup(name='vulcls', version='1.0', packages=find_packages(exclude=('tests',)))
