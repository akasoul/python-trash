from setuptools import setup, find_packages

setup(name='trainer',
      version='0.23',
      packages=find_packages(),
      description='example to run keras on gcloud ml-engine',
      author='Voloshuk Anton',
      install_requires=[
          'keras',
          'h5py',
          'numpy',
          'sklearn',
          'argparse'
      ],
      zip_safe=False)