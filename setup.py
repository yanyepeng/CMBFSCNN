# -*- coding: utf-8 -*-



import os
from setuptools import setup, find_packages

VERSION = '0.0.1'
def find_data_files(directory):
    """
    Using glob patterns in ``package_data`` that matches a directory can
    result in setuptools trying to install that directory as a file and
    the installation to fail.

    This function walks over the contents of *directory* and returns a list
    of only filenames found.
    """

    strip = os.path.dirname(os.path.abspath(__file__))

    result = []
    for root, dirs, files in os.walk(directory):
        for filename in files:
          filename = os.path.join(root, filename)
          result.append(os.path.relpath(filename, strip))

    print("\n".join(result))
    return result
#print('+++++++', find_data_files('pysm/template/'))
setup(name='cmbfscnn',
      version=VERSION,
      description="a component separation of CMB using CNN ",
      long_description='a python command tool for camel case',
      author='Ye-Peng Yan',
      author_email='yanyepengphy@163.com',
      license='MIT',
      packages=['cmbfscnn', 'pysm', 'examples','Utils_ILC','pysm/template','pysm/test','Utils_ILC/ILC'],
      include_package_data=True,
      zip_safe=False,
      python_requires = '>=3.8',)
      #data_files=[('pysm',find_data_files('pysm/template/')), ('Utils_ILC/ILC',find_data_files('Utils_ILC/ILC/needlet_data'))]
      #)



