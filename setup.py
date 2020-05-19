import setuptools

setuptools.setup(name='pymlogit',
      version='0.0.2',
      description='Estimation of discrete choice models in python',
      url='https://github.com/arteagac/pymlogit',
      author='Cristian Arteaga',
      author_email='cristiandavidarteaga@gmail.com',
      license='MIT',
      packages=['pymlogit'],
      zip_safe=False,
      python_requires='>=3.5',
      install_requires=[
          'numpy>=1.13.1',
          'scipy>=1.0.0'
      ])