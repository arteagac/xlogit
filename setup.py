import setuptools
import codecs

with codecs.open('README.rst', encoding='utf8') as fh:
    long_description = fh.read()

setuptools.setup(name='xlogit',
                 version='0.2.2',
                 description='A Python package for GPU-accelerated ' +
                 'estimation of mixed logit models.',
                 long_description=long_description,
                 long_description_content_type="text/x-rst",
                 url='https://github.com/arteagac/xlogit',
                 author='Cristian Arteaga',
                 author_email='cristiandavidarteaga@gmail.com',
                 license='MIT',
                 packages=['xlogit'],
                 zip_safe=False,
                 python_requires='>=3.5',
                 install_requires=[
                     'numpy>=1.13.1',
                     'scipy>=1.0.0'
                     ])
