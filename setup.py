from setuptools import setup, find_packages

classifiers = [
    'Development Status :: ', 'Intended Audience :: Education/Research',
    'Operating System :: Microsoft :: Windows :: Windows 10 :: Linux :: ',
    'License :: MIT License',
    'Programming Language :: Python :: 3'
    ]

setup(
    name='MLTSA',
    version='0.0.1',
    description='Machine Learning Transition State Analysis',
    long_description=open('README.txt').read() + '\n\n' + open('CHANGELOG.txt').read(),
    url='',
    author='Pedro Buigues',
    author_email='pedrojuanbj@gmail.com',
    license="MIT",
    classifiers=classifiers,
    keywords="MLTSA",
    packages=find_packages(),
    install_requires=[""]
)
