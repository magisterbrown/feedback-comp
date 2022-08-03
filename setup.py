from setuptools import setup, find_packages

f = open("requirements.txt", "r")
deps = f.readlines()
global_pack = list(filter(lambda x:not ('file' in x), deps))
print(global_pack)

f.close()

setup(name='feedback',
      version='1.0',
      description='',
      author='magisterbrownie',
      author_email='magisterbrownie@gmail.com',
      url='',
      packages=find_packages(),
      install_requires=global_pack
     )
