import setuptools
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# with open("README.md", "r") as fh:
#     long_description = fh.read()

setuptools.setup(
    name='mappingdistr',
    version="0.0.1",
    author='Mikail Yayla',
    author_email='mikail.yayla@tu-dortmund.de',
    description='Mapping based on distribution',
    # long_description=long_description,
    # long_description_content_type="text/markdown",
    # url="",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Operating System :: OS Independent",
    ],
    ext_modules=[
        CUDAExtension('mappingdistr', [
            'mappingdistr.cpp',
            'mappingdistr_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
