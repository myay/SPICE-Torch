import setuptools
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# with open("README.md", "r") as fh:
#     long_description = fh.read()

setuptools.setup(
    name='binarizePM1',
    version="0.0.1",
    author='Mikail Yayla',
    author_email='mikail.yayla@tu-dortmund.de',
    description='Binarization to PM1 For PyTorch Tensors',
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
        CUDAExtension('binarizePM1', [
            'binarizePM1.cpp',
            'binarizePM1_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
