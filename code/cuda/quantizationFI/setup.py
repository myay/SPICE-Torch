import setuptools
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# with open("README.md", "r") as fh:
#     long_description = fh.read()

setuptools.setup(
    name='quantizeFI',
    version="0.0.1",
    author='Mikail Yayla',
    author_email='mikail.yayla@tu-dortmund.de',
    description='Bit Flip Injection into Quantized Tensors For PyTorch',
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
        CUDAExtension('quantizationFI', [
            'quantizeFI.cpp',
            'quantizeFI_kernels.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
