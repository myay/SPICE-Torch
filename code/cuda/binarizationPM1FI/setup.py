import setuptools
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setuptools.setup(
    name='binarizePM1FI',
    version="0.0.1",
    author='Mikail Yayla',
    author_email='mikail.yayla@tu-dortmund.de',
    description='Fault Injection for binarizationPM1 package',
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
        CUDAExtension('binarizePM1FI', [
            'binarizePM1FI.cpp',
            'binarizePM1FI_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
