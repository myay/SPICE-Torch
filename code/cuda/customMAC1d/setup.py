from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='custommac1d',
    ext_modules=[
        CUDAExtension('custommac1d', [
            'custommac1d.cpp',
            'custommac1d_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
