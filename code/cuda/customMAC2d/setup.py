from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='custommac2d',
    ext_modules=[
        CUDAExtension('custommac2d', [
            'custommac2d.cpp',
            'custommac2d_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
