from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='custommac1dmappingdistr',
    ext_modules=[
        CUDAExtension('custommac1dmappingdistr', [
            'custommac1dmappingdistr.cpp',
            'custommac1dmappingdistr_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
