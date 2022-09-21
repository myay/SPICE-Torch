from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='custommac2dmappingdistr',
    ext_modules=[
        CUDAExtension('custommac2dmappingdistr', [
            'custommac2dmappingdistr.cpp',
            'custommac2dmappingdistr_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
