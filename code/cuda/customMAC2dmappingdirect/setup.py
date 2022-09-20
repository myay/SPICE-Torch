from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='custommac2dmappingdirect',
    ext_modules=[
        CUDAExtension('custommac2dmappingdirect', [
            'custommac2dmappingdirect.cpp',
            'custommac2dmappingdirect_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
