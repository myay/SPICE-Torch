from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='custommac1dmappingdirect',
    ext_modules=[
        CUDAExtension('custommac1dmappingdirect', [
            'custommac1dmappingdirect.cpp',
            'custommac1dmappingdirect_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
