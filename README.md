# Tensora backend using CFFI

Starting in Tensora 0.4, [Tensora](https://tensora.drhagen.com/) uses LLVM to compile tensora kernels.
This Python package provides an alternative way to compile the generated kernels using CFFI.
It can be installed with the `tensora[cffi]` extra, and used via the `backend=BackendCompiler.cffi` argument to `TensorMethod`.
This is the only backend that can be used with the `compiler=TensorCompiler.taco` argument to `TensorMethod`.
This package is unlikely to be useful outside of that context.
