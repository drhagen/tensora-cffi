__all__ = ["compile_evaluate", "tensor_cdefs"]

import re
import tempfile
import threading
from typing import Any
from weakref import WeakKeyDictionary

from cffi import FFI

lock = threading.Lock()

global_weakkeydict = WeakKeyDictionary()

# order: The number of dimensions of the tensor
# dimensions: The size of each dimension of the tensor; has length `order`
# csize: No idea what this is
# mode_ordering: The dimension that each level refers to; has length `order` and
#   has exactly the numbers 0 to `order` - 1; e.g. if `mode_ordering` is (2, 0, 1), then
#   the first level describes the last dimension, the second level describes the first
#   dimension, etc.
# mode_types: The type (dense, compressed sparse) of each level; importantly, the ith
#   element in `mode_types` describes the ith element in `indices`, not `dimensions`
# indices: A complex data structure storing all the index information of the structure;
#   it has length `order`; each element is one level of the data structure; each level
#   conceptually stores one dimension worth of indexes;
#   *   if a level is dense, then the element in indices is a null pointer or a pointer to
#       a length 0 array or a pointer to a length 1 array, which contains a pointer to a
#       length 1 array, which contains the size of this level's dimension. It does not
#       really matter what goes here because it is never used.
#   *   if a level is compressed (sparse), then the element in indices is a pointer to a
#       length 2 array
#       *   the first element is the `pos` arrays of compressed sparse matrix
#           representation; it has a number of elements equal to the number of indexes
#           in the previous level plus 1; each element `i` in `pos` is the starting
#           position in `idx` for indexes in this dimension associated with the `i`th
#           index of the previous dimension; the last element is the total length of
#           `idx` (that is, the first index not in `idx`)
#       *   the second element is the `idx` array of compressed sparse matrix
#           representation; each element is an index that has a value in this dimension
# vals: The actual values of the sparse matrix; has a length equal to the number of indexes
#   in the last dimension; one has to traverse the `indices` structure to determine the
#   coordinate of each value
# vals_size: Deprecated https://github.com/tensor-compiler/taco/issues/208#issuecomment-476314322

taco_define_header = """
    #ifndef TACO_C_HEADERS
    #define TACO_C_HEADERS
    #define TACO_MIN(_a,_b) ((_a) < (_b) ? (_a) : (_b))
    #define TACO_MAX(_a,_b) ((_a) > (_b) ? (_a) : (_b))
    #endif
"""

taco_type_header = """
    typedef enum { taco_mode_dense, taco_mode_sparse } taco_mode_t;

    typedef struct {
      int32_t      order;         // tensor order (number of modes)
      int32_t*     dimensions;    // tensor dimensions
      int32_t      csize;         // component size
      int32_t*     mode_ordering; // mode storage ordering
      taco_mode_t* mode_types;    // mode storage types
      int32_t***   indices;       // tensor index data (per mode)
      double*     vals;          // tensor values
      int32_t      vals_size;     // values array size
    } taco_tensor_t;

    void free(void *ptr);
"""

# Define ffi for building the tensors.
tensor_cdefs = FFI()

# This `FFI` is `include`d in each kernel so objects created with its `new` can be passed to the kernels.
tensor_cdefs.cdef(taco_type_header)

# This library only has definitions, in order to `include` it elsewhere, `set_source` must be called with empty `source` first
tensor_cdefs.set_source("_main", "")

# Open the base C library, which works on Linux and Mac
tensor_lib = tensor_cdefs.dlopen(None)


def compile_evaluate(source: str) -> Any:
    """Compile evaluate kernel in C code using CFFI.

    Args:
        source: C code containing one evaluate function

    Returns:
        The compiled FFILibrary which has a single method `evaluate` which
        expects cffi pointers to taco_tensor_t instances.
    """
    # Extract signature
    # This needs to be provided alone to cdef
    signature_match = re.search(r"int(32_t)? evaluate\(([^)]*)\)", source)
    signature = signature_match.group(0)

    # Use cffi to compile the kernels
    ffibuilder = FFI()
    ffibuilder.include(tensor_cdefs)
    ffibuilder.cdef(signature + ";")
    ffibuilder.set_source(
        "taco_kernel",
        taco_define_header + taco_type_header + source,
        extra_compile_args=["-Wno-unused-variable", "-Wno-unknown-pragmas"],
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        # Lock because FFI.compile is not thread safe: https://foss.heptapod.net/pypy/cffi/-/issues/490
        with lock:
            # Create shared object in temporary directory
            lib_path = ffibuilder.compile(tmpdir=temp_dir)

        # Load the shared object
        lib = ffibuilder.dlopen(lib_path)

    # Return the entire library rather than just the function because it appears that the memory containing the compiled
    # code is freed as soon as the library goes out of scope: https://stackoverflow.com/q/55323592/1485877
    return lib
