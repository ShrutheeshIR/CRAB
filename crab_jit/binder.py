"""The one Python<->native binding mechanism in this codebase: every JIT'd C++
function -- whether auto-generated (Path A, crab_jit.simple) or hand-written/fused
(Path B, e.g. crab_codegen.collision_kernel) -- exports `void fn(void** inputs, void**
outputs)` and gets bound through JitFunction, the same way. See REDESIGN.md.

Python-side inputs may be a list, tuple, numpy array, or a bare scalar; each is
flattened into its own flat double buffer and boxed into the `inputs` void* array
positionally. Outputs always come back as np.ndarray -- a single array if there's one
output, a tuple of arrays if there's more than one. Eigen only ever appears inside the
C++ on the other side of this boundary -- Python never needs to know it exists.
"""

import ctypes
from typing import Sequence, Union

import numpy as np

from .engine import CrabJitEngine

ArrayLike = Union[Sequence[float], np.ndarray, float, int]


def _flatten_input(value: ArrayLike, expected_size: int) -> ctypes.Array:
    arr = np.atleast_1d(np.asarray(value, dtype=np.float64)).ravel()
    if arr.size != expected_size:
        raise ValueError(f"expected input of size {expected_size}, got {arr.size}")
    return (ctypes.c_double * expected_size)(*arr)


class JitFunction:
    """Callable wrapper around one JIT'd `void fn(void** inputs, void** outputs)`
    function with a fixed, known number/size of flat double inputs and outputs."""

    def __init__(self, engine: CrabJitEngine, wrapper_name: str,
                 input_sizes: Sequence[int], output_sizes: Sequence[int]):
        self._engine = engine
        self._input_sizes = list(input_sizes)
        self._output_sizes = list(output_sizes)
        fn_t = ctypes.CFUNCTYPE(None, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p))
        self._fn = fn_t(engine.lookup(wrapper_name))

    def __call__(self, *args: ArrayLike):
        if len(args) != len(self._input_sizes):
            raise ValueError(f"expected {len(self._input_sizes)} argument(s), got {len(args)}")

        input_bufs = [_flatten_input(arg, size) for arg, size in zip(args, self._input_sizes)]
        output_bufs = [(ctypes.c_double * size)() for size in self._output_sizes]

        inputs = (ctypes.c_void_p * len(input_bufs))(
            *(ctypes.cast(buf, ctypes.c_void_p) for buf in input_bufs)
        )
        outputs = (ctypes.c_void_p * len(output_bufs))(
            *(ctypes.cast(buf, ctypes.c_void_p) for buf in output_bufs)
        )

        self._fn(inputs, outputs)

        results = tuple(np.array(buf, dtype=np.float64) for buf in output_bufs)
        return results[0] if len(results) == 1 else results


def bind_jit_function(engine: CrabJitEngine, wrapper_name: str,
                       input_sizes: Sequence[int], output_sizes: Sequence[int]) -> JitFunction:
    return JitFunction(engine, wrapper_name, input_sizes, output_sizes)
