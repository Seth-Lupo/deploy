"""
TensorRT Runtime Utilities
Provides engine loading and inference for TensorRT models
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# TensorRT imports - available in TRT container
try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False
    logger.warning("TensorRT not available - running in CPU fallback mode")


@dataclass
class TRTBinding:
    """Represents a TensorRT engine binding"""
    name: str
    dtype: np.dtype
    shape: Tuple[int, ...]
    is_input: bool
    size: int  # Size in bytes


class TRTEngine:
    """
    TensorRT Engine wrapper for inference
    Handles dynamic shapes and manages GPU memory
    """

    def __init__(self, engine_path: str):
        if not TRT_AVAILABLE:
            raise RuntimeError("TensorRT not available")

        self.engine_path = engine_path
        self.engine = None
        self.context = None
        self.bindings: Dict[str, TRTBinding] = {}
        self.input_names: List[str] = []
        self.output_names: List[str] = []
        self._device_buffers: Dict[str, cuda.DeviceAllocation] = {}
        self._host_buffers: Dict[str, np.ndarray] = {}
        self.stream = None

    def load(self):
        """Load TensorRT engine from file"""
        logger.info(f"Loading TensorRT engine: {self.engine_path}")

        trt_logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(trt_logger)

        with open(self.engine_path, 'rb') as f:
            engine_data = f.read()

        self.engine = runtime.deserialize_cuda_engine(engine_data)
        if self.engine is None:
            raise RuntimeError(f"Failed to load engine: {self.engine_path}")

        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()

        # Parse bindings
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            shape = self.engine.get_tensor_shape(name)
            is_input = self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT

            # Handle dynamic shapes (-1 values)
            static_shape = tuple(max(1, s) for s in shape)
            size = int(np.prod(static_shape) * np.dtype(dtype).itemsize)

            binding = TRTBinding(
                name=name,
                dtype=dtype,
                shape=tuple(shape),
                is_input=is_input,
                size=size
            )
            self.bindings[name] = binding

            if is_input:
                self.input_names.append(name)
            else:
                self.output_names.append(name)

        logger.info(f"Engine loaded - Inputs: {self.input_names}, Outputs: {self.output_names}")

    def _allocate_buffers(self, input_shapes: Dict[str, Tuple[int, ...]]):
        """Allocate GPU buffers for given input shapes"""
        # Set input shapes and compute output shapes
        for name, shape in input_shapes.items():
            if name in self.bindings:
                self.context.set_input_shape(name, shape)

        # Allocate buffers
        for name, binding in self.bindings.items():
            if binding.is_input:
                shape = input_shapes.get(name, binding.shape)
            else:
                shape = self.context.get_tensor_shape(name)
                # Convert TRT Dims to tuple if needed
                if hasattr(shape, '__iter__') and not isinstance(shape, tuple):
                    shape = tuple(shape)

            size = int(np.prod(shape) * np.dtype(binding.dtype).itemsize)

            # Reallocate if size changed
            if name not in self._device_buffers or self._host_buffers[name].nbytes < size:
                if name in self._device_buffers:
                    self._device_buffers[name].free()

                self._host_buffers[name] = np.zeros(shape, dtype=binding.dtype)
                self._device_buffers[name] = cuda.mem_alloc(size)
            else:
                # Reshape existing buffer
                self._host_buffers[name] = self._host_buffers[name].ravel()[:int(np.prod(shape))].reshape(shape)

    def infer(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Run inference with given inputs

        Args:
            inputs: Dictionary mapping input names to numpy arrays

        Returns:
            Dictionary mapping output names to numpy arrays
        """
        # Get input shapes
        input_shapes = {name: arr.shape for name, arr in inputs.items()}

        # Allocate buffers
        self._allocate_buffers(input_shapes)

        # Copy inputs to device
        for name, arr in inputs.items():
            arr_contiguous = np.ascontiguousarray(arr, dtype=self.bindings[name].dtype)
            cuda.memcpy_htod_async(
                self._device_buffers[name],
                arr_contiguous,
                self.stream
            )
            self.context.set_tensor_address(name, int(self._device_buffers[name]))

        # Set output addresses
        for name in self.output_names:
            self.context.set_tensor_address(name, int(self._device_buffers[name]))

        # Execute
        self.context.execute_async_v3(self.stream.handle)

        # Copy outputs to host
        outputs = {}
        for name in self.output_names:
            shape = self.context.get_tensor_shape(name)
            # Convert TRT Dims to tuple if needed
            if hasattr(shape, '__iter__') and not isinstance(shape, tuple):
                shape = tuple(shape)
            host_buffer = np.zeros(shape, dtype=self.bindings[name].dtype)
            cuda.memcpy_dtoh_async(host_buffer, self._device_buffers[name], self.stream)
            outputs[name] = host_buffer

        self.stream.synchronize()
        return outputs

    def warmup(self, input_shapes: Optional[Dict[str, Tuple[int, ...]]] = None):
        """Warmup the engine with dummy inputs"""
        if input_shapes is None:
            # Use default shapes from bindings
            input_shapes = {}
            for name in self.input_names:
                shape = self.bindings[name].shape
                # Replace -1 with reasonable defaults
                shape = tuple(32 if s == -1 else s for s in shape)
                input_shapes[name] = shape

        dummy_inputs = {
            name: np.zeros(shape, dtype=self.bindings[name].dtype)
            for name, shape in input_shapes.items()
        }

        # Run a few warmup iterations
        for _ in range(3):
            self.infer(dummy_inputs)

        logger.info(f"Engine warmup complete: {self.engine_path}")

    def shutdown(self):
        """Release GPU resources"""
        for buffer in self._device_buffers.values():
            buffer.free()
        self._device_buffers.clear()
        self._host_buffers.clear()

        if self.stream:
            self.stream = None
        if self.context:
            self.context = None
        if self.engine:
            self.engine = None


class TRTEngineManager:
    """
    Manages multiple TensorRT engines
    Provides centralized loading and inference
    """

    def __init__(self, model_base_path: str = "/workspace/models"):
        self.model_base_path = model_base_path
        self.engines: Dict[str, TRTEngine] = {}

    def load_engine(self, name: str, engine_path: str) -> TRTEngine:
        """Load and register an engine"""
        if name in self.engines:
            return self.engines[name]

        engine = TRTEngine(engine_path)
        engine.load()
        self.engines[name] = engine
        return engine

    def get_engine(self, name: str) -> Optional[TRTEngine]:
        """Get a loaded engine by name"""
        return self.engines.get(name)

    def warmup_all(self):
        """Warmup all loaded engines"""
        for name, engine in self.engines.items():
            logger.info(f"Warming up engine: {name}")
            engine.warmup()

    def shutdown(self):
        """Shutdown all engines"""
        for engine in self.engines.values():
            engine.shutdown()
        self.engines.clear()


# Singleton manager instance
_engine_manager: Optional[TRTEngineManager] = None

def get_engine_manager(model_base_path: str = "/workspace/models") -> TRTEngineManager:
    """Get or create the global engine manager"""
    global _engine_manager
    if _engine_manager is None:
        _engine_manager = TRTEngineManager(model_base_path)
    return _engine_manager
