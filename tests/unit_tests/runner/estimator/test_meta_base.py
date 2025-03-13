import pytest
import sys
import os
from unittest.mock import patch, MagicMock

# Import the module under test
from flagscale.runner.estimator.meta_base import (
    ShardedDim,
    MetaTensor,
    ModelStatsRegistry,
    MetaModule,
    register_model,
    get_registry
)


class TestShardedDim:
    """Test suite for ShardedDim class."""

    def test_init(self):
        """Test initialization with different values."""
        # Basic initialization
        sdim = ShardedDim(10)
        assert sdim.dim == 10
        assert sdim.shard == 1

        # With sharding
        sdim = ShardedDim(10, 2)
        assert sdim.dim == 10
        assert sdim.shard == 2

        # Negative sharding should default to 1
        sdim = ShardedDim(10, -1)
        assert sdim.dim == 10
        assert sdim.shard == 1

    def test_copy(self):
        """Test copy method."""
        sdim = ShardedDim(10, 2)
        copy = sdim.copy()
        assert copy.dim == 10
        assert copy.shard == 2
        assert copy is not sdim

    def test_string_representation(self):
        """Test string representation."""
        # Without sharding
        sdim = ShardedDim(10)
        assert str(sdim) == "ShardedDim(10, 1)"
        assert repr(sdim) == "ShardedDim(10, 1)"

        # With sharding
        sdim = ShardedDim(10, 2)
        assert str(sdim) == "ShardedDim(10, 2)"
        assert repr(sdim) == "ShardedDim(10, 2)"

    def test_add(self):
        """Test addition operation."""
        sdim1 = ShardedDim(10, 2)
        sdim2 = ShardedDim(20, 2)
        result = sdim1 + sdim2
        assert result.dim == 30
        assert result.shard == 2

        # Different shards should raise error
        sdim3 = ShardedDim(20, 5)
        with pytest.raises(ValueError):
            sdim1 + sdim3

        # Non-ShardedDim should raise error
        with pytest.raises(TypeError):
            sdim1 + 10

    def test_radd(self):
        """Test reverse addition operation."""
        sdim = ShardedDim(10, 2)
        with pytest.raises(TypeError):
            10 + sdim

    def test_sub(self):
        """Test subtraction operation."""
        sdim1 = ShardedDim(20, 2)
        sdim2 = ShardedDim(10, 2)
        result = sdim1 - sdim2
        assert result.dim == 10
        assert result.shard == 2

        # Different shards should raise error
        sdim3 = ShardedDim(10, 5)
        with pytest.raises(ValueError):
            sdim1 - sdim3

        # Non-ShardedDim should raise error
        with pytest.raises(TypeError):
            sdim1 - 10

    def test_mul(self):
        """Test multiplication operation."""
        sdim1 = ShardedDim(10, 2)
        sdim2 = ShardedDim(20, 2)
        result = sdim1 * sdim2
        assert result.dim == 200  # 10 * 20
        assert result.shard == 2  # Keeps first shard

        # Non-ShardedDim should raise error
        with pytest.raises(TypeError):
            sdim1 * 10

    def test_rmul(self):
        """Test reverse multiplication operation."""
        sdim = ShardedDim(10, 2)
        with pytest.raises(TypeError):
            10 * sdim
    
    def test_truediv(self):
        """Test division operation."""
        sdim1 = ShardedDim(12, 2)
        sdim2 = ShardedDim(2, 2)
        result = sdim1 / sdim2
        assert result.dim == 6  # 12 / 2
        assert result.shard == 2  # Sharding preserved

        # Different shards should raise error
        sdim3 = ShardedDim(5, 1)
        with pytest.raises(ValueError):
            sdim1 / sdim3
        
        # Division by zero should raise error
        sdim_zero = ShardedDim(0, 2)
        with pytest.raises(ZeroDivisionError):
            sdim1 / sdim_zero
        
        # Non-integer division should raise error
        sdim4 = ShardedDim(3, 1)
        with pytest.raises(ValueError):
            sdim1 / sdim4  # 10 / 3 is not an integer

        # Non-ShardedDim should raise error
        with pytest.raises(TypeError):
            sdim1 / 5

    def test_rtruediv(self):
        """Test reverse division operation."""
        sdim = ShardedDim(10, 2)
        with pytest.raises(TypeError):
            5 / sdim

    def test_floordiv(self):
        """Test floor division operation."""
        sdim1 = ShardedDim(12, 2)
        sdim2 = ShardedDim(2, 2)
        result = sdim1 // sdim2
        assert result.dim == 6  # 12 // 2
        assert result.shard == 2  # Sharding preserved

        # Floor division should be equivalent to true division for integer results
        assert (sdim1 // sdim2).dim == (sdim1 / sdim2).dim
        assert (sdim1 // sdim2).shard == (sdim1 / sdim2).shard

        # Different shards should raise error
        sdim3 = ShardedDim(5, 1)
        with pytest.raises(ValueError):
            sdim1 // sdim3

    def test_rfloordiv(self):
        """Test reverse floor division operation."""
        sdim = ShardedDim(10, 2)
        with pytest.raises(TypeError):
            5 // sdim
    
    def test_eq(self):
        """Test equality check."""
        sdim1 = ShardedDim(10, 2)
        sdim2 = ShardedDim(10, 2)
        sdim3 = ShardedDim(20, 4)
        sdim4 = ShardedDim(20, 2)

        assert sdim1 == sdim2
        assert sdim1 != sdim3
        assert sdim1 != sdim4
        assert sdim1 != 10


class TestMetaTensor:
    """Test suite for MetaTensor class."""

    def test_init(self):
        """Test initialization with different shapes and shard specs."""
        # Basic initialization
        tensor = MetaTensor([2, 4, 4])
        assert tensor.shape == [2, 4, 4]
        assert tensor.shard_spec == [1, 1, 1]
        
        # With sharding - ensure dimensions are divisible by their shards
        tensor = MetaTensor([2, 4, 4], [1, 2, 4])
        assert tensor.shape == [2, 4, 4]
        assert tensor.shard_spec == [1, 2, 4]
        
        # Mismatched lengths should raise error
        with pytest.raises(ValueError):
            MetaTensor([2, 4, 4], [1, 2])
        
        # Dimensions not divisible by shards should raise error
        with pytest.raises(ValueError):
            MetaTensor([2, 3, 4], [1, 2, 1])
    
    def test_shape_property(self):
        """Test shape property getter and setter."""
        tensor = MetaTensor([2, 4, 4], [1, 2, 4])
        
        # Check getter
        assert tensor.shape == [2, 4, 4]
        
        # Test setter - extending shape
        tensor.shape = [2, 4, 4, 6]
        assert tensor.shape == [2, 4, 4, 6]
        assert tensor.shard_spec == [1, 2, 4, 1]  # Original sharding preserved + new dimension with shard=1
        
        # Test setter - shrinking shape
        tensor.shape = [2, 4]
        assert tensor.shape == [2, 4]
        assert tensor.shard_spec == [1, 2]  # Original sharding preserved for remaining dimensions
        
        # Test setter with dimension not divisible by shard
        with pytest.raises(ValueError):
            tensor.shape = [2, 3]  # 3 not divisible by shard 2
    
    def test_shard_spec_property(self):
        """Test shard_spec property getter and setter."""
        tensor = MetaTensor([2, 4, 4])
        
        # Check getter
        assert tensor.shard_spec == [1, 1, 1]
        
        # Test setter with divisible dimensions
        tensor.shard_spec = [1, 2, 4]
        assert tensor.shard_spec == [1, 2, 4]
        
        # Test setter with non-divisible dimensions
        with pytest.raises(ValueError):
            tensor.shard_spec = [1, 3, 1]  # 4 not divisible by 3
        
        # Mismatched length should raise error
        with pytest.raises(ValueError):
            tensor.shard_spec = [1, 2]

    def test_total_elements(self):
        """Test total_elements method with and without sharding."""
        # Use dimensions divisible by their shards
        tensor = MetaTensor([2, 4, 4], [1, 2, 1])  # Now 4 is divisible by 2
        
        # Without sharding
        assert tensor.total_elements(apply_sharding=False) == 32  # 2*4*4
        
        # With sharding
        assert tensor.total_elements(apply_sharding=True) == 16  # 2*(4/2)*4 = 2*2*4 = 16

    def test_unshard(self):
        """Test unshard method."""
        tensor = MetaTensor([10, 20, 30, 40], [2, 4, 2, 1])
        
        # Test unshard all dimensions
        tensor_copy = tensor.copy()
        tensor_copy.unshard()
        assert tensor_copy.shard_spec == [1, 1, 1, 1]
        
        # Test unshard single dimension
        tensor_copy = tensor.copy()
        tensor_copy.unshard(1)
        assert tensor_copy.shard_spec == [2, 1, 2, 1]
        
        # Test unshard with negative index
        tensor_copy = tensor.copy()
        tensor_copy.unshard(-2)
        assert tensor_copy.shard_spec == [2, 4, 1, 1]
        
        # Test unshard range
        tensor_copy = tensor.copy()
        tensor_copy.unshard(start=1, end=2)
        assert tensor_copy.shard_spec == [2, 1, 1, 1]
        
        # Test unshard with negative indices in range
        tensor_copy = tensor.copy()
        tensor_copy.unshard(start=0, end=-2)
        assert tensor_copy.shard_spec == [1, 1, 1, 1]
        
        # Test out-of-bounds indices
        with pytest.raises(IndexError):
            tensor.copy().unshard(4)
        with pytest.raises(IndexError):
            tensor.copy().unshard(start=5)
        with pytest.raises(ValueError):
            tensor.copy().unshard(start=2, end=1)

    def test_transpose(self):
        """Test transpose method."""
        tensor = MetaTensor([10, 20, 30], [2, 4, 1])
        
        # Transpose first and last dimensions
        transposed = tensor.transpose(0, 2)
        assert transposed.shape == [30, 20, 10]
        assert transposed.shard_spec == [1, 4, 2]
        
        # Test with negative indices
        transposed = tensor.transpose(-3, -1)
        assert transposed.shape == [30, 20, 10]
        assert transposed.shard_spec == [1, 4, 2]
        
        # Out-of-bounds indices should raise error
        with pytest.raises(IndexError):
            tensor.transpose(0, 3)

    def test_permute(self):
        """Test permute method."""
        tensor = MetaTensor([10, 20, 30], [2, 4, 1])
        
        # Permute dimensions
        permuted = tensor.permute(2, 0, 1)
        assert permuted.shape == [30, 10, 20]
        assert permuted.shard_spec == [1, 2, 4]
        
        # Permute with negative indices
        permuted = tensor.permute(-1, -3, -2)
        assert permuted.shape == [30, 10, 20]
        assert permuted.shard_spec == [1, 2, 4]
        
        # Passing indices as list
        permuted = tensor.permute([2, 0, 1])
        assert permuted.shape == [30, 10, 20]
        
        # Invalid permutations
        with pytest.raises(ValueError):
            tensor.permute(0, 1)  # Too few dimensions
        with pytest.raises(ValueError):
            tensor.permute(0, 1, 1)  # Duplicate dimension
        with pytest.raises(IndexError):
            tensor.permute(0, 1, 3)  # Out-of-bounds index

    def test_len(self):
        """Test __len__ method."""
        tensor = MetaTensor([2, 3, 4])
        assert len(tensor) == 3

    def test_getitem(self):
        """Test __getitem__ method."""
        tensor = MetaTensor([10, 20, 30], [2, 4, 1])
        
        # Get single dimension
        assert tensor[1].dim == 20
        assert tensor[1].shard == 4
        
        # Get slice
        slice_tensor = tensor[1:]
        assert slice_tensor.shape == [20, 30]
        assert slice_tensor.shard_spec == [4, 1]

    def test_setitem(self):
        """Test __setitem__ method."""
        tensor = MetaTensor([10, 20, 30], [2, 4, 1])
        
        # Set single dimension
        tensor[1] = ShardedDim(40, 2)
        assert tensor.shape == [10, 40, 30]
        assert tensor.shard_spec == [2, 2, 1]
        
        # Set with invalid type
        with pytest.raises(TypeError):
            tensor[1] = 40
        
        # Set slice
        other_tensor = MetaTensor([50, 60], [1, 2])
        tensor[1:] = other_tensor
        assert tensor.shape == [10, 50, 60]
        assert tensor.shard_spec == [2, 1, 2]
        
        # Set slice with invalid type
        with pytest.raises(TypeError):
            tensor[1:] = [50, 60]
        
        # Set slice with mismatched length
        with pytest.raises(ValueError):
            tensor[1:] = MetaTensor([50], [1])

    def test_iter(self):
        """Test __iter__ method."""
        tensor = MetaTensor([10, 20, 30], [2, 4, 1])
        dims = list(tensor)
        assert len(dims) == 3
        assert dims[0].dim == 10 and dims[0].shard == 2
        assert dims[1].dim == 20 and dims[1].shard == 4
        assert dims[2].dim == 30 and dims[2].shard == 1

    def test_append(self):
        """Test append method."""
        tensor = MetaTensor([10, 20])
        
        # Append valid dimension
        tensor.append(ShardedDim(30, 2))
        assert tensor.shape == [10, 20, 30]
        assert tensor.shard_spec == [1, 1, 2]
        
        # Append invalid type
        with pytest.raises(TypeError):
            tensor.append(30)

    def test_extend(self):
        """Test extend method."""
        tensor = MetaTensor([10, 20])
        other_tensor = MetaTensor([30, 40], [2, 4])
        
        # Extend with valid tensor
        tensor.extend(other_tensor)
        assert tensor.shape == [10, 20, 30, 40]
        assert tensor.shard_spec == [1, 1, 2, 4]
        
        # Extend with invalid type
        with pytest.raises(TypeError):
            tensor.extend([30, 40])

    def test_pop(self):
        """Test pop method."""
        tensor = MetaTensor([10, 20, 30], [2, 4, 1])
        
        # Pop last dimension
        dim = tensor.pop()
        assert dim.dim == 30 and dim.shard == 1
        assert tensor.shape == [10, 20]
        
        # Pop specific index
        dim = tensor.pop(0)
        assert dim.dim == 10 and dim.shard == 2
        assert tensor.shape == [20]

    def test_insert(self):
        """Test insert method."""
        tensor = MetaTensor([10, 20])
        
        # Insert valid dimension
        tensor.insert(1, ShardedDim(30, 2))
        assert tensor.shape == [10, 30, 20]
        assert tensor.shard_spec == [1, 2, 1]
        
        # Insert invalid type
        with pytest.raises(TypeError):
            tensor.insert(1, 30)

    def test_remove(self):
        """Test remove method."""
        tensor = MetaTensor([10, 20, 30], [1, 2, 1])
        
        # Remove existing dimension
        tensor.remove(ShardedDim(20, 2))
        assert tensor.shape == [10, 30]
        
        # Remove non-existent dimension
        with pytest.raises(ValueError):
            tensor.remove(ShardedDim(20, 2))
        
        # Remove with invalid type
        with pytest.raises(TypeError):
            tensor.remove(20)

    def test_clear(self):
        """Test clear method."""
        tensor = MetaTensor([10, 20, 30])
        tensor.clear()
        assert tensor.shape == []

    def test_copy(self):
        """Test copy method."""
        tensor = MetaTensor([10, 20, 30], [2, 4, 1])
        copy_tensor = tensor.copy()
        
        # Verify it's a different object with the same values
        assert tensor is not copy_tensor
        assert tensor.shape == copy_tensor.shape
        assert tensor.shard_spec == copy_tensor.shard_spec
        
        # Modify copy should not affect original
        copy_tensor[0] = ShardedDim(40, 1)
        assert tensor.shape == [10, 20, 30]
        assert copy_tensor.shape == [40, 20, 30]

    def test_index(self):
        """Test index method."""
        tensor = MetaTensor([10, 20, 30, 20], [1, 2, 1, 2])
        
        # Find existing dimension
        idx = tensor.index(ShardedDim(20, 2))
        assert idx == 1
        
        # Find non-existent dimension
        with pytest.raises(ValueError):
            tensor.index(ShardedDim(20, 3))
        
        # Find with invalid type
        with pytest.raises(TypeError):
            tensor.index(20)

    def test_count(self):
        """Test count method."""
        tensor = MetaTensor([10, 20, 30, 20], [1, 2, 1, 2])
        
        # Count existing dimension
        count = tensor.count(ShardedDim(20, 2))
        assert count == 2
        
        # Count non-existent dimension
        count = tensor.count(ShardedDim(20, 4))
        assert count == 0
        
        # Count with invalid type
        with pytest.raises(TypeError):
            tensor.count(20)

    def test_contains(self):
        """Test __contains__ method."""
        tensor = MetaTensor([10, 20, 30], [1, 2, 1])
        
        # Check existing dimension
        assert ShardedDim(20, 2) in tensor
        
        # Check non-existent dimension
        assert ShardedDim(20, 4) not in tensor
        
        # Check with invalid type
        with pytest.raises(TypeError):
            20 in tensor

    def test_eq(self):
        """Test __eq__ method."""
        tensor1 = MetaTensor([10, 20], [2, 1])
        tensor2 = MetaTensor([10, 20], [2, 1])
        tensor3 = MetaTensor([10, 20], [1, 1])
        tensor4 = MetaTensor([10, 30], [2, 1])
        
        # Equal tensors
        assert tensor1 == tensor2
        
        # Different sharding
        assert tensor1 != tensor3
        
        # Different dimensions
        assert tensor1 != tensor4
        
        # Different type
        assert tensor1 != [10, 20]
    
    def test_add(self):
        """Test element-wise addition of two tensors with compatible shapes."""
        tensor1 = MetaTensor([10, 20, 30], [2, 4, 1])
        tensor2 = MetaTensor([10, 20, 30], [2, 4, 1])
        
        # Element-wise addition
        result = tensor1 + tensor2
        
        # Result should have same shape and sharding
        assert result.shape == [10, 20, 30]
        assert result.shard_spec == [2, 4, 1]
        
        # Incompatible sharding should raise error
        tensor3 = MetaTensor([10, 20, 30], [1, 4, 1])
        with pytest.raises(ValueError):
            tensor1 + tensor3
            
    def test_sub(self):
        """Test element-wise subtraction of two tensors with compatible shapes."""
        tensor1 = MetaTensor([10, 20, 30], [2, 4, 1])
        tensor2 = MetaTensor([10, 20, 30], [2, 4, 1])
        
        # Element-wise subtraction
        result = tensor1 - tensor2
        
        # Result should have same shape and sharding
        assert result.shape == [10, 20, 30]
        assert result.shard_spec == [2, 4, 1]
        
        # Non-MetaTensor should raise TypeError
        with pytest.raises(TypeError):
            tensor1 - 10
    
    def test_mul(self):
        """Test element-wise multiplication of two tensors with compatible shapes."""
        tensor1 = MetaTensor([10, 20, 30], [2, 4, 1])
        tensor2 = MetaTensor([10, 20, 30], [2, 4, 1])
        
        # Element-wise multiplication
        result = tensor1 * tensor2
        
        # Result should have same shape and sharding
        assert result.shape == [10, 20, 30]
        assert result.shard_spec == [2, 4, 1]
        
        # Incompatible sharding should raise error
        tensor3 = MetaTensor([10, 20, 30], [1, 4, 1])
        with pytest.raises(ValueError):
            tensor1 * tensor3
            
        # Scalar multiplication should return copy of tensor
        result = tensor1 * 10
        assert result.shape == [10, 20, 30] 
        assert result.shard_spec == tensor1.shard_spec
        assert result is not tensor1  # Should be a copy
        
        # Non-number/tensor should raise TypeError
        with pytest.raises(TypeError):
            tensor1 * "string"
    
    def test_div(self):
        """Test element-wise division of two tensors with compatible shapes."""
        tensor1 = MetaTensor([10, 20, 30], [2, 4, 1])
        tensor2 = MetaTensor([10, 20, 30], [2, 4, 1])
        
        # Element-wise division
        result = tensor1 / tensor2
        
        # Result should have same shape and sharding
        assert result.shape == [10, 20, 30]
        assert result.shard_spec == [2, 4, 1]
        
        # Incompatible sharding should raise error
        tensor3 = MetaTensor([10, 20, 30], [1, 4, 1])
        with pytest.raises(ValueError):
            tensor1 / tensor3
            
        # Different shapes should raise error
        tensor4 = MetaTensor([10, 20, 40], [2, 4, 1])
        with pytest.raises(ValueError):
            tensor1 / tensor4
            
        # Scalar division should return copy of tensor
        result = tensor1 / 10
        assert result.shape == [10, 20, 30] 
        assert result.shard_spec == tensor1.shard_spec
        assert result is not tensor1  # Should be a copy
        
        # Division by zero should raise error
        with pytest.raises(ZeroDivisionError):
            tensor1 / 0
            
        # Non-number/tensor should raise TypeError
        with pytest.raises(TypeError):
            tensor1 / "string"
    
    def test_rtruediv(self):
        """Test right-side division."""
        tensor1 = MetaTensor([10, 20, 30], [2, 4, 1])
        tensor2 = MetaTensor([10, 20, 30], [2, 4, 1])
        
        # Element-wise right division (tensor2 / tensor1)
        result = tensor1.__rtruediv__(tensor2)
        
        # Result should have same shape and sharding
        assert result.shape == [10, 20, 30]
        assert result.shard_spec == [2, 4, 1]
        
        # Scalar division
        result = tensor1.__rtruediv__(10)
        assert result.shape == tensor1.shape
        assert result.shard_spec == tensor1.shard_spec
        
        # Non-tensor/number should raise TypeError
        with pytest.raises(TypeError):
            tensor1.__rtruediv__("string")    


class TestMetaModule:
    """Test suite for MetaModule class."""
    def setup_method(self):
        """Set up test environment for each test."""
        # Reset registries before each test
        import flagscale.runner.estimator.meta_base as meta_base
        meta_base._model_registries = {}
        
        # Reset module counter for predictable tests
        MetaModule._counter = 0
        MetaModule._path = None
        
        # Register models used in tests
        register_model("default")
        register_model("test_model")

    def test_init(self):
        """Test initialization."""
        # Basic init
        module = MetaModule()
        assert module.shard_specs is None
        assert module.model_id == "default"
        
        # With shard_specs and model_id
        module = MetaModule([[1, 2], [3, 4]], "test_model")
        assert module.shard_specs == [[1, 2], [3, 4]]
        assert module.model_id == "test_model"

    def test_compute_methods(self):
        """Test default compute_* methods."""
        module = MetaModule()
        
        # Default implementations return 0
        assert module.add_flops() == 0
        assert module.add_params() == 0
        assert module.add_acts() == 0

    @patch('flagscale.runner.estimator.meta_base.get_registry')
    def test_update_registry(self, mock_get_registry):
        """Test update_registry method."""
        # Setup mock registry
        mock_registry = MagicMock()
        mock_get_registry.return_value = mock_registry
        
        # Create module with compute methods that return non-zero values
        class TestModule(MetaModule):
            def add_flops(self, *args, **kwargs):
                return 100
                
            def add_params(self, *args, **kwargs):
                return 200
                
            def add_acts(self, *args, **kwargs):
                return 300
                
            def forward(self, *args, **kwargs):
                return "result"
        
        module = TestModule()
        MetaModule._path = "TestModule_1"  # Set the path
        
        # Call update_registry
        module.update_registry("arg1", arg2="value")
        
        # Verify registry methods were called with correct values
        mock_registry.add_flops.assert_called_once_with(100, path="TestModule_1")
        mock_registry.add_params.assert_called_once_with(200, path="TestModule_1")
        mock_registry.add_acts.assert_called_once_with(300, path="TestModule_1")

    @patch('flagscale.runner.estimator.meta_base.get_registry')
    def test_call(self, mock_get_registry):
        """Test __call__ method."""
        # Setup mock registry
        mock_registry = MagicMock()
        mock_get_registry.return_value = mock_registry
        
        # Create module with mocked methods
        class TestModule(MetaModule):
            def add_flops(self, *args, **kwargs):
                return 100
                
            def add_params(self, *args, **kwargs):
                return 200
                
            def add_acts(self, *args, **kwargs):
                return 300
                
            def forward(self, *args, **kwargs):
                return "result"
        
        module = TestModule()
        
        # Call the module
        result = module("arg1", arg2="value")
        
        # Verify registry methods were called (level now derived from path)
        mock_registry.add_flops.assert_called_once_with(100, path="TestModule_1")
        mock_registry.add_params.assert_called_once_with(200, path="TestModule_1")
        mock_registry.add_acts.assert_called_once_with(300, path="TestModule_1")
        
        # Verify result is from forward method
        assert result == "result"

    def test_forward_not_implemented(self):
        """Test forward method raises NotImplementedError if not overridden."""
        module = MetaModule()
        with pytest.raises(NotImplementedError):
            module()

    def test_zero_metrics_handling(self):
        """Test handling of zero metric values."""
        # Create test registry to avoid using mock
        register_model("zero_test")
        registry = get_registry("zero_test")
        
        # Create module that returns zero for some metrics
        class ZeroModule(MetaModule):
            def __init__(self):
                super().__init__(model_id="zero_test")
                
            def add_flops(self, *args, **kwargs):
                return 0
                
            def add_params(self, *args, **kwargs):
                return 100
                
            def add_acts(self, *args, **kwargs):
                return 0
                
            def forward(self, *args, **kwargs):
                return None
        
        module = ZeroModule()
        module()
        
        # Verify metrics are added to logs
        assert len(registry.flops_logs) == 1
        assert len(registry.params_logs) == 1
        assert len(registry.acts_logs) == 1
        
        # Verify registry totals
        assert registry.total_flops == 0
        assert registry.total_params == 100
        assert registry.total_acts == 0


class TestModelStatsRegistry:
    """Test suite for ModelStatsRegistry class."""

    def test_init(self):
        """Test initialization."""
        # With model_id
        registry = ModelStatsRegistry("test_model")
        assert registry.model_id == "test_model"
        assert registry.total_flops == 0
        assert registry.total_params == 0
        assert registry.total_acts == 0
        assert registry.module_ids == {}
        assert registry.flops_counter == 0
        
        # Without model_id (should use default)
        registry = ModelStatsRegistry()
        assert registry.model_id == "default"

    def test_add_flops(self):
        """Test adding FLOPs."""
        registry = ModelStatsRegistry()
        
        # Add with path
        registry.add_flops(100, "op1")
        assert registry.total_flops == 100
        assert len(registry.flops_logs) == 1
        assert registry.flops_logs[0][0] == 100
        assert registry.flops_logs[0][1] == "op1"
        assert registry.flops_logs[0][2] == 0  # level derived from path
        assert "op1" in registry.flops_by_module
        assert registry.flops_by_module["op1"] == 100
        
        # Add without path (should generate one)
        registry.add_flops(50)
        assert registry.total_flops == 150
        assert len(registry.flops_logs) == 2
        assert registry.flops_logs[1][0] == 50
        assert "unnamed_flop" in registry.flops_logs[1][1]
        
        # Test negative value handling
        with pytest.raises(ValueError):
            registry.add_flops(-10, "op2")
        
        # Test path already exists handling - should now raise ValueError 
        with pytest.raises(ValueError):
            registry.add_flops(30, "op1")

    def test_add_params(self):
        """Test adding parameters."""
        registry = ModelStatsRegistry()
        
        # Add with path
        registry.add_params(1000, "layer1")
        assert registry.total_params == 1000
        assert len(registry.params_logs) == 1
        assert registry.params_logs[0][0] == 1000
        assert registry.params_logs[0][1] == "layer1"
        assert registry.params_logs[0][2] == 0  # level derived from path
        assert "layer1" in registry.params_by_module
        assert registry.params_by_module["layer1"] == 1000
        
        # Add without path (should generate one)
        registry.add_params(500)
        assert registry.total_params == 1500
        assert len(registry.params_logs) == 2
        assert registry.params_logs[1][0] == 500
        assert "unnamed_param" in registry.params_logs[1][1]
        
        # Test negative value handling
        with pytest.raises(ValueError):
            registry.add_params(-100, "layer2")

    def test_add_acts(self):
        """Test adding activations."""
        registry = ModelStatsRegistry()
        
        # Add with path
        registry.add_acts(200, "activation1")
        assert registry.total_acts == 200
        assert len(registry.acts_logs) == 1
        assert registry.acts_logs[0][0] == 200
        assert registry.acts_logs[0][1] == "activation1"
        assert registry.acts_logs[0][2] == 0  # level derived from path
        assert "activation1" in registry.acts_by_module
        assert registry.acts_by_module["activation1"] == 200
        
        # Add without path (should generate one)
        registry.add_acts(300)
        assert registry.total_acts == 500
        assert len(registry.acts_logs) == 2
        assert registry.acts_logs[1][0] == 300
        assert "unnamed_act" in registry.acts_logs[1][1]
        
        # Test negative value handling
        with pytest.raises(ValueError):
            registry.add_acts(-50, "activation2")

    def test_reset(self):
        """Test resetting the registry."""
        registry = ModelStatsRegistry()
        
        # Add some metrics
        registry.add_flops(100, "op1")
        registry.add_params(200, "layer1")
        registry.add_acts(300, "activation1")
        
        # Verify that metrics were added
        assert registry.total_flops == 100
        assert registry.total_params == 200
        assert registry.total_acts == 300
        assert len(registry.flops_logs) == 1
        assert len(registry.params_logs) == 1
        assert len(registry.acts_logs) == 1
        assert "op1" in registry.flops_by_module
        assert "layer1" in registry.params_by_module
        assert "activation1" in registry.acts_by_module
        assert registry.flops_counter == 1
        assert registry.params_counter == 1
        assert registry.acts_counter == 1
        assert len(registry.module_ids) == 3
        
        # Reset
        registry.reset()
        
        # Verify that everything was reset
        assert registry.total_flops == 0
        assert registry.total_params == 0
        assert registry.total_acts == 0
        assert len(registry.flops_logs) == 0
        assert len(registry.params_logs) == 0
        assert len(registry.acts_logs) == 0
        assert len(registry.flops_by_module) == 0
        assert len(registry.params_by_module) == 0
        assert len(registry.acts_by_module) == 0
        assert registry.flops_counter == 0
        assert registry.params_counter == 0
        assert registry.acts_counter == 0
        assert len(registry.module_ids) == 0
       
    def test_hierarchical_paths(self):
        """Test handling of hierarchical paths."""
        registry = ModelStatsRegistry()
        
        # Add metrics with hierarchical paths
        registry.add_flops(100, "parent")
        registry.add_params(200, "parent/child1")
        registry.add_acts(300, "parent/child1/grandchild")
        registry.add_flops(150, "parent/child2")
        
        # Verify modules are tracked correctly
        assert "parent" in registry.flops_by_module
        assert "parent/child1" in registry.params_by_module
        assert "parent/child1/grandchild" in registry.acts_by_module
        assert "parent/child2" in registry.flops_by_module
        
        # Verify levels are derived correctly
        assert registry.flops_logs[0][2] == 0  # parent: 0 slashes
        assert registry.params_logs[0][2] == 1  # parent/child1: 1 slash
        assert registry.acts_logs[0][2] == 2   # parent/child1/grandchild: 2 slashes
        assert registry.flops_logs[1][2] == 1  # parent/child2: 1 slash

    @patch('builtins.print')
    def test_print_logs(self, mock_print):
        """Test print_logs method."""
        registry = ModelStatsRegistry("test_model")
        
        # Add metrics with hierarchical structure
        registry.add_flops(100, "parent")
        registry.add_params(200, "parent/child1")
        registry.add_acts(300, "parent/child1/grandchild")
        registry.add_flops(150, "parent/child2")
        
        # Test printing all metrics
        registry.print_logs()
        mock_print.assert_called()
        
        # Test printing specific metric type
        mock_print.reset_mock()
        registry.print_logs(metric_type="flops")
        mock_print.assert_called()
        
        # Test printing with summary
        mock_print.reset_mock()
        registry.print_logs(include_summary=True)
        mock_print.assert_called()
        
        # Test with invalid metric type
        with pytest.raises(ValueError):
            registry.print_logs(metric_type="invalid")


    def test_hierarchical_module_tracking(self):
        """Test hierarchical module tracking across registries."""
        # Register model
        register_model("hierarchy_test")

        # Create hierarchical module structure with a more complex structure
        class RootModule(MetaModule):
            def __init__(self):
                super().__init__(model_id="hierarchy_test")
                self.child1 = ChildModule()
                self.child2 = ChildModule()

            def add_flops(self, *args, **kwargs):
                return 100

            def forward(self, *args, **kwargs):
                # Call children
                self.child1()
                self.child2()
                return None

        class ChildModule(MetaModule):
            def __init__(self):
                super().__init__(model_id="hierarchy_test")
                self.grandchild = GrandchildModule()

            def add_params(self, *args, **kwargs):
                return 200

            def forward(self, *args, **kwargs):
                # Call grandchild
                self.grandchild()
                return None

        class GrandchildModule(MetaModule):
            def __init__(self):
                super().__init__(model_id="hierarchy_test")

            def add_acts(self, *args, **kwargs):
                return 300

            def forward(self, *args, **kwargs):
                return None

        # Create root module and call it
        root = RootModule()
        root()

        # Get registry
        registry = get_registry("hierarchy_test")

        # Verify hierarchical paths were created correctly
        root_path = None
        child_paths = []
        grandchild_paths = []

        for log in registry.flops_logs + registry.params_logs + registry.acts_logs:
            path = log[1]
            if "RootModule_" in path and "/" not in path:
                root_path = path
            elif "ChildModule_" in path and "/" in path and path.count("/") == 1:
                if path not in child_paths:
                    child_paths.append(path)
            elif "GrandchildModule_" in path and path.count("/") == 2:
                if path not in grandchild_paths:
                    grandchild_paths.append(path)

        assert root_path is not None, "Root module path not found"
        assert len(child_paths) == 2, f"Expected 2 child paths, got {len(child_paths)}: {child_paths}"
        assert len(grandchild_paths) == 2, f"Expected 2 grandchild paths, got {len(grandchild_paths)}: {grandchild_paths}"

        # Verify each child path has correct parent
        for child_path in child_paths:
            assert child_path.startswith(root_path + "/"), f"Child path {child_path} doesn't start with {root_path}/"

        # Verify each grandchild path has correct parent
        for grandchild_path in grandchild_paths:
            # Extract the parent path by removing the last component
            parent_path = "/".join(grandchild_path.split("/")[:-1])
            assert parent_path in child_paths, f"Grandchild's parent {parent_path} not in child_paths: {child_paths}"

        # Verify levels are derived correctly (now based on path)
        for log in registry.flops_logs:
            path = log[1]
            level = log[2]
            assert level == path.count("/"), f"Path {path} has wrong level {level}, expected {path.count('/')}"

        for log in registry.params_logs:
            path = log[1]
            level = log[2]
            assert level == path.count("/"), f"Path {path} has wrong level {level}, expected {path.count('/')}"

        for log in registry.acts_logs:
            path = log[1]
            level = log[2]
            assert level == path.count("/"), f"Path {path} has wrong level {level}, expected {path.count('/')}"

        # Verify correct metrics were added at each level
        for log in registry.flops_logs:
            path = log[1]
            if "RootModule_" in path and "/" not in path:
                assert log[0] == 100, f"Root module should have 100 FLOPs, got {log[0]}"

        for log in registry.params_logs:
            print(log)
            path = log[1]
            if "RootModule_1/ChildModule_2" == path: 
                assert log[0] == 200, f"Child module should have 200 params, got {log[0]}"

        for log in registry.acts_logs:
            path = log[1]
            if "GrandchildModule_" in path:
                assert log[0] == 300, f"Grandchild module should have 300 acts, got {log[0]}"