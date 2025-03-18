from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union


class ShardedDim:
    """
    A class representing a single dimension with its sharding factor.

    Provides arithmetic operations that enforce consistent sharding.
    Operations between ShardedDim objects require matching shard values.
    """

    def __init__(self, dim, shard=1):
        self.dim = dim
        self.shard = shard if shard > 0 else 1

        # Ensure dimension is divisible by the shard
        if self.dim % self.shard != 0:
            raise ValueError(
                f"Dimension {dim} is not divisible by shard factor {shard}"
            )

    def sharded_dim(self):
        """
        Get the effective size after sharding is applied.

        Returns:
        --------
        int
            Dimension size divided by sharding factor
        """
        return self.dim // self.shard

    def copy(self):
        """
        Create a copy of this ShardedDim.

        Returns:
        --------
        ShardedDim
            New ShardedDim with the same dimension and sharding factor
        """
        return ShardedDim(self.dim, self.shard)

    def __repr__(self):
        """Formal string representation."""
        return f"ShardedDim({self.dim}, {self.shard})"

    def __str__(self):
        """String representation showing dimension and sharding."""
        return f"ShardedDim({self.dim}, {self.shard})"

    # Arithmetic operations that enforce matching sharding
    def __add__(self, other):
        """
        Add operation (returns a ShardedDim).

        Parameters:
        -----------
        other : ShardedDim
            Another ShardedDim object to add to this one

        Returns:
        --------
        ShardedDim
            Result of addition with matching sharding

        Raises:
        -------
        ValueError
            If sharding factors don't match
        TypeError
            If attempting to add with non-ShardedDim object
        """
        if not isinstance(other, ShardedDim):
            raise TypeError(
                f"Can only add ShardedDim with another ShardedDim, not {type(other)}"
            )

        if self.shard != other.shard:
            raise ValueError(
                f"Cannot add ShardedDim with different sharding: {self.shard} vs {other.shard}"
            )

        # Shards match, maintain sharding
        return ShardedDim(self.dim + other.dim, self.shard)

    def __radd__(self, other):
        """
        Reverse add operation.

        Raises:
        -------
        TypeError
            Always raises TypeError as only ShardedDim objects can be added together
        """
        raise TypeError(
            f"Can only add ShardedDim with another ShardedDim, not {type(other)}"
        )

    def __sub__(self, other):
        """
        Subtract operation.

        Parameters:
        -----------
        other : ShardedDim
            Another ShardedDim object to subtract from this one

        Returns:
        --------
        ShardedDim
            Result of subtraction with matching sharding

        Raises:
        -------
        ValueError
            If sharding factors don't match
        TypeError
            If attempting to subtract with non-ShardedDim object
        """
        if not isinstance(other, ShardedDim):
            raise TypeError(
                f"Can only subtract ShardedDim with another ShardedDim, not {type(other)}"
            )

        if self.shard != other.shard:
            raise ValueError(
                f"Cannot subtract ShardedDim with different sharding: {self.shard} vs {other.shard}"
            )

        # Shards match, maintain sharding
        return ShardedDim(self.dim - other.dim, self.shard)

    def __mul__(self, other):
        """
        Multiply operation.

        When multiplying ShardedDim objects, the result uses the sharding factor
        of self since we want to maintain consistent sharding.

        Parameters:
        -----------
        other : ShardedDim
            Another ShardedDim object to multiply with this one

        Returns:
        --------
        ShardedDim
            Result of multiplication with sharding factor preserved

        Raises:
        -------
        TypeError
            If attempting to multiply with non-ShardedDim object
        """
        if not isinstance(other, ShardedDim):
            raise TypeError(
                f"Can only multiply ShardedDim with another ShardedDim, not {type(other)}"
            )

        if self.shard != other.shard:
            raise ValueError(
                f"Cannot subtract ShardedDim with different sharding: {self.shard} vs {other.shard}"
            )

        # When multiplying dimensions, we multiply the sharded values and preserve the sharding
        return ShardedDim(self.dim * other.dim, self.shard)

    def __rmul__(self, other):
        """
        Reverse multiply operation.

        Raises:
        -------
        TypeError
            Always raises TypeError as only ShardedDim objects can be multiplied together
        """
        raise TypeError(
            f"Can only multiply ShardedDim with another ShardedDim, not {type(other)}"
        )

    def __truediv__(self, other):
        """
        Division operation.

        When dividing ShardedDim objects, the result uses the sharding factor
        of self since we want to maintain consistent sharding.

        Parameters:
        -----------
        other : ShardedDim
            Another ShardedDim object to divide by

        Returns:
        --------
        ShardedDim
            Result of division with sharding factor preserved

        Raises:
        -------
        TypeError
            If attempting to divide with non-ShardedDim object
        ValueError
            If sharding factors don't match
        ZeroDivisionError
            If dividing by zero
        """
        if not isinstance(other, ShardedDim):
            raise TypeError(
                f"Can only divide ShardedDim with another ShardedDim, not {type(other)}"
            )

        if self.shard != other.shard:
            raise ValueError(
                f"Cannot divide ShardedDim with different sharding: {self.shard} vs {other.shard}"
            )

        if other.dim == 0:
            raise ZeroDivisionError("division by zero")

        # Check if division results in an integer
        if self.dim % other.dim != 0:
            raise ValueError(
                f"Division of {self.dim} by {other.dim} must result in an integer value"
            )

        # When dividing dimensions, we divide the values and preserve the sharding
        return ShardedDim(self.dim // other.dim, self.shard)

    def __rtruediv__(self, other):
        """
        Reverse division operation.

        Raises:
        -------
        TypeError
            Always raises TypeError as only ShardedDim objects can be divided together
        """
        raise TypeError(
            f"Can only divide ShardedDim with another ShardedDim, not {type(other)}"
        )

    def __floordiv__(self, other):
        """
        Floor division operation (same as __truediv__ for integer dimensions).

        Parameters:
        -----------
        other : ShardedDim
            Another ShardedDim object to divide by

        Returns:
        --------
        ShardedDim
            Result of floor division with sharding factor preserved

        Raises:
        -------
        TypeError
            If attempting to divide with non-ShardedDim object
        ValueError
            If sharding factors don't match
        ZeroDivisionError
            If dividing by zero
        """
        return self.__truediv__(other)

    def __rfloordiv__(self, other):
        """
        Reverse floor division operation.

        Raises:
        -------
        TypeError
            Always raises TypeError as only ShardedDim objects can be divided together
        """
        return self.__rtruediv__(other)

    def __eq__(self, other):
        """
        Equality check (compares both dimension and sharding values).

        Two ShardedDim objects are equal only when both their dimension
        and sharding values match exactly.

        Parameters:
        -----------
        other : ShardedDim or any
            Object to compare with

        Returns:
        --------
        bool
            True if dimensions and shards match, False otherwise
        """
        if isinstance(other, ShardedDim):
            return self.dim == other.dim and self.shard == other.shard
        return False  # Only equal to other ShardedDim objects


class MetaTensor:
    """
    A tensor-like object that maintains shape and sharding specification.
    Uses ShardedDim objects as its basic elements.
    Works like a list where each element is a ShardedDim object.
    """

    def __init__(self, shape, shard_spec=None, model_id="default"):
        # Create from shape and shard_spec
        shape_list = list(shape)
        if shard_spec is None:
            shard_spec = [1] * len(shape_list)
        else:
            shard_spec = list(shard_spec)

        # Validate the shard spec matches shape dimensions
        if len(shard_spec) != len(shape_list):
            raise ValueError(
                f"Shard spec length ({len(shard_spec)}) must match shape dimensions ({len(shape_list)})"
            )

        # Create ShardedDim objects for each dimension
        # The ShardedDim constructor will validate divisibility
        try:
            self._sharded_dims = [
                ShardedDim(d, s) for d, s in zip(shape_list, shard_spec)
            ]
        except ValueError as e:
            # Re-raise with more context about which dimension caused the issue
            for i, (d, s) in enumerate(zip(shape_list, shard_spec)):
                if d % s != 0:
                    raise ValueError(
                        f"Dimension {i} with size {d} is not divisible by shard factor {s}"
                    ) from e
            raise  # Re-raise if the error was for another reason

        self._model_id = model_id

    # Shape and sharding properties
    @property
    def shape(self):
        """Get the unsharded shape as a list."""
        return [sdim.dim for sdim in self._sharded_dims]

    @shape.setter
    def shape(self, value):
        """Set the shape, preserving sharding where possible."""
        value = list(value)
        old_length = len(self._sharded_dims)
        new_length = len(value)

        if new_length >= old_length:
            # Keep existing sharding for common dimensions
            new_dims = []
            # For existing dimensions, preserve sharding but validate divisibility
            for i in range(old_length):
                new_dims.append(ShardedDim(value[i], self._sharded_dims[i].shard))
            # Add new dimensions with default sharding
            for i in range(old_length, new_length):
                new_dims.append(ShardedDim(value[i], 1))
            self._sharded_dims = new_dims
        else:
            # Truncate to new size
            self._sharded_dims = [
                ShardedDim(value[i], self._sharded_dims[i].shard)
                for i in range(new_length)
            ]

    @property
    def shard_spec(self):
        """Get the shard specification as a list."""
        return [sdim.shard for sdim in self._sharded_dims]

    @shard_spec.setter
    def shard_spec(self, value):
        """Set the shard specification for each dimension."""
        value = list(value)
        if len(value) != len(self._sharded_dims):
            raise ValueError(
                f"Shard spec length ({len(value)}) must match shape dimensions ({len(self._sharded_dims)})"
            )

        # Create new ShardedDims with updated sharding, which will validate divisibility
        new_dims = []
        for i, shard in enumerate(value):
            new_dims.append(ShardedDim(self._sharded_dims[i].dim, shard))
        self._sharded_dims = new_dims

    @property
    def model_id(self):
        """Get the model identifier."""
        return self._model_id

    @model_id.setter
    def model_id(self, value):
        """Set the model identifier."""
        self._model_id = value

    def total_elements(self, apply_sharding=True):
        """
        Calculate the total number of elements in the tensor.

        Parameters:
        -----------
        apply_sharding : bool, optional (default=True)
            If True, uses sharded dimensions
            If False, uses raw dimensions

        Returns:
        --------
        int
            Total number of elements
        """
        total = 1
        for sdim in self._sharded_dims:
            total *= (sdim.dim // sdim.shard) if apply_sharding else sdim.dim
        return total

    def unshard(self, index=None, start=None, end=None):
        """
        Convert tensor dimensions to unsharded state.

        This method supports unsharding:
        1. All dimensions (when no arguments provided)
        2. A single dimension at given index
        3. A range of dimensions from start to end (inclusive)

        Parameters:
        -----------
        index : int, optional
            Index of single dimension to unshard. If provided, start and end are ignored.
        start : int, optional
            Start index for range unsharding (inclusive)
        end : int, optional
            End index for range unsharding (inclusive)

        Returns:
        --------
        MetaTensor
            Self reference for method chaining

        Raises:
        -------
        IndexError
            If the provided indices are out of bounds
        """
        # Case 1: Unshard a single dimension if index is provided
        if index is not None:
            # Handle negative index
            if index < 0:
                index = len(self._sharded_dims) + index

            if 0 <= index < len(self._sharded_dims):
                self._sharded_dims[index].shard = 1
            else:
                raise IndexError(
                    f"Dimension index {index} out of range for tensor with {len(self._sharded_dims)} dimensions"
                )

        # Case 2: Unshard a range of dimensions if start and/or end are provided
        elif start is not None or end is not None:
            # Use defaults if not provided
            if start is None:
                start = 0
            if end is None:
                end = len(self._sharded_dims) - 1

            # Handle negative indices
            if start < 0:
                start = len(self._sharded_dims) + start
            if end < 0:
                end = len(self._sharded_dims) + end

            # Check bounds
            if start < 0 or start >= len(self._sharded_dims):
                raise IndexError(
                    f"Start index {start} out of range for tensor with {len(self._sharded_dims)} dimensions"
                )
            if end < 0 or end >= len(self._sharded_dims):
                raise IndexError(
                    f"End index {end} out of range for tensor with {len(self._sharded_dims)} dimensions"
                )
            if start > end:
                raise ValueError(
                    f"Start index {start} cannot be greater than end index {end}"
                )

            # Unshard the specified range
            for i in range(start, end + 1):
                self._sharded_dims[i].shard = 1

        # Case 3: Unshard all dimensions (default behavior)
        else:
            for sdim in self._sharded_dims:
                sdim.shard = 1

        return self

    def __len__(self):
        """Get the number of dimensions."""
        return len(self._sharded_dims)

    def __getitem__(self, index):
        """
        Get ShardedDim object(s) at the specified index/slice.

        Parameters:
        -----------
        index : int or slice
            Index or slice to access

        Returns:
        --------
        ShardedDim or MetaTensor
            For single indices, returns a ShardedDim object
            For slices, returns a new MetaTensor
        """
        if isinstance(index, slice):
            # Return a new MetaTensor for slices
            new_tensor = MetaTensor.__new__(MetaTensor)
            new_tensor._sharded_dims = self._sharded_dims[index]
            # Initialize _model_id to avoid the attribute error
            new_tensor._model_id = self._model_id
            return new_tensor
        # Return the ShardedDim object for single indices
        return self._sharded_dims[index]

    def __setitem__(self, index, value):
        """
        Set dimension(s) at the specified index/slice.

        Parameters:
        -----------
        index : int or slice
            Index or slice to modify
        value : ShardedDim or MetaTensor
            Value(s) to set. For single indices, must be a ShardedDim.
            For slices, must be a MetaTensor with matching slice length.

        Raises:
        -------
        TypeError
            If value is not a ShardedDim (for single index) or MetaTensor (for slice)
        ValueError
            If slice assignment length doesn't match
        """
        if isinstance(index, slice):
            # Handle slice assignment
            if isinstance(value, MetaTensor):
                # Extract ShardedDim objects directly
                new_dims = value._sharded_dims

                # Calculate the length of the slice
                start, stop, step = index.indices(len(self._sharded_dims))
                slice_indices = list(range(start, stop, step))

                # Check if the lengths match
                if len(new_dims) != len(slice_indices):
                    raise ValueError(
                        f"Cannot assign {len(new_dims)} values to slice of length {len(slice_indices)}"
                    )

                # Replace dimensions in the slice
                for i, sdim in zip(slice_indices, new_dims):
                    self._sharded_dims[i] = sdim
            else:
                # Only MetaTensor objects allowed for slice assignment
                raise TypeError(
                    f"Slice assignment requires a MetaTensor, got {type(value)}"
                )
        else:
            # Simple index assignment - must be ShardedDim
            if isinstance(value, ShardedDim):
                self._sharded_dims[index] = value
            else:
                raise TypeError(
                    f"Single index assignment requires a ShardedDim object, got {type(value)}"
                )

    def __iter__(self):
        """
        Iterator over ShardedDim objects.

        Yields:
        -------
        ShardedDim
            Each dimension with its sharding
        """
        return iter(self._sharded_dims)

    def __contains__(self, value):
        """
        Check if a ShardedDim exists in the tensor.

        Parameters:
        -----------
        value : ShardedDim
            The ShardedDim to check for

        Returns:
        --------
        bool
            True if found, False otherwise

        Raises:
        -------
        TypeError
            If value is not a ShardedDim object
        """
        if not isinstance(value, ShardedDim):
            raise TypeError(f"Can only check for ShardedDim objects, not {type(value)}")

        return any(
            sdim.dim == value.dim and sdim.shard == value.shard
            for sdim in self._sharded_dims
        )

    def __repr__(self):
        """
        Formal string representation for debugging.

        Returns:
        --------
        str
            Representation showing class name, shape, and shard specs
        """
        return f"MetaTensor(shape={self.shape}, shard_spec={self.shard_spec})"

    def index(self, value):
        """
        Return the index of the first occurrence of a ShardedDim.

        Parameters:
        -----------
        value : ShardedDim
            The ShardedDim to find

        Returns:
        --------
        int
            Index of the first occurrence

        Raises:
        -------
        TypeError
            If value is not a ShardedDim object
        ValueError
            If the dimension is not found
        """
        if not isinstance(value, ShardedDim):
            raise TypeError(
                f"Can only find index of ShardedDim objects, not {type(value)}"
            )

        for i, sdim in enumerate(self._sharded_dims):
            if sdim.dim == value.dim and sdim.shard == value.shard:
                return i
        raise ValueError(f"ShardedDim {value} not found in tensor")

    def expand(self, *sizes):
        """
        Returns a new tensor with singleton dimensions expanded to the specified size.

        This method allows expanding dimensions of size 1 (singleton dimensions) to larger sizes.
        Dimensions that are not size 1 cannot be expanded to a different size.
        If -1 is specified for a dimension, it remains unchanged.

        Parameters:
        -----------
        *sizes : ints or tuple of ints
            The desired size for each dimension. If -1, the size remains unchanged.

        Returns:
        --------
        MetaTensor
            A new tensor with expanded dimensions

        Raises:
        -------
        ValueError
            If trying to expand a non-singleton dimension to a different size,
            or if the number of dimensions doesn't match
        """
        # Handle the case where a single tuple/list is provided
        if len(sizes) == 1 and isinstance(sizes[0], (tuple, list)):
            sizes = sizes[0]

        # Convert sizes to a list
        sizes = list(sizes)

        # Check if the number of dimensions matches
        if len(sizes) != len(self):
            raise ValueError(
                f"expand: Invalid number of dimensions: got {len(sizes)} but tensor has {len(self)}"
            )

        # Create a new shape, preserving dimensions that shouldn't change
        new_shape = []
        for i, (size, sdim) in enumerate(zip(sizes, self._sharded_dims)):
            # If size is -1, keep the original dimension size
            if size == -1:
                new_shape.append(sdim.dim)
            # If the original dimension is 1, expand it to the new size
            elif sdim.dim == 1:
                new_shape.append(size)
            # If trying to change a non-singleton dimension to a different size
            elif size != sdim.dim:
                raise ValueError(
                    f"expand: trying to expand dimension {i} from {sdim.dim} to {size}"
                )
            # Keep the same size if the dimension matches
            else:
                new_shape.append(sdim.dim)

        # Create the new expanded tensor, preserving sharding where possible
        expanded = MetaTensor(new_shape, model_id=self.model_id)

        # Handle sharding for the expanded tensor:
        # - For dimensions that were not expanded, preserve the original sharding
        # - For expanded dimensions (from 1 to larger size), use no sharding (shard=1)
        new_shard_spec = []
        for i, (orig_dim, new_dim, orig_shard) in enumerate(
            zip(self.shape, new_shape, self.shard_spec)
        ):
            if orig_dim == new_dim:
                # Dimension didn't change, keep original sharding
                new_shard_spec.append(orig_shard)
            else:
                # Dimension was expanded from 1, use no sharding
                new_shard_spec.append(1)

        # Apply sharding specification to the expanded tensor
        expanded.shard_spec = new_shard_spec

        return expanded

    def clone(self):
        """
        Return a shallow copy of the tensor.

        Returns:
        --------
        MetaTensor
            A new tensor with the same ShardedDim objects
        """
        return clone(self)

    def unsqueeze(self, dim):
        """
        Returns a new tensor with a dimension of size one inserted at the
        specified position.

        This method wraps the module-level unsqueeze function, providing
        a more convenient object-oriented interface.

        Parameters:
        -----------
        dim : int
            The dimension to insert. Can be negative (counting from the end)

        Returns:
        --------
        MetaTensor
            A tensor with an additional dimension of size 1

        Raises:
        -------
        IndexError
            If dimension index is out of range
        """
        return unsqueeze(self, dim)

    def squeeze(self, dim=None):
        """
        Returns a new tensor with all singleton dimensions removed.

        This method wraps the module-level squeeze function, providing
        a more convenient object-oriented interface.

        Parameters:
        -----------
        dim : int, optional
            If specified, only removes the singleton dimension at the given index

        Returns:
        --------
        MetaTensor
            A tensor with singleton dimensions removed

        Raises:
        -------
        IndexError
            If dimension index is out of range
        """
        return squeeze(self, dim)

    def permute(self, *dims):
        """
        Returns a view of the original tensor with its dimensions permuted.

        Parameters:
        -----------
        *dims : ints
            The desired ordering of dimensions

        Returns:
        --------
        MetaTensor
            Tensor with permuted dimensions

        Raises:
        -------
        ValueError
            If dimensions don't match or contain duplicates
        """
        return permute(self, dims)

    def transpose(self, dim0, dim1):
        """
        Returns a tensor that is a transposed version of this tensor.

        The given dimensions are swapped.

        Parameters:
        -----------
        dim0 : int
            First dimension to be transposed
        dim1 : int
            Second dimension to be transposed

        Returns:
        --------
        MetaTensor
            Transposed tensor

        Raises:
        -------
        IndexError
            If dimensions are out of range
        """

        return transpose(self, dim0, dim1)

    def reshape(self, *shape):
        """
        Returns a new tensor with the same data but different shape.

        This method wraps the module-level reshape function, providing
        a more convenient object-oriented interface.

        Parameters:
        -----------
        *shape : ints or tuple of ints
            The desired shape of the new tensor

        Returns:
        --------
        MetaTensor
            A tensor with the same data but different shape

        Raises:
        -------
        ValueError
            If the total number of elements doesn't match
        """
        return reshape(self, shape)

    def split(self, split_size_or_sections, dim=0):
        """
        Splits the tensor into multiple sub-tensors along the specified dimension.

        Parameters:
        -----------
        split_size_or_sections : int or list of ints
            If int, the size of each split. If list, the sizes of each split.
        dim : int, optional
            The dimension along which to split the tensor (default is 0).

        Returns:
        --------
        list of MetaTensor
            A list of sub-tensors resulting from the split.
        """
        return split(self, split_size_or_sections, dim)

    def concat(self, tensors, dim=0, out=None):
        """
        Concatenates a sequence of tensors along the specified dimension.

        Parameters:
        -----------
        tensors : list of MetaTensor
            The tensors to concatenate
        dim : int, optional
            The dimension along which to concatenate (default is 0)

        Returns:
        --------
        MetaTensor
            A tensor resulting from the concatenation
        """
        return concat([self] + tensors, dim, out)

    def __add__(self, other):
        """
        Overload + operator for element-wise addition.

        Parameters:
        -----------
        other : MetaTensor, numeric, list of ShardedDim, or ShardedDim
            The tensor, value, or dimensions to add

        Returns:
        --------
        MetaTensor
            Result of self + other

        Raises:
        -------
        ValueError
            If tensors have different non-default model_ids
        """
        # Import here to avoid circular imports
        from flagscale.runner.estimator.meta_modules import Elementwise

        # Handle scalar values
        if isinstance(other, (int, float)):
            # For scalar addition, create a tensor with the same shape as self
            # This is a simplification - in practice, broadcasting logic would be here
            scalar_tensor = self.clone()
            # In a real implementation, you'd fill this with the scalar value
            return Elementwise(operation="add", model_id=self.model_id).forward(
                self, scalar_tensor
            )

        # Handle tensor addition
        elif isinstance(other, MetaTensor):
            # Use helper function to check model IDs
            target_model_id = _check_model_ids(self, other, "addition")

            # Perform the addition with the determined model_id
            return Elementwise(operation="add", model_id=target_model_id).forward(
                self, other
            )

        # Handle adding a list of ShardedDim objects (tensor concatenation)
        elif isinstance(other, list):
            # Check if all elements are ShardedDim objects
            if all(isinstance(item, ShardedDim) for item in other):
                # Create a new tensor with the additional dimensions
                result_shape = self.shape + [item.dim for item in other]
                result_shard_spec = self.shard_spec + [item.shard for item in other]
                return MetaTensor(
                    result_shape, result_shard_spec, model_id=self.model_id
                )
            else:
                raise TypeError(
                    "When adding a list to MetaTensor, all elements must be ShardedDim objects"
                )

        # Handle adding a single ShardedDim (adding a dimension)
        elif isinstance(other, ShardedDim):
            # Create a new tensor with the additional dimension
            result_shape = self.shape + [other.dim]
            result_shard_spec = self.shard_spec + [other.shard]
            return MetaTensor(result_shape, result_shard_spec, model_id=self.model_id)

        # Handle unsupported types
        else:
            raise TypeError(f"Unsupported operand type for +: {type(other)}")

    def __sub__(self, other):
        """
        Overload - operator for element-wise subtraction.

        Parameters:
        -----------
        other : MetaTensor or numeric
            The tensor or value to subtract

        Returns:
        --------
        MetaTensor
            Result of self - other

        Raises:
        -------
        ValueError
            If tensors have different non-default model_ids
        """
        # Import here to avoid circular imports
        from flagscale.runner.estimator.meta_modules import Elementwise

        # Handle scalar values
        if isinstance(other, (int, float)):
            # For scalar subtraction, create a tensor with the same shape as self
            scalar_tensor = self.clone()
            # In a real implementation, you'd fill this with the scalar value
            return Elementwise(operation="sub", model_id=self.model_id).forward(
                self, scalar_tensor
            )

        # Handle tensor subtraction
        elif isinstance(other, MetaTensor):
            # Use helper function to check model IDs
            target_model_id = _check_model_ids(self, other, "subtraction")

            # Perform the subtraction with the determined model_id
            return Elementwise(operation="sub", model_id=target_model_id).forward(
                self, other
            )

        # Handle unsupported types
        else:
            raise TypeError(f"Unsupported operand type for -: {type(other)}")

    def __mul__(self, other):
        """
        Overload * operator for element-wise multiplication.

        Parameters:
        -----------
        other : MetaTensor or numeric
            The tensor or value to multiply by

        Returns:
        --------
        MetaTensor
            Result of self * other

        Raises:
        -------
        ValueError
            If tensors have different non-default model_ids
        """
        # Import here to avoid circular imports
        from flagscale.runner.estimator.meta_modules import Elementwise

        # Handle scalar values
        if isinstance(other, (int, float)):
            # For scalar multiplication, create a tensor with the same shape as self
            scalar_tensor = self.clone()
            # In a real implementation, you'd fill this with the scalar value
            return Elementwise(operation="mul", model_id=self.model_id).forward(
                self, scalar_tensor
            )

        # Handle tensor multiplication
        elif isinstance(other, MetaTensor):
            # Use helper function to check model IDs
            target_model_id = _check_model_ids(self, other, "multiplication")

            # Perform the multiplication with the determined model_id
            return Elementwise(operation="mul", model_id=target_model_id).forward(
                self, other
            )

        # Handle unsupported types
        else:
            raise TypeError(f"Unsupported operand type for *: {type(other)}")

    def __truediv__(self, other):
        """
        Overload / operator for element-wise division.

        Parameters:
        -----------
        other : MetaTensor or numeric
            The tensor or value to divide by

        Returns:
        --------
        MetaTensor
            Result of self / other

        Raises:
        -------
        ValueError
            If tensors have different non-default model_ids
        ZeroDivisionError
            If dividing by zero
        TypeError
            If operand type is unsupported
        """
        # Import here to avoid circular imports
        from flagscale.runner.estimator.meta_modules import Elementwise

        # Handle scalar values
        if isinstance(other, (int, float)):
            # Check for division by zero
            if other == 0:
                raise ZeroDivisionError("division by zero")

            # For scalar division, create a tensor with the same shape as self
            scalar_tensor = self.clone()
            # In a real implementation, you'd fill this with the scalar value
            return Elementwise(operation="div", model_id=self.model_id).forward(
                self, scalar_tensor
            )

        # Handle tensor division
        elif isinstance(other, MetaTensor):
            # Use helper function to check model IDs
            target_model_id = _check_model_ids(self, other, "division")

            # In a real implementation, we'd check for zeros in the tensor
            return Elementwise(operation="div", model_id=target_model_id).forward(
                self, other
            )

        # Handle unsupported types
        else:
            raise TypeError(f"Unsupported operand type for /: {type(other)}")

    def __rtruediv__(self, other):
        """
        Reverse division, called when left operand doesn't support division.

        Parameters:
        -----------
        other : numeric or MetaTensor
            The value to divide by self

        Returns:
        --------
        MetaTensor
            Result of other / self

        Raises:
        -------
        ValueError
            If tensors have different non-default model_ids
        ZeroDivisionError
            If dividing by zero (i.e., if self contains zeros)
        TypeError
            If operand type is unsupported
        """
        # Import here to avoid circular imports
        from flagscale.runner.estimator.meta_modules import Elementwise

        # Handle numeric types (int, float)
        if isinstance(other, (int, float)):
            # Check for division by zero in the numerator
            if other == 0:
                raise ZeroDivisionError("division by zero")

            scalar_tensor = self.clone()
            # Swap order for reverse division
            return Elementwise(operation="div", model_id=self.model_id).forward(
                scalar_tensor, self
            )

        # Handle MetaTensor division
        elif isinstance(other, MetaTensor):
            # Use helper function to check model IDs (note the order is reversed for rtruediv)
            target_model_id = _check_model_ids(other, self, "division")

            # In a real implementation, we'd check for zeros in self (denominator)
            return Elementwise(operation="div", model_id=target_model_id).forward(
                other, self
            )

        # Handle unsupported types
        else:
            raise TypeError(f"Unsupported operand type for /: {type(other)}")

    def __radd__(self, other):
        """
        Reverse add operation, called when left operand doesn't support addition.

        Parameters:
        -----------
        other : numeric
            The value to add

        Returns:
        --------
        MetaTensor
            Result of other + self
        """
        # Addition is commutative, so we can just call __add__
        return self.__add__(other)

    def __rsub__(self, other):
        """
        Reverse subtraction, called when left operand doesn't support subtraction.

        Parameters:
        -----------
        other : numeric
            The value from which to subtract self

        Returns:
        --------
        MetaTensor
            Result of other - self
        """
        # Import here to avoid circular imports
        from flagscale.runner.estimator.meta_modules import Elementwise

        # Only numeric types would use __rsub__, create a tensor with the same shape as self
        if isinstance(other, (int, float)):
            scalar_tensor = self.clone()
            # Swap order for reverse subtraction
            return Elementwise(operation="sub", model_id=self.model_id).forward(
                scalar_tensor, self
            )
        else:
            raise TypeError(f"Unsupported operand type for -: {type(other)}")

    def __rmul__(self, other):
        """
        Reverse multiplication, called when left operand doesn't support multiplication.

        Parameters:
        -----------
        other : numeric
            The value to multiply with self

        Returns:
        --------
        MetaTensor
            Result of other * self
        """
        # Multiplication is commutative, so we can just call __mul__
        return self.__mul__(other)

    def __eq__(self, other):
        """
        Check equality with another object.

        Two MetaTensor objects are equal if they have the same dimensions
        with the same sharding.

        Parameters:
        -----------
        other : MetaTensor or any
            Object to compare with

        Returns:
        --------
        bool
            True if equal, False otherwise
        """
        if isinstance(other, MetaTensor):
            return len(self._sharded_dims) == len(other._sharded_dims) and all(
                a == b for a, b in zip(self._sharded_dims, other._sharded_dims)
            )
        return False

    def matmul(self, other):
        """
        Matrix multiplication of two tensors.

        Parameters:
        -----------
        other : MetaTensor
            The tensor to multiply with

        Returns:
        --------
        MetaTensor
            Result of matrix multiplication
        """
        from flagscale.runner.estimator.meta_functional import matmul

        return matmul(self, other)


def _check_model_ids(tensor1, tensor2, operation_name):
    """
    Check if tensors have consistent model_ids and raise error if not.

    Parameters:
    -----------
    tensor1 : MetaTensor
        First tensor
    tensor2 : MetaTensor
        Second tensor
    operation_name : str
        Name of the operation being performed (for error messages)

    Returns:
    --------
    str
        The resolved model_id to use

    Raises:
    -------
    ValueError
        If tensors have different non-default model_ids
    """
    # Determine which model_id to use
    target_model_id = tensor1.model_id

    # If this tensor has default model_id but the other doesn't, use the other's
    if target_model_id == "default" and tensor2.model_id != "default":
        target_model_id = tensor2.model_id
    # If models are different and neither is default, raise error
    elif (
        target_model_id != "default"
        and tensor2.model_id != "default"
        and target_model_id != tensor2.model_id
    ):
        raise ValueError(
            f"Cannot perform {operation_name} on tensors with different model_ids: "
            f"'{tensor1.model_id}' and '{tensor2.model_id}'. "
            f"Either use consistent model_ids or explicitly set one tensor to use 'default'."
        )

    return target_model_id


def clone(input, *, memory_format=None):
    """
    Returns a copy of the input tensor.

    Creates a new tensor with the same shape and sharding specifications
    as the input tensor.

    Parameters:
    -----------
    input : MetaTensor
        Tensor to clone
    memory_format : optional
        Parameter included for API compatibility (not used)

    Returns:
    --------
    MetaTensor
        A new tensor with the same shape and sharding as the input

    Raises:
    -------
    TypeError
        If input is not a MetaTensor
    """
    # Validate input is a MetaTensor
    if not isinstance(input, MetaTensor):
        raise TypeError(f"Expected a MetaTensor, got {type(input)}")

    # Create a new tensor by copying each ShardedDim
    new_tensor = MetaTensor.__new__(MetaTensor)
    new_tensor._sharded_dims = [
        ShardedDim(sdim.dim, sdim.shard) for sdim in input._sharded_dims
    ]
    # Copy the model_id from the input tensor
    new_tensor._model_id = input._model_id

    return new_tensor


def unsqueeze(input: MetaTensor, dim: int):
    """
    Returns a new tensor with a dimension of size one inserted at the
    specified position.

    The returned tensor shares the same underlying data with this tensor,
    but with a changed shape.

    Parameters:
    -----------
    input : MetaTensor
        Input tensor
    dim : int
        The dimension to insert. Can be negative (counting from the end)

    Returns:
    --------
    MetaTensor
        A tensor with an additional dimension of size 1

    Raises:
    -------
    IndexError
        If dimension index is out of range
    """
    # Validate input is a MetaTensor
    if not isinstance(input, MetaTensor):
        raise TypeError(f"Expected a MetaTensor, got {type(input)}")

    # Get the current number of dimensions
    ndim = len(input)

    # Handle negative dimension index
    if dim < 0:
        dim = ndim + dim + 1

    # Validate the dimension index
    if dim < 0 or dim > ndim:
        raise IndexError(
            f"Dimension out of range (expected to be in range of [-{ndim+1}, {ndim}], but got {dim})"
        )

    # Create new shape with an additional dimension
    new_shape = list(input.shape)
    new_shape.insert(dim, 1)

    # Create new shard specification - the new dimension has no sharding (shard=1)
    new_shard_spec = list(input.shard_spec)
    new_shard_spec.insert(dim, 1)

    # Create the unsqueezed tensor
    unsqueezed = MetaTensor(new_shape, new_shard_spec, model_id=input.model_id)

    return unsqueezed


def squeeze(input: MetaTensor, dim: Optional[Union[int, List[int]]] = None):
    """
    Returns a new tensor with all singleton dimensions (size 1) removed,
    or with specified singleton dimensions removed if dim is provided.

    Parameters:
    -----------
    input : MetaTensor
        Input tensor
    dim : int or list of ints, optional
        If specified, only the dimension(s) of size 1 at the given position(s)
        are removed. If None (default), all dimensions of size 1 are removed.

    Returns:
    --------
    MetaTensor
        A tensor with specified or all singleton dimensions removed

    Raises:
    -------
    IndexError
        If a specified dimension is out of range
    ValueError
        If a specified dimension is not of size 1
    """
    # Validate input is a MetaTensor
    if not isinstance(input, MetaTensor):
        raise TypeError(f"Expected a MetaTensor, got {type(input)}")

    # Get the current shape and number of dimensions
    shape = input.shape
    ndim = len(shape)

    # Handle the case where dim is None (squeeze all singleton dimensions)
    if dim is None:
        # Find all dimensions of size 1
        dims_to_squeeze = [i for i, size in enumerate(shape) if size == 1]
    else:
        # Convert single integer to list
        if isinstance(dim, int):
            dims_to_squeeze = [dim]
        else:
            dims_to_squeeze = list(dim)

        # Handle negative indices and validate dimensions
        for i in range(len(dims_to_squeeze)):
            if dims_to_squeeze[i] < 0:
                dims_to_squeeze[i] = ndim + dims_to_squeeze[i]

            # Validate the dimension index
            if dims_to_squeeze[i] < 0 or dims_to_squeeze[i] >= ndim:
                raise IndexError(
                    f"Dimension out of range (expected to be in range of [{-ndim}, {ndim-1}], but got {dims_to_squeeze[i]})"
                )

            # Check that the dimension is actually of size 1
            if shape[dims_to_squeeze[i]] != 1:
                raise ValueError(
                    f"Cannot squeeze dimension {dims_to_squeeze[i]} with size {shape[dims_to_squeeze[i]]} (must be 1)"
                )

    # Sort dimensions in descending order to avoid index shifting issues
    dims_to_squeeze.sort(reverse=True)

    # Create new shape and sharding specification without the squeezed dimensions
    new_shape = []
    new_shard_spec = []

    for i in range(ndim):
        if i not in dims_to_squeeze:
            new_shape.append(shape[i])
            new_shard_spec.append(input.shard_spec[i])

    # If all dimensions were squeezed, create a scalar (0-dimensional) tensor
    # represented as a 1-dimensional tensor with size 1
    if not new_shape:
        new_shape = [1]
        new_shard_spec = [1]

    # Create the squeezed tensor
    squeezed = MetaTensor(new_shape, new_shard_spec, model_id=input.model_id)

    return squeezed


def permute(input: MetaTensor, dims):
    """
    Returns a new tensor with the dimensions permuted according to the specified order.

    Parameters:
    -----------
    input : MetaTensor
        Input tensor
    dims : tuple or list of ints
        The desired ordering of dimensions

    Returns:
    --------
    MetaTensor
        A tensor with dimensions permuted according to the specified order

    Raises:
    -------
    TypeError
        If input is not a MetaTensor or dims is not a valid sequence
    ValueError
        If the number of dimensions in dims doesn't match the input tensor,
        or if dims contains duplicates or out-of-range values
    """
    # Validate input is a MetaTensor
    if not isinstance(input, MetaTensor):
        raise TypeError(f"Expected a MetaTensor, got {type(input)}")

    # Handle the case where dims is passed as a list or tuple
    if not isinstance(dims, (list, tuple)):
        raise TypeError(f"Expected dims to be a list or tuple, got {type(dims)}")

    # Convert dims to list
    dims = list(dims)

    # Get the number of dimensions in the input tensor
    ndim = len(input)

    # Validate dimensions
    if len(dims) != ndim:
        raise ValueError(
            f"Number of dimensions in permutation ({len(dims)}) must match the number of dimensions in tensor ({ndim})"
        )

    # Check for duplicates
    if len(set(dims)) != len(dims):
        raise ValueError("Repeated dim in permute")

    # Verify all dimensions are valid and handle negative indices
    for i in range(len(dims)):
        if dims[i] < -ndim or dims[i] >= ndim:
            raise IndexError(
                f"Dimension out of range (expected to be in range of [{-ndim}, {ndim-1}], but got {dims[i]})"
            )
        # Convert negative indices to positive
        if dims[i] < 0:
            dims[i] = ndim + dims[i]

    # Create new shape and sharding specification based on the permutation
    new_shape = [input.shape[i] for i in dims]
    new_shard_spec = [input.shard_spec[i] for i in dims]

    # Create the permuted tensor
    permuted = MetaTensor(new_shape, new_shard_spec, model_id=input.model_id)

    return permuted


def transpose(input: MetaTensor, dim0: int, dim1: int):
    """
    Returns a new tensor that is a transposed version of the input tensor.
    The dimensions dim0 and dim1 are swapped.

    This function provides a standalone version of the tensor.transpose method.

    Parameters:
    -----------
    input : MetaTensor
        Input tensor
    dim0 : int
        First dimension to be transposed
    dim1 : int
        Second dimension to be transposed

    Returns:
    --------
    MetaTensor
        A tensor with the dimensions transposed

    Raises:
    -------
    TypeError
        If input is not a MetaTensor
    IndexError
        If dimensions are out of range
    """
    # Validate input is a MetaTensor
    if not isinstance(input, MetaTensor):
        raise TypeError(f"Expected a MetaTensor, got {type(input)}")

    # Get the number of dimensions
    ndim = len(input)

    # Handle negative indices
    if dim0 < 0:
        dim0 = ndim + dim0
    if dim1 < 0:
        dim1 = ndim + dim1

    # Validate the dimension indices
    if dim0 < 0 or dim0 >= ndim:
        raise IndexError(
            f"Dimension 0 out of range (expected to be in range of [{-ndim}, {ndim-1}], but got {dim0})"
        )
    if dim1 < 0 or dim1 >= ndim:
        raise IndexError(
            f"Dimension 1 out of range (expected to be in range of [{-ndim}, {ndim-1}], but got {dim1})"
        )

    # Create a permutation that swaps the specified dimensions
    dims = list(range(ndim))
    dims[dim0], dims[dim1] = dims[dim1], dims[dim0]

    # Use permute to create the transposed tensor
    return permute(input, dims)


def reshape(input: MetaTensor, shape):
    """
    Returns a new tensor with the same data but a different shape.

    The reshape function handles two key cases:
    1. When input is completely unsharded: arbitrary reshaping is allowed
    2. When input has sharding: reshaping follows these rules:
       - Allowed: Merging a sharded dim with unsharded dims
       - Allowed: Splitting a sharded dim into multiple dims (sharding preserved on first dim)
       - Not allowed: Merging two or more sharded dims

    Parameters:
    -----------
    input : MetaTensor
        Input tensor
    shape : tuple or list
        New shape

    Returns:
    --------
    MetaTensor
        A tensor with the same data but reshaped

    Raises:
    -------
    TypeError
        If input is not a MetaTensor
    ValueError
        If the new shape is incompatible with the original shape,
        or if trying to merge multiple sharded dimensions
    """
    # Validate input is a MetaTensor
    if not isinstance(input, MetaTensor):
        raise TypeError(f"Expected a MetaTensor, got {type(input)}")

    # Convert shape to a list if it's not already
    if isinstance(shape, int):
        shape = [shape]
    elif isinstance(shape, tuple):
        shape = list(shape)

    # Handle -1 in the new shape (infer dimension)
    if -1 in shape:
        # Find the index of -1
        neg_idx = shape.index(-1)
        # Calculate the product of all other dimensions
        known_product = 1
        for i, dim_size in enumerate(shape):
            if i != neg_idx and dim_size != 0:
                known_product *= dim_size

        # Calculate the size of the inferred dimension
        total_elements = input.total_elements(apply_sharding=False)
        if total_elements % known_product != 0:
            raise ValueError(
                f"Cannot reshape tensor of size {input.shape} to size {shape}: "
                f"{total_elements} is not divisible by {known_product}"
            )

        # Set the inferred dimension
        shape[neg_idx] = total_elements // known_product

    # Check if the total number of elements matches
    new_total = 1
    for s in shape:
        new_total *= s
    if new_total != input.total_elements(apply_sharding=False):
        raise ValueError(
            f"Cannot reshape tensor of size {input.shape} to size {shape}: "
            f"total elements don't match ({input.total_elements(apply_sharding=False)} != {new_total})"
        )

    # Case 1: Input is completely unsharded - arbitrary reshaping is allowed
    if all(shard == 1 for shard in input.shard_spec):
        return MetaTensor(shape, model_id=input.model_id)

    # Case 2: Input has sharding - need to track sharded dimensions carefully

    # Analyze input dimensions to identify sharded ones
    sharded_dims = [i for i, shard in enumerate(input.shard_spec) if shard > 1]

    # Map from input positions to output positions
    input_pos = 0
    output_pos = 0
    output_shard_spec = [1] * len(shape)
    input_shape = input.shape
    input_shard_spec = input.shard_spec

    remaining_input_elements = input.total_elements(apply_sharding=False)
    elements_in_current_output_dim = 1

    # Track whether we've merged sharded dimensions
    merged_sharded_dim_indices = []

    while input_pos < len(input_shape) and output_pos < len(shape):
        input_dim_size = input_shape[input_pos]
        output_dim_size = shape[output_pos]

        # Case 2.1: Input and output dimensions match exactly
        if input_dim_size == output_dim_size:
            # Transfer sharding directly
            output_shard_spec[output_pos] = input_shard_spec[input_pos]
            input_pos += 1
            output_pos += 1
            remaining_input_elements //= input_dim_size
            elements_in_current_output_dim = 1

        # Case 2.2: Input dimension is smaller than output (merging multiple dims into one)
        elif input_dim_size < output_dim_size:
            # We're combining multiple input dimensions into one output dimension
            elements_in_current_output_dim *= input_dim_size

            # If this is a sharded dimension being merged, track it
            if input_shard_spec[input_pos] > 1:
                merged_sharded_dim_indices.append(input_pos)

            # If we've accumulated enough elements for this output dimension
            if elements_in_current_output_dim == output_dim_size:
                # If we merged multiple sharded dimensions, raise error
                if len(merged_sharded_dim_indices) > 1:
                    raise ValueError(
                        f"Cannot reshape tensor: merging multiple sharded dimensions is not supported. "
                        f"Tried to merge sharded dimensions at indices {merged_sharded_dim_indices}"
                    )
                # If we merged exactly one sharded dimension, transfer its sharding
                elif len(merged_sharded_dim_indices) == 1:
                    output_shard_spec[output_pos] = input_shard_spec[
                        merged_sharded_dim_indices[0]
                    ]

                    # Verify the output dimension is divisible by its shard factor
                    if output_dim_size % output_shard_spec[output_pos] != 0:
                        raise ValueError(
                            f"Cannot reshape tensor: output dimension {output_pos} of size {output_dim_size} "
                            f"is not divisible by shard factor {output_shard_spec[output_pos]}"
                        )

                # Move to next output dimension
                output_pos += 1
                elements_in_current_output_dim = 1
                merged_sharded_dim_indices = []

            input_pos += 1
            remaining_input_elements //= input_dim_size

        # Case 2.3: Input dimension is larger than output (splitting one dim into multiple)
        else:  # input_dim_size > output_dim_size
            # We're splitting one input dimension across multiple output dimensions
            if input_dim_size % output_dim_size != 0:
                raise ValueError(
                    f"Cannot reshape tensor: input dimension {input_pos} of size {input_dim_size} "
                    f"is not divisible by output dimension {output_pos} of size {output_dim_size}"
                )

            # If splitting a sharded dimension:
            if input_shard_spec[input_pos] > 1:
                # Apply sharding to the first output dimension from this split
                shard_factor = input_shard_spec[input_pos]

                # Verify the output dimension is divisible by the shard factor
                if output_dim_size % shard_factor != 0:
                    raise ValueError(
                        f"Cannot reshape tensor: output dimension {output_pos} of size {output_dim_size} "
                        f"is not divisible by shard factor {shard_factor}"
                    )

                # Apply sharding to the first output dimension from this split
                output_shard_spec[output_pos] = shard_factor

                # Track how many remaining elements we need to handle from this dimension
                remaining_from_this_input = input_dim_size // output_dim_size - 1

                # Move to next output position for the current split
                output_pos += 1
                remaining_input_elements //= output_dim_size

                # Handle all remaining positions from the current split
                # (these will all be unsharded - shard=1)
                temp_pos = output_pos
                for _ in range(remaining_from_this_input):
                    if temp_pos < len(shape):
                        # No sharding for dimensions beyond the first
                        output_shard_spec[temp_pos] = 1
                        temp_pos += 1
            else:
                # For unsharded dimensions, no special handling needed
                output_pos += 1
                remaining_input_elements //= output_dim_size

            # Only move to the next input dimension when we've processed all output dims
            # that came from the current input dim
            if remaining_input_elements % input_dim_size == 0:
                input_pos += 1

    # Create the reshaped tensor with appropriate sharding
    reshaped = MetaTensor(shape, output_shard_spec, model_id=input.model_id)

    return reshaped


def split(tensor: MetaTensor, split_size_or_sections, dim=0):
    """
    Splits the tensor into multiple sub-tensors along the specified dimension.

    Parameters:
    -----------
    tensor : MetaTensor
        The tensor to split.
    split_size_or_sections : int or list of ints
        If int, the size of each split. If list, the sizes of each split.
    dim : int, optional
        The dimension along which to split the tensor (default is 0).

    Returns:
    --------
    list of MetaTensor
        A list of sub-tensors resulting from the split.
    """
    if isinstance(split_size_or_sections, int):
        if tensor.shape[dim] % split_size_or_sections != 0:
            raise ValueError(
                "The tensor cannot be evenly split along the specified dimension."
            )
        # Calculate the number of splits
        num_splits = tensor.shape[dim] // split_size_or_sections
        split_sizes = [split_size_or_sections] * num_splits
    else:
        # Use the provided list of split sizes
        split_sizes = split_size_or_sections

    # Validate the split sizes
    if sum(split_sizes) != tensor.shape[dim]:
        raise ValueError(
            "The sum of split sizes must equal the size of the specified dimension."
        )

    # Create the sub-tensors
    sub_tensors = []
    for size in split_sizes:
        new_shape = tensor.shape[:dim] + [size] + tensor.shape[dim + 1 :]
        new_shard_spec = (
            tensor.shard_spec[:dim]
            + [tensor.shard_spec[dim]]
            + tensor.shard_spec[dim + 1 :]
        )
        sub_tensor = MetaTensor(new_shape, new_shard_spec, model_id=tensor.model_id)
        sub_tensors.append(sub_tensor)

    return sub_tensors


def concat(tensors, dim=0, out=None):
    """
    Concatenates the given sequence of tensors in the given dimension.

    Parameters:
    -----------
    tensors : sequence of MetaTensor
        Sequence of tensors to concatenate
    dim : int, optional
        Dimension along which to concatenate (default: 0)
    out : MetaTensor, optional
        Output tensor (not used for MetaTensor, included for API compatibility)

    Returns:
    --------
    MetaTensor
        Concatenated tensor

    Raises:
    -------
    TypeError
        If tensors are not all MetaTensor objects
    ValueError
        If tensors have different number of dimensions or mismatched dimensions
        other than the concatenation dimension, or if tensors have inconsistent model_ids
    """
    # Validate input is a sequence
    if not isinstance(tensors, (list, tuple)):
        raise TypeError(f"Expected a sequence of tensors, got {type(tensors)}")

    # Handle empty sequence
    if len(tensors) == 0:
        raise ValueError("Cannot concatenate empty sequence of tensors")

    # Handle single tensor
    if len(tensors) == 1:
        return tensors[0].clone()

    # Check all inputs are MetaTensors
    if not all(isinstance(t, MetaTensor) for t in tensors):
        raise TypeError("All tensors must be MetaTensor objects")

    # Determine which model_id to use for the output tensor
    target_model_id = "default"
    model_ids = {}

    # Collect model_ids from input tensors
    for i, tensor in enumerate(tensors):
        if tensor.model_id != "default":
            if tensor.model_id not in model_ids:
                model_ids[tensor.model_id] = []
            model_ids[tensor.model_id].append(i)

    # If there are non-default model_ids
    if model_ids:
        if len(model_ids) == 1:
            # If all tensors with non-default model_id have the same one, use that
            target_model_id = next(iter(model_ids.keys()))
        else:
            # If there are multiple different model_ids, raise an error
            model_id_details = []
            for mid, indices in model_ids.items():
                tensor_indices = ", ".join(str(idx) for idx in indices)
                model_id_details.append(
                    f"  - '{mid}' from tensors at indices [{tensor_indices}]"
                )

            raise ValueError(
                f"Cannot concatenate tensors with different non-default model_ids:\n"
                f"{chr(10).join(model_id_details)}\n"
                f"Either use consistent model_ids or explicitly set all tensors to use 'default'."
            )

    # Get the number of dimensions from the first tensor
    ndim = len(tensors[0])

    # Convert negative dimension index to positive
    if dim < 0:
        dim = ndim + dim

    # Validate the dimension index
    if dim < 0 or dim >= ndim:
        raise IndexError(
            f"Dimension out of range (expected to be in range of [-{ndim}, {ndim-1}], but got {dim})"
        )

    # Check all tensors have the same number of dimensions
    if not all(len(t) == ndim for t in tensors):
        raise ValueError("All tensors must have the same number of dimensions")

    # Check all dimensions except concatenation dimension match
    for i in range(ndim):
        if i != dim:
            sizes = [t.shape[i] for t in tensors]
            shards = [t.shard_spec[i] for t in tensors]

            if not all(size == sizes[0] for size in sizes):
                raise ValueError(
                    f"All tensors must have the same size in non-concatenated dimensions, "
                    f"but found different sizes at dimension {i}: {sizes}"
                )

            if not all(shard == shards[0] for shard in shards):
                raise ValueError(
                    f"All tensors must have the same sharding in non-concatenated dimensions, "
                    f"but found different sharding at dimension {i}: {shards}"
                )

    # Calculate new shape and check sharding compatibility along concat dimension
    new_shape = list(tensors[0].shape)
    concat_dim_size = sum(t.shape[dim] for t in tensors)
    new_shape[dim] = concat_dim_size

    # Get sharding for the concatenated dimension
    concat_dim_shards = [t.shard_spec[dim] for t in tensors]

    # Use the same sharding factor if all tensors have the same,
    # otherwise default to 1 (no sharding)
    if all(shard == concat_dim_shards[0] for shard in concat_dim_shards):
        concat_dim_shard = concat_dim_shards[0]

        # Check if the concatenated size is divisible by this shard
        if concat_dim_size % concat_dim_shard != 0:
            concat_dim_shard = 1  # Fall back to no sharding
    else:
        concat_dim_shard = 1  # Different sharding factors, use no sharding

    # Create new shard specification
    new_shard_spec = list(tensors[0].shard_spec)
    new_shard_spec[dim] = concat_dim_shard

    # Create new tensor with concatenated dimensions and the determined model_id
    concat_tensor = MetaTensor(new_shape, new_shard_spec, model_id=target_model_id)

    return concat_tensor


def repeat_interleave(input: MetaTensor, repeats, dim=None, *, output_size=None):
    """
    Repeats elements of a tensor along a specified dimension.

    The repeats argument can be:
    - A single integer, repeating all elements the same number of times
    - A list/tensor of integers, specifying repetition for each element along the dimension

    Parameters:
    -----------
    input : MetaTensor
        Input tensor
    repeats : int or list of ints
        Number of repetitions for each element
    dim : int, optional
        Dimension along which to repeat values. If None, the tensor is flattened
        and then repeated
    output_size : int, optional
        Size of the output tensor along the given dimension (not used for MetaTensor,
        included for API compatibility)

    Returns:
    --------
    MetaTensor
        Tensor with repeated values

    Raises:
    -------
    TypeError
        If input is not a MetaTensor or repeats is not an int or list/tuple of ints
    ValueError
        If dimension is out of range or repeats contains invalid values
    """
    # Validate input is a MetaTensor
    if not isinstance(input, MetaTensor):
        raise TypeError(f"Expected a MetaTensor, got {type(input)}")

    model_id = input.model_id
    # Handle scalar repetition vs per-element repetition
    is_scalar_repeat = isinstance(repeats, int)

    # If dim is None, we're working with a flattened tensor
    if dim is None:
        # Calculate the total elements in the original tensor
        total_elements = input.total_elements(apply_sharding=False)

        if is_scalar_repeat:
            # Simple case: repeat each element the same number of times
            new_size = total_elements * repeats
        else:
            # Must provide exactly one repeat value for each element in the flattened tensor
            if not isinstance(repeats, (list, tuple)):
                raise TypeError(
                    f"Expected repeats to be int, list, or tuple, got {type(repeats)}"
                )

            if len(repeats) != total_elements:
                raise ValueError(
                    f"Expected repeats to have length {total_elements}, but got {len(repeats)}"
                )

            # Calculate new size based on sum of repetitions
            new_size = sum(repeats)

        # Create a 1D tensor with the new size
        return MetaTensor([new_size], model_id=model_id)

    # We're repeating along a specific dimension
    ndim = len(input)

    # Handle negative dimension index
    if dim < 0:
        dim = ndim + dim

    # Validate the dimension index
    if dim < 0 or dim >= ndim:
        raise IndexError(
            f"Dimension out of range (expected to be in range of [{-ndim}, {ndim-1}], but got {dim})"
        )

    # Get the size of the dimension we're repeating along
    dim_size = input.shape[dim]

    # Calculate the new dimension size
    if is_scalar_repeat:
        # Simple case: each element repeated the same number of times
        new_dim_size = dim_size * repeats
    else:
        # Complex case: different repetition for each element
        if not isinstance(repeats, (list, tuple)):
            raise TypeError(
                f"Expected repeats to be int, list, or tuple, got {type(repeats)}"
            )

        if len(repeats) != dim_size:
            raise ValueError(
                f"Expected repeats to have length {dim_size} for dimension {dim}, but got {len(repeats)}"
            )

        # Check all values are non-negative
        if any(r < 0 for r in repeats):
            raise ValueError("Repeat values must be non-negative")

        # Calculate new size based on sum of repetitions
        new_dim_size = sum(repeats)

    # Create the new shape with the updated dimension
    new_shape = list(input.shape)
    new_shape[dim] = new_dim_size

    # For sharding, if the original dimension was sharded and we're using scalar repetition,
    # we can preserve the sharding if it divides the new dimension evenly
    new_shard_spec = list(input.shard_spec)
    orig_shard = input.shard_spec[dim]

    if is_scalar_repeat and orig_shard > 1:
        # Check if new dimension size is divisible by the original sharding
        if new_dim_size % orig_shard == 0:
            # We can keep the same sharding
            new_shard_spec[dim] = orig_shard
        else:
            # Not divisible, must unshard
            new_shard_spec[dim] = 1
    else:
        # For per-element repetition or already unsharded, use no sharding
        new_shard_spec[dim] = 1

    # Create the repeated tensor
    return MetaTensor(new_shape, new_shard_spec, model_id=model_id)
