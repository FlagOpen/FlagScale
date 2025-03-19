from dataclasses import dataclass, field
from typing import Any, Dict, Optional


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

    def __init__(self, shape: list, shard_spec=None):
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
        if dim0 < 0:
            dim0 = len(self._sharded_dims) + dim0
        if dim1 < 0:
            dim1 = len(self._sharded_dims) + dim1

        if dim0 >= len(self._sharded_dims) or dim1 >= len(self._sharded_dims):
            raise IndexError(
                f"Dimension out of range (expected to be in range of [{-len(self._sharded_dims)}, {len(self._sharded_dims)-1}], but got {dim0} and {dim1})"
            )

        # Create a new tensor and swap the dimensions
        new_tensor = self.copy()
        new_tensor._sharded_dims[dim0], new_tensor._sharded_dims[dim1] = (
            new_tensor._sharded_dims[dim1],
            new_tensor._sharded_dims[dim0],
        )

        return new_tensor

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
        # Handle the case where dims is passed as a list or tuple
        if len(dims) == 1 and isinstance(dims[0], (list, tuple)):
            dims = dims[0]

        # Convert dims to list
        dims = list(dims)

        # Validate dimensions
        if len(dims) != len(self._sharded_dims):
            raise ValueError(
                f"Number of dimensions in permutation ({len(dims)}) must match the number of dimensions in tensor ({len(self._sharded_dims)})"
            )

        # Check for duplicates
        if len(set(dims)) != len(dims):
            raise ValueError("Repeated dim in permute")

        # Verify all dimensions are valid
        for d in dims:
            if d < -len(self._sharded_dims) or d >= len(self._sharded_dims):
                raise IndexError(
                    f"Dimension out of range (expected to be in range of [{-len(self._sharded_dims)}, {len(self._sharded_dims)-1}], but got {d})"
                )

        # Handle negative indices
        for i in range(len(dims)):
            if dims[i] < 0:
                dims[i] = len(self._sharded_dims) + dims[i]

        # Create a new tensor with permuted dimensions
        new_tensor = MetaTensor.__new__(MetaTensor)
        new_tensor._sharded_dims = [
            (
                self._sharded_dims[dim].copy()
                if hasattr(self._sharded_dims[dim], "copy")
                else ShardedDim(
                    self._sharded_dims[dim].dim, self._sharded_dims[dim].shard
                )
            )
            for dim in dims
        ]

        return new_tensor

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

    def append(self, value):
        """
        Append a new dimension to the end of the tensor.

        Parameters:
        -----------
        value : ShardedDim
            The dimension to append

        Raises:
        -------
        TypeError
            If value is not a ShardedDim object
        """
        if isinstance(value, ShardedDim):
            self._sharded_dims.append(value)
        else:
            raise TypeError(f"Can only append ShardedDim objects, not {type(value)}")

    def extend(self, values):
        """
        Extend the tensor by appending elements from an iterable of ShardedDim objects.

        Parameters:
        -----------
        values : MetaTensor
            MetaTensor containing ShardedDim objects to append

        Raises:
        -------
        TypeError
            If values is not a MetaTensor or contains non-ShardedDim objects
        """
        if isinstance(values, MetaTensor):
            self._sharded_dims.extend(values._sharded_dims)
        else:
            raise TypeError(f"Can only extend with MetaTensor, not {type(values)}")

    def pop(self, index=-1):
        """
        Remove and return the ShardedDim at position index.

        Parameters:
        -----------
        index : int, optional
            Position to remove (default is last item)

        Returns:
        --------
        ShardedDim
            Removed dimension
        """
        return self._sharded_dims.pop(index)

    def insert(self, index, value):
        """
        Insert a ShardedDim at a given position.

        Parameters:
        -----------
        index : int
            Position to insert at
        value : ShardedDim
            The dimension to insert

        Raises:
        -------
        TypeError
            If value is not a ShardedDim object
        """
        if isinstance(value, ShardedDim):
            self._sharded_dims.insert(index, value)
        else:
            raise TypeError(f"Can only insert ShardedDim objects, not {type(value)}")

    def remove(self, value):
        """
        Remove the first occurrence of a ShardedDim.

        Parameters:
        -----------
        value : ShardedDim
            The ShardedDim object to remove

        Raises:
        -------
        TypeError
            If value is not a ShardedDim object
        ValueError
            If the dimension is not found
        """
        if not isinstance(value, ShardedDim):
            raise TypeError(f"Can only remove ShardedDim objects, not {type(value)}")

        for i, sdim in enumerate(self._sharded_dims):
            if sdim.dim == value.dim and sdim.shard == value.shard:
                self._sharded_dims.pop(i)
                return
        raise ValueError(f"ShardedDim {value} not found in tensor")

    def clear(self):
        """Remove all ShardedDim objects from the tensor."""
        self._sharded_dims.clear()

    def copy(self):
        """
        Return a shallow copy of the tensor.

        Returns:
        --------
        MetaTensor
            A new tensor with the same ShardedDim objects
        """
        new_tensor = MetaTensor.__new__(MetaTensor)
        new_tensor._sharded_dims = [
            ShardedDim(sdim.dim, sdim.shard) for sdim in self._sharded_dims
        ]
        return new_tensor

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

    def count(self, value):
        """
        Count occurrences of a ShardedDim.

        Parameters:
        -----------
        value : ShardedDim
            The ShardedDim to count

        Returns:
        --------
        int
            Number of occurrences

        Raises:
        -------
        TypeError
            If value is not a ShardedDim object
        """
        if not isinstance(value, ShardedDim):
            raise TypeError(f"Can only count ShardedDim objects, not {type(value)}")

        return sum(
            1
            for sdim in self._sharded_dims
            if sdim.dim == value.dim and sdim.shard == value.shard
        )

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

    def __add__(self, other):
        """
        Add operation with another tensor or concatenate with ShardedDim objects.

        This method supports three different operations:
        1. Element-wise addition when adding two MetaTensor objects with identical shapes
        2. Concatenation when adding a MetaTensor with a list of ShardedDim objects
        3. Concatenation when adding a MetaTensor with a single ShardedDim object

        For element-wise addition, shapes and sharding must be compatible.
        For concatenation, a new tensor is created with combined dimensions.

        Parameters:
        -----------
        other : MetaTensor, list, or ShardedDim
            Tensor to add element-wise or concatenate,
            or list of ShardedDim objects,
            or single ShardedDim object

        Returns:
        --------
        MetaTensor
            Result of addition or concatenation

        Raises:
        -------
        TypeError
            If other is not a MetaTensor, list of ShardedDim objects, or ShardedDim
        ValueError
            If attempting element-wise addition with incompatible shapes
        """
        # Case 1: Element-wise addition of two MetaTensor objects
        if isinstance(other, MetaTensor) and len(self) == len(other):
            # Check if shapes match (element-wise addition)
            if self.shape == other.shape:
                # Create a new tensor with the same shape
                result = self.copy()

                # Verify sharding compatibility
                for i, (sdim1, sdim2) in enumerate(
                    zip(self._sharded_dims, other._sharded_dims)
                ):
                    if sdim1.shard != sdim2.shard:
                        raise ValueError(
                            f"Cannot perform element-wise addition with tensors that have different "
                            f"sharding at dimension {i}: {sdim1.shard} vs {sdim2.shard}"
                        )

                # For element-wise addition, dimensions remain the same
                return result

        # Case 2: Concatenation with a list containing ShardedDim objects
        elif isinstance(other, list):
            # Create a new tensor by concatenating dimensions
            new_tensor = self.copy()

            # Process each item in the list
            for item in other:
                if isinstance(item, ShardedDim):
                    new_tensor._sharded_dims.append(item.copy())
                elif isinstance(item, MetaTensor):
                    # Extend with all dimensions from the MetaTensor
                    for sdim in item._sharded_dims:
                        new_tensor._sharded_dims.append(sdim.copy())
                else:
                    raise TypeError(
                        f"List items must be ShardedDim or MetaTensor objects, not {type(item)}"
                    )

            return new_tensor

        # Case 3: Concatenation with a single ShardedDim object
        elif isinstance(other, ShardedDim):
            new_tensor = self.copy()
            new_tensor._sharded_dims.append(other.copy())
            return new_tensor

        # Invalid type for addition
        raise TypeError(
            f"Can only add MetaTensor with another MetaTensor, a ShardedDim, or a list of ShardedDim objects, not {type(other)}"
        )

    def __radd__(self, other):
        """
        Right-side addition to support both element-wise operations and concatenation.

        This method supports:
        1. Element-wise addition when other is a MetaTensor with matching shape
        2. Concatenation when other is a list of ShardedDim objects

        Parameters:
        -----------
        other : MetaTensor or list
            MetaTensor to add element-wise, or list of ShardedDim objects to concatenate

        Returns:
        --------
        MetaTensor
            New tensor with combined dimensions or element-wise operation result

        Raises:
        -------
        TypeError
            If other is not a compatible type for the operation
        ValueError
            If attempting element-wise addition with incompatible shapes
        """
        if isinstance(other, MetaTensor):
            # Delegate to standard add method for MetaTensor + MetaTensor
            return other + self
        else:
            raise TypeError(
                f"Right-side addition only supports MetaTensor or list of ShardedDim objects, not {type(other)}"
            )

    def __sub__(self, other):
        """
        Subtract operation with another tensor.

        For element-wise subtraction, shapes and sharding must be compatible.

        Parameters:
        -----------
        other : MetaTensor
            Tensor to subtract element-wise

        Returns:
        --------
        MetaTensor
            Result of subtraction

        Raises:
        -------
        TypeError
            If other is not a MetaTensor
        ValueError
            If attempting subtraction with incompatible shapes or sharding
        """
        if not isinstance(other, MetaTensor):
            raise TypeError(
                f"Can only subtract MetaTensor with another MetaTensor, not {type(other)}"
            )

        if len(self) != len(other):
            raise ValueError(
                f"Cannot subtract tensors with different shapes: {self.shape} vs {other.shape}"
            )

        if self.shape != other.shape:
            raise ValueError(
                f"Cannot subtract tensors with different dimensions: {self.shape} vs {other.shape}"
            )

        # Create a new tensor with the same shape
        result = self.copy()

        # Verify sharding compatibility
        for i, (sdim1, sdim2) in enumerate(
            zip(self._sharded_dims, other._sharded_dims)
        ):
            if sdim1.shard != sdim2.shard:
                raise ValueError(
                    f"Cannot perform element-wise subtraction with tensors that have different "
                    f"sharding at dimension {i}: {sdim1.shard} vs {sdim2.shard}"
                )

        # For element-wise subtraction, dimensions remain the same
        return result

    def __rsub__(self, other):
        """
        Right-side subtraction operation.

        Parameters:
        -----------
        other : MetaTensor
            Tensor from which to subtract this tensor

        Returns:
        --------
        MetaTensor
            Result of subtraction

        Raises:
        -------
        TypeError
            If other is not a MetaTensor
        """
        if isinstance(other, MetaTensor):
            return other - self
        else:
            raise TypeError(
                f"Right-side subtraction only supports MetaTensor, not {type(other)}"
            )

    def __mul__(self, other):
        """
        Element-wise multiplication with a scalar or another tensor.

        Parameters:
        -----------
        other : int, float, or MetaTensor
            Scalar or tensor to multiply with

        Returns:
        --------
        MetaTensor
            Result of multiplication

        Raises:
        -------
        TypeError
            If other is not a scalar or MetaTensor
        ValueError
            If attempting element-wise multiplication with incompatible shapes
        """
        # Handle scalar multiplication (broadcast)
        if isinstance(other, (int, float)):
            result = self.copy()
            return result

        # Handle tensor multiplication
        if isinstance(other, MetaTensor):
            if len(self) != len(other):
                raise ValueError(
                    f"Cannot multiply tensors with different shapes: {self.shape} vs {other.shape}"
                )

            if self.shape != other.shape:
                raise ValueError(
                    f"Cannot multiply tensors with different dimensions: {self.shape} vs {other.shape}"
                )

            # Create a new tensor with the same shape
            result = self.copy()

            # Verify sharding compatibility
            for i, (sdim1, sdim2) in enumerate(
                zip(self._sharded_dims, other._sharded_dims)
            ):
                if sdim1.shard != sdim2.shard:
                    raise ValueError(
                        f"Cannot perform element-wise multiplication with tensors that have different "
                        f"sharding at dimension {i}: {sdim1.shard} vs {sdim2.shard}"
                    )

            # For element-wise multiplication, dimensions remain the same
            return result

        raise TypeError(
            f"Unsupported operand type for *: 'MetaTensor' and '{type(other)}'"
        )

    def __rmul__(self, other):
        """
        Right-side multiplication to support scalar * tensor.

        Parameters:
        -----------
        other : int, float, or MetaTensor
            Scalar or tensor to multiply with

        Returns:
        --------
        MetaTensor
            Result of multiplication

        Raises:
        -------
        TypeError
            If other is not a scalar or MetaTensor
        """
        if isinstance(other, (int, float)):
            return self * other
        elif isinstance(other, MetaTensor):
            return other * self
        else:
            raise TypeError(
                f"Unsupported operand type for *: '{type(other)}' and 'MetaTensor'"
            )

    def __truediv__(self, other):
        """
        Element-wise division with a scalar or another tensor.

        Parameters:
        -----------
        other : int, float, or MetaTensor
            Scalar or tensor to divide by

        Returns:
        --------
        MetaTensor
            Result of division

        Raises:
        -------
        TypeError
            If other is not a scalar or MetaTensor
        ValueError
            If attempting element-wise division with incompatible shapes
        ZeroDivisionError
            If dividing by zero
        """
        # Handle scalar division (broadcast)
        if isinstance(other, (int, float)):
            if other == 0:
                raise ZeroDivisionError("division by zero")
            result = self.copy()
            return result

        # Handle tensor division
        if isinstance(other, MetaTensor):
            if len(self) != len(other):
                raise ValueError(
                    f"Cannot divide tensors with different shapes: {self.shape} vs {other.shape}"
                )

            if self.shape != other.shape:
                raise ValueError(
                    f"Cannot divide tensors with different dimensions: {self.shape} vs {other.shape}"
                )

            # Create a new tensor with the same shape
            result = self.copy()

            # Verify sharding compatibility
            for i, (sdim1, sdim2) in enumerate(
                zip(self._sharded_dims, other._sharded_dims)
            ):
                if sdim1.shard != sdim2.shard:
                    raise ValueError(
                        f"Cannot perform element-wise division with tensors that have different "
                        f"sharding at dimension {i}: {sdim1.shard} vs {sdim2.shard}"
                    )

            # For element-wise division, dimensions remain the same
            return result

        raise TypeError(
            f"Unsupported operand type for /: 'MetaTensor' and '{type(other)}'"
        )

    def __rtruediv__(self, other):
        """
        Right-side division to support scalar / tensor or tensor / tensor.

        Parameters:
        -----------
        other : int, float, or MetaTensor
            Dividend

        Returns:
        --------
        MetaTensor
            Result of division

        Raises:
        -------
        TypeError
            If other is not a scalar or MetaTensor
        """
        # Handle scalar / tensor division
        if isinstance(other, (int, float)):
            return self.copy()
        elif isinstance(other, MetaTensor):
            # For other_tensor / self_tensor, delegate to other's __truediv__
            # This would be handled by Python's operator dispatch, but we include it for clarity
            return other / self
        else:
            raise TypeError(
                f"Unsupported operand type for /: '{type(other)}' and 'MetaTensor'"
            )

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

    def __repr__(self):
        """
        Formal string representation for debugging.

        Returns:
        --------
        str
            Representation showing class name, shape, and shard specs
        """
        return f"MetaTensor(shape={self.shape}, shard_spec={self.shard_spec})"


class MetaModule:
    """
    Base class for model components that track computational resources.

    This class provides automatic tracking of FLOPs, parameters, and activations
    through the ModelStatsRegistry system with hierarchical level information.
    Derived classes can either provide their own implementations of compute_* methods,
    or rely on the default behavior.
    """

    _path = None
    _counter = 0

    def __init__(self, shard_specs=None, model_id="default"):
        """
        Initialize a MetaModule with sharding specifications.

        Parameters:
        -----------
        shard_specs : list
            Specifications for tensor sharding
        model_id : str, optional
            Identifier for the model, used to retrieve the correct registry
        name : str, optional
            Name of this module component (defaults to class name + counter)
        """
        self.shard_specs = shard_specs
        self.model_id = model_id
        self.registry = get_registry(model_id)
        self.shared_params = False

    def add_flops(self, *args, **kwargs):
        """
        Add FLOPs to the registry with level information.

        Parameters:
        -----------
        *args
            Positional arguments for FLOP calculation
        level : bool, optional
            Whether to include level information in the tag
        **kwargs
            Keyword arguments for FLOP calculation

        Returns:
        --------
        int
            Number of FLOPs added
        """
        return 0

    def add_params(self, *args, **kwargs):
        """
        Add parameters to the registry with level information.

        Parameters:
        -----------
        *args
            Positional arguments for parameter calculation
        level : bool, optional
            Whether to include level information in the tag
        **kwargs
            Keyword arguments for parameter calculation

        Returns:
        --------
        int
            Number of parameters added
        """
        return 0

    def add_acts(self, *args, **kwargs):
        """
        Add activation elements to the registry with level information.

        Parameters:
        -----------
        *args
            Positional arguments for activation calculation
        level : bool, optional
            Whether to include level information in the tag
        **kwargs
            Keyword arguments for activation calculation

        Returns:
        --------
        int
            Number of activation elements added
        """
        return 0

    def get_flops(self):
        """
        Get previously computed FLOPs from module's registry.

        This method returns the tracked FLOPs from the module's registry rather
        than computing them. To compute and add FLOPs to the registry, use add_flops().

        Returns:
        --------
        int or float
            Number of floating-point operations previously recorded in registry
        """
        return self.registry.total_flops

    def get_params(self):
        """
        Get previously computed parameter count from module's registry.

        This method returns the tracked parameter count from the module's registry
        rather than computing them. To compute and add parameters to the registry,
        use add_params().

        Returns:
        --------
        int
            Number of parameters previously recorded in registry
        """
        return self.registry.total_params

    def get_acts(self):
        """
        Get previously computed activation memory from module's registry.

        This method returns the tracked activation memory from the module's registry
        rather than computing them. To compute and add activations to the registry,
        use add_acts().

        Returns:
        --------
        int
            Number of activation elements previously recorded in registry
        """
        return self.registry.total_acts

    def share_params(self):
        """
        Share parameters with another module.

        This method allows sharing parameters between two modules, which is useful
        for weight tying in some models. The derived class should implement this method
        to handle the sharing logic.

        Returns:
        --------
        MetaModule
            Reference to the current module for method chaining
        """
        self.shared_params = True

    def update_registry(self, *args, **kwargs):
        """
        Update registry with computed metrics.

        This method calculates metrics using the add_* methods and adds them
        to the registry with hierarchy information.

        Parameters:
        -----------
        *args, **kwargs
            Arguments passed to the add_* methods

        Returns:
        --------
        tuple
            (flops, params, acts) added to the registry
        """
        # Compute metrics with level information
        flops = self.add_flops(*args, **kwargs)
        params = 0
        if not self.shared_params:
            params = self.add_params(*args, **kwargs)
        acts = self.add_acts(*args, **kwargs)

        # Add to registry with appropriate tags
        self.registry.add_flops(flops, path=MetaModule._path)
        self.registry.add_params(params, path=MetaModule._path)
        self.registry.add_acts(acts, path=MetaModule._path)

        return (flops, params, acts)

    def __call__(self, *args, **kwargs):
        """
        Execute the module's operation with automatic resource tracking.

        This method:
        1. Automatically updates the registry with computed metrics
        2. Calls the forward method that derived classes should implement

        Parameters:
        -----------
        *args, **kwargs
            Arguments to pass to the module implementation

        Returns:
        --------
        object
            Result of the module's operation
        """
        # Save parent path before modifying
        parent_path = MetaModule._path

        # Update global counter
        MetaModule._counter += 1

        name = f"{self.__class__.__name__}_{MetaModule._counter}"

        # Update path using the module's name (includes class name and instance ID)
        if parent_path is None:
            MetaModule._path = name
        else:
            MetaModule._path = f"{parent_path}/{name}"

        # Update registry with metrics
        self.update_registry(*args, **kwargs)

        # Call the implementation
        output = self.forward(*args, **kwargs)

        # Restore parent path when exiting
        MetaModule._path = parent_path

        return output

    def forward(self, *args, **kwargs):
        """
        Implement the actual module operation.

        Derived classes should override this method to provide their functionality.

        Parameters:
        -----------
        *args, **kwargs
            Arguments to the module

        Returns:
        --------
        object
            Result of the operation

        Raises:
        -------
        NotImplementedError
            If the derived class doesn't implement this method
        """
        raise NotImplementedError(
            f"Class {self.__class__.__name__} must implement forward method"
        )


class ModelStatsRegistry:
    """Registry for tracking model statistics with detailed operation logging and hierarchy."""

    def __init__(self, model_id=None):
        """
        Initialize a model statistics registry.

        Parameters:
        -----------
        model_id : str, optional
            Identifier for the model (defaults to "default")
        """
        self.model_id = model_id or "default"
        self.reset()

    def reset(self):
        """Reset all statistics and logs."""
        self.total_flops = 0
        self.total_params = 0
        self.total_acts = 0

        self.flops_logs = []
        self.params_logs = []
        self.acts_logs = []

        self.flops_counter = 0
        self.params_counter = 0
        self.acts_counter = 0

        self.flops_by_module = {}
        self.params_by_module = {}
        self.acts_by_module = {}

        self.module_ids = {}

    def add_flops(self, value, path=None):
        """
        Add FLOPs to the registry with path information.

        Parameters:
        -----------
        value : int or float
            Number of FLOPs to add
        path : str, optional
            Full path identifier for the module
        """
        if value < 0:
            raise ValueError(f"Cannot add negative FLOPs: {value}")

        self.total_flops += value

        # Increment counter for this operation type
        self.flops_counter += 1

        # Generate a path if none provided
        if path is None:
            path = f"unnamed_flop_{self.flops_counter}"

        # Calculate level from path (number of / characters)
        level = path.count("/")

        # Extract module ID from path for sorting
        # Format is typically: ModuleName_ID/SubmoduleName_ID/...
        try:
            module_name = path.split("/")[-1]
            module_id = int(module_name.split("_")[-1]) if "_" in module_name else 0
            self.module_ids[path] = module_id
        except (ValueError, IndexError):
            self.module_ids[path] = 999999  # Fallback ID for malformed paths

        # Update module-specific flops
        if path not in self.flops_by_module:
            self.flops_by_module[path] = 0
        else:
            raise ValueError(f"Path already exists: {path}")
        self.flops_by_module[path] += value

        # Log the operation with path, level and accumulated value
        self.flops_logs.append((value, path, level, self.total_flops))

    def add_params(self, value, path=None):
        """
        Add parameters to the registry with path information.

        Parameters:
        -----------
        value : int or float
            Number of parameters to add
        path : str, optional
            Full path identifier for the module
        """
        if value < 0:
            raise ValueError(f"Cannot add negative parameters: {value}")

        self.total_params += value

        # Increment counter for this operation type
        self.params_counter += 1

        # Generate a path if none provided
        if path is None:
            path = f"unnamed_param_{self.params_counter}"

        # Calculate level from path (number of / characters)
        level = path.count("/")

        # Extract module ID from path for sorting
        # Format is typically: ModuleName_ID/SubmoduleName_ID/...
        try:
            module_name = path.split("/")[-1]
            module_id = int(module_name.split("_")[-1]) if "_" in module_name else 0
            self.module_ids[path] = module_id
        except (ValueError, IndexError):
            self.module_ids[path] = 999999  # Fallback ID for malformed paths

        # Update module-specific parameters
        if path not in self.params_by_module:
            self.params_by_module[path] = 0
        self.params_by_module[path] += value

        # Log the operation with path, level and accumulated value
        self.params_logs.append((value, path, level, self.total_params))

    def add_acts(self, value, path=None):
        """
        Add activation elements to the registry with path information.

        Parameters:
        -----------
        value : int or float
            Number of activation elements to add
        path : str, optional
            Full path identifier for the module
        """
        if value < 0:
            raise ValueError(f"Cannot add negative activations: {value}")

        self.total_acts += value

        # Increment counter for this operation type
        self.acts_counter += 1

        # Generate a path if none provided
        if path is None:
            path = f"unnamed_act_{self.acts_counter}"

        # Calculate level from path (number of / characters)
        level = path.count("/")

        # Extract module ID from path for sorting
        # Format is typically: ModuleName_ID/SubmoduleName_ID/...
        try:
            module_name = path.split("/")[-1]
            module_id = int(module_name.split("_")[-1]) if "_" in module_name else 0
            self.module_ids[path] = module_id
        except (ValueError, IndexError):
            self.module_ids[path] = 999999  # Fallback ID for malformed paths

        # Update module-specific activations
        if path not in self.acts_by_module:
            self.acts_by_module[path] = 0
        self.acts_by_module[path] += value

        # Log the operation with path, level and accumulated value
        self.acts_logs.append((value, path, level, self.total_acts))

    def print_logs(self, metric_type=None, include_summary=False):
        """
        Print logs in hierarchical format while preserving module creation order within each level.

        Parameters:
        -----------
        metric_type : str or list, optional
            Type(s) of metrics to print. Can be "flops", "params", "acts",
            or a list containing any of these. If None, prints all metrics.
        include_summary : bool, optional
            Whether to include summary information at the end (defaults to False)
        """
        # Parse and validate metric_type
        all_metric_types = ["flops", "params", "acts"]

        if metric_type is None:
            metric_types = all_metric_types
        elif isinstance(metric_type, str):
            if metric_type not in all_metric_types:
                raise ValueError(
                    f"Invalid metric_type: {metric_type}. Must be one of {all_metric_types}"
                )
            metric_types = [metric_type]
        elif isinstance(metric_type, (list, tuple)):
            for m in metric_type:
                if m not in all_metric_types:
                    raise ValueError(
                        f"Invalid metric_type: {m}. Must be one of {all_metric_types}"
                    )
            metric_types = metric_type
        else:
            raise ValueError(
                f"metric_type must be None, str, list, or tuple, got {type(metric_type)}"
            )

        # Print header
        metric_list = ", ".join(metric_types).upper()
        print(f"\n===== {metric_list} Statistics for '{self.model_id}' =====")

        # Collect all unique module paths with their levels and module IDs
        module_info = {}

        for logs, metric in [
            (self.flops_logs, "flops"),
            (self.params_logs, "params"),
            (self.acts_logs, "acts"),
        ]:
            if metric not in metric_types:
                continue

            for value, path, level, _ in logs:
                if path not in module_info:
                    # Extract module name (last part of path) for display
                    name = path.split("/")[-1] if "/" in path else path
                    module_info[path] = {
                        "level": level,
                        "name": name,
                        "path": path,
                        "parent_path": (
                            "/".join(path.split("/")[:-1]) if "/" in path else ""
                        ),
                        "module_id": self.module_ids.get(path, 999999),
                    }

        # Organize modules into a hierarchy while preserving module ID order
        hierarchy = {}

        # First, identify all root modules (level 0 or no parent in module_info)
        root_modules = []
        for path, info in module_info.items():
            if info["level"] == 0 or not info["parent_path"]:
                root_modules.append(path)

        # Sort root modules by module ID
        root_modules.sort(key=lambda p: module_info[p]["module_id"])

        # Build hierarchy dictionary
        for root in root_modules:
            hierarchy[root] = self._build_module_hierarchy(root, module_info)

        # Calculate column widths
        max_name_len = max(
            [len(info["name"]) + info["level"] * 2 for info in module_info.values()],
            default=30,
        )
        max_name_len = max(max_name_len, 30)  # Minimum width

        # Print column headers
        header = f"{'Module':<{max_name_len + 5}}"
        for metric in metric_types:
            header += f"{metric.upper():>15} {metric.upper()+'-TOTAL':>15}"
        print(f"\n{header}")
        print("-" * (max_name_len + 5 + len(metric_types) * 30))

        # Calculate accumulated metrics
        accumulated = self._calculate_accumulated_metrics()

        # Print hierarchy
        for root in root_modules:
            self._print_module_hierarchy(
                root,
                hierarchy[root],
                module_info,
                metric_types,
                max_name_len + 5,
                accumulated,
            )

        # Print summary if requested
        if include_summary:
            print("\n===== Summary =====")
            for metric in metric_types:
                total = getattr(self, f"total_{metric}")
                print(f"Total {metric.upper()}: {total:,}")

    def _calculate_accumulated_metrics(self):
        """
        Calculate accumulated metrics for each module by summing its children's metrics.

        This is used to display total resource usage per module including its submodules.

        Returns:
        --------
        dict
            Dictionary with accumulated metrics for each metric type
        """
        accumulated = {"flops": {}, "params": {}, "acts": {}}

        # Get all module paths from all metric dictionaries
        all_paths = set()
        all_paths.update(self.flops_by_module.keys())
        all_paths.update(self.params_by_module.keys())
        all_paths.update(self.acts_by_module.keys())

        # Define a function to get all children of a path
        def get_children(path):
            return {p for p in all_paths if p.startswith(path + "/") or p == path}

        # Calculate accumulated metrics for each path
        for path in all_paths:
            children = get_children(path)

            # Sum up metrics from all children for each metric type
            for metric, module_dict in [
                ("flops", self.flops_by_module),
                ("params", self.params_by_module),
                ("acts", self.acts_by_module),
            ]:
                accumulated[metric][path] = sum(
                    module_dict.get(child, 0) for child in children
                )

        return accumulated

    def _build_module_hierarchy(self, parent_path, module_info):
        """
        Recursively build a hierarchy of modules while preserving module ID order.

        Parameters:
        -----------
        parent_path : str
            Path of the parent module
        module_info : dict
            Dictionary of module information

        Returns:
        --------
        dict
            Hierarchical structure of modules
        """
        # Find all children of this parent
        children = []
        for path, info in module_info.items():
            if info["parent_path"] == parent_path:
                children.append(path)

        # Sort children by module ID
        children.sort(key=lambda p: module_info[p]["module_id"])

        # Build hierarchy recursively
        result = {}
        for child in children:
            result[child] = self._build_module_hierarchy(child, module_info)

        return result

    def _print_module_hierarchy(
        self,
        path,
        children,
        module_info,
        metric_types,
        name_width,
        accumulated,
        indent=0,
    ):
        """
        Recursively print a module hierarchy with metrics.

        Parameters:
        -----------
        path : str
            Path of the current module
        children : dict
            Dictionary of child modules
        module_info : dict
            Dictionary of module information
        metric_types : list
            List of metric types to display
        name_width : int
            Width for the name column
        accumulated : dict
            Dictionary of accumulated metrics
        indent : int, optional
            Current indentation level
        """
        # Skip if module doesn't exist in module_info (should never happen)
        if path not in module_info:
            return

        info = module_info[path]

        # Create indentation
        indent_str = "  " * indent

        # Display name with indentation
        display_name = f"{indent_str}{info['name']}"

        # Prepare line with metrics
        line = f"{display_name:<{name_width}}"

        # Add metrics with their accumulated values
        has_metrics = False
        for metric in metric_types:
            module_dict = getattr(self, f"{metric}_by_module")
            direct_value = module_dict.get(path, 0)
            accum_value = accumulated[metric].get(path, 0)

            if direct_value > 0 or accum_value > 0:
                has_metrics = True

            # Format for display
            direct_str = f"{direct_value:,}" if direct_value > 0 else "-"
            accum_str = f"{accum_value:,}" if accum_value > 0 else "-"

            # Add to output line
            line += f"{direct_str:>15} {accum_str:>15}"

        # Print line if module has any metrics
        if has_metrics:
            print(line)

        # Recursively print children sorted by module_id
        for child_path in sorted(
            children.keys(), key=lambda p: module_info[p]["module_id"]
        ):
            self._print_module_hierarchy(
                child_path,
                children[child_path],
                module_info,
                metric_types,
                name_width,
                accumulated,
                indent + 1,
            )


# Registry to store stats for different models
_model_registries = {}


def register_model(model_id="default"):
    """
    Register a model with a specific configuration.

    Parameters:
    -----------
    model_id : str
        Unique identifier for the model

    Returns:
    --------
    ModelStatsRegistry
        The newly created registry

    Raises:
    -------
    ValueError
        If model_id already exists in registry
    """
    global _model_registries
    if model_id in _model_registries:
        if model_id == "default":
            return _model_registries[model_id]
        else:
            raise ValueError(f"Model ID {model_id} already exists in registry")
    _model_registries[model_id] = ModelStatsRegistry(model_id)
    return _model_registries[model_id]


def get_registry(model_id="default"):
    """
    Get or create a registry for the specified model.

    If the specified model_id doesn't exist but "default" does exist,
    returns the default registry.

    Parameters:
    -----------
    model_id : str, optional
        Identifier for the model (defaults to "default")

    Returns:
    --------
    ModelStatsRegistry
        Registry for the specified model or default registry
    """
    global _model_registries

    # Remove debug print statement

    # Ensure default registry exists
    if "default" not in _model_registries:
        _model_registries["default"] = ModelStatsRegistry("default")

    # Return requested registry if it exists, otherwise default
    if model_id not in _model_registries:
        raise ValueError(f"Model ID {model_id} not found in registry")

    return _model_registries[model_id]


def reset_registry(model_id="default"):
    """
    Reset the registry for the specified model.

    Parameters:
    -----------
    model_id : str, optional
        Identifier for the model (defaults to "default")

    Raises:
    -------
    ValueError
        If model_id not found in registry
    """
    global _model_registries

    # Ensure default registry exists
    if "default" not in _model_registries:
        _model_registries["default"] = ModelStatsRegistry("default")

    # Reset requested registry
    if model_id not in _model_registries:
        raise ValueError(f"Model ID {model_id} not found in registry")
    _model_registries[model_id].reset()


@dataclass
class ModelConfig:
    """
    Base configuration class for model architectures.

    This class serves as a base for all model configurations, providing
    common parameters and type hints for model initialization.
    Model-specific configurations should inherit from this class.
    """

    # Model identification
    model_id: str = "default"

    # Training configuration
    batch_size: int = 1
    seq_length: int = 512

    # Parallelism configuration
    data_parallel_size: int = 1
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    expert_parallel_size: int = 1
    pipeline_rank: int = 0

    # Other common parameters
    dtype: str = "float16"

    # Additional model-specific configurations
    kwargs: Dict[str, Any] = field(default_factory=dict)
