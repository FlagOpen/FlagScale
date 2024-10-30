# Copyright (c) 2024, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any, Callable, ClassVar, Dict, Optional, Union

from megatron.energon.epathlib import EPath


class ModuleLoader:
    """Class providing access to python files to be loaded as modules."""

    #: Loaded modules. Global singleton, no need to load a module multiple times.
    _modules: ClassVar[Dict[EPath, Dict[str, Any]]] = {}

    def _load_module(self, path: Union[str, EPath]):
        """Internally loads a module"""
        path = EPath(path)
        with path.open("r") as rf:
            src_code = rf.read()
        code = compile(src_code, path.url, "exec")
        mod = {}
        eval(code, mod)
        self._modules[path] = mod

    def __getitem__(self, path: Union[str, EPath]) -> Any:
        """
        Gets a module by path.

        Args:
            path: The path to load

        Returns:
            Loaded module dict.
        """
        path = EPath(path)
        if path not in self._modules:
            self._load_module(path)
        return self._modules[path]

    def get_function(
        self,
        module_fn: str,
        default_fn_name: Optional[str] = None,
        relative_path: Optional[EPath] = None,
    ) -> Callable:
        """
        Loads a function in the form `"module/submodule.py:function_name"`.

        Args:
            module_fn: The function string to load. Must contain the path of the python module
                and a function name separated by a colon (might be omitted if `default_fn_name` is
                specified).
            default_fn_name: Default name of the function if not given in `module_fn` string.
            relative_path: The relative parent path to the module. If not specified, the current
                working directory / absolute path is used.

        Returns:
            The function from the module
        """
        if ":" in module_fn:
            module, fn_name = module_fn.rsplit(":", 1)
        else:
            if default_fn_name is None:
                raise ValueError("Function name must be specified")
            fn_name = default_fn_name
            module = module_fn
        if relative_path is not None:
            module = str(relative_path / module)
        fn = self[module][fn_name]
        assert callable(fn)
        return fn
