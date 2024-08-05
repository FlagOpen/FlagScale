class PathNotFound(BaseException):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class GitNotFound(BaseException):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class CommitShort(BaseException):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class DeviceError(BaseException):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class UnpatchDiffError(BaseException):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)
        

class GitApplyError(BaseException):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class DirNotFound(BaseException):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)
