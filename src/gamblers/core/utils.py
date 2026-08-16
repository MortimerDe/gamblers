from typing import Never


def todo(msg: str = "todo") -> Never:
    raise NotImplementedError(msg)
