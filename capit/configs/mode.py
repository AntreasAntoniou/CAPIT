from dataclasses import dataclass


@dataclass
class BaseMode:
    fit: bool = True
    test: bool = True
