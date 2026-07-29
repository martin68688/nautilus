"""Typed protocol-runtime failures."""


class ProtocolRuntimeError(RuntimeError):
    pass


class CollectorUnavailable(ProtocolRuntimeError):
    pass


class CollectorRejected(ProtocolRuntimeError):
    pass


class InvalidViewHandle(ProtocolRuntimeError):
    pass


class ProtocolStateError(ProtocolRuntimeError):
    pass


__all__ = [
    "CollectorRejected",
    "CollectorUnavailable",
    "InvalidViewHandle",
    "ProtocolRuntimeError",
    "ProtocolStateError",
]
