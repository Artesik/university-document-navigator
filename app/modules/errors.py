"""Errors that can safely be shown in the local UI."""


class NavigatorError(Exception):
    pass


class DocumentError(NavigatorError):
    pass


class IndexError(NavigatorError):
    pass


class ModelError(NavigatorError):
    pass
