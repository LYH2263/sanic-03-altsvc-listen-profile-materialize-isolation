from __future__ import annotations

from ast import Assign, Constant, NodeTransformer, Subscript
from typing import TYPE_CHECKING, Any

from sanic.http.constants import HTTP
from sanic.touchup.materialize import AltSvcOutcome, Materialization

from .base import BaseScheme


if TYPE_CHECKING:
    from sanic import Sanic


class AltSvcCheck(BaseScheme):
    ident = "ALTSVC"

    def visitors(self) -> list[NodeTransformer]:
        return [RemoveAltSvc(self.app, self.app.state.verbosity)]


class RemoveAltSvc(NodeTransformer):
    def __init__(self, app: Sanic, verbosity: int = 0) -> None:
        self._app = app
        self._verbosity = verbosity
        if app.config.ALT_SVC_ISOLATION:
            # Freeze the listening profile at TouchUp time into an
            # append-only materialization. Later prepare calls or TouchUp
            # toggles cannot rewrite this observed terminal state.
            self._materialization: Materialization | None = (
                app.state.profile_ledger.materialize(True)
            )
        else:
            self._materialization = None
            self._versions = {
                info.settings["version"] for info in app.state.server_info
            }

    def visit_Assign(self, node: Assign) -> Any:
        if any(self._matches(target) for target in node.targets):
            if self._should_remove():
                return None
            assert isinstance(node.value, Constant)
            node.value.value = self.value()
        return node

    def _should_remove(self) -> bool:
        if self._materialization is not None:
            return self._materialization.outcome is AltSvcOutcome.STRIPPED
        return len(self._versions) == 1

    @staticmethod
    def _matches(node) -> bool:
        return (
            isinstance(node, Subscript)
            and isinstance(node.slice, Constant)
            and node.slice.value == "alt-svc"
        )

    def value(self):
        if self._materialization is not None:
            return self._materialization.value
        values = []
        for info in self._app.state.server_info:
            port = info.settings["port"]
            version = info.settings["version"]
            if version is HTTP.VERSION_3:
                values.append(f'h3=":{port}"')
        return ", ".join(values)
