from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from sanic.http.constants import HTTP


if TYPE_CHECKING:
    from sanic import Sanic


class AltSvcOutcome(Enum):
    """Terminal state of the ``alt-svc`` assignment on a response path."""

    # The assignment is removed from the registered response path.
    STRIPPED = "stripped"
    # The assignment is materialized as one or more ``h3`` announcements.
    H3 = "h3"
    # The source-level empty assignment is retained and still emitted.
    EMPTY = "empty"


@dataclass(frozen=True)
class ProfileEntry:
    """One ``prepare`` registration, frozen in call order."""

    version: HTTP
    port: int


@dataclass(frozen=True)
class ListenerProfile:
    """Immutable snapshot of the listening profile at one observation."""

    entries: tuple[ProfileEntry, ...]

    @property
    def versions(self) -> frozenset[HTTP]:
        return frozenset(entry.version for entry in self.entries)

    @property
    def is_single_version(self) -> bool:
        return len(self.versions) == 1

    def h3_value(self) -> str:
        return ", ".join(
            f'h3=":{entry.port}"'
            for entry in self.entries
            if entry.version is HTTP.VERSION_3
        )


@dataclass(frozen=True)
class Materialization:
    """A frozen, append-only observation of an alt-svc terminal state."""

    seq: int
    profile: ListenerProfile
    touchup: bool
    outcome: AltSvcOutcome
    value: str


@dataclass(frozen=True)
class PrepareRejection:
    """A frozen record of a rejected ``prepare`` call.

    Rejections live on a separate stream from materializations so that a
    failed registration cannot be collapsed into the response terminal
    states.
    """

    seq: int
    version: HTTP
    registered: int
    reason: str


def decide(
    profile: ListenerProfile, touchup: bool
) -> tuple[AltSvcOutcome, str]:
    """Resolve the alt-svc terminal state from a frozen profile/switch pair."""

    if not touchup:
        return AltSvcOutcome.EMPTY, ""
    if profile.is_single_version:
        return AltSvcOutcome.STRIPPED, ""
    return AltSvcOutcome.H3, profile.h3_value()


class ProfileMaterializationLedger:
    """Append-only ledger of listening profiles and their materializations.

    Every observation is frozen at creation time. Later ``prepare`` calls,
    TouchUp toggles or HTTP/1 responses append new observations but never
    mutate an already observed terminal state.
    """

    def __init__(self, app: Sanic) -> None:
        self._app = app
        self._registrations: list[ProfileEntry] = []
        self._materializations: list[Materialization] = []
        self._rejections: list[PrepareRejection] = []

    # ------------------------------------------------------------------
    # prepare plane: listener registration and rejection
    # ------------------------------------------------------------------
    def record_registration(self, version: HTTP, port: int) -> ProfileEntry:
        entry = ProfileEntry(version=version, port=port)
        self._registrations.append(entry)
        return entry

    def record_rejection(self, version: HTTP, reason: str) -> PrepareRejection:
        rejection = PrepareRejection(
            seq=len(self._rejections),
            version=version,
            registered=len(self._app.state.server_info),
            reason=reason,
        )
        self._rejections.append(rejection)
        return rejection

    # ------------------------------------------------------------------
    # TouchUp / HTTP/1 response plane: profile materialization
    # ------------------------------------------------------------------
    def snapshot_profile(self) -> ListenerProfile:
        return ListenerProfile(
            entries=tuple(
                ProfileEntry(
                    version=info.settings["version"],
                    port=info.settings["port"],
                )
                for info in self._app.state.server_info
            )
        )

    def materialize(self, touchup: bool) -> Materialization:
        profile = self.snapshot_profile()
        # A TouchUp run rebuilds every registered method, so the same
        # profile is observed repeatedly. Collapse consecutive identical
        # observations instead of appending duplicate terminal states.
        if self._materializations:
            latest = self._materializations[-1]
            if latest.touchup is touchup and latest.profile == profile:
                return latest
        outcome, value = decide(profile, touchup)
        materialization = Materialization(
            seq=len(self._materializations),
            profile=profile,
            touchup=touchup,
            outcome=outcome,
            value=value,
        )
        self._materializations.append(materialization)
        return materialization

    def observe_response(self) -> Materialization:
        """Observe the response path without rewriting prior observations.

        Under an unchanged profile/TouchUp switch this resolves to the
        latest frozen observation, so re-sending responses never appends
        or mutates anything. When the switch or the live profile changes,
        a new observation is materialized for the current conditions.
        """

        return self.materialize(self._app.config.TOUCHUP is True)

    # ------------------------------------------------------------------
    # read-only views
    # ------------------------------------------------------------------
    @property
    def registrations(self) -> tuple[ProfileEntry, ...]:
        return tuple(self._registrations)

    @property
    def materializations(self) -> tuple[Materialization, ...]:
        return tuple(self._materializations)

    @property
    def rejections(self) -> tuple[PrepareRejection, ...]:
        return tuple(self._rejections)

    @property
    def latest(self) -> Materialization | None:
        return self._materializations[-1] if self._materializations else None
