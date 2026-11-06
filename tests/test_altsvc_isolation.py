import inspect
import socket
import sys

from ast import parse, unparse
from dataclasses import FrozenInstanceError
from pathlib import Path
from textwrap import dedent

import pytest

from sanic.application.state import ApplicationServerInfo
from sanic.http.constants import HTTP
from sanic.http.http1 import Http
from sanic.response import empty
from sanic.touchup import TouchUp
from sanic.touchup.materialize import (
    AltSvcOutcome,
    Materialization,
    PrepareRejection,
    ProfileEntry,
)
from sanic.touchup.schemes.altsvc import RemoveAltSvc
from tests.client import RawClient


parent_dir = Path(__file__).parent
localhost_dir = parent_dir / "certs/localhost"

HOST = "127.0.0.1"
REQUEST = "GET / HTTP/1.1\nhost: localhost\nconnection: close\n\n"
SECONDARY = "secondary server"


def _free_port() -> int:
    sock = socket.socket()
    sock.bind((HOST, 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def _pristine_source() -> str:
    target = getattr(
        Http, "_http1_response_header", Http.http1_response_header
    )
    return inspect.getsource(target)


def _render_response_header(app) -> str:
    node = parse(dedent(_pristine_source()))
    node = RemoveAltSvc(app, app.state.verbosity).visit(node)
    return unparse(node)


async def _serve_and_raw_get(app, port, times=1):
    server = await app.create_server(
        host=HOST, port=port, return_asyncio_server=True
    )
    await server.startup()
    try:
        responses = []
        for _ in range(times):
            client = RawClient(HOST, port)
            await client.connect()
            await client.send(REQUEST)
            responses.append(await client.recv(4096))
            await client.close()
        return responses
    finally:
        server.close()
        await server.wait_closed()
        app.stop()


# ---------------------------------------------------------------------
# Carrier 1: prepare() listener registration and rejection
# ---------------------------------------------------------------------
def test_prepare_registers_listener_profile_in_order(app):
    app.config.ALT_SVC_ISOLATION = True
    port_one, port_two = _free_port(), _free_port()

    app.prepare(port=port_one)
    app.prepare(port=port_two)

    assert app.state.profile_ledger.registrations == (
        ProfileEntry(HTTP.VERSION_1, port_one),
        ProfileEntry(HTTP.VERSION_1, port_two),
    )
    assert [info.settings["port"] for info in app.state.server_info] == [
        port_one,
        port_two,
    ]
    assert [info.settings["version"] for info in app.state.server_info] == [
        HTTP.VERSION_1,
        HTTP.VERSION_1,
    ]
    # The prepare plane never creates response materializations itself.
    assert app.state.profile_ledger.materializations == ()


def test_http3_non_first_prepare_rejection_is_separate_stream(app):
    app.config.ALT_SVC_ISOLATION = True
    app.prepare(port=_free_port())

    with pytest.raises(RuntimeError, match=SECONDARY):
        app.prepare(version=3, port=_free_port())

    rejections = app.state.profile_ledger.rejections
    assert len(rejections) == 1
    rejection = rejections[0]
    assert isinstance(rejection, PrepareRejection)
    assert rejection.version is HTTP.VERSION_3
    assert rejection.registered == 1
    assert rejection.seq == 0
    assert SECONDARY in rejection.reason
    # A rejection neither mutates registrations nor materializes a response
    # terminal state, and it does not roll back server_info.
    assert len(app.state.profile_ledger.registrations) == 1
    assert app.state.profile_ledger.materializations == ()
    assert len(app.state.server_info) == 1


def test_prepare_plane_pinned_without_isolation(app):
    # ALT_SVC_ISOLATION defaults to False: nothing is recorded.
    app.prepare(port=_free_port())
    with pytest.raises(RuntimeError, match=SECONDARY):
        app.prepare(version=3, port=_free_port())

    ledger = app.state.profile_ledger
    assert ledger.registrations == ()
    assert ledger.rejections == ()
    assert ledger.materializations == ()
    assert len(app.state.server_info) == 1


# ---------------------------------------------------------------------
# Carrier 2: startup TouchUp rewrite of the response path
# ---------------------------------------------------------------------
@pytest.mark.skipif(
    sys.version_info < (3, 9), reason="ast.unparse requires 3.9+"
)
def test_touchup_strips_alt_svc_for_single_version_profile(app):
    app.config.ALT_SVC_ISOLATION = True
    app.config.TOUCHUP = True
    app.prepare(port=_free_port())

    rendered = _render_response_header(app)

    materialization = app.state.profile_ledger.latest
    assert isinstance(materialization, Materialization)
    assert materialization.touchup is True
    assert materialization.outcome is AltSvcOutcome.STRIPPED
    assert materialization.value == ""
    assert materialization.profile.versions == frozenset({HTTP.VERSION_1})
    assert "alt-svc" not in rendered


@pytest.mark.skipif(
    sys.version_info < (3, 9), reason="ast.unparse requires 3.9+"
)
def test_touchup_materializes_h3_port_string_for_multi_version(app):
    app.config.ALT_SVC_ISOLATION = True
    app.config.TOUCHUP = True
    h3_port, h1_port = _free_port(), _free_port()

    app.prepare(
        version=3,
        port=h3_port,
        ssl={
            "cert": localhost_dir / "fullchain.pem",
            "key": localhost_dir / "privkey.pem",
        },
    )
    app.prepare(version=1, port=h1_port)

    rendered = _render_response_header(app)

    materialization = app.state.profile_ledger.latest
    assert materialization.outcome is AltSvcOutcome.H3
    assert materialization.value == f'h3=":{h3_port}"'
    assert materialization.profile.is_single_version is False
    assert materialization.profile.entries == (
        ProfileEntry(HTTP.VERSION_3, h3_port),
        ProfileEntry(HTTP.VERSION_1, h1_port),
    )
    assert "alt-svc" in rendered
    assert f'h3=":{h3_port}"' in rendered


def test_touchup_run_appends_frozen_materialization(app):
    app.config.ALT_SVC_ISOLATION = True
    app.config.TOUCHUP = True
    app.prepare(port=_free_port())

    TouchUp.run(app)

    ledger = app.state.profile_ledger
    assert len(ledger.materializations) == 1
    assert ledger.latest.outcome is AltSvcOutcome.STRIPPED
    with pytest.raises(FrozenInstanceError):
        ledger.latest.value = 'h3=":4000"'  # type: ignore[misc]


def test_touchup_off_empty_assignment_is_its_own_terminal_state(app):
    app.config.ALT_SVC_ISOLATION = True
    app.config.TOUCHUP = False
    app.prepare(port=_free_port())

    materialization = app.state.profile_ledger.observe_response()

    assert materialization.outcome is AltSvcOutcome.EMPTY
    assert materialization.touchup is False
    assert materialization.value == ""
    assert 'headers["alt-svc"] = ""' in _pristine_source()
    # Re-triggering the response path only re-reads the frozen observation.
    assert app.state.profile_ledger.observe_response() is materialization
    assert len(app.state.profile_ledger.materializations) == 1


@pytest.mark.skipif(
    sys.version_info < (3, 9), reason="ast.unparse requires 3.9+"
)
def test_touchup_legacy_behavior_pinned_without_isolation(app):
    app.config.TOUCHUP = True
    app.prepare(port=_free_port())

    TouchUp.run(app)

    assert app.state.profile_ledger.materializations == ()
    assert "alt-svc" not in _render_response_header(app)


@pytest.mark.skipif(
    sys.version_info < (3, 9), reason="ast.unparse requires 3.9+"
)
def test_touchup_legacy_h3_value_pinned_without_isolation(app):
    app.config.TOUCHUP = True
    h3_port = _free_port()
    app.state.server_info.append(
        ApplicationServerInfo(
            settings={"version": HTTP.VERSION_3, "port": h3_port}
        )
    )
    app.state.server_info.append(
        ApplicationServerInfo(
            settings={"version": HTTP.VERSION_1, "port": _free_port()}
        )
    )

    TouchUp.run(app)

    assert app.state.profile_ledger.materializations == ()
    assert f'h3=":{h3_port}"' in _render_response_header(app)


# ---------------------------------------------------------------------
# Carrier 3: HTTP/1 raw response bytes on the send path
# ---------------------------------------------------------------------
@pytest.mark.asyncio
async def test_response_bytes_stripped_on_single_version(app, port):
    app.config.ALT_SVC_ISOLATION = True
    app.config.TOUCHUP = True

    @app.get("/")
    async def handler(*_):
        return empty()

    app.prepare(host=HOST, port=port)
    responses = await _serve_and_raw_get(app, port, times=2)

    assert all(b"alt-svc" not in response for response in responses)
    materializations = app.state.profile_ledger.materializations
    assert len(materializations) == 1
    assert materializations[0].outcome is AltSvcOutcome.STRIPPED


@pytest.mark.asyncio
async def test_response_bytes_empty_assignment_when_touchup_off(app, port):
    app.config.ALT_SVC_ISOLATION = True
    app.config.TOUCHUP = False

    @app.get("/")
    async def handler(*_):
        return empty()

    app.prepare(host=HOST, port=port)
    responses = await _serve_and_raw_get(app, port, times=2)

    assert all(b"alt-svc: \r\n" in response for response in responses)
    materializations = app.state.profile_ledger.materializations
    assert len(materializations) == 1
    assert materializations[0].outcome is AltSvcOutcome.EMPTY


@pytest.mark.asyncio
async def test_response_bytes_h3_string_without_http3_socket(app, port):
    app.config.ALT_SVC_ISOLATION = True
    app.config.TOUCHUP = True
    h3_port = _free_port()

    @app.get("/")
    async def handler(*_):
        return empty()

    # The profile carries an HTTP/3 listener, but only the HTTP/1 socket
    # is ever opened: no HTTP/3 packet is received, no certificate is used.
    app.state.server_info.append(
        ApplicationServerInfo(
            settings={"version": HTTP.VERSION_3, "port": h3_port}
        )
    )
    app.prepare(host=HOST, port=port)

    responses = await _serve_and_raw_get(app, port)

    expected = f'alt-svc: h3=":{h3_port}"\r\n'.encode()
    assert expected in responses[0]
    latest = app.state.profile_ledger.latest
    assert latest is not None
    assert latest.outcome is AltSvcOutcome.H3
    assert latest.value == f'h3=":{h3_port}"'


@pytest.mark.asyncio
async def test_response_path_is_pinned_when_isolation_disabled(app, port):
    app.config.TOUCHUP = False

    @app.get("/")
    async def handler(*_):
        return empty()

    app.prepare(host=HOST, port=port)
    responses = await _serve_and_raw_get(app, port)

    assert b"alt-svc: \r\n" in responses[0]
    ledger = app.state.profile_ledger
    assert ledger.materializations == ()
    assert ledger.registrations == ()
    assert ledger.rejections == ()


@pytest.mark.skipif(
    sys.version_info < (3, 9), reason="ast.unparse requires 3.9+"
)
def test_response_path_keeps_hook_through_strip_rewrite(app):
    app.config.ALT_SVC_ISOLATION = True
    app.prepare(port=_free_port())

    assert "observe_response" in _pristine_source()
    assert 'headers["alt-svc"] = ""' in _pristine_source()

    rendered = _render_response_header(app)
    assert "observe_response" in rendered
    assert "alt-svc" not in rendered


# ---------------------------------------------------------------------
# Same-app outer/inner materialization isolation: the centerpiece
# ---------------------------------------------------------------------
def test_same_app_outer_inner_terminal_states_never_write_through(app):
    app.config.ALT_SVC_ISOLATION = True
    app.config.TOUCHUP = True
    ledger = app.state.profile_ledger

    # Outer observation A: a single-version profile is stripped.
    port_one = _free_port()
    app.prepare(port=port_one)
    TouchUp.run(app)
    outer = ledger.materializations[0]
    assert outer.outcome is AltSvcOutcome.STRIPPED
    assert len(outer.profile.entries) == 1

    # The prepare-failure fork lives on its own stream.
    with pytest.raises(RuntimeError, match=SECONDARY):
        app.prepare(version=3, port=_free_port())
    assert len(ledger.rejections) == 1
    assert len(ledger.materializations) == 1

    # Inner cycle: reset the live profile as a new serve cycle does, then
    # build a multi-listener profile headed by HTTP/3, which materializes
    # the h3 port string as a new observation.
    app.state.server_info.clear()
    h3_port, port_two = _free_port(), _free_port()
    app.prepare(
        version=3,
        port=h3_port,
        ssl={
            "cert": localhost_dir / "fullchain.pem",
            "key": localhost_dir / "privkey.pem",
        },
    )
    app.prepare(version=1, port=port_two)
    TouchUp.run(app)
    h3_materialization = ledger.materializations[1]
    assert h3_materialization.outcome is AltSvcOutcome.H3
    assert h3_materialization.value == f'h3=":{h3_port}"'

    # The outer stripped observation is frozen and still describes the
    # original single-listener profile.
    assert ledger.materializations[0] == outer
    assert ledger.materializations[0].profile.entries == (
        ProfileEntry(HTTP.VERSION_1, port_one),
    )
    assert len(h3_materialization.profile.entries) == 2

    # Switch TouchUp off: a new observation becomes the empty assignment.
    app.config.TOUCHUP = False
    empty_materialization = ledger.observe_response()
    assert empty_materialization.outcome is AltSvcOutcome.EMPTY
    assert len(ledger.materializations) == 3

    # Write-through attempts: TouchUp toggling and repeated response
    # observations under unchanged conditions neither append nor mutate.
    app.config.TOUCHUP = True
    app.config.TOUCHUP = False
    assert ledger.observe_response() is empty_materialization
    assert ledger.observe_response() is empty_materialization
    assert len(ledger.materializations) == 3
    assert len(h3_materialization.profile.entries) == 2

    # A later prepare changes the live profile: the next response is a
    # NEW observation materialized from the current profile/switch, while
    # every earlier terminal state remains frozen.
    port_three = _free_port()
    app.prepare(version=1, port=port_three)
    refreshed = ledger.observe_response()
    assert refreshed is not empty_materialization
    assert refreshed.outcome is AltSvcOutcome.EMPTY
    assert refreshed.profile.entries[-1] == ProfileEntry(
        HTTP.VERSION_1, port_three
    )
    assert len(ledger.snapshot_profile().entries) == 3
    assert len(h3_materialization.profile.entries) == 2

    # The terminal states stay distinct and ordered on one application.
    assert [
        materialization.outcome for materialization in ledger.materializations
    ] == [
        AltSvcOutcome.STRIPPED,
        AltSvcOutcome.H3,
        AltSvcOutcome.EMPTY,
        AltSvcOutcome.EMPTY,
    ]
    assert ledger.materializations[0] == outer
    assert ledger.materializations[2] == empty_materialization
    assert ledger.registrations[0] == ProfileEntry(HTTP.VERSION_1, port_one)
    assert len(ledger.registrations) == 4
    with pytest.raises(FrozenInstanceError):
        outer.outcome = AltSvcOutcome.H3  # type: ignore[misc]
