"""Tests for VeNCrypt security type (19) with subtype Plain (256)."""

from asyncio import StreamReader
from io import BytesIO

import pytest

from asyncvnc2 import Client


def _reader_with(data: bytes) -> StreamReader:
    """StreamReader pre-loaded with canned bytes. Requires a running loop."""
    reader = StreamReader()
    reader._buffer.extend(bytearray(data))
    return reader


# Server-side byte sequence up to the point where the client picks Plain and
# sends credentials. Composable per-test so we can vary the subtype list and
# SecurityResult without repeating the boilerplate.

def _server_prefix(subtypes: list[int]) -> bytes:
    return (
        b'RFB 003.008\n' +                     # banner
        (1).to_bytes(1, 'big') +               # one security type offered
        (19).to_bytes(1, 'big') +              # VeNCrypt
        bytes([0, 2]) +                        # VeNCrypt version 0.2
        bytes([0]) +                           # version ack
        bytes([len(subtypes)]) +               # subtype count
        b''.join(s.to_bytes(4, 'big') for s in subtypes)
    )


def _security_result_failed() -> bytes:
    # SecurityResult = 1 (failed). Existing asyncvnc2 handling raises
    # PermissionError('Auth failed') on this value without draining the
    # reason -- fine for testing that our VeNCrypt code path completes.
    return (1).to_bytes(4, 'big')


# -- happy path: reaches SecurityResult, exchange bytes are correct ------

@pytest.mark.asyncio
async def test_plain_writes_correct_bytes():
    reader = _reader_with(_server_prefix([256]) + _security_result_failed())
    writer = BytesIO()
    with pytest.raises(PermissionError):
        # Fails at the (mocked) SecurityResult, but only after our client
        # has completed the full VeNCrypt handshake and sent credentials.
        await Client.create(reader, writer, username='alice', password='hunter2')

    sent = writer.getvalue()
    # Client responses in order:
    #   RFB 003.008\n           handshake reply
    #   \x13                    security type = 19 (VeNCrypt)
    #   \x00\x02                echo version 0.2
    #   \x00\x00\x01\x00        subtype 256 (Plain), uint32 BE
    #   creds block             len(u)=5, len(p)=7, "alice", "hunter2"
    expected = (
        b'RFB 003.008\n' +
        bytes([19]) +
        bytes([0, 2]) +
        (256).to_bytes(4, 'big') +
        (5).to_bytes(4, 'big') +
        (7).to_bytes(4, 'big') +
        b'alice' + b'hunter2'
    )
    assert sent == expected


# -- error path: server offers 19 but no Plain subtype -------------------

@pytest.mark.asyncio
async def test_no_plain_subtype_raises_specific_error():
    # Server offers only TLS-wrapped subtypes.
    reader = _reader_with(_server_prefix([257, 259]))
    writer = BytesIO()
    with pytest.raises(ValueError, match=r'Plain \(256\) not offered'):
        await Client.create(reader, writer, username='alice', password='hunter2')


# -- error path: Plain offered but no credentials supplied ---------------

@pytest.mark.asyncio
async def test_plain_requires_credentials():
    reader = _reader_with(_server_prefix([256]))
    writer = BytesIO()
    with pytest.raises(ValueError, match='username and password'):
        await Client.create(reader, writer, username=None, password=None)
    # The credential check must fire BEFORE the subtype selection reaches
    # the wire. Otherwise the server sees "Plain requested" and blocks on
    # length fields that never arrive, holding a connection slot until
    # timeout. Verify by asserting the subtype bytes (256 as big-endian
    # uint32 = 00 00 01 00) are absent from what the writer emitted.
    sent = writer.getvalue()
    assert (256).to_bytes(4, 'big') not in sent


# -- error path: version-ack byte non-zero (server rejects our version) --

@pytest.mark.asyncio
async def test_version_rejection():
    reader = _reader_with(
        b'RFB 003.008\n' +
        (1).to_bytes(1, 'big') +
        (19).to_bytes(1, 'big') +
        bytes([0, 2]) +
        bytes([1])  # version ack = 1 (rejected)
    )
    writer = BytesIO()
    with pytest.raises(ValueError, match=r'server rejected VeNCrypt'):
        await Client.create(reader, writer, username='alice', password='hunter2')


# -- regression: no ack read between subtype selection and credentials ---

@pytest.mark.asyncio
async def test_no_ack_expected_after_subtype_selection():
    # VeNCrypt 0.2 does NOT ack the client's subtype choice. If our code
    # ever regressed to reading a byte between step 4 and step 5, the
    # SecurityResult read would consume shifted bytes and the parse would
    # desync. This test ensures the byte-for-byte sequence stays correct
    # -- effectively the same check as test_plain_writes_correct_bytes but
    # named so a future contributor sees why the asymmetry matters.
    reader = _reader_with(_server_prefix([256]) + _security_result_failed())
    writer = BytesIO()
    with pytest.raises(PermissionError):
        await Client.create(reader, writer, username='u', password='p')
    # SecurityResult was consumed cleanly; buffer should be empty.
    assert len(reader._buffer) == 0
