"""CI-safe tests: RESP server HELLO handshake replies.

RESP3-capable clients (e.g. redis-py >= 8, where RESP3 became the default
protocol) send a HELLO command on connect and fail the connection when it
is not answered.
"""

from roboml.resp_server.server import ERROR_INVALID_PROTOCOL, hello_reply


def test_hello_resp3_returns_map():
    """A HELLO 3 handshake gets a RESP3 map advertising proto 3."""
    reply = hello_reply([b"3"])
    assert reply.startswith(b"%4\r\n")
    assert b"$6\r\nserver\r\n$6\r\nroboml\r\n" in reply
    assert b"$5\r\nproto\r\n:3\r\n" in reply


def test_hello_resp2_returns_flat_array():
    """HELLO 2 and argument-less HELLO get a RESP2 flat key-value array."""
    for args in ([b"2"], []):
        reply = hello_reply(args)
        assert reply.startswith(b"*8\r\n")
        assert b"$5\r\nproto\r\n:2\r\n" in reply


def test_hello_unsupported_protocol_rejected():
    """Protocol versions other than 2 and 3 get a NOPROTO error."""
    assert hello_reply([b"4"]) == ERROR_INVALID_PROTOCOL
    assert hello_reply([b"abc"]) == ERROR_INVALID_PROTOCOL


def test_hello_auth_arguments_ignored():
    """AUTH/SETNAME arguments are accepted and ignored (no authentication)."""
    reply = hello_reply([b"3", b"AUTH", b"user", b"pass"])
    assert reply.startswith(b"%4\r\n")
    assert b"$5\r\nproto\r\n:3\r\n" in reply
