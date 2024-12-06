import asyncio


class StreamReader(asyncio.StreamReader):
    """
    Overrides the official StreamReader to address the
    following issue: http://bugs.python.org/issue30861

    Also it leverages to get rid of the double buffer and
    get rid of one coroutine step. Data flows from the buffer
    to the Redis parser directly.
    Copyright (c) 2014-2017 Alexey Popravka
    """

    _parser = None

    def set_parser(self, parser):
        self._parser = parser
        if self._buffer:
            self._parser.feed(self._buffer)
            del self._buffer[:]

    def feed_data(self, data):
        assert not self._eof, "feed_data after feed_eof"

        if not data:
            return
        if self._parser is None:
            # Ignore small errors from parser. Probably should remove this
            self._buffer.extend(data)
            return
        self._parser.feed(data)
        self._wakeup_waiter()

    async def readobj(self):
        """
        Return a parsed object or an exception
        """
        assert self._parser is not None, "set_parser must be called"
        while True:
            obj = self._parser.gets()

            if obj is not False:
                # Return object if parsing is done
                # else wait for more data to be parsed
                return obj

            if self._exception:
                raise self._exception

            if self._eof:
                break

            await self._wait_for_data("readobj")
            # NOTE: after break we return None which must be handled as b''

    async def _read_not_allowed(self, *args, **kwargs):
        raise RuntimeError("Use readobj")

    read = _read_not_allowed
    readline = _read_not_allowed
    readuntil = _read_not_allowed
    readexactly = _read_not_allowed
