#!/usr/bin/env python3

"""
Mod9 ASR WebSocket server: wrapper over the ASR Engine.

This application-level protocol over WebSocket is similar to the Engine's
custom application-level protocol over TCP, but leverages message events.

The first message should be sent by the client as a JSON-formatted object
representing the request options, as in the protocol over TCP. Instead of
strictly requiring single-line newline-terminated JSON, however, clients
of the WebSocket interface may format the JSON with arbitrary whitespace.

Subsequent messages sent by the client should be bytes of the audio data,
as in the protocol over TCP. Each of these messages must be non-empty.

Upon receipt of an empty message, this WebSocket server will consider a
client to have finished sending its audio data. This functionality is in
addition to the underlying Engine's potential to terminate requests as
per the custom application-level protocol over TCP, i.e. after:
 - receiving the number of bytes set by a ``content-length`` option;
 - receiving the number of samples indicated in a WAV header;
 - receiving an "END-OF-FILE" sequence (cf. ``eof`` option);
 - or if no data has been received for over 10 seconds.

The WebSocket client asynchronously receives a series of messages relayed
from the Engine, each encoding a single JSON object. This is similar to
the Engine's response structure for the protocol over TCP, except that no
newlines are required to terminate each reply.

Multiple clients can be served concurrently by this WebSocket server, but
beware that its scalability and robustness is somewhat uncertain:
See https://websockets.readthedocs.io/en/stable/deployment.html

TODO: SSL support coming soon!
"""

import argparse
import asyncio
import json
import logging
import uuid
import sys

import websockets

import mod9.reformat.config as config
from mod9.reformat import utils

DEBUG_LOGGING_MESSAGE_INTERVAL = 10000

# Incrementing request ID counter and logger adapter to prepend to each request.
request_id = 1


class CustomLoggerAdapter(logging.LoggerAdapter):
    """
    Add ``request_id`` and ``rest_uuid`` to logs.

    E.g. see:
    https://docs.python.org/3/howto/logging-cookbook.html#using-loggeradapters-to-impart-contextual-information
    https://github.com/python/cpython/blob/96cf5a63d2dbadaebf236362b4c7c09c51fda55c/Lib/logging/__init__.py#L1799-L1883
    """
    def process(self, msg, kwargs):
        """Prepend [{ws_uuid}:{request_id}:{level}] to message."""
        ws_uuid = self.extra.get('ws_uuid', str(uuid.UUID(int=0)))[:8]
        request_id = self.extra.get('request_id', 0)
        level = logging.getLevelName(self.extra['level']).lower()
        return f"[{ws_uuid}:{request_id}:{level}]\t{msg}", kwargs

    def log(self, level, msg, *args, **kwargs):
        """Override: pass level as element of ``self.extra`` dict."""
        if self.isEnabledFor(level):
            self.extra['level'] = level
            msg, kwargs = self.process(msg, kwargs)
            self.logger.log(level, msg, *args, **kwargs)


async def relay_options(websocket, sock_writer, logger):
    """
    Relay options from client's (first) WebSocket message to Engine TCP socket.

    Args:
        websocket (websockets.server.WebSocketServerProtocol):
            Client WebSocket to receive options from.
        sock_writer (asyncio.StreamWriter):
            Engine TCP socket to write options to.

    Returns:
        bytes:
            End-of-file byte sequence, possibly set by client options.

    Raises:
        utils.Mod9BadRequestError:
            If client's first message cannot be parsed properly.
    """
    options_message = await websocket.recv()
    logger.debug("Received options message: %s", options_message)

    # Re-serialize WebSocket-specified options as single-line JSON terminated with a newline.
    try:
        options_json = options_message.decode()
        options = json.loads(options_json)
        options_json = json.dumps(options, separators=(',', ':'))
        options_line = options_json.encode('utf-8') + b'\n'
    except json.decoder.JSONDecodeError as e:
        raise utils.Mod9BadRequestError(
            f"Could not parse options from first message (JSON decode error: {e})."
        )
    except Exception:
        raise utils.Mod9BadRequestError(
            f"Could not parse options from first message (comprising {len(options_message)} bytes)."
        )

    # Write the single-line newline-terminated options to Engine over TCP.
    sock_writer.write(options_line)
    await sock_writer.drain()

    # The WebSocket server will need this later in order to properly terminate the request.
    eof = options.get('eof', 'END-OF-FILE').encode('utf-8')
    return eof


async def receive_messages(websocket, logger):
    """
    Receive messages from WebSocket and yield as async bytes generator.
    Stop iteration upon receipt of an empty message.

    Args:
        websocket (websockets.server.WebSocketServerProtocol):
            WebSocket to read from.

    Yields:
        Iterable[bytes]
    """
    num_messages = 0
    total_bytes = 0

    while True:
        message = await websocket.recv()
        num_messages += 1
        total_bytes += len(message)

        if message:
            if num_messages % DEBUG_LOGGING_MESSAGE_INTERVAL == 0:
                logger.debug(
                    "Received %d messages totaling %d bytes.",
                    num_messages,
                    total_bytes,
                )
            yield message
        else:
            # Received empty message, indicating end of messages.
            logger.debug("Received %d messages totaling %d bytes.", num_messages, total_bytes)
            return


async def write_chunks_to_socket(async_bytes_generator, sock_writer, eof):
    """
    Write contents of async bytes generator to given socket.
    The bytes from the generator might be re-chunked.

    Args:
        async_bytes_generator (Iterable[bytes]):
            Data to send to socket.
        sock_writer (asyncio.StreamWriter):
            Socket to send to.
        eof (bytes):
            End-of-file byte sequence.

    Returns:
        None
    """
    async for chunk in async_bytes_generator:
        for i in range(0, len(chunk), config.MAX_CHUNK_SIZE):
            subchunk = chunk[i:i+config.MAX_CHUNK_SIZE]
            sock_writer.write(subchunk)
            await sock_writer.drain()

    # This may not be necessary if the client sent WAV, or already sent their own EOF,
    #  but it seems the Engine doesn't mind receiving an extra EOF in those situations.
    sock_writer.write(eof)
    await sock_writer.drain()


async def read_lines_from_socket(sock_reader):
    """
    Read lines from socket and yield as async lines generator.

    Args:
        sock_reader (asyncio.StreamReader):
            Socket to read lines from.

    Yields:
        Iterable[bytes]
    """
    while True:
        try:
            line = await sock_reader.readline()
        except ValueError:
            raise utils.Mod9LimitOverrunError(
                'Could not read in reply from Engine due to current limit of'
                f" {config.WEBSOCKET_LIMIT_BYTES} bytes."
                ' Please retry the request with a larger value passed to the'
                ' --websocket-limit-bytes command-line argument.'
            )
        if not line:
            return
        yield line


async def send_messages(async_lines_generator, websocket, logger):
    """
    Send contents of async lines generator to given WebSocket.
    Each line is stripped before being sent as a message.

    Args:
        async_lines_generator (Iterable[bytes]):
            Lines to send to socket.
        websocket (websockets.server.WebSocketServerProtocol):
            WebSocket to send to.

    Returns:
        None
    """
    first_response_line = True
    async for line in async_lines_generator:
        message = line.decode().strip().encode('utf-8')
        await websocket.send(message)
        logger.debug("Sent message: %s", message)

        if first_response_line:
            # To pass through Engine UUID in logging.
            engine_uuid = json.loads(message.decode()).get('uuid')
            engine_version = json.loads(message.decode()).get('engine', {}).get('version')
            logger.info(
                "Engine version %s and UUID %s received from first Engine reply.",
                'missing' if not engine_version else engine_version,
                'not' if not engine_uuid else engine_uuid,
            )
            first_response_line = False


async def send_error(websocket, details, logger):
    """
    Send an error message to the WebSocket client, formatted like Engine reply.

    Args:
        websocket (websockets.server.WebSocketServerProtocol):
            WebSocket to send to.
        details (str):
            Information to be included in the .error field, to be prefixed with
            a [WebSocket] tag clarifying that this didn't come from the Engine.

    Returns:
        None
    """
    reply = {'status': 'failed', 'error': '[WebSocket] '+details}
    try:
        reply_json = json.dumps(reply, separators=(',', ':'))
        reply_message = reply_json.encode('utf-8')
        await websocket.send(reply_message)
    except Exception:
        logger.error('Could not send exception info to client.', exc_info=True)
    else:
        logger.debug("Sent error message: %s", reply_message)


async def handle_request(websocket, path):
    """
    Asynchronously send options and audio from WebSocket to Engine
    socket, then send response from Engine socket to WebSocket.

    Args:
        websocket (websockets.server.WebSocketServerProtocol):
            WebSocket from client request.
        path (string):
            WebSocket URI path from client request.

    Returns:
        None
    """
    global request_id
    ws_uuid = str(uuid.uuid4())
    logger = CustomLoggerAdapter(logging.root, {'ws_uuid': ws_uuid, 'request_id': request_id})
    request_id += 1

    logger.info(
        "Request from client %s at path %s assigned WS UUID %s.",
        websocket.remote_address[0],
        path,
        ws_uuid,
    )

    # Resources that might be created below, and should then be cleaned up later.
    sock_writer, websocket_to_socket = None, None

    try:
        # Connect to the ASR Engine over TCP.
        sock_reader, sock_writer = await asyncio.open_connection(
            host=config.ASR_ENGINE_HOST,
            port=config.ASR_ENGINE_PORT,
            limit=config.WEBSOCKET_LIMIT_BYTES,
        )
    except Exception:
        logger.error('Could not connect to ASR Engine.')
        await send_error(
            websocket,
            'Could not connect to ASR Engine; contact server operator.',
            logger,
        )
        return

    try:
        # Send the request options, and parse the EOF byte sequence.
        eof = await relay_options(websocket, sock_writer, logger)

        # Receive messages from WebSocket & write chunks (i.e. audio) to Engine socket.
        websocket_to_socket = asyncio.get_event_loop().create_task(
            write_chunks_to_socket(receive_messages(websocket, logger), sock_writer, eof),
        )

        # Read lines (i.e. replies) from Engine socket & send messages to WebSocket.
        socket_to_websocket = asyncio.get_event_loop().create_task(
            send_messages(read_lines_from_socket(sock_reader), websocket, logger),
        )

        # Wait until the Engine closes the socket.
        # TODO: it might be nice if we inspected and logged the .status of the final Engine message.
        await socket_to_websocket
        logger.debug('ASR Engine closed TCP connection.')
    except utils.Mod9BadRequestError as e:
        logger.error('Request failed due to bad request from client.')
        await send_error(websocket, str(e), logger)
    except websockets.ConnectionClosed:
        logger.error('WebSocket closed unexpectedly; client disconnected?')
        # We cannot send_error to the client in this case, since the WebSocket is closed.
    except Exception:
        # Include the traceback so operators can contact support@mod9.com with helpful information.
        logger.error('Request failed unexpectedly.', exc_info=True)
        # Do not relay details to the client, as it's likely too server-specific or sensitive.
        await send_error(
            websocket,
            'Request failed unexpectedly; contact server operator.',
            logger,
        )
    finally:
        # Cancel any audio remaining to be sent, since the Engine can't accept it anymore.
        if websocket_to_socket:
            websocket_to_socket.cancel()

        # Close the underlying socket; this is not a TCP half-close (cf. write_eof).
        if sock_writer:
            sock_writer.close()
            await sock_writer.wait_closed()

        # Drain messages left over in WebSocket recv buffer to ensure reply CLOSE frame is received,
        #  e.g. if the client has sent more data than the Engine was able to accept.
        websocket_close_task = asyncio.get_event_loop().create_task(websocket.close())
        try:
            while True:
                _ = await websocket.recv()  # Discard received message.
        except websockets.ConnectionClosedOK:
            logger.debug('WebSocket connection drained successfully.')
        except Exception:
            logger.warning('WebSocket connection was not drained successfully.')

        await websocket_close_task
        logger.info("WebSocket connection closed.")


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--engine-host',
        metavar='HOST',
        help='ASR Engine host name.'
             ' Can also be set by ASR_ENGINE_HOST environment variable.',
        default=config.ASR_ENGINE_HOST,
    )
    parser.add_argument(
        '--engine-port',
        metavar='PORT',
        help='ASR Engine port number.'
             ' Can also be set by ASR_ENGINE_PORT environment variable.',
        type=int,
        default=config.ASR_ENGINE_PORT,
    )
    parser.add_argument(
        '--host',
        help='WebSocket host address. Can be set to 0.0.0.0 for external access.',
        default='127.0.0.1'  # Internal access only.
    )
    parser.add_argument(
        '--port',
        help='WebSocket port number.',
        type=int,
        default=9980,  # The ASR Engine is typically at 9900, so add 80 (i.e. standard HTTP port).
        # NOTE: the default for the REST API should be 8080, so try to avoid a conflict with that.
    )
    parser.add_argument(
        '--log-level',
        metavar='LEVEL',
        help='Verbosity of logging.',
        default='INFO',
    )
    parser.add_argument(
        '--skip-engine-check',
        action='store_true',
        help='When starting server, do not wait for ASR Engine.',
        default=False,
    )
    parser.add_argument(
        '--websocket-limit-bytes',
        help='Limit on number of bytes allowed per reply line read from Engine. Default: 1 MiB.',
        type=int,
    )
    args = parser.parse_args()

    # TODO: we should instead adjust the log level of our own logger rather than the root logger.
    # TODO: Trace request IDs in logging.
    args.log_level = args.log_level.upper()
    logging.basicConfig(
        format="[%(asctime)s] %(message)s",
        level=args.log_level,
    )

    # Use local time with ISO-8601 format for time output.
    logging.Formatter.formatTime = utils.format_log_time

    if args.log_level == 'DEBUG':
        # These are overly verbose for our purposes.
        logging.getLogger('asyncio').setLevel(logging.INFO)
        logging.getLogger('websockets').setLevel(logging.INFO)

    config.ASR_ENGINE_HOST = args.engine_host
    config.ASR_ENGINE_PORT = args.engine_port

    if args.websocket_limit_bytes:
        config.WEBSOCKET_LIMIT_BYTES = args.websocket_limit_bytes

    logger = CustomLoggerAdapter(logging.root, dict())

    if not args.skip_engine_check:
        utils.test_host_port(logger=logger)

    logger.info(
        "Running a WebSocket server (Python SDK version %s) at %s on port %d.",
        config.WRAPPER_VERSION,
        args.host,
        args.port,
    )
    start_server = websockets.serve(handle_request, host=args.host, port=args.port)
    asyncio.get_event_loop().run_until_complete(start_server)

    try:
        asyncio.get_event_loop().run_forever()
    except KeyboardInterrupt:
        logger.warning('Exiting on interrupt signal.')
        sys.exit(130)  # Don't show Python traceback.


if __name__ == '__main__':
    main()
