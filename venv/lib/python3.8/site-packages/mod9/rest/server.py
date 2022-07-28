#!/usr/bin/env python3

"""
Mod9 ASR REST API: wrapper over the ASR Engine.

This server implements a RESTful API compatible with Google Cloud STT.
It can run as a standalone Flask server, but is best deployed via WSGI.
"""

import argparse
from collections import OrderedDict
from datetime import datetime
import itertools
import logging
import logging.config
from random import randint
import threading
import uuid

from flask import Flask
from flask import Response
from flask_restful import (
    reqparse,
    abort,
    Api,
    Resource,
)

from mod9.asr import speech_mod9
import mod9.reformat.config as config
from mod9.reformat import utils
from mod9.reformat import google as reformat

# Use local time with ISO-8601 format for time output.
logging.Formatter.formatTime = utils.format_log_time

# Configure logging to work well with either WSGI or development webserver.
#  See: https://flask.palletsprojects.com/en/2.0.x/logging/#basic-configuration
logging.config.dictConfig({
    'version': 1,
    'formatters': {
        'default': {
            # For desired formatting, add log level to message below in ``CustomLoggerAdapter``.
            'format': "[%(asctime)s] %(message)s",
        }
    },
    'handlers': {
        'wsgi': {
            'class': 'logging.StreamHandler',
            'stream': 'ext://flask.logging.wsgi_errors_stream',
            'formatter': 'default'
        }
    },
    'root': {
        'level': 'INFO',
        'handlers': ['wsgi']
    }
})


# Use the standard name for a Flask app, at highest scope, to faciliate WSGI integration.
app = Flask(__name__)

# Make JSON output readable.
#  ``RESTFUL_JSON`` is a config var from ``flask_restful`` that
#  allows JSON formatting similar to ``json.dumps()``, see docs:
#  https://flask-restful.readthedocs.io/en/latest/extending.html#:~:text=RESTFUL_JSON
app.config['RESTFUL_JSON'] = {
    'ensure_ascii': False,  # Pass Engine's UTF-8 encoded bytes as-is, not escaped as ASCII.
    'indent': 2,            # Match Google's formatting.
    'sort_keys': False,     # Enable use of OrderedDict, to match Google's formatting.
}

if config.FLASK_ENV is not None:
    app.config['ENV'] = config.FLASK_ENV
else:
    # Stop output of warning message on server start.
    #  Note this is not recommended by Flask devs, but should be fine for our purposes:
    #  https://flask.palletsprojects.com/en/1.1.x/config/#environment-and-debug-features
    app.config['ENV'] = 'development'

# Use Flask-RESTful, a well-designed framework with convenient utilities.
api = Api(app)

# Input parser for Recognize and LongRunningRecognize resources.
parser = reqparse.RequestParser()
parser.add_argument(
    'config',
    required=True,
    help="Required 'config' contains ASR settings in JSON format.",
    location='json',
    type=dict,
)
parser.add_argument(
    'audio',
    required=True,
    help="Required 'audio' contains audio file path or encoded bytes.",
    location='json',
    type=dict,
)

# Storage for operation names and results used by LongRunningRecognize.
operation_names = []
operation_results = {}

# Audio URI schemes to accept. Operator can set at server launch; default is to accept none.
allowed_uri_schemes = config.ASR_REST_API_ALLOWED_URI_SCHEMES


class CustomLoggerAdapter(logging.LoggerAdapter):
    """
    Add ``rest_uuid`` and log level to logs.

    Add log level here so we can format it how we want.

    E.g. see:
    https://docs.python.org/3/howto/logging-cookbook.html#using-loggeradapters-to-impart-contextual-information
    https://github.com/python/cpython/blob/96cf5a63d2dbadaebf236362b4c7c09c51fda55c/Lib/logging/__init__.py#L1799-L1883
    """
    def process(self, msg, kwargs):
        """Prepend [{rest_uuid}:{level}] to message."""
        rest_uuid = self.extra.get('rest_uuid', str(uuid.UUID(int=0)))[:8]
        level = logging.getLevelName(self.extra['level']).lower()
        # Space before tab to avoid jagged logs for, e.g., INFO vs WARNING.
        return f"[{rest_uuid}:{level}] \t{msg}", kwargs

    def log(self, level, msg, *args, **kwargs):
        """Override: pass level as element of ``self.extra`` dict."""
        if self.isEnabledFor(level):
            self.extra['level'] = level
            msg, kwargs = self.process(msg, kwargs)
            self.logger.log(level, msg, *args, **kwargs)


@app.before_first_request
def log_pysdk_version_and_schemes():
    """
    Log server information on start, e.g. version, enabled URI schemes.

    Args:
        None

    Returns:
        None
    """

    logger = CustomLoggerAdapter(logging.root, dict())
    logger.info("Mod9 ASR REST API (Python SDK version %s).", config.WRAPPER_VERSION)

    logger.info(
        "Accepting URI schemes: %s.",
        ', '.join(allowed_uri_schemes) if len(allowed_uri_schemes) else None,
    )

    if 'http' in allowed_uri_schemes and 'https' not in allowed_uri_schemes:
        logger.warning('REST API set to allow http:// but NOT https:// audio URIs.')


def make_request_logger():
    """
    Create a REST UUID and logger for a request.

    Args:
        None

    Returns:
        CustomLoggerAdapter:
            The logger to be used for the request.
    """

    rest_uuid = str(uuid.uuid4())
    logger = CustomLoggerAdapter(logging.root, {'rest_uuid': rest_uuid})
    logger.info("REST UUID: %s.", rest_uuid)
    return logger


def make_response_headers(rest_uuid, engine_uuid=None, engine_version=None):
    """
    Prepare response headers.

    Args:
        rest_uuid (str):
            Request UUID assigned by REST API.
        engine_uuid (Union[str, None]):
            UUID received from Engine for request, or None.
        engine_version (Union[str, None]):
            Version received from Engine as part of response, or None.

    Returns:
        dict:
            The headers.
    """

    headers = {
        'X-Mod9-ASR-REST-API-UUID': rest_uuid,
        'X-Mod9-ASR-REST-API-Version': config.WRAPPER_VERSION,
    }
    if engine_uuid:
        headers['X-Mod9-ASR-Engine-UUID'] = engine_uuid
    if engine_version:
        headers['X-Mod9-ASR-Engine-Version'] = engine_version

    return headers


def place_reformatted_mod9_response_in_operation_results(
    mod9_config_settings,
    mod9_audio_settings,
    operation_name,
    logger,
):
    """
    Get and format response from Mod9 ASR Engine. Place response in
    proper ``operation_result``.

    Args:
        mod9_config_settings (dict[str, Union[dict, float, int, str]]):
            Mod9-style transcription options.
        mod9_audio_settings (Union[
                GeneratorType,
                TeeGeneratorType,
                str,
        ]):
            Audio content to be transcribed.
        operation_name (str):
            Transcription operation UUID.

    Returns:
        None
    """

    # Response type metadata constant string.
    response_type = 'type.googleapis.com/google.cloud.speech.v1p1beta1.LongRunningRecognizeResponse'

    # Talk to Mod9 ASR Engine.
    try:
        engine_response = utils.get_transcripts_mod9(mod9_config_settings, mod9_audio_settings)
    except (KeyError, ConnectionError):
        logger.exception('Error communicating with Mod9 ASR Engine.')
        abort(Response(status=500, headers=make_response_headers(logger.extra['rest_uuid'])))

    # Peek at first Engine response line containing Engine UUID and version.
    first_response_line = next(engine_response)
    engine_response = itertools.chain([first_response_line], engine_response)
    engine_uuid = first_response_line.get('uuid')
    engine_version = first_response_line.get('engine', dict()).get('version')
    logger.info(
        "Engine version %s replied with UUID: %s.",
        engine_version,
        engine_uuid,
    )

    # Place response in operation_results.
    operation_results[operation_name]['response'] = OrderedDict(
        [
            ('@type', response_type),
            ('results', list(reformat.result_from_mod9(engine_response, logger=logger))),
        ]
    )
    operation_results[operation_name]['done'] = True
    operation_results[operation_name].move_to_end('response')

    # Hack to remove .isFinal field from response.
    [result.pop('isFinal') for result in operation_results[operation_name]['response']['results']]

    # Update metadata.
    current_time = datetime.utcnow().isoformat() + 'Z'
    operation_results[operation_name]['metadata']['lastUpdateTime'] = current_time
    operation_results[operation_name]['metadata']['progressPercent'] = 100
    operation_results[operation_name]['metadata'].move_to_end('startTime')
    operation_results[operation_name]['metadata'].move_to_end('lastUpdateTime')

    logger.info('Request completed.')


class Recognize(Resource):
    """Implement synchronous ASR for ``/recognize`` endpoint."""

    def post(self):
        """Perform synchronous ASR on POSTed config and audio."""
        logger = make_request_logger()

        # Translate request -> Mod9 format.
        args = parser.parse_args()
        try:
            mod9_config_settings, mod9_audio_settings = reformat.input_to_mod9(
                args,
                module=speech_mod9,
                logger=logger,
            )
        except Exception as e:
            logger.exception('Invalid arguments.')
            abort(
                Response(
                    status=400,
                    response=f"Invalid arguments: {e}",
                    headers=make_response_headers(logger.extra['rest_uuid']),
                ),
            )

        if isinstance(mod9_audio_settings, str):
            try:
                utils.validate_uri_scheme(mod9_audio_settings, allowed_uri_schemes)
            except utils.Mod9DisabledAudioURISchemeError as e:
                logger.error(e)
                abort(
                    Response(
                        status=403,
                        response=str(e) + '\n',
                        headers=make_response_headers(logger.extra['rest_uuid']),
                    ),
                )

        # Talk to Mod9 ASR Engine.
        try:
            engine_response = utils.get_transcripts_mod9(mod9_config_settings, mod9_audio_settings)
        except (KeyError, ConnectionError):
            logger.exception('Error communicating with Mod9 ASR Engine.')
            abort(Response(status=500, headers=make_response_headers(logger.extra['rest_uuid'])))

        # Peek at first Engine response line containing Engine UUID and version.
        first_response_line = next(engine_response)
        engine_response = itertools.chain([first_response_line], engine_response)
        engine_uuid = first_response_line.get('uuid')
        engine_version = first_response_line.get('engine', dict()).get('version')
        logger.info(
            "Engine version %s replied with UUID: %s.",
            engine_version,
            engine_uuid,
        )

        # Translate response -> external API format.
        response = {
            'results': list(reformat.result_from_mod9(engine_response, logger=logger))
        }

        # Hack to remove .isFinal field from response.
        [result.pop('isFinal') for result in response['results']]

        logger.info('Request completed.')

        headers = make_response_headers(logger.extra['rest_uuid'], engine_uuid, engine_version)
        return response, 200, headers


class Operations(Resource):
    """
    Implement operations name fetching for ``/operations`` endpoint.
    """

    def get(self):
        """Output finished and presently running operation names."""
        logger = make_request_logger()

        logger.info('Fetched operations.')
        headers = make_response_headers(logger.extra['rest_uuid'])
        return {'operations': list(reversed(operation_names))}, 200, headers


class GetOperationByName(Resource):
    """
    Implement operation result fetching for
    ``/operations/<operation_name>`` endpoint.
    """

    def get(self, operation_name):
        """Output operation with given operation name."""
        logger = make_request_logger()

        logger.info("Fetched operation %s.", operation_name)
        headers = make_response_headers(logger.extra['rest_uuid'])
        return operation_results[str(operation_name)], 200, headers


class LongRunningRecognize(Resource):
    """
    Implement asynchronous ASR for ``/longrunningrecognize`` endpoint.
    """

    def post(self):
        """Perform asynchronous ASR on POSTed config and audio."""
        logger = make_request_logger()

        # Translate request -> Mod9 format.
        args = parser.parse_args()
        try:
            mod9_config_settings, mod9_audio_settings = reformat.input_to_mod9(
                args,
                module=speech_mod9,
                logger=logger,
            )
        except Exception as e:
            logger.exception('Invalid arguments.')
            abort(
                Response(
                    status=400,
                    response=f"Invalid arguments: {e}",
                    headers=make_response_headers(logger.extra['rest_uuid']),
                ),
            )

        if isinstance(mod9_audio_settings, str):
            try:
                utils.validate_uri_scheme(mod9_audio_settings, allowed_uri_schemes)
            except utils.Mod9DisabledAudioURISchemeError as e:
                logger.error(e)
                abort(
                    Response(
                        status=403,
                        response=str(e) + '\n',
                        headers=make_response_headers(logger.extra['rest_uuid']),
                    ),
                )

        # Generate a name for operation and append to list of operations.
        # Similar to Google STT, make this a 19-character string of random integers.
        operation_name = str(randint(1, 9)) + ''.join(str(randint(0, 9)) for _ in range(18))
        operation_names.append({'name': operation_name})

        # Set up initial operation_result.
        # Reponse type metadata constant string.
        meta_type = 'type.googleapis.com/google.cloud.speech.v1p1beta1.LongRunningRecognizeMetadata'
        start_time = datetime.utcnow().isoformat() + 'Z'
        operation_metadata = OrderedDict(
            [
                ('@type', meta_type),
                ('startTime', start_time),
                ('lastUpdateTime', start_time),
            ]
        )

        operation_results[operation_name] = OrderedDict([('name', operation_name)])
        operation_results[operation_name]['metadata'] = operation_metadata

        # Send request to Mod9 ASR Engine. Result will appear in operation_results when done.
        request_thread = threading.Thread(
            target=place_reformatted_mod9_response_in_operation_results,
            args=(
                mod9_config_settings,
                mod9_audio_settings,
                operation_name,
                logger,
            ),
        )
        request_thread.start()

        headers = make_response_headers(logger.extra['rest_uuid'])
        return {'name': operation_name}, 200, headers


api.add_resource(
    Recognize,
    '/speech:recognize',
    '/speech:recognize/',
)
api.add_resource(
    LongRunningRecognize,
    '/speech:longrunningrecognize',
    '/speech:longrunningrecognize/',
)
api.add_resource(
    Operations,
    '/operations',
    '/operations/',
)
api.add_resource(
    GetOperationByName,
    '/operations/<int:operation_name>',
    '/operations/<int:operation_name>/',
)


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--engine-host',
        help='ASR Engine host name.'
             ' Can also be set by ASR_ENGINE_HOST environment variable.',
        default=config.ASR_ENGINE_HOST,
    )
    parser.add_argument(
        '--engine-port',
        help='ASR Engine port.'
             ' Can also be set by ASR_ENGINE_PORT environment variable.',
        type=int,
        default=config.ASR_ENGINE_PORT,
    )
    parser.add_argument(
        '--host',
        help='REST API host address. Can be set to 0.0.0.0 for external access.',
        default='127.0.0.1'  # Internal access only.
    )
    parser.add_argument(
        '--port',
        help='REST API port number.',
        type=int,
        default=8080,
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
    uri_group = parser.add_argument_group(
        'Audio URI schemes',
        'Allow clients to input audio using a variety of URI schemes.',
    )
    uri_group.add_argument(
        '--allow-file-uri',
        help='Allow clients to use `file://` URI scheme.',
        action='store_const',
        const='file',
    )
    uri_group.add_argument(
        '--allow-gs-uri',
        help='Allow clients to use Google `gs://` URI scheme.',
        action='store_const',
        const='gs',
    )
    uri_group.add_argument(
        '--allow-http-uri',
        help='Allow clients to use `http://` URI scheme.',
        action='store_const',
        const='http',
    )
    uri_group.add_argument(
        '--allow-https-uri',
        help='Allow clients to use `https://` URI scheme.',
        action='store_const',
        const='https',
    )
    uri_group.add_argument(
        '--allow-s3-uri',
        help='Allow clients to use AWS `s3://` URI scheme.',
        action='store_const',
        const='s3',
    )
    args = parser.parse_args()

    global allowed_uri_schemes
    allowed_uri_schemes = allowed_uri_schemes.union(
        {
            args.allow_file_uri,
            args.allow_gs_uri,
            args.allow_http_uri,
            args.allow_https_uri,
            args.allow_s3_uri,
        }
    )
    if None in allowed_uri_schemes:
        allowed_uri_schemes.remove(None)

    logger = CustomLoggerAdapter(logging.root, dict())
    logger.setLevel(args.log_level.upper())

    config.ASR_ENGINE_HOST = args.engine_host
    config.ASR_ENGINE_PORT = args.engine_port

    if not args.skip_engine_check:
        utils.test_host_port(logger=logger)

    # See https://flask.palletsprojects.com/en/2.0.x/deploying/index.html
    logger.warning(
        'This REST API is being run directly as a Python Flask server.'
        ' For production deployment, use WSGI.'
    )
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == '__main__':
    main()
