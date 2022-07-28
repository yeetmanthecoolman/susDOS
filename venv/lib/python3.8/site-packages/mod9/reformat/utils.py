"""
Utilities including functions to talk to the Mod9 ASR Engine TCP Server.
"""

import datetime
import io
from itertools import _tee as TeeGeneratorType
import json
import logging
import os
import requests
import socket
import threading
import time
from types import GeneratorType
from urllib.parse import urlparse
from urllib.parse import quote

import boto3
from botocore.exceptions import ClientError as AWSClientError
import google.auth
import google.auth.transport.requests
import google.cloud.storage
import google.resumable_media.requests
from packaging import version

from mod9.reformat import config


class Mod9IncompatibleEngineVersionError(Exception):
    pass


class Mod9InsufficientEngineCapacityError(Exception):
    pass


class Mod9UnexpectedEngineResponseError(Exception):
    pass


class Mod9EngineFirstResponseNotProcessingError(Mod9UnexpectedEngineResponseError):
    pass


class Mod9EngineResponseNotCompletedError(Mod9UnexpectedEngineResponseError):
    pass


class Mod9EngineCouldNotGetModelNameError(Mod9UnexpectedEngineResponseError):
    pass


class Mod9UnexpectedEngineStateError(Exception):
    pass


class Mod9DisabledAudioURISchemeError(Exception):
    pass


class Mod9BadRequestError(Exception):
    pass


class Mod9LimitOverrunError(Exception):
    pass


class Mod9CouldNotAccessFileError(Exception):
    pass


class PropagatingThread(threading.Thread):
    """
    Thread class that propagates any exception that occurred during
    target function execution.

    See: https://stackoverflow.com/a/31614591
    """

    def run(self):
        self.exc = None
        try:
            self.ret = self._target(*self._args, **self._kwargs)
        except BaseException as e:
            self.exc = e

    def join(self):
        super().join()
        if self.exc:
            raise self.exc
        return self.ret


def get_local_path(local_path_with_prefix):
    """
    Get absolute path to file, assuming it begins with 'file://'.

    Args:
        local_path_with_prefix (str):
            Prefixed, possibly non-absolute path to file.

    Returns:
        str:
            Un-prefixed absolute path to file.
    """

    local_path = local_path_with_prefix.replace('file://', '', 1)
    return os.path.abspath(os.path.expanduser(local_path))


def parse_wav_encoding(audio_settings):
    """
    Determine if audio_settings has WAV file header.

    Args:
        audio_settings (Union[None, str, TeeGeneratorType[bytes]]):
            Audio content or location.

    Returns:
        Union[bool, str]:
            False or truthy value, which may be the encoding name.
        int:
            Sample rate if WAV header successfully parsed, else 0.
        int:
            Channel count if WAV header is successfully parsed, else 0.
    """

    if audio_settings is None:
        return False, 0, 0
    elif isinstance(audio_settings, TeeGeneratorType):
        audio_settings_header = next(audio_settings)
        return extract_encoding_rate_channel_from_wav_header(audio_settings_header)
    elif isinstance(audio_settings, str):
        uri_scheme = urlparse(audio_settings).scheme
        if uri_scheme == 'http' or uri_scheme == 'https':
            headers = {'Range': 'bytes=0-28'}
            with requests.get(audio_settings, headers=headers) as r:
                r.raise_for_status()
                return extract_encoding_rate_channel_from_wav_header(r.content)
        # TODO: try to read the WAV header for gs://, s3://, and file:// schemes.
        else:
            # Will fail in the pathological case of a non-WAV file stored with a ``.wav`` suffix,
            #  and when a WAV file is stored without a ``.wav`` suffix.
            return audio_settings.endswith('.wav'), 0, 0
    else:
        raise TypeError('Expected type None or TeeGeneratorType or str.')


def extract_encoding_rate_channel_from_wav_header(wav_header):
    """
    Attempt to extract encoding, sample rate from possible WAV header.

    Args:
        wav_header (bytes):
            Possible WAV file header.

    Returns:
        Union[bool, str]:
            False or truthy value, which may be the encoding name.
        int:
            Sample rate if WAV header successfully parsed, else 0.
        int:
            Channel count if WAV header is successfully parsed, else 0.
    """
    if wav_header[:4] == b'RIFF' and wav_header[8:12] == b'WAVE':
        # Get the encoding type, which is always truthy.
        encoding_bytes = wav_header[20:22]
        if encoding_bytes == b'\x01\x00':
            wav_encoding = 'pcm_s16le'
        elif encoding_bytes == b'\x03\x00':
            wav_encoding = 'pcm_f32le'
        elif encoding_bytes == b'\x06\x00':
            wav_encoding = 'a-law'
        elif encoding_bytes == b'\x07\x00':
            wav_encoding = 'mu-law'
        else:
            # TODO: handle WAVE_FORMAT_EXTENSIBLE with subformats.
            wav_encoding = True
        # Get number of channels.
        channels = int.from_bytes(wav_header[22:24], byteorder='little')
        # Get the sample rate.
        sample_rate = int.from_bytes(wav_header[24:28], byteorder='little')
        return wav_encoding, sample_rate, channels
    else:
        return False, 0, 0


def camel_case_to_snake_case(camel_case_string):
    """
    Convert a camelCase string to a snake_case string.

    Args:
        camel_case_string (str):
            A string using lowerCamelCaseConvention.

    Returns:
        str:
            A string using snake_case_convention.
    """

    # https://stackoverflow.com/a/44969381
    return ''.join(['_' + c.lower() if c.isupper() else c for c in camel_case_string]).lstrip('_')


def snake_case_to_camel_case(snake_case_string):
    """
    Convert a snake_case string to a camelCase string. Initial '_' will
    cause capitalization.

    Args:
        snake_case_string (str):
            A string using snake_case_convention.

    Returns:
        str:
            A string using lowerCamelCase or UpperCamelCase convention.
    """

    lower_case_words = snake_case_string.split(sep='_')
    camel_case_string = lower_case_words[0] \
        + ''.join(word.capitalize() for word in lower_case_words[1:])
    return camel_case_string


def recursively_convert_dict_keys_case(dict_in, convert_key):
    """
    Convert all the keys in a (potentially) recursive dict using
    function convert_key.

    Args:
        dict_in (dict[str, Union[dict, str]]):
            Dict of dicts and strs with str keys.
        convert_key (Callable[[str], str]):
            Function to convert str to str, to be applied to keys.

    Returns:
        dict[str, str]:
            Dict of dicts and strs with str keys; same structure as
            dict_in, with unchanged values, but with values transformed
            according to convert_key.
    """
    if not isinstance(dict_in, dict):
        return dict_in
    dict_out = {convert_key(key): value for key, value in dict_in.items()}
    for key, value in dict_out.items():
        if isinstance(value, dict):
            dict_out[key] = recursively_convert_dict_keys_case(dict_out[key], convert_key)
        if isinstance(value, list):
            dict_out[key] = [
                recursively_convert_dict_keys_case(item, convert_key) for item in dict_out[key]
            ]
    return dict_out


def get_bucket_key_from_path(bucketed_path_with_prefix, prefix):
    """
    Get bucket and key from path, assuming it begins with given prefix.

    Args:
        bucketed_path_with_prefix (str):
            Prefixed path including bucket and key.
        prefix (str):
            Prefix to look for in bucketed_path_with_prefix.

    Returns:
        tuple:
            bucket_name (str):
                Parsed name of bucket.
            key_name (str):
                Parsed name of key.
    """

    bucket_key = bucketed_path_with_prefix.replace(prefix, '', 1)
    bucket_name, key_name = bucket_key.split(sep='/', maxsplit=1)
    return bucket_name, key_name


def convert_gs_uri_to_http_url(uri):
    """
    Convert URI, assumed to begin with 'gs://', to URL for use with
    Google Resumable Media.

    Args:
        uri (str):
            URI to Google Cloud Storage beginning with ``gs://``.

    Returns:
        str:
            URL to Google Cloud Storage usable with Google Resumable
            Media.
    """

    bucket, key = get_bucket_key_from_path(uri, 'gs://')

    # https://cloud.google.com/storage/docs/request-endpoints#encoding
    key = quote(key, safe='')

    url = f"https://storage.googleapis.com/download/storage/v1/b/{bucket}" \
          f"/o/{key}?alt=media"
    return url


def generator_producer(generator, sock):
    """
    Send contents of generator to given socket.

    Args:
        generator (Iterable[bytes]):
            Data to send to socket.
        sock (socket.socket):
            Socket to send to.

    Returns:
        None
    """

    for chunk in generator:
        for i in range(0, len(chunk), config.MAX_CHUNK_SIZE):
            sock.sendall(chunk[i:i+config.MAX_CHUNK_SIZE])


def file_producer(uri, sock):
    """
    Send contents of file in chunks to given socket.

    Args:
        uri (str):
            Prefixed path to local file to be sent to socket.
        sock (socket.socket):
            Socket to send to.

    Returns:
        None
    """

    with open(get_local_path(uri), 'rb') as fin:
        for chunk in iter(lambda: fin.read(config.MAX_CHUNK_SIZE), b''):
            sock.sendall(chunk)


def http_producer(uri, sock):
    """
    Send contents of file hosted at URL in chunks to given socket.

    Args:
        uri (str):
            Prefixed public URL file to be sent to socket.
        sock (socket.socket):
            Socket to send to.

    Returns:
        None
    """

    with requests.get(uri, stream=True) as r:
        r.raise_for_status()
        for chunk in r.iter_content(chunk_size=config.MAX_CHUNK_SIZE):
            sock.sendall(chunk)


def google_cloud_producer(uri, sock):
    """
    Send contents of file hosted on GCS in chunks to given socket.

    Args:
        uri (str):
            Prefixed Google Cloud Storage file to to be sent to socket.
            Note you must have access to the file with currently loaded
            Google credentials.
        sock (socket.socket):
            Socket to send to.

    Returns:
        None
    """

    url = convert_gs_uri_to_http_url(uri)

    # https://googleapis.dev/python/google-resumable-media/latest/resumable_media/requests.html#google.resumable_media.requests.ChunkedDownload
    ro_scope = 'https://www.googleapis.com/auth/devstorage.read_only'
    credentials, _ = google.auth.default(scopes=(ro_scope,))
    transport = google.auth.transport.requests.AuthorizedSession(credentials)

    chunk_start = 0
    total_bytes = float('inf')
    while chunk_start < total_bytes:
        # ChunkedDownload appends bytes to stream.
        #  Use new stream each chunk to avoid holding entire file in memory.
        with io.BytesIO() as f:
            download = google.resumable_media.requests.ChunkedDownload(
                url,
                config.GS_CHUNK_SIZE,
                f,
                start=chunk_start,
            )
            download.consume_next_chunk(transport)
            chunk = f.getvalue()
            sock.sendall(chunk)

            chunk_start += config.GS_CHUNK_SIZE
            total_bytes = download.total_bytes


def aws_s3_producer(uri, sock):
    """
    Send contents of file hosted on AWS S3 in chunks to given socket.

    Args:
        uri (str):
            Prefixed AWS S3 file to to be sent to socket. Note you must
            have access to the file with currently loaded AWS
            credentials.
        sock (socket.socket):
            Socket to send to.

    Returns:
        None
    """

    bucket, key = get_bucket_key_from_path(uri, 's3://')
    # https://stackoverflow.com/a/40854612
    # https://botocore.amazonaws.com/v1/documentation/api/latest/reference/response.html#botocore.response.StreamingBody.iter_chunks
    s3c = boto3.client('s3')
    for chunk in s3c.get_object(Bucket=bucket, Key=key)['Body'].iter_chunks(
        chunk_size=config.MAX_CHUNK_SIZE
    ):
        sock.sendall(chunk)


def _make_eof_producer(producer):
    """
    Send a special EOF byte sequence to terminate the request.

    Args:
        producer (Callable[[str, socket.socket], None]):
            Request producer.

    Returns:
        Callable[[str, socket.socket], None]:
            Request producer that sends 'END-OF-FILE' at end of request.
    """

    def new_producer(audio_input, sock):
        producer(audio_input, sock)
        sock.sendall(b'END-OF-FILE')
    return new_producer


def connect_to_engine(host=None, port=None):
    """
    Create a socket connection to Engine at ``host``, ``port``.

    Args:
        host (Union[str, None]):
            Engine host, or None to default to config.
        port (Union[int, None]):
            Engine port, or None to default to config.

    Returns:
        socket.socket:
            The socket connection to the Engine.
    """

    if host is None:
        host = config.ASR_ENGINE_HOST
    if port is None:
        port = config.ASR_ENGINE_PORT

    return socket.create_connection(
        (host, port),
        timeout=config.SOCKET_CONNECTION_TIMEOUT_SECONDS,
    )


prefix_producer_map = {
    'file':  file_producer,
    'gs':    google_cloud_producer,
    'http':  http_producer,
    'https': http_producer,
    's3':    aws_s3_producer,
}


def get_transcripts_mod9(
    options,
    audio_input,
    host=None,
    port=None,
):
    """
    Open TCP connection to Mod9 server, send input, and yield output
    generator.

    Args:
        options (dict[str, Union[dict, float, int, str]]):
            Transcription options for Engine.
        audio_input (Union[GeneratorType, TeeGeneratorType, str]):
            Audio content to be transcribed.
        host (Union[str, None]):
            Engine host, or None to default to config.
        port (Union[int, None]):
            Engine port, or None to default to config.

    Yields:
        dict[str, Union[dict, float, int, str]]:
            Result from Mod9 ASR Engine TCP Server.
    """

    with connect_to_engine(host=host, port=port) as sock:
        sock.settimeout(config.SOCKET_INACTIVITY_TIMEOUT_SECONDS)

        # Start by sending the options as JSON on the first line (terminated w/ newline character).
        first_request_line = json.dumps(options, separators=(',', ':')) + '\n'
        sock.sendall(first_request_line.encode())

        # The Engine should respond with an initial 'processing' status message.
        sockfile = sock.makefile(mode='r', encoding='utf-8')
        first_response_line = json.loads(sockfile.readline())
        if first_response_line.get('status') != 'processing':
            raise Mod9EngineFirstResponseNotProcessingError(
                f"Did not receive 'processing' from Mod9 ASR Engine. Got '{first_response_line}'."
            )

        # Select proper producer given audio input type.
        if isinstance(audio_input, GeneratorType) or isinstance(audio_input, TeeGeneratorType):
            producer = generator_producer
        elif isinstance(audio_input, str):
            producer = prefix_producer_map.get(urlparse(audio_input).scheme)
            if producer is None:
                allowed_prefixes = '://, '.join(prefix_producer_map) + '://'
                raise NotImplementedError(
                    f"URI '{audio_input}' has unrecognized prefix."
                    f" Allowed prefixes: {allowed_prefixes}."
                )
        else:
            raise TypeError(f"Audio input should be generator or str; got '{type(audio_input)}'.")

        if options.get('format') == 'raw':
            producer = _make_eof_producer(producer)

        # Launch producer thread to stream from audio input source to Engine.
        producer_thread = PropagatingThread(target=producer, args=(audio_input, sock))
        producer_thread.start()

        yield first_response_line
        for line in sockfile:
            yield json.loads(line)

        producer_thread.join()


def get_loaded_models_mod9(host=None, port=None):
    """
    Query Engine for a list of models and return loaded models.

    Args:
        host (Union[str, None]):
            Engine host, or None to default to config.
        port (Union[int, None]):
            Engine port, or None to default to config.

    Returns:
        dict[str, dict[str, Union[bool, dict, int, str]]]:
            Metadata about models currently loaded in Engine.
    """

    with connect_to_engine(host=host, port=port) as sock:
        sock.settimeout(config.SOCKET_INACTIVITY_TIMEOUT_SECONDS)

        # Start by sending request to list models (terminated w/ newline character).
        get_model_request = '{"command": "get-models-info"}\n'
        sock.sendall(get_model_request.encode())

        sockfile = sock.makefile(mode='r', encoding='utf-8')

        response_raw = sockfile.read()  # expected to be single-line JSON
        get_models_response = json.loads(response_raw)
        if get_models_response.get('status') != 'completed':
            raise Mod9EngineResponseNotCompletedError(
                f"Got response '{get_models_response}'; must have `.status` field `completed`."
            )
        if 'asr_models' not in get_models_response:
            raise Mod9UnexpectedEngineResponseError(
                f"Got response '{get_models_response}'; must have `.asr_models` field."
            )

        sockfile.close()

        return get_models_response['asr_models']


def find_loaded_models_with_rate(rate, host=None, port=None):
    """
    Get all models loaded in Engine that have the specified rate.

    Args:
        rate (int):
            Check for loaded Engine models with this rate.
        host (Union[str, None]):
            Engine host, or None to default to config.
        port (Union[int, None]):
            Engine port, or None to default to config.

    Returns:
        list[dict[str, Union[bool, dict, int, str]]]:
            Metadata about models currently loaded in Engine with specified rate (or empty list).
    """
    try:
        models = get_loaded_models_mod9(host=host, port=port)
    except Mod9UnexpectedEngineResponseError:
        return []
    return [model for model in models if model['rate'] == rate]


def find_loaded_models_with_language(language_code, host=None, port=None):
    """
    Get all models loaded in Engine with the specified language code prefix.

    Args:
        language_code (str):
            Check for loaded Engine models with the prefix of this language code.
        host (Union[str, None]):
            Engine host, or None to default to config.
        port (Union[int, None]):
            Engine port, or None to default to config.

    Returns:
        list[dict[str, Union[bool, dict, int, str]]]:
            Metadata about models currently loaded in Engine with
            specified language code prefix (or empty list).
    """

    try:
        models = get_loaded_models_mod9(host=host, port=port)
    except Mod9UnexpectedEngineResponseError:
        return []
    language_code_region = language_code.lower().split(sep='-')
    possible_models = []
    possible_model_language_code_regions = []
    for model in models:
        model_language_code = model.get('language')
        if model_language_code:
            model_language_code_region = model_language_code.lower().split(sep='-')
            if language_code_region[0] == model_language_code_region[0]:
                possible_models.append(model)
                possible_model_language_code_regions.append(model_language_code_region)
    return possible_models


def get_model_languages(host=None, port=None):
    """
    Get the language code of all models loaded in the Engine.

    Args:
        host (Union[str, None]):
            Engine host, or None to default to config.
        port (Union[int, None]):
            Engine port, or None to default to config.

    Returns:
        set[str]:
            Language code of all models loaded in the Engine.
    """

    try:
        models = get_loaded_models_mod9(host=host, port=port)
    except Mod9UnexpectedEngineResponseError:
        return set()
    languages = set()
    for model in models:
        model_language_code = model.get('language')
        if model_language_code:
            languages.add(model_language_code)
    return languages


def select_best_model_for_language_code(models, language_code, model_type=None):
    """
    Get model with language code that best matches given tag.

    Args:
        models (list[dict[str, Union[bool, dict, int, str]]]):
            Model metadata from which to select best match.
        language_code (str):
            Check for loaded Engine models with the prefix of this language code.

    Returns:
        dict[str, Union[bool, dict, int, str]]:
            Model corresponding to the best match (or first model).
    """

    def use_model_type(models_subset):
        if model_type:
            model_type_mod9 = 'phone' if model_type == 'phone_call' else 'video'
            for model in models_subset:
                if model_type_mod9 in model['name']:
                    return model
        return None

    language_code_region = language_code.lower().split(sep='-')
    model_language_code_regions = [model['language'].lower().split(sep='-') for model in models]
    possible_models = []
    if len(language_code_region) == 1:
        for i, model_language_code_region in enumerate(model_language_code_regions):
            if len(model_language_code_region) == 1:
                possible_models.append(models[i])
        if len(possible_models) == 1:
            return possible_models[0]
        elif len(possible_models) > 1:
            model_via_type = use_model_type(possible_models)
            if model_via_type:
                return model_via_type
        model_via_type = use_model_type(models)
        if model_via_type:
            return model_via_type
        return models[0]

    general_models = []
    model_type_models = []
    for i, model_language_code_region in enumerate(model_language_code_regions):
        if len(model_language_code_region) == 1:
            general_models.append(models[i])
        elif language_code_region[1] == model_language_code_region[1]:
            model_type_models.append(models[i])
    if len(model_type_models) == 1:
        return model_type_models[0]
    elif len(model_type_models) > 1:
        model_via_type = use_model_type(model_type_models)
        if model_via_type:
            return model_via_type
    if len(general_models) == 1:
        return general_models[0]
    elif len(general_models) > 1:
        model_via_type = use_model_type(general_models)
        if model_via_type:
            return model_via_type
    return models[0]


def get_version_mod9(host=None, port=None):
    """
    Query Engine for version number.

    Args:
        host (Union[str, None]):
            Engine host, or None to default to config.
        port (Union[int, None]):
            Engine port, or None to default to config.

    Returns:
        string:
            The version of the Engine.

    Raises:
        OSError:
            Socket errors.
        TypeError:
            Version parsing errors.
    """

    with connect_to_engine(host=host, port=port) as sock:
        sock.settimeout(config.SOCKET_INACTIVITY_TIMEOUT_SECONDS)

        sock.sendall('{"command": "get-version"}\n'.encode())

        with sock.makefile(mode='r', encoding='utf-8') as sockfile:
            response_raw = sockfile.read()  # expected to be single-line JSON
            response = json.loads(response_raw)

        return response.get('version')


def is_compatible_mod9(engine_version_string):
    """
    Determine if present wrappers are compatible with Engine version.

    Args:
        engine_version_string (Union[str, None]):
            The Engine version to compare to wrapper allowed range.

    Returns:
        bool:
            Whether the wrappers and Engine are compatible.

    Raises:
        OSError:
            Socket errors.
        ValueError:
            Invalid semantic version given to comparator.
    """

    engine_version = version.parse(engine_version_string)

    lower_bound_string, upper_bound_string = config.WRAPPER_ENGINE_COMPATIBILITY_RANGE

    is_within_lower_bound = True
    is_within_upper_bound = True

    if lower_bound_string is not None:
        lower_bound = version.parse(lower_bound_string)
        is_within_lower_bound = lower_bound <= engine_version  # Lower bound is inclusive.
    if upper_bound_string is not None:
        upper_bound = version.parse(upper_bound_string)
        is_within_upper_bound = engine_version < upper_bound  # Upper bound is exclusive.

    return is_within_lower_bound and is_within_upper_bound


def test_host_port(logger=None, host=None, port=None):
    """
    Check if Mod9 ASR Engine is online. Loop until get-info command
    provides a ``ready`` response. Log stats.

    Args:
        host (Union[str, None]):
            Engine host, or None to default to config.
        port (Union[int, None]):
            Engine port, or None to default to config.

    Returns:
        None
    """

    if not logger:
        logger = logging.root

    engine_version = get_version_mod9(host=host, port=port)
    if not is_compatible_mod9(engine_version):
        raise Mod9IncompatibleEngineVersionError(
            f"This Python SDK version {config.WRAPPER_VERSION} is not compatible with"
            f" ASR Engine version {engine_version} (which should be"
            f" >={config.WRAPPER_ENGINE_COMPATIBILITY_RANGE[0]}"
            + (').' if config.WRAPPER_ENGINE_COMPATIBILITY_RANGE[1] is None
               else f" and <{config.WRAPPER_ENGINE_COMPATIBILITY_RANGE[1]}).")
        )

    logger.info(
        "Checking for ASR Engine running at %s on port %d ...",
        config.ASR_ENGINE_HOST, config.ASR_ENGINE_PORT
    )

    # Loop sending get-info until receive ``state`` in response as "ready".
    response = dict()
    response_raw = ''
    while response.get('state') != 'ready':
        with connect_to_engine(host=host, port=port) as sock:
            sock.settimeout(config.SOCKET_INACTIVITY_TIMEOUT_SECONDS)

            sock.sendall('{"command": "get-info"}\n'.encode())
            with sock.makefile(mode='r', encoding='utf-8') as sockfile:
                response_raw = sockfile.read()  # expected to be single-line JSON
                response = json.loads(response_raw)

        # Log and sleep except when receiving a ready response.
        if response.get('state') == 'starting':
            logger.warning(
                "The Engine is still starting. Will attempt to connect again in %s seconds...",
                config.ENGINE_CONNECTION_RETRY_SECONDS,
            )
            time.sleep(config.ENGINE_CONNECTION_RETRY_SECONDS)
        elif response.get('state') != 'ready':
            raise Mod9UnexpectedEngineStateError(
                'Engine responded with unexpected state: ' + response.get('state')
            )

    if response['requests']['limit'] - response['requests']['active'] == 0:
        logger.warning('The Engine is at its limit, and unable to accept new requests.')

    logger.info("The ASR Engine is ready: %s", response_raw.strip())
    # TODO: also report the name of loaded models, from the get-models-info command?


def validate_uri_scheme(uri, allowed_uri_schemes):
    """
    Check if a URI has scheme in allowed set.

    Args:
        uri (str):
            URI with a scheme that may or may not be allowed.
        allowed_uri_schemes (set[str]):
            Set of schemes the URI is allowed to have.

    Raises:
        Mod9DisabledAudioURISchemeError:
            The error message indicating which schemes are allowed,
            if given URI had a different scheme, or None if it did not.
    """

    uri_scheme = urlparse(uri).scheme
    if uri_scheme not in allowed_uri_schemes:
        error_message = f"URI '{uri_scheme}' not allowed; server configured to allow:" + \
                        f" {', '.join(allowed_uri_schemes) if len(allowed_uri_schemes) else None}."

        raise Mod9DisabledAudioURISchemeError(error_message)


def format_log_time(self, record, datefmt):
    """
    Helper function to format logging record times in ISO-8601 format.

    Usage is with a ``logging.Formatter``, e.g.,
    ```
    logging.Formatter.formatTime = format_log_time
    ```

    See:
    https://stackoverflow.com/a/58777937

    Args:
        self (logging.Formatter):
            The log formatter calling this function.
        record (logging.LogRecord):
            The log record whose time to format.
        datefmt (Union[str, None]):
            Optional string to specify time format; overriden here.

    Returns:
        str:
            The time, formatted as 2006-01-02T15:04:05.999-07:00.
    """

    log_time = datetime.datetime.fromtimestamp(record.created, datetime.timezone.utc)
    return log_time.astimezone().isoformat(timespec='milliseconds')


def uri_exists(uri):
    """
    Attempt to access remote file at given uri.

    Args:
        uri (str):
            URI where remote file resides.

    Returns:
        bool:
            Whether the remote file is reachable.
    """

    parsed_uri = urlparse(uri)
    try:
        if parsed_uri.scheme == 'file':
            file_path = parsed_uri.netloc + parsed_uri.path
            if not os.path.exists(file_path):
                raise Mod9CouldNotAccessFileError
        elif parsed_uri.scheme == 'gs':
            bucket, key = get_bucket_key_from_path(uri, 'gs://')
            ro_scope = 'https://www.googleapis.com/auth/devstorage.read_only'
            credentials, _ = google.auth.default(scopes=(ro_scope,))
            gsc = google.cloud.storage.Client(credentials=credentials)
            blob = google.cloud.storage.Blob(bucket=gsc.bucket(bucket), name=key)
            if not blob.exists(client=gsc):
                raise Mod9CouldNotAccessFileError
        elif parsed_uri.scheme == 'http' or parsed_uri.scheme == 'https':
            with requests.head(uri) as r:
                r.raise_for_status()
        elif parsed_uri.scheme == 's3':
            bucket, key = get_bucket_key_from_path(uri, 's3://')
            s3c = boto3.client('s3')
            s3c.head_object(Bucket=bucket, Key=key)
    except (AWSClientError, requests.HTTPError, Mod9CouldNotAccessFileError) as e:
        raise Mod9CouldNotAccessFileError(f"Could not access file at {uri}.") from e
