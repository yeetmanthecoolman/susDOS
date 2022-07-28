"""
Common functions used by the Google-compatible Mod9 ASR Python SDK
wrappers.
"""

import google.cloud.speech_v1p1beta1 as cloud_speech

from mod9.reformat.utils import get_transcripts_mod9
from mod9.reformat import google as reformat


class Mod9NotImplementedError(Exception):
    pass


def long_running_recognize(
    request,
    retry=None,
    timeout=None,
    metadata=None,
    module=cloud_speech,
    host=None,
    port=None,
    *args,
    **kwargs,
):
    """
    Performs asynchronous speech recognition.

    Mod9: Not currently implemented.
    """

    raise Mod9NotImplementedError('Refer to the REST wrapper for long-running recognition.')


def recognize(
    request,
    retry=None,
    timeout=None,
    metadata=None,
    module=cloud_speech,
    host=None,
    port=None,
    *args,
    **kwargs,
):
    """
    Performs synchronous speech recognition: receive results after all
    audio has been sent and processed.

    Args:
        request (Union[dict, RecognizeRequest]):
            The request object. The top-level message sent by the client
            for the `Recognize` method.
        config (RecognitionConfig):
            Required. Provides information to the recognizer that
            specifies how to process the request. This corresponds to
            the ``config`` field on the ``request`` instance; if
            ``request`` is provided, this should not be set.
        audio (RecognitionAudio):
            Required. The audio data to be recognized. This corresponds
            to the ``audio`` field on the ``request`` instance; if
            ``request`` is provided, this should not be set.
        module (module):
            Module to read Google-like types from, in case of
            subclassing. Default is ``google.cloud.speech_v1p1beta1``.
        host (Union[str, None]):
            Engine host, or None to default to config.
        port (Union[int, None]):
            Engine port, or None to default to config.

    Returns:
        RecognizeResponse:
            The only message returned to the client by the ``Recognize``
            method. It contains the result as zero or more sequential
            ``SpeechRecognitionResult`` messages.
    """

    if retry or timeout or metadata:
        # NOTE: Mod9 ASR Python SDK wrapper does not support retry,
        #  timeout, or metadata arguments, which are currently ignored.
        # TODO: Add support for retry and timeout arguments.
        pass

    # Parse inputs to ensure they are the expected encoding, have allowed arguments.
    options, requests = reformat.input_to_mod9(
        {'config': request.config, 'audio': request.audio},
        module=module,
        host=host,
        port=port,
    )

    # Read Engine responses.
    mod9_results = get_transcripts_mod9(
        options,
        requests,
        host=host,
        port=port,
    )

    # Convert Mod9 style to Google style.
    google_result_dicts = reformat.result_from_mod9(mod9_results, host=host, port=port)

    # Convert Mod9 type to Google type.
    google_results = reformat.google_type_result_from_dict(
        google_result_dicts,
        google_result_type=module.SpeechRecognitionResult,
        module=module,
    )

    # Yield the expected response object.
    response = module.RecognizeResponse()
    for google_result in google_results:
        response.results.append(google_result)

    return response


def streaming_recognize(
    requests,
    retry=None,
    timeout=None,
    metadata=None,
    module=cloud_speech,
    host=None,
    port=None,
    *args,
    **kwargs,
):
    """
    Perform streaming speech recognition.

    This method allows you to receive results while sending audio.

    Example:
      >>> from mod9.asr import speech
      >>> client = speech.SpeechClient()
      >>> encoding = speech.RecognitionConfig.AudioEncoding.LINEAR16
      >>> config = speech.StreamingRecognitionConfig(
      ...     config=speech.RecognitionConfig(
      ...         encoding=encoding,
      ...     ),
      ... )
      >>> # Byte-encoded content goes in ``audio_content`` below:
      >>> request = speech.StreamingRecognizeRequest(audio_content=b'')
      >>> requests = [request]
      >>> for element in client.streaming_recognize(config, requests):
      ...     # process element
      ...     pass

    Args:
        config (StreamingRecognitionConfig):
            The configuration to use for the stream.
        requests (Iterable[StreamingRecognizeRequest]):
            The input objects.
        module (module):
            Module to read Google-like types from, in case of
            subclassing. Default is ``google.cloud.speech_v1p1beta1``.
        host (Union[str, None]):
            Engine host, or None to default to config.
        port (Union[int, None]):
            Engine port, or None to default to config.

    Returns:
        Iterable[StreamingRecognizeResponse]
    """

    if retry or timeout or metadata:
        # NOTE: Mod9 ASR Python SDK wrapper does not support retry,
        #  timeout, or metadata arguments, which are currently ignored.
        # TODO: Add support for retry and timeout arguments.
        pass

    # The first request should have the config and no audio.
    request = next(requests)

    # Parse inputs to ensure they are the expected encoding, have allowed arguments.
    options, _ = reformat.input_to_mod9(
        {'config': request['streaming_config']},
        module=module,
        host=host,
        port=port,
    )

    # Read Engine responses.
    audio_requests = (request.audio_content for request in requests)
    mod9_results = get_transcripts_mod9(
        options,
        audio_requests,
        host=host,
        port=port,
    )

    # Convert Mod9 style to Google style.
    google_result_dicts = reformat.result_from_mod9(mod9_results, host=host, port=port)

    # Convert Mod9 type to Google type.
    google_results = reformat.google_type_result_from_dict(
        google_result_dicts,
        google_result_type=module.StreamingRecognitionResult,
        module=module,
    )

    # Yield the expected response object.
    for google_result in google_results:
        response = module.StreamingRecognizeResponse()
        response.results.append(google_result)
        yield response
