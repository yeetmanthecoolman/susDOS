"""
A drop-in replacement for Google Cloud STT Python Client Library.

The speech module provides a strict subset of Google Cloud STT
functionality, and uses Google's provided objects directly. Input in
Google-format objects (or dict) is converted internally to Mod9-format
JSON. Output is converted from Mod9-format JSON to Google-format
objects.
"""

import google.cloud.speech_v1p1beta1 as cloud_speech

from mod9.asr.common import (
    long_running_recognize,
    recognize,
    streaming_recognize,
)
from mod9.reformat import config

__version__ = config.WRAPPER_VERSION

# Copy Google's types into this namespace.
# Not all are used/implemented by Mod9 ASR; they may be silently ignored.
LongRunningRecognizeMetadata = cloud_speech.LongRunningRecognizeMetadata
LongRunningRecognizeRequest = cloud_speech.LongRunningRecognizeRequest
LongRunningRecognizeResponse = cloud_speech.LongRunningRecognizeResponse
RecognitionAudio = cloud_speech.RecognitionAudio
RecognitionConfig = cloud_speech.RecognitionConfig
RecognitionMetadata = cloud_speech.RecognitionMetadata
RecognizeRequest = cloud_speech.RecognizeRequest
RecognizeResponse = cloud_speech.RecognizeResponse
SpeakerDiarizationConfig = cloud_speech.SpeakerDiarizationConfig
SpeechContext = cloud_speech.SpeechContext
SpeechRecognitionAlternative = cloud_speech.SpeechRecognitionAlternative
SpeechRecognitionResult = cloud_speech.SpeechRecognitionResult
StreamingRecognitionConfig = cloud_speech.StreamingRecognitionConfig
StreamingRecognitionResult = cloud_speech.StreamingRecognitionResult
StreamingRecognizeRequest = cloud_speech.StreamingRecognizeRequest
StreamingRecognizeResponse = cloud_speech.StreamingRecognizeResponse
WordInfo = cloud_speech.WordInfo


class Mod9ASREngineTransport(object):
    """Duck-typed SpeechTransport"""
    _wrapped_methods = {}

    def __init__(self, host=None, port=None):
        self._wrapped_methods[self.recognize] = \
            self.recognize
        self._wrapped_methods[self.streaming_recognize] = \
            self.streaming_recognize
        self.host = host
        self.port = port

    def long_running_recognize(self, *args, **kwargs):
        """
        Performs asynchronous speech recognition.

        Mod9: Not currently implemented.
        """
        return long_running_recognize(
            *args,
            host=self.host,
            port=self.port,
            **kwargs,
        )

    def recognize(self, *args, **kwargs):
        """
        Performs synchronous speech recognition: receive results after
        all audio has been sent and processed.

        Args:
            request (Union[dict, RecognizeRequest]):
                The request object. The top-level message sent by the
                client for the `Recognize` method.
            config (RecognitionConfig):
                Required. Provides information to the recognizer that
                specifies how to process the request. This corresponds
                to the ``config`` field on the ``request`` instance; if
                ``request`` is provided, thi should not be set.
            audio (RecognitionAudio):
                Required. The audio data to be recognized. This
                corresponds to the ``audio`` field on the ``request``
                instance; if ``request`` is provided, this should not
                be set.

        Returns:
            RecognizeResponse:
                The only message returned to the client by the
                ``Recognize`` method. It contains the result as zero or
                more sequential ``SpeechRecognitionResult`` messages.
        """
        return recognize(
            *args,
            host=self.host,
            port=self.port,
            **kwargs,
        )

    def streaming_recognize(self, *args, **kwargs):
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

        Returns:
            Iterable[StreamingRecognizeResponse]
        """
        return streaming_recognize(
            *args,
            host=self.host,
            port=self.port,
            **kwargs,
        )


class SpeechClient(cloud_speech.SpeechClient):
    """OVERRIDE: drop-in replacement for Google Cloud Speech."""

    def __init__(self, host=None, port=None, *args, **kwargs):
        """OVERRIDE: ignore arguments, set Mod9's custom transport."""
        self._transport = Mod9ASREngineTransport(host=host, port=port)
