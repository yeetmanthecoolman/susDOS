"""
A drop-in replacement for Google Cloud STT Python Client Library.

The speech_mod9 module provides a subset of Google Cloud STT
functionality, with additional Mod9-exclusive features, such as phrase
alternatives, and uses subclassed Google objects directly. Input in
Google-type objects (or Google-style dict) is converted internally to
Mod9-style dict. Output is converted from Mod9-style dict to subclassed
Google-style and -type objects.
"""

import sys

import google.cloud.speech_v1p1beta1 as cloud_speech
from google.protobuf import duration_pb2 as duration
import proto

from mod9.asr import common
from mod9.reformat import config

__version__ = config.WRAPPER_VERSION

# Used to compile protobufs into Python classes.
__protobuf__ = proto.module(
    package='mod9.asr.speech_mod9',
    manifest={
        "LongRunningRecognizeMetadata",
        "LongRunningRecognizeRequest",
        "LongRunningRecognizeResponse",
        "RecognitionAudio",                    # Held by subclass
        "RecognitionConfig",                   # Mod9-only attribute(s)
        "RecognitionMetadata",
        "RecognizeRequest",                    # Holds subclass with Mod9-only attribute(s)
        "RecognizeResponse",                   # Holds subclass with Mod9-only attribute(s)
        "SpeakerDiarizationConfig",
        "SpeechContext",
        "SpeechRecognitionAlternative",
        'SpeechRecognitionPhrase',             # Mod9-only message
        'SpeechRecognitionPhraseAlternative',  # Mod9-only message
        "SpeechRecognitionResult",             # Mod9-only attribute(s)
        "StreamingRecognitionConfig",          # Holds subclass with Mod9-only attribute(s)
        "StreamingRecognitionResult",          # Mod9-only attribute(s)
        "StreamingRecognizeRequest",
        "StreamingRecognizeResponse",          # Holds subclass with Mod9-only attribute(s)
        "WordInfo",
    },
)


# Copy Google's types into this namespace.
# Not all are used/implemented by Mod9 ASR; they may be silently ignored.
LongRunningRecognizeMetadata = cloud_speech.LongRunningRecognizeMetadata
LongRunningRecognizeRequest = cloud_speech.LongRunningRecognizeRequest
LongRunningRecognizeResponse = cloud_speech.LongRunningRecognizeResponse
RecognitionAudio = cloud_speech.RecognitionAudio
RecognitionMetadata = cloud_speech.RecognitionMetadata
SpeakerDiarizationConfig = cloud_speech.SpeakerDiarizationConfig
SpeechContext = cloud_speech.SpeechContext
SpeechRecognitionAlternative = cloud_speech.SpeechRecognitionAlternative
StreamingRecognizeRequest = cloud_speech.StreamingRecognizeRequest
WordInfo = cloud_speech.WordInfo


# Subclass Google's types required to extend
#  functionality to include Mod9-only options.


# Mod9-only message
class SpeechRecognitionPhraseBias(proto.Message):
    """
    Phrase alternative model scores. Mod9-only message.

    Attributes:
        am (float):
            The acoustic model score of the phrase, shifted
            so that the most likely phrase has score 0.0.
        lm (float):
            The language model score of the phrase, shifted
            so that the most likely phrase has score 0.0.
    """

    am = proto.Field(proto.FLOAT, number=1)
    lm = proto.Field(proto.FLOAT, number=2)


# Mod9-only message
class SpeechRecognitionPhrase(proto.Message):
    """
    Phrase alternative hypotheses and model scores. Mod9-only message.

    Attributes:
        bias (SpeechRecognitionPhraseBias):
            The acoustic and language model scores.
        phrase (str):
            Phrase text representing the words the user spoke.
    """

    bias = proto.Field(SpeechRecognitionPhraseBias, number=1)
    phrase = proto.Field(proto.STRING, number=2)


# Mod9-only message
class SpeechRecognitionPhraseAlternative(proto.Message):
    """
    Set of phrase alternatives for a portion of the audio. Mod9-only
    message.

    Attributes:
        alternatives (Sequence[SpeechRecognitionPhrase]):
            Set of phrase hypotheses (up to the maximum specified in
            ``max_phrase_alternatives``). These phrase alternatives are
            ordered in terms of accuracy, with the top (first) phrase
            alternative being the most probable, as ranked by the
            recognizer.
        start_time (duration.Duration):
            Time offset relative to the beginning of the audio, and
            corresponding to the start of the spoken phrase.
        end_time (duration.Duration):
            Time offset relative to the beginning of the audio, and
            corresponding to the end of the spoken phrase.
        phrase (str):
            Most likely phrase text representing the words the user
            spoke.
    """

    alternatives = proto.RepeatedField(SpeechRecognitionPhrase, number=1)
    start_time = proto.Field(duration.Duration, number=2)
    end_time = proto.Field(duration.Duration, number=3)
    phrase = proto.Field(proto.STRING, number=4)


# Mod9-only message
class SpeechRecognitionWord(proto.Message):
    """
    Word alternative hypotheses. Mod9-only message.

    Attributes:
        word (str):
            Word text representing the words the user spoke.
        confidence (float):
            Confidence score of the word.
    """

    word = proto.Field(proto.STRING, number=1)
    confidence = proto.Field(proto.FLOAT, number=2)


# Mod9-only message
class SpeechRecognitionWordAlternative(proto.Message):
    """
    Set of word alternatives for a portion of the audio. Mod9-only
    message.

    Attributes:
        alternatives (Sequence[SpeechRecognitionWord]):
            Set of word hypotheses (up to the maximum specified in
            ``max_word_alternatives``). These word alternatives are
            ordered in terms of confidence, with the top (first) word
            alternative being the most probable, as ranked by the
            recognizer.
        start_time (duration.Duration):
            Time offset relative to the beginning of the audio, and
            corresponding to the start of the spoken word.
        end_time (duration.Duration):
            Time offset relative to the beginning of the audio, and
            corresponding to the end of the spoken word.
        confidence (float):
            Confidence score of the most likely word.
        word (str):
            Most likely word text representing the words the user
            spoke.
    """

    alternatives = proto.RepeatedField(SpeechRecognitionWord, number=1)
    start_time = proto.Field(duration.Duration, number=2)
    end_time = proto.Field(duration.Duration, number=3)
    confidence = proto.Field(proto.FLOAT, number=4)
    word = proto.Field(proto.STRING, number=5)


class SpeechRecognitionResult(proto.Message):
    """
    OVERRIDE: A speech recognition result corresponding to a portion of
    the audio. This is a Mod9 subclass with some Mod9-only attributes.

    Attributes:
        alternatives (Sequence[SpeechRecognitionAlternative]):
            May contain one or more recognition hypotheses (up to the
            maximum specified in ``max_alternatives``). These
            alternatives are ordered in terms of accuracy, with the top
            (first) alternative being the most probable, as ranked by
            the recognizer.
        channel_tag (int):
            Channel number of the result transcript.
        language_code (str):
            Output only. The
            `BCP-47 <https://www.rfc-editor.org/rfc/bcp/bcp47.txt>`__
            language tag of the language in this result. This language
            code was detected to have the most likelihood of being
            spoken in the audio.
        phrases (Sequence[SpeechRecognitionPhraseAlternative]):
            Sequence of phrase alternatives in increasing time order.
            Mod9-only attribute.
        words (Sequence[SpeechRecognitionWordAlternative]):
            Sequence of word alternatives in increasing time order.
            Mod9-only attribute.
        asr_model (str):
            Indicate which ASR model is being used.
            Mod9-only attribute.
    """

    alternatives = proto.RepeatedField(SpeechRecognitionAlternative, number=1)
    channel_tag = proto.Field(proto.INT32, number=2)
    language_code = proto.Field(proto.STRING, number=5)

    # Mod9-only attributes:
    phrases = proto.RepeatedField(SpeechRecognitionPhraseAlternative, number=901)
    words = proto.RepeatedField(SpeechRecognitionWordAlternative, number=902)
    asr_model = proto.Field(proto.STRING, number=903)


class RecognizeResponse(proto.Message):
    """
    OVERRIDE: The only message returned to the client by the
    ``Recognize`` method. It contains the result as zero or more
    sequential ``SpeechRecognitionResult`` messages. Mod9 subclass to
    hold Mod9 subclassed SpeechRecognitionResult.

    Attributes:
        results (Sequence[SpeechRecognitionResult]):
            Sequential list of transcription results corresponding to
            sequential portions of audio.
    """

    results = proto.RepeatedField(SpeechRecognitionResult, number=2)


class StreamingRecognitionResult(proto.Message):
    """
    OVERRIDE: A streaming speech recognition result corresponding to a
    portion of audio that is currently being processed.
    This is a Mod9 subclass with some Mod9-only attributes.

    Attributes:
        alternatives (Sequence[SpeechRecognitionAlternative]):
            May contain one or more recognition hypotheses (up to the
            maximum specified in ``max_alternatives``). These
            alternatives are ordered in terms of accuracy, with the top
            (first) alternative being the most probable, as ranked by
            the recognizer.
        is_final (bool):
            If ``false``, this ``StreamingRecognitionResult`` represents
            an interim result that may change. If ``true``, this is the
            final time the speech service will return this particular
            ``StreamingRecognitionResult``, the recognizer will not
            return any further hypotheses for this portion of the
            transcript and corresponding audio.
        stability (float):
            An estimate of the likelihood that the recognizer will not
            change its guess about this interim result. Values range
            from 0.0 (completely unstable) to 1.0 (completely stable).
            This field is only provided for interim results
            (``is_final=false``). The default of 0.0 is a sentinel value
            indicating ``stability`` was not set.
        result_end_time (duration.Duration):
            Time offset of the end of this result relative to the
            beginning of the audio.
        channel_tag (int):
            Channel number of the result transcript.
        language_code (str):
            Output only. The
            `BCP-47 <https://www.rfc-editor.org/rfc/bcp/bcp47.txt>`__
            language tag of the language in this result. This language
            code was detected to have the most likelihood of being
            spoken in the audio.
        phrases (Sequence[SpeechRecognitionPhraseAlternative]):
            Sequence of phrase alternatives in increasing time order.
            Mod9-only attribute.
        words (Sequence[SpeechRecognitionWordAlternative]):
            Sequence of word alternatives in increasing time order.
            Mod9-only attribute.
        asr_model (str):
            Indicate which ASR model is being used.
            Mod9-only attribute.
    """

    alternatives = proto.RepeatedField(SpeechRecognitionAlternative, number=1)
    is_final = proto.Field(proto.BOOL, number=2)
    stability = proto.Field(proto.FLOAT, number=3)
    result_end_time = proto.Field(duration.Duration, number=4)
    channel_tag = proto.Field(proto.INT32, number=5)
    language_code = proto.Field(proto.STRING, number=6)

    # Mod9-only attributes:
    phrases = proto.RepeatedField(SpeechRecognitionPhraseAlternative, number=901)
    words = proto.RepeatedField(SpeechRecognitionWordAlternative, number=902)
    asr_model = proto.Field(proto.STRING, number=903)


class StreamingRecognizeResponse(proto.Message):
    """
    OVERRIDE: ``StreamingRecognizeResponse`` is the only message
    returned to the client by ``StreamingRecognize``. A series of zero
    or more ``StreamingRecognizeResponse`` messages are streamed back
    to the client. If there is no recognizable audio, and
    ``single_utterance`` is set to false, then no messages are streamed
    back to the client. Mod9 subclass to hold Mod9 subclassed
    ``StreamingRecognitionResult``.

    Here's an example of a series of ten
    ``StreamingRecognizeResponse`` s that might be returned while
    processing audio:

    1. results { alternatives { transcript: "tube" } stability: 0.01 }

    2. results { alternatives { transcript: "to be a" } stability: 0.01
       }

    3. results { alternatives { transcript: "to be" } stability: 0.9 }
       results { alternatives { transcript: " or not to be" } stability:
       0.01 }

    4. results { alternatives { transcript: "to be or not to be"
       confidence: 0.92 } alternatives { transcript: "to bee or not to
       bee" } is_final: true }

    5. results { alternatives { transcript: " that's" } stability: 0.01
       }

    6. results { alternatives { transcript: " that is" } stability: 0.9
       } results { alternatives { transcript: " the question" }
       stability: 0.01 }

    7. results { alternatives { transcript: " that is the question"
       confidence: 0.98 } alternatives { transcript: " that was the
       question" } is_final: true }

    Notes:

    -  Only two of the above responses #4 and #7 contain final results;
       they are indicated by ``is_final: true``. Concatenating these
       together generates the full transcript: "to be or not to be that
       is the question".

    -  The others contain interim ``results``. #3 and #6 contain two
       interim ``results``: the first portion has a high stability and
       is less likely to change; the second portion has a low stability
       and is very likely to change. A UI designer might choose to show
       only high stability ``results``.

    -  The specific ``stability`` and ``confidence`` values shown above
       are only for illustrative purposes. Actual values may vary.
       (Mod9 does not currently support ``stability`` and returns a
       placeholder ``0.5`` value).

    -  In each response, only one of these fields will be set:
       ``error``, ``speech_event_type``, or one or more (repeated)
       ``results``.

    Attributes:
        results (Sequence[StreamingRecognitionResult]):
            This repeated list contains zero or more results that
            correspond to consecutive portions of the audio currently
            being processed. It contains zero or one ``is_final=true``
            result (the newly settled portion), followed by zero or more
            ``is_final=false`` results (the interim results).
        error (status.Status):
            Mod9: not available at present.
        speech_event_type (StreamingRecognizeResponse.SpeechEventType):
            Mod9: not available at present.
    """

    results = proto.RepeatedField(StreamingRecognitionResult, number=2)


class RecognitionConfig(proto.Message):
    """
    OVERRIDE: Provides information to the recognizer that specifies how
    to process the request.
    This is a Mod9 subclass with some Mod9-only attributes.

    Attributes:
        encoding (RecognitionConfig.AudioEncoding):
            Encoding of audio data sent in all ``RecognitionAudio``
            messages. This field is optional for ``WAV``
            audio files and required for all other audio formats. For
            details, see
            [AudioEncoding][RecognitionConfig.AudioEncoding].
        sample_rate_hertz (int):
            Sample rate in Hertz of the audio data sent in all
            ``RecognitionAudio`` messages. Valid values are 8000, 16000.
            16000 is optimal. For best results, set the sampling rate of
            the audio source to 16000 Hz. If that's not possible, use
            the native sample rate of the audio source (instead of
            re-sampling). This field is optional for WAV audio
            files, but is required for all other audio formats. For
            details, see
            [AudioEncoding][RecognitionConfig.AudioEncoding].
        language_code (str):
            The language of the supplied audio as a
            `BCP-47 <https://www.rfc-editor.org/rfc/bcp/bcp47.txt>`__
            language tag. Example: "en-US". Mod9 currently supports
            ``en-US``.
        max_alternatives (int):
            Maximum number of recognition hypotheses to be returned.
            Specifically, the maximum number of
            ``SpeechRecognitionAlternative`` messages within each
            ``SpeechRecognitionResult``. The server may return fewer
            than ``max_alternatives``. Valid values are ``0``-``1000``.
            A value of ``0`` or ``1`` will return a maximum of one. If
            omitted, will return a maximum of one.
        enable_word_time_offsets (bool):
            If ``true``, the top result includes a list of words and the
            start and end time offsets (timestamps) for those words. If
            ``false``, no word-level time offset information is
            returned. The default is ``false``.
        enable_word_confidence (bool):
            If ``true``, the top result includes a list of words and the
            confidence for those words. If ``false``, no word-level
            confidence information is returned. The default is
            ``false``.
        enable_automatic_punctuation (bool):
            If 'true', adds punctuation to recognition result
            hypotheses. This feature is only available in select
            languages. Setting this for requests in other languages
            has no effect at all. The default 'false' value does not add
            punctuation to result hypotheses.
        audio_channel_count (int):
            Number of channels present in the audio.
        enable_separate_recognition_per_channel (bool):
            Mod9: Only supports ``true`` value for multi-channel audio, i.e.
            it only supports transcribing all channels.
            This is different from GSTT which transcribes only the first
            channel if the value is ``false``.
        max_phrase_alternatives (int):
            Mod9-only attribute. Maximum number of phrase hypotheses to
            be returned. Specifically, the maximum number of
            ``SpeechRecognitionPhrase`` messages within each
            ``SpeechRecognitionPhraseAlternative``. The server may
            return fewer than ``max_phrase_alternatives``. Valid values
            are ``0``-``10000``. A value of ``0`` or ``1`` will return a
            maximum of one. If omitted, will return a maximum of one.
        max_word_alternatives (int):
            Mod9-only attribute. Maximum number of word hypotheses to
            be returned. Specifically, the maximum number of
            ``SpeechRecognitionWord`` messages within each
            ``SpeechRecognitionWordAlternative``. The server may
            return fewer than ``max_word_alternatives``. Valid values
            are ``0``-``10000``. A value of ``0`` or ``1`` will return a
            maximum of one. If omitted, will return a maximum of one.
        latency (float):
            Mod9-only attribute. Chunk size for ASR processing
            (in seconds). Low values increase CPU usage; high
            values degrade endpointing. Default is ``0.24``.
            Valid values are between ``0.01`` and ``3.0``.
        speed (int):
            Mod9-only attribute. Increasing speed will trade-off accuracy
            and diversity of recognition alternatives.  Default is ``5``.
            Valid values are between ``1`` and ``9``.
        options_json (str):
            Mod9-only attribute. Additional request options specified as
            a JSON object. This will override options set by other means.
        asr_model (str):
            Mod9-only attribute. Set ASR model to use. This will
            override ``language_code`` setting, but will be overridden by
            ``options_json``.
        intervals_json (str):
            Mod9-only attribute. Specific intervals to be transcribed,
            specified as a JSON object.
        alternative_language_codes (Sequence[str]):
            Mod9: not available at present.
        profanity_filter (bool):
            Mod9: not available at present.
        adaptation (~.resource.SpeechAdaptation):
            Mod9: not available at present.
        speech_contexts (Sequence[~.cloud_speech.SpeechContext]):
            Mod9: not available at present.
        enable_speaker_diarization (bool):
            Mod9: not available at present.
        diarization_speaker_count (int):
            Mod9: not available at present.
        diarization_config (~.cloud_speech.SpeakerDiarizationConfig):
            Mod9: not available at present.
        metadata (~.cloud_speech.RecognitionMetadata):
            Mod9: not available at present.
        model (str):
            Mod9: not available at present.
        use_enhanced (bool):
            Mod9: not available at present.
    """

    class AudioEncoding(proto.Enum):
        """
        OVERRIDE: The encoding of the audio data sent in the request.
        Mod9 subclass to support Mod9-only attributes:
        ``ALAW``, ``LINEAR24``, ``LINEAR32``, and ``FLOAT32``.

        For best results, the audio source should be captured and
        transmitted using a lossless encoding (e.g. ``LINEAR16``).
        The accuracy of the speech recognition can be reduced if lossy
        codecs are used to capture or transmit audio, particularly if
        background noise is present. Lossy codecs include ``ALAW``, and
        ``MULAW``.

        The ``WAV`` audio file formats include a header that describes
        the included audio content. You can request recognition for
        ``WAV`` files that contain ``LINEAR16``, ``ALAW``, or
        ``MULAW`` encoded audio, among several others.  If ``WAV`` audio
        file format is specified in a request, it is not needed to set
        ``AudioEncoding``; the audio encoding format is determined from
        the file header. If you specify an ``AudioEncoding`` when you
        send ``WAV`` audio, the encoding configuration must match the
        encoding described in the audio header.
        """

        ENCODING_UNSPECIFIED = 0
        LINEAR16 = 1
        MULAW = 3

        # Mod9-only attributes.
        ALAW = 901
        LINEAR24 = 902
        LINEAR32 = 903
        FLOAT32 = 904

    # Google-compatible attributes:
    encoding = proto.Field(proto.ENUM, number=1, enum=AudioEncoding,)
    sample_rate_hertz = proto.Field(proto.INT32, number=2)
    language_code = proto.Field(proto.STRING, number=3)
    max_alternatives = proto.Field(proto.INT32, number=4)
    audio_channel_count = proto.Field(proto.INT32, number=7)
    enable_word_time_offsets = proto.Field(proto.BOOL, number=8)
    enable_automatic_punctuation = proto.Field(proto.BOOL, number=11)
    enable_separate_recognition_per_channel = proto.Field(proto.BOOL, number=12)
    model = proto.Field(proto.STRING, number=13)
    enable_word_confidence = proto.Field(proto.BOOL, number=15)

    # Mod9-only attributes:
    max_phrase_alternatives = proto.Field(proto.INT32, number=901)
    latency = proto.Field(proto.FLOAT, number=902)
    speed = proto.Field(proto.INT32, number=903)
    options_json = proto.Field(proto.STRING, number=904)
    max_word_alternatives = proto.Field(proto.INT32, number=905)
    asr_model = proto.Field(proto.STRING, number=906)
    intervals_json = proto.Field(proto.STRING, number=907)


class StreamingRecognitionConfig(proto.Message):
    """
    OVERRIDE: Provides information to the recognizer that specifies how
    to process the request. Mod9 subclass to hold Mod9 subclassed
    ``RecognitionConfig``.

    Attributes:
        config (RecognitionConfig):
            Required. Provides information to the recognizer that
            specifies how to process the request.
        interim_results (bool):
            If ``true``, interim results (tentative hypotheses) may be
            returned as they become available (these interim results are
            indicated with the ``is_final=false`` flag). If ``false`` or
            omitted, only ``is_final=true`` result(s) are returned.
        single_utterance (bool):
            Mod9: not available at present.
    """

    config = proto.Field(RecognitionConfig, number=1)
    interim_results = proto.Field(proto.BOOL, number=3)


class RecognitionAudio(proto.Message):
    """
    OVERRIDE: Contains audio data in the encoding specified in the
    ``RecognitionConfig``. Either ``content`` or ``uri`` must be
    supplied. Mod9 does not limit content size. Mod9 subclass.

    Attributes:
        content (bytes):
            The audio data bytes encoded as specified in
            ``RecognitionConfig``. Note: as with all bytes fields, proto
            buffers use a pure binary representation, whereas JSON
            representations use base64.
        uri (str):
            URI that points to a file that contains audio data bytes as
            specified in ``RecognitionConfig``. The file must not be
            compressed (for example, gzip). Mod9 supports a variety of
            storage backends: ``file://``, ``gs://``, ``http://``,
            ``s3://``.
    """

    content = proto.Field(proto.BYTES, number=1, oneof="audio_source")
    uri = proto.Field(proto.STRING, number=2, oneof="audio_source")


class RecognizeRequest(proto.Message):
    """
    OVERRIDE: The top-level message sent by the client for the
    ``Recognize`` method. Mod9 subclass to hold subclassed objects.

    Attributes:
        config (RecognitionConfig):
            Required. Provides information to the recognizer that
            specifies how to process the request.
        audio (RecognitionAudio):
            Required. The audio data to be recognized.
    """

    config = proto.Field(RecognitionConfig, number=1)
    audio = proto.Field(RecognitionAudio, number=2)


class SpeechClient(object):
    """A duck-typed extension of google.cloud.speech.SpeechClient."""

    def __init__(self, host=None, port=None, *args, **kwargs):
        """Ignore arguments other than host and port, and set Mod9's custom transport."""
        self.host = host
        self.port = port

    def long_running_recognize(self, request=None, *, config=None, audio=None, **kwargs):
        """
        Performs asynchronous speech recognition.

        Mod9: Not currently implemented.
        """
        # TODO: similar logic to recognize for handling various input types.
        return common.long_running_recognize(
            request,
            host=self.host,
            port=self.port,
            **kwargs,
        )

    def recognize(self, request=None, *, config=None, audio=None, **kwargs):
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
                ``request`` is provided, this should not be set.
            audio (RecognitionAudio):
                Required. The audio data to be recognized. This
                corresponds to the ``audio`` field on the ``request``
                instance; if ``request`` is provided, this should not be
                set.

        Returns:
            RecognizeResponse:
                The only message returned to the client by the
                ``Recognize`` method. It contains the result as zero or
                more sequential ``SpeechRecognitionResult`` messages.
        """

        # Replicate the logic for handling various input types.
        has_flattened_params = any([config, audio])
        if request is not None and has_flattened_params:
            raise ValueError(
                "If the ``request`` argument is set, then none of "
                "the individual field arguments should be set."
            )
        if not isinstance(request, RecognizeRequest):
            if request is None:
                request = RecognizeRequest(config=config, audio=audio)
            else:
                request = RecognizeRequest(**request)
        return common.recognize(
            request,
            module=sys.modules[__name__],
            host=self.host,
            port=self.port,
            **kwargs,
        )

    def streaming_recognize(self, config, requests, **kwargs):
        """
        Perform streaming speech recognition.

        This method allows you to receive results while sending audio.

        Example:
          >>> from mod9.asr import speech_mod9 as speech
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

        # Replicate the logic of google.cloud.speech_v1.helpers.SpeechHelpers.
        return common.streaming_recognize(
            self._streaming_request_iterable(config, requests),
            module=sys.modules[__name__],
            host=self.host,
            port=self.port,
            **kwargs,
        )

    def _streaming_request_iterable(self, config, requests):
        # Replicate the logic of google.cloud.speech_v1.helpers.SpeechHelpers.
        yield {'streaming_config': config}
        for request in requests:
            yield request


# Used to compile protobufs into Python classes.
__all__ = tuple(sorted(__protobuf__.manifest))
