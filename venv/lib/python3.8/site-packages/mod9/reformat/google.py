"""
Functions for converting Google-style and -type input and output to/from
Mod9-style and -type.
"""

import base64
from binascii import Error as BinasciiError
from collections import OrderedDict
import itertools
import json
import logging
from types import GeneratorType

from mod9.reformat import utils
from mod9.reformat.config import (
    WRAPPER_ENGINE_COMPATIBILITY_RANGE,
    WRAPPER_VERSION,
)

CHUNKSIZE = 8 * 1024


class ConflictingGoogleAudioSettingsError(Exception):
    pass


class Mod9EngineFailedStatusError(Exception):
    pass


# Used to allow arbitrary `model`s to be loaded into Engine and requested by user.
class ObjectContainingEverything:
    """Object that always returns True to `if x in ObjectContainingEverything()` queries."""
    def __contains__(self, x):
        return True


# Used to allow number ranges to be requested by user.
class RealNumericalRangeObject:
    """Object that returns True if ``x`` is in given numerical range."""
    def __init__(self, low, high):
        self.low = low
        self.high = high

    def __contains__(self, x):
        return x >= self.low and x <= self.high


class GoogleConfigurationSettingsAndMappings:
    """
    Hold dicts used to translate keys and values from Google to Mod9
    forms.

    Internally use camelCase for google_allowed_keys. We chose to use
    camelCase internally for compatibility with Google's
    ``protobuf.json_format`` funtions (which are called by ``to_json()``
    and ``from_json()``). We need to make sure we use the proper casing,
    both internally and for output, or can run into non-obvious bugs.
    These bugs can be non-obvious because many of the primary attributes
    are one-word attributes (like ``.transcript``) and so functionality
    will only break if it is dependent on multi-word attributes.

    Input from REST API come in camelCase. Input from Python SDK come in
    snake_case, are converted to camelCase using either ``from_json()``
    (in the case of Google protobuf object input), or
    ``mod9.reformat.utils.recursively_convert_dict_keys_case()`` and
    ``reformat.utils.snake_case_to_camel_case()`` (in the case of dict
    input). Output from REST API use camelCase. Output from Python SDK
    converted to snake_case in ``to_json()``.
    """
    def __init__(self):
        """
        Define dicts used to translate keys and values from Google to
        Mod9 forms.
        """

        # Mod9 and Google input key names for use in following dict's.
        # NOTE: options-json and intervals-json are exceptional; they are not valid Engine options.
        self.mod9_allowed_keys = (
            'encoding',
            'rate',
            'word-confidence',
            'word-intervals',
            'transcript-formatted',
            'transcript-alternatives',
            'channels',
            'language-code',          # transformed to a real Mod9 option
            'model',                  # transformed to a real Mod9 option
            'recognition-per-channel',  # only works in speech_mod9 or REST. This setting
                                        # is ignored if true and error if false.
            'phrase-alternatives',    # only works in speech_mod9 or REST
            'word-alternatives',      # only works in speech_mod9 or REST
            'latency',                # only works in speech_mod9
            'speed',                  # only works in speech_mod9 or REST
            'options-json',           # only works in speech_mod9 or REST
            'intervals-json',         # only works in speech_mod9 or REST
            'asr-model',              # only works in speech_mod9 or REST
        )
        self.google_allowed_keys = (
            'encoding',
            'sampleRateHertz',
            'enableWordConfidence',
            'enableWordTimeOffsets',
            'enableAutomaticPunctuation',
            'maxAlternatives',
            'audioChannelCount',
            'languageCode',
            'model',
            'enableSeparateRecognitionPerChannel',
            'maxPhraseAlternatives',  # only works in speech_mod9 or REST
            'maxWordAlternatives',    # only works in speech_mod9 or REST
            'latency',                # only works in speech_mod9
            'speed',                  # only works in speech_mod9 or REST
            'optionsJson',            # only works in speech_mod9 or REST
            'intervalsJson',          # only works in speech_mod9 or REST
            'asrModel',               # only works in speech_mod9 or REST
        )

        # Map from (Google key) to (Mod9 key).
        self.google_to_mod9_key_translations = dict(
            zip(
                self.google_allowed_keys,
                self.mod9_allowed_keys,
            )
        )

        # Allowed Google values.
        self.google_encoding_allowed_values = {
            'ENCODING_UNSPECIFIED',
            'LINEAR16',
            'MULAW',
            # Only in Mod9:
            'ALAW',
            'LINEAR24',
            'LINEAR32',
            'FLOAT32',
        }
        self.google_rate_allowed_values = range(8000, 48001)
        self.google_confidence_allowed_values = {True, False}
        self.google_timestamp_allowed_values = {True, False}
        self.google_punctuation_allowed_values = {True, False}
        self.google_max_alternatives_allowed_values = range(1001)
        self.google_audio_channel_count_allowed_values = ObjectContainingEverything()
        self.google_language_code_allowed_values = ObjectContainingEverything()
        self.google_model_allowed_values = {'phone_call', 'video', 'default'}
        self.google_enable_separate_recognition_per_channel_allowed_values = {True, False},
        self.google_max_phrase_alternatives_allowed_values = range(10001)
        self.google_max_word_alternatives_allowed_values = range(10001)
        self.latency_allowed_values = RealNumericalRangeObject(0.01, 3.0)
        self.speed_allowed_values = range(1, 10)
        self.options_json_allowed_values = ObjectContainingEverything()
        self.intervals_json_allowed_values = ObjectContainingEverything()
        self.asr_model_allowed_values = ObjectContainingEverything()

        # Group allowed Google values. To be used in following dict.
        self.google_allowed_values = (
            self.google_encoding_allowed_values,
            self.google_rate_allowed_values,
            self.google_confidence_allowed_values,
            self.google_timestamp_allowed_values,
            self.google_punctuation_allowed_values,
            self.google_max_alternatives_allowed_values,
            self.google_audio_channel_count_allowed_values,
            self.google_language_code_allowed_values,
            self.google_model_allowed_values,
            self.google_enable_separate_recognition_per_channel_allowed_values,
            self.google_max_phrase_alternatives_allowed_values,
            self.google_max_word_alternatives_allowed_values,
            self.latency_allowed_values,
            self.speed_allowed_values,
            self.options_json_allowed_values,
            self.intervals_json_allowed_values,
            self.asr_model_allowed_values,
        )

        # Map from (Mod9 key) to (allowed Google values) for given key.
        self.mod9_keys_to_allowed_values = dict(
            zip(
                self.mod9_allowed_keys,
                self.google_allowed_values,
            )
        )

        # Map from (allowed Google encodings) to (Mod9 encodings)
        self.google_encoding_to_mod9_encoding = {
            'LINEAR16': 'pcm_s16le',
            'MULAW': 'mu-law',
            'ALAW': 'a-law',
            'LINEAR24': 'pcm_s24le',
            'LINEAR32': 'pcm_s32le',
            'FLOAT32': 'pcm_f32le',
        }


def input_to_mod9(
    google_input_settings,
    module,
    logger=logging.getLogger(),
    host=None,
    port=None
):
    """
    Wrapper method to take Google inputs of various types and return
    Mod9-compatible inputs.

    Args:
        google_input_settings (dict):
            Contains dicts or Google-like-types ``.config`` and
            ``.audio``: options for transcription and audio to be
            transcribed, respectively.
        module (module):
            Module to read Google-like-types from, in case of
            subclassing.
        host (Union[str, None]):
            Engine host, or None to default to config.
        port (Union[int, None]):
            Engine port, or None to default to config.

    Returns:
        tuple:
            mod9_config_settings (dict):
                Mod9-style options to pass to Mod9 ASR Engine TCP Server.
            mod9_audio_settings (dict):
                Mod9-style audio to pass to Mod9 ASR Engine TCP Server.
    """

    engine_version = utils.get_version_mod9(host=host, port=port)
    if not utils.is_compatible_mod9(engine_version):
        raise utils.Mod9IncompatibleEngineVersionError(
            f"Python SDK version {WRAPPER_VERSION} compatible range"
            f" {WRAPPER_ENGINE_COMPATIBILITY_RANGE}"
            f" does not include given Engine of version {engine_version}."
            ' Please use a compatible SDK-Engine pairing. Exiting.'
        )

    # Convert keys from snake_case to camelCase (if necessary), which we use internally.
    #  See docstring for GoogleConfigurationSettingsAndMappings for more info.
    google_input_settings = utils.recursively_convert_dict_keys_case(
        google_input_settings,
        utils.snake_case_to_camel_case,
    )

    # Convert Google-type inputs to dict-type inputs, if necessary.
    if isinstance(google_input_settings['config'], dict):
        google_config_settings_dict = google_input_settings['config']
    elif isinstance(google_input_settings['config'], module.RecognitionConfig):
        google_config_settings_dict = json.loads(
            module.RecognitionConfig.to_json(
                google_input_settings['config'],
                use_integers_for_enums=False,
            )
        )
    elif isinstance(google_input_settings['config'], module.StreamingRecognitionConfig):
        google_config_settings_dict = json.loads(
            module.StreamingRecognitionConfig.to_json(
                google_input_settings['config'],
                use_integers_for_enums=False,
            )
        )

    if 'audio' not in google_input_settings or google_input_settings['audio'] is None:
        # Empty dict will lead to mod9_audio_settings returning None.
        google_audio_settings_dict = dict()
    elif isinstance(google_input_settings['audio'], dict):
        google_audio_settings_dict = google_input_settings['audio']
    elif isinstance(google_input_settings['audio'], module.RecognitionAudio):
        google_audio_settings_dict = json.loads(
            module.RecognitionAudio.to_json(google_input_settings['audio'])
        )

    # Convert Google-style inputs to Mod9-style inputs.
    mod9_config_settings = google_config_settings_to_mod9(
        google_config_settings_dict,
        logger=logger,
        host=host,
        port=port,
    )
    mod9_audio_settings = google_audio_settings_to_mod9(google_audio_settings_dict)

    # None is a placeholder since we need to inspect the audio_settings to determine file format.
    if 'format' in mod9_config_settings and mod9_config_settings['format'] is None:
        # Set file type based on file header.
        if isinstance(mod9_audio_settings, GeneratorType):
            # Split generator so utils.parse_wav_encoding() can look at content header.
            mod9_audio_settings, mod9_audio_settings_clone = itertools.tee(mod9_audio_settings)
        wav_encoding, wav_sample_rate, wav_channels = utils.parse_wav_encoding(mod9_audio_settings)
        if wav_encoding:
            mod9_config_settings['format'] = 'wav'
            if 'encoding' in mod9_config_settings:
                if wav_encoding != mod9_config_settings['encoding']:
                    # The Google Cloud STT API complains if WAV and encoding are mismatched.
                    raise ConflictingGoogleAudioSettingsError(
                        "WAV file format encoded as %s should match config specified as %s."
                        % (wav_encoding, mod9_config_settings['encoding'])
                    )
                del mod9_config_settings['encoding']
            if wav_sample_rate:
                if 'rate' in mod9_config_settings:
                    if mod9_config_settings['rate'] != wav_sample_rate:
                        raise ValueError(f"Specified rate, {mod9_config_settings['rate']},"
                                         f" differs from WAV header rate, {wav_sample_rate}.")
                else:
                    mod9_config_settings['rate'] = wav_sample_rate
            if 'channels' in mod9_config_settings:
                if wav_channels and wav_channels != mod9_config_settings['channels']:
                    # The Google Cloud STT API complains if WAV header and channels are mismatched.
                    raise ConflictingGoogleAudioSettingsError(
                        f"Specified channel count {mod9_config_settings['channels']}, "
                        f"differs from WAV header channel count {wav_channels}.")
                # The Mod9 ASR Engine complains if both WAV format and audio channels are specified.
                del mod9_config_settings['channels']
        else:
            mod9_config_settings['format'] = 'raw'
            if 'encoding' not in mod9_config_settings:
                raise KeyError('Must specify an audio encoding for non-WAV file formats')
        # Reset audio using tee'd clone if it exists.
        try:
            mod9_audio_settings = mod9_audio_settings_clone
        except UnboundLocalError:
            pass

    # Mod9 TCP does not accept 'rate' argument for 'wav' format.
    if 'format' not in mod9_config_settings or mod9_config_settings['format'] == 'wav':
        if 'rate' in mod9_config_settings:
            del mod9_config_settings['rate']

    # Need transcript intervals to mirror Google response format.
    mod9_config_settings['transcript-intervals'] = True

    return mod9_config_settings, mod9_audio_settings


def google_config_settings_to_mod9(
    google_config_settings,
    logger=logging.getLogger(),
    host=None,
    port=None
):
    """
    Map from Google-style key:value inputs to Mod9 ASR TCP server-style
    key:value inputs.

    Args:
        google_config_settings (dict):
            Google-style options.
        host (Union[str, None]):
            Engine host, or None to default to config.
        port (Union[int, None]):
            Engine port, or None to default to config.

    Returns:
        dict:
            Mod9-style options to pass to Mod9 ASR Engine TCP Server.
    """

    settings = GoogleConfigurationSettingsAndMappings()
    mod9_config_settings = dict()

    # StreamingRecognitionConfig has config attribute of type RecognitionConfig.
    if 'config' in google_config_settings:
        # Grab streaming options and assign google_config_settings to RecognitionConfig attribute.
        if google_config_settings.get('singleUtterance') is True:
            raise NotImplementedError(
                'Streaming recognize not yet implemented for Google option single_utterance: True.'
            )
        mod9_config_settings['partial'] = google_config_settings.get('interimResults', False)
        google_config_settings = google_config_settings['config']

        # Turn off batch mode for streaming, otherwise partial will not work.
        mod9_config_settings['batch-threads'] = 0
    else:
        # Use max number of threads available for best speed.
        mod9_config_settings['batch-threads'] = -1

        # This option only applies with speech_mod9.
        if google_config_settings.get('latency'):
            raise KeyError("Option key 'latency' only supported for streaming requests.")

    if 'languageCode' not in google_config_settings and 'asrModel' not in google_config_settings:
        log_string = 'No languageCode given. Defaulting to first loaded model: %s of language %s.'
        models = utils.get_loaded_models_mod9(host=host, port=port)
        default_model_name = models[0].get('name', 'model')
        default_model_language = models[0]['language']
        logger.warning(log_string, default_model_name, default_model_language)
        google_config_settings['languageCode'] = default_model_language
        # Preempt setting model below based on ``languageCode``.
        mod9_config_settings['asr-model'] = default_model_name

    # ``to_json()`` populates absent attributes of config with falsy values -> exceptions later.
    google_config_settings = {
        key: value for key, value in google_config_settings.items() if value
    }

    # Accept ``"useEnhanced"`` for compatibility with Google, but discard.
    if 'useEnhanced' in google_config_settings:
        del google_config_settings['useEnhanced']

    # A subset of possible Google keys are supported by this wrapper and the Mod9 Engine.
    for google_key in google_config_settings:
        if google_key not in settings.google_allowed_keys:
            raise KeyError(f"Option key '{google_key}' not supported.")

    # Translate google_config_settings keys to corresponding Mod9 keys and values.
    for google_key, google_value in google_config_settings.items():
        mod9_key = settings.google_to_mod9_key_translations[google_key]
        # A subset of possible Google values are supported by this wrapper and the Mod9 Engine.
        if google_value in settings.mod9_keys_to_allowed_values[mod9_key]:
            # Some Mod9 values are equivalent to Google values. Others to be translated later.
            mod9_config_settings[mod9_key] = google_value
        else:
            raise KeyError(f"Option value '{google_value}' not supported for key '{google_key}'.")

    # Do first step of translating Mod9 'encoding' value from Google to Mod9 format + encoding.
    # Format will map to 'wav' or 'raw'. Set placeholder until determined by file header.
    mod9_config_settings['format'] = None
    if 'encoding' in mod9_config_settings:
        if mod9_config_settings['encoding'] == 'ENCODING_UNSPECIFIED':
            mod9_config_settings['format'] = 'wav'
            del mod9_config_settings['encoding']
        else:
            mod9_config_settings['encoding'] = \
                settings.google_encoding_to_mod9_encoding[mod9_config_settings['encoding']]

    # Set N-best settings.
    #  ``result_from_mod9()`` iterates through alternatives.
    n_best_N = mod9_config_settings.get('transcript-alternatives')
    if n_best_N is not None:
        if n_best_N == 0:
            # Google sets n_best_N: 0 -> 1.
            n_best_N = 1
        mod9_config_settings['transcript-alternatives'] = n_best_N

    # These options only apply with speech_mod9 and REST.
    if mod9_config_settings.get('phrase-alternatives'):
        # Phrase alternatives always include AM/LM bias (presented as "confidence") and timestamps.
        mod9_config_settings['phrase-alternatives-bias'] = True
        mod9_config_settings['phrase-intervals'] = True
    if mod9_config_settings.get('word-alternatives'):
        # Word alternatives might include confidence and timestamps, if they are enabled for 1-best.
        mod9_config_settings['word-alternatives-confidence'] = \
            mod9_config_settings.get('word-confidence', False)

    if not mod9_config_settings.get('asr-model'):
        # Set appropriate model based on user input ``languageCode``.
        models = utils.find_loaded_models_with_language(
            google_config_settings['languageCode'],
            host=host,
            port=port,
        )
        if len(models) == 0:
            raise ValueError(
                f"Language {google_config_settings['languageCode']} not supported or "
                f"loaded. Currently loaded: {sorted(utils.get_model_languages())}"
            )
        elif len(models) == 1:
            mod9_config_settings['asr-model'] = models[0]['name']
        else:
            model = utils.select_best_model_for_language_code(
                models,
                google_config_settings['languageCode'],
                model_type=google_config_settings.get('model'),
            )
            mod9_config_settings['asr-model'] = model['name']
    if mod9_config_settings.get('language-code'):
        del mod9_config_settings['language-code']
    if mod9_config_settings.get('model'):
        del mod9_config_settings['model']

    if 'intervals-json' in mod9_config_settings:
        try:
            mod9_config_settings['batch-intervals'] = \
                json.loads(mod9_config_settings['intervals-json'])
        except Exception:
            raise ValueError('Could not parse intervals JSON (should be an array of arrays).')
        del mod9_config_settings['intervals-json']

    # These speech_mod9 options override all prior options that might have been set.
    if 'options-json' in mod9_config_settings:
        try:
            extra_options = json.loads(mod9_config_settings['options-json'])
        except Exception:
            raise ValueError('Could not parse additional options JSON.')
        mod9_config_settings.update(extra_options)
        del mod9_config_settings['options-json']

    # This option does not have an equivalent Mod9 option.
    # For multi-channel audio, Mod9 only supports transcribing all channels. The setting is true
    # by default and raises an exception if it's set to false.
    # GSTT sets enableSeparateRecognitionPerChannel to false by default and only transcribes
    # the first channel in that case.
    if 'recognition-per-channel' in mod9_config_settings:
        if not mod9_config_settings['recognition-per-channel'] and \
           mod9_config_settings.get('channels', 1) > 1:
            # Only `true` value is supported.
            raise ValueError('Setting enableSeparateRecognitionPerChannel must be set to '
                             "'true' for multichannel audio.")
        del mod9_config_settings['recognition-per-channel']

    return mod9_config_settings


def google_audio_settings_to_mod9(google_audio_settings):
    """
    Map from Google-style audio input to Mod9 TCP server-style audio
    input.

    Args:
        google_audio_settings (dict):
            Google-style audio.

    Returns:
        Union[str, Iterable[bytes]]:
            Mod9-style audio to pass to Mod9 ASR Engine TCP Server.
    """

    if not google_audio_settings:
        return None

    # Require one, and only one, of 'uri' or 'content'.
    if 'uri' in google_audio_settings and 'content' in google_audio_settings:
        raise ConflictingGoogleAudioSettingsError("Got both 'uri' and 'content' keys.")
    if 'uri' not in google_audio_settings and 'content' not in google_audio_settings:
        raise KeyError("Got neither 'uri' nor 'content' key.")

    if 'uri' in google_audio_settings:
        mod9_audio_settings = google_audio_settings['uri']
        utils.uri_exists(mod9_audio_settings)
    else:
        # Decode google_audio_settings byte string if Base64 encoded; send chunks in generator.
        try:
            byte_string = base64.b64decode(google_audio_settings['content'], validate=True)
        except BinasciiError:
            byte_string = google_audio_settings['content']
        mod9_audio_settings = (
            byte_string[i:i+CHUNKSIZE] for i in range(0, len(byte_string), CHUNKSIZE)
        )

    return mod9_audio_settings


def result_from_mod9(
    mod9_results,
    logger=logging.getLogger(),
    host=None,
    port=None,
):
    """
    Map from Mod9 TCP server-style output to Google-style output.

    Args:
        mod9_results (Iterable[dict]):
            Mod9-style results from the Mod9 ASR Engine TCP Server.
        host (Union[str, None]):
            Engine host, or None to default to config.
        port (Union[int, None]):
            Engine port, or None to default to config.

    Yields:
        dict:
            Google-style result.
    """

    # Set language below based on first Engine reply. Raise exception if cannot set.
    language_code = None

    # In speech_mod9, results should indicate which ASR model was used.
    asr_model = None

    # Longer audio comes chopped into segments.
    for mod9_result in mod9_results:
        if mod9_result['status'] != 'processing':
            # Non-'processing' status -> failure or transcription is complete.
            if mod9_result['status'] == 'failed':
                raise Mod9EngineFailedStatusError(f"Mod9 server issues 'failed': {mod9_result}.")
            elif mod9_result['status'] != 'completed':
                logger.error("Unexpected Mod9 server response: %s.", mod9_result)
            else:
                # Status 'completed' is final response (with no transcript).
                break

        if 'result_index' not in mod9_result:
            if mod9_result.get('asr_model'):
                # This is expected to be the first reply.
                asr_model = mod9_result['asr_model']
                models = utils.get_loaded_models_mod9(host=host, port=port)
                for model in models:
                    if model['name'] == asr_model:
                        language_code = model['language']
                        break
                if not language_code:
                    raise utils.Mod9EngineCouldNotGetModelNameError(
                        'Could not set language code '
                        f"(asr_model field not found in {mod9_result})."
                    )
            continue

        alternatives = []
        if 'alternatives' in mod9_result:
            for alternative_number, mod9_alternative in enumerate(mod9_result['alternatives']):
                alternative = build_google_alternative(mod9_result['result_index'])
                if alternative_number == 0 and 'transcript_formatted' in mod9_result:
                    alternative['transcript'] += mod9_result['transcript_formatted']
                else:
                    alternative['transcript'] += mod9_alternative['transcript']
                alternatives.append(alternative)
        else:
            # Partial results (``.final`` == ``False``) from Mod9 do not have alternatives
            #  or ``.transcript_formatted``.
            #  Requests without transcript alternatives do not have alternatives,
            #  but may have ``.transcript_formatted``.
            alternative = build_google_alternative(mod9_result['result_index'])
            if 'transcript_formatted' in mod9_result:
                alternative['transcript'] += mod9_result['transcript_formatted']
            else:
                alternative['transcript'] += mod9_result['transcript']

            alternatives.append(alternative)

        # Build the WordInfo if Mod9 has returned word-level results.
        #  If returning N-best, only 1-best gets word alternatives.
        if 'words' in mod9_result:
            words = []
            for mod9word in mod9_result['words']:
                new_word = OrderedDict()
                if 'interval' in mod9word:
                    start_time, end_time = mod9word['interval']
                    new_word['startTime'] = "{:.3f}s".format(start_time)
                    new_word['endTime'] = "{:.3f}s".format(end_time)

                new_word['word'] = mod9word['word']

                if 'confidence' in mod9word:
                    new_word['confidence'] = mod9word['confidence']

                words.append(new_word)
            alternatives[0]['words'] = words

        google_result = {
            'alternatives': alternatives,
            'isFinal': mod9_result['final'],
        }

        if 'channel' in mod9_result:
            google_result['channelTag'] = int(mod9_result['channel'])

        # NOTE: this is only returned in v1p1beta1, and it's lowercase for some reason.
        google_result['languageCode'] = language_code

        google_result['resultEndTime'] = "{:.3f}s".format(mod9_result['interval'][1])

        # NOTE: this will be retained only in speech_mod9.
        google_result['asrModel'] = asr_model

        if not mod9_result['final']:
            google_result['stability'] = 0.0  # Google's default.

        if 'phrases' in mod9_result:
            google_result['phrases'] = mod9_result['phrases']
            for phrase in google_result['phrases']:
                interval = phrase.pop('interval')
                phrase['startTime'] = "{:.3f}s".format(interval[0])
                phrase['endTime'] = "{:.3f}s".format(interval[1])

        if 'words' in mod9_result and len(mod9_result['words']) > 0:
            if 'alternatives' in mod9_result['words'][0]:
                google_result['words'] = mod9_result['words']
                for word in google_result['words']:
                    if 'interval' in word:
                        interval = word.pop('interval')
                        word['startTime'] = "{:.3f}s".format(interval[0])
                        word['endTime'] = "{:.3f}s".format(interval[1])

        yield google_result


def build_google_alternative(transcript_number, confidence_value=1.0):
    """
    Build template for Google alternative.

    Args:
        transcript_number (int):
            Indicate the segment/endpoint the alternative is a part of.
        confidence_value (float):
            Rating in [0.0, 1.0] indicating confidence this alternative
            is the true transcript/one-best (default is 1.0).

    Returns:
        dict:
            Google-style transcript alternative (i.e. one of N-best list).
    """

    alternative = OrderedDict([('transcript', '')])
    # Google transcripts after the first start with a space.
    if transcript_number > 0:
        alternative['transcript'] += ' '

    # Add placeholder value for transcript-level confidence.
    alternative['confidence'] = confidence_value

    return alternative


def google_type_result_from_dict(
        google_result_dicts,
        google_result_type,
        module,
):
    """
    Convert dict-type result iterable to Google-type result generator.

    Args:
        google_result_dicts (Iterable[dict]):
            Google-style results.
        google_result_type (
            Union[
                module.SpeechRecognitionResult,
                module.StreamingRecognitionResult,
            ]
        ):
            Google-like-type to return.
        module (module):
            Module to read Google-like-types from, in case of
            subclassing.

    Yields:
        Union[
            module.SpeechRecognitionResult,
            module.StreamingRecognitionResult,
        ]:
            Google-like-type result.
    """

    for google_result_dict in google_result_dicts:
        # ``from_json()`` complains if attributes that do not exist in a protobuf are passed.
        if google_result_type == module.SpeechRecognitionResult:
            if 'isFinal' in google_result_dict:
                google_result_dict.pop('isFinal')
            if 'resultEndTime' in google_result_dict:
                google_result_dict.pop('resultEndTime')
        if module.__name__ != 'mod9.asr.speech_mod9':
            google_result_dict.pop('asrModel')

        # Internal camelCase keys are converted to snake_case by from_json().
        #  See docstring in GoogleConfigurationSettingsAndMappings for more info.
        yield google_result_type.from_json(json.dumps(google_result_dict))
