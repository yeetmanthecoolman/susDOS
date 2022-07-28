"""
Provides defaults used throughout mod9-asr.
"""

import logging
import os

# Current wrappers version.  Note that this is not the same as the Engine version.
WRAPPER_VERSION = '1.11.3'

# CHANGELOG:
#   1.11.3 (19 May 2022):
#   - Fix major bugs in Python SDK caused by an internal refactoring circa version 1.7.0.
#   1.11.2 (28 Mar 2022):
#   - Fix minor bug to allow multiple rescoring after running with --sclite option.
#   1.11.1 (22 Mar 2022):
#   - Fix minor bug to allow asrModel request option to be specified.
#   1.11.0 (21 Mar 2022):
#   - Allow longer connection timeout, useful for benchmarking with speed=3.
#   - Support multichannel audio for any number of channels, unlike Google that has restrictions
#     based on the encoding. Internally, the Engine supports a max of 128 byte-aligned samples.
#     for 16-bit encoding, that is 64 channels.
#     - The request option "audioChannelCount" indicates the number of channels.
#     - The request option "enableSeparateRecognitionPerChannel" must be `true` for multichannel
#       audio recognized via Mod9. This is unlike Google that recognizes only the first channel
#       for the default value of `false`.
#     - The response field "channelTag" indicates the channel number of the segment.
#   1.10.0 (16 Feb 2022):
#   - Ignore empty word in Switchboard benchmark scoring script (for legacy Remeeting ASR API).
#   - Add Rev.ai formatted result handling to Switchboard benchmark script.
#   1.9.0 (09 Feb 2022):
#   - Add Deepgram formatted result handling to Switchboard benchmark script.
#   1.8.0 (08 Feb 2022):
#   - Add Microsoft formatted result handling to Switchboard benchmark script.
#   1.7.0 (02 Feb 2022):
#   - Allow multiple clients to connect to different Engine hosts and ports.
#   - Support more audio encodings: 24- and 32-bit signed integers and 32-bit float.
#   - Add IBM Watson format conversion to mod9-asr-switchboard-benchmark.
#   1.6.0 (10 Jan 2022):
#   - Add "asrModel" to results using speech_mod9 and REST API.
#   - Minor fixes to mod9-asr-switchboard-benchmark and new features:
#     - Enable scoring of 1-best results in Amazon Transcribe formatted JSON.
#     - Set --optional-backchannels to make %BCACK and %BCNACK optionally deletable in the STM.
#     - Set --omit-backchannels to remove from the CTM, since it won't help if optionally deletable.
#     - Set --omit-hesitations to remove %HESITATION from the CTM, since it's optionally deletable.
#     - Change --exclude-words to --omit-words, now defaulting to non-speech w/o hesitations.
#   1.5.1 (06 Jan 2022):
#   - Minor fix to mod9-asr-switchboard-benchmark.
#   1.5.0 (06 Jan 2022):
#   - Fixed handling of gs:// URIs in which the blob name requires percent-encoded URLs.
#   - Make "languageCode" optional, in contrast to Google; default is first Engine model loaded.
#   - Do not accept "command_and_search" as model type.
#   - The speech_mod9 module and REST API now extend support for "asrModel".
#     - This will override Google-compatible "languageCode" and "model", if specified.
#   - The speech_mod9 module and REST API now extend support for "maxWordAlternatives".
#   - Only allow "maxAlternatives" up to 1000 transcript-level alternatives.
#   - Improve determination of WAV files by checking header of HTTP(S) URI files.
#   - Confirm existence of audio via URI to avoid waiting for timeout if audio does not exist.
#   - The speech_mod9 module and REST API now extend support for "intervalsJson".
#   - Remove "enablePhraseConfidence" option from speech_mod9 and REST API; always report biases.
#   - The REST API now always reports 19-digit operation names.
#   - Various minor changes to mod9-asr-switchboard-benchmark:
#     - Default of 0 for --max-expansions, fully-expanded alternatives assuming bugfixed SCTK.
#     - Improved parsing of Google STT-formatted results, and optimization of refiltered CTM.
#     - Added --alternatives-max to allow scoring variable depth lists of alternatives.
#     - Added --verbose option, false by default since the tool was otherwise too verbose.
#     - When verbose, report statistics about the size, depth, and width of alternatives.
#   1.4.2 (16 Dec 2021):
#   - Enable mod9-asr-websocket-client to request non-recognize commands without audio data.
#   1.4.1 (27 Nov 2021):
#   - Minor bugfixes.
#   1.4.0 (23 Nov 2021):
#   - Add mod9-asr-switchboard-benchmark to replicate results at rmtg.co/benchmark.
#   1.3.0 (18 Nov 2021):
#   - Add mod9-asr-elasticsearch-client to demonstrate indexing of phrase alternatives.
#   - Enable non-English languages to be specified with "languageCode" option.
#     Unlike Google STT, a region suffix may be omitted, e.g. "en" instead of "en-US".
#   - Support the "model" option, similarly to Google STT.
#   1.2.1 (11 Nov 2021):
#   - Bugfix to allow WebSocket server to handle responses up to 1 MiB (instead of 64KiB).
#   - This setting may be overriden with the --websocket-limit-bytes option.
#   1.2.0 (30 Aug 2021):
#   - Improved logging.
#   - Allow "rate" option to be in the range [8000,48000], as with Google STT.
#   - Added "speed" option to speech_mod9.
#   - Added "options_json" to speech_mod9.
#   1.1.1 (11 Aug 2021):
#   - Rebuild correctly (after `rm -rf build/ dist/ *.egg-info`)
#   1.1.0 (11 Aug 2021):
#   - Released in coordination with Engine version 1.1.0 (coincidental version match, not causal).
#   - Added "latency" request option to speech_mod9.
#   - REST API now logs to a file, with UUIDs both for itself and the proxied Engine.
#   1.0.0 (31 Jul 2021):
#   - This version is not compatible with Engine version < 1.0.0 (due to "asr-model" option).
#   - Bugfixes to WebSocket interface; also add --skip-engine-check and --allow-*-uri (for REST).
#   0.5.0 (28 May 2021): Add Websocket Interface.
#   0.4.1 (20 May 2021): Additional minor documentation fixes; Flask-RESTful version pinning.
#   0.4.0 (30 Apr 2021): Rename mod9-rest-server to mod9-asr-rest-api; minor documentation fixes.

# Range of compatible Engine versions for current wrappers.
#  Lower bound is inclusive, upper bound is exclusive.
#  ``None`` indicates no bound.
# PySDK 1.11.0 requires Engine 1.9.0+ to support multichannel audio.
WRAPPER_ENGINE_COMPATIBILITY_RANGE = ('1.9.0', None)

ASR_ENGINE_HOST = os.getenv('ASR_ENGINE_HOST', 'localhost')
ASR_ENGINE_PORT = int(os.getenv('ASR_ENGINE_PORT', 9900))

SOCKET_CONNECTION_TIMEOUT_SECONDS = 10.0
SOCKET_INACTIVITY_TIMEOUT_SECONDS = 300.0
ENGINE_CONNECTION_RETRY_SECONDS = 1.0

# These should be small enough so that it doesn't trigger the Engine's read timeout (10s default).
MAX_CHUNK_SIZE = 128 * 1024  # Used as chunk size for URI producers; limits generators.
GS_CHUNK_SIZE = 262144  # Google requires chunks be multiples of 262144

FLASK_ENV = os.getenv('FLASK_ENV', None)

# Audio URI prefixes to accept, used by REST only (PySDK allows all).
#  Operator can set at server launch; default is allow none.
ASR_REST_API_ALLOWED_URI_SCHEMES = os.getenv('ASR_REST_API_ALLOWED_URI_SCHEMES', set())
if ASR_REST_API_ALLOWED_URI_SCHEMES:
    ASR_REST_API_ALLOWED_URI_SCHEMES = ASR_REST_API_ALLOWED_URI_SCHEMES.lower().split(sep=',')
    ASR_REST_API_ALLOWED_URI_SCHEMES = set(
        scheme.replace('://', '') for scheme in ASR_REST_API_ALLOWED_URI_SCHEMES
    )

if 'http' in ASR_REST_API_ALLOWED_URI_SCHEMES and 'https' not in ASR_REST_API_ALLOWED_URI_SCHEMES:
    logging.warning('REST API set to allow http:// but NOT https:// audio URIs.')

# Limit on number of bytes allowed per reply line read by WebSocket server.
WEBSOCKET_LIMIT_BYTES = 1024 * 1024  # 1 MiB
