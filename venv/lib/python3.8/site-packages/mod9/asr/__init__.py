"""
Python SDK wrappers over the Mod9 ASR Engine TCP Server.

Requires a Mod9 ASR Engine TCP server backend to work. Contact
sales@mod9.com for licensing inquiries.

More documentation and example usage can be found at
http://mod9.io/python-sdk


``mod9.asr.speech`` is referred to as the Mod9 ASR Python SDK wrapper,
and is a fully-compatible drop-in replacement for the Google STT Python
Client Library, making use of Google's objects. Send input in Google-
style objects or dicts as you would to the Google STT Python Client
Library and the request will be converted to Mod9-compatible form. The
results from the Mod9 ASR Engine TCP server will be converted to Google-
type objects.


``mod9.asr.speech_mod9`` extends the Mod9 ASR Python SDK wrapper with
Mod9-exclusive options, such as phrase alternatives and A-law audio
encodings (in addition to Google offerings such as transcript
alternatives/N-best lists and 16-but linear PCM and μ-law encodings).
Input can use either Google-type or Google-like subclasses within the
module (required to access Mod9-only options) or dict. Output is Google-
like subclasses defined within the module.
"""
from mod9.reformat.config import WRAPPER_VERSION as __version__
