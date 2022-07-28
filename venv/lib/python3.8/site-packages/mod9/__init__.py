"""
Python SDK and REST API, high-level interfaces to the Mod9 ASR Engine.

Requires a Mod9 ASR Engine server; see https://mod9.io for details.

More documentation and example usage can be found at
  https://mod9.io/python
  https://mod9.io/rest

Within you will find the following modules:

``mod9.asr`` implements the Mod9 ASR Python SDK.
Two versions are offered within this module, both of which are drop-in
replacements for the Google STT Python Client Library"
  - ``mod9.asr.speech``, uses Google's objects for input/output and
    offers a strict subset of Google functionality.
  - ``mod9.asr.speech_mod9`` extends Google's functionality with
     Mod9-exclusive options, such as phrase alternatives.

``mod9.rest.server`` implements the Mod9 ASR REST API, a compatible
drop-in replacement for the Google Cloud STT REST API.
Run ``mod9-asr-rest-api`` from the command line to launch a REST server.
This script can be installed by ``pip --user`` at ``. ~/.local/bin``.

``mod9.reformat`` contains the internal library support. For example,
``mod9.reformat.google`` converts between Mod9 and Google data formats,
while ``mod9.reformat.utils`` communicates with the Mod9 ASR Engine.
"""
