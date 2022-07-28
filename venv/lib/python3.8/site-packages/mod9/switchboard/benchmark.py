#!/usr/bin/env python3

import argparse
import itertools
import json
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request

DESCRIPTION = """
This specialized tool can be used to score the Switchboard benchmark by suitably formatting and
scoring output from the Mod9 ASR Engine. It reads lines of JSON (i.e. JSONL format) from stdin and
prints a report on stdout, saving files in a work directory.
It can also read results formatted by Google, Amazon, Microsoft, Deepgram, Rev.ai or IBM.
This uses the official NIST SCTK software, which is expected to be installed on the system, and also
requires certain reference data files which might be downloaded. These dependencies are already
installed in the mod9/asr Docker image for convenience.
The Switchboard audio data is needed for meaningful demonstration and could be obtained from the
Linguistic Data Consortium (https://catalog.ldc.upenn.edu/LDC2002S09).
"""

ALTERNATIVES_LEVELS = [
    'word',        # e.g. MBR-derived sausages
    'phrase',      # patent-pending Mod9 data structure
    'transcript',  # i.e. N-best
]

OMITTED_WORDS = [
    # These non-speech words are not transcribed in the reference, and will always hurt the score.
    '[cough]',        # Kaldi recipes
    '[laughter]',     # Kaldi recipes
    '[noise]',        # Kaldi recipes
    '<affirmative>',  # Rev.ai
    '<laugh>',        # Rev.ai
]

REFERENCE_GLM = 'switchboard-benchmark.glm'
REFERENCE_STM = 'switchboard-benchmark.stm'
REFERENCE_URL = 'https://mod9.io/switchboard-benchmark'

SCLITE_OPTS = '-F -D'

SCTK_TOOLS = [
    'csrfilt.sh',
    'rfilter1',
    'sclite',
]

SWITCHBOARD_SPEAKER_IDS = [
    "sw_4390_A",
    "sw_4390_B",
    "sw_4484_A",
    "sw_4484_B",
    "sw_4507_A",
    "sw_4507_B",
    "sw_4520_A",
    "sw_4520_B",
    "sw_4537_A",
    "sw_4537_B",
    "sw_4543_A",
    "sw_4543_B",
    "sw_4547_A",
    "sw_4547_B",
    "sw_4560_A",
    "sw_4560_B",
    "sw_4577_A",
    "sw_4577_B",
    "sw_4580_A",
    "sw_4580_B",
    "sw_4601_A",
    "sw_4601_B",
    "sw_4604_A",
    "sw_4604_B",
    "sw_4683_A",
    "sw_4683_B",
    "sw_4686_A",
    "sw_4686_B",
    "sw_4689_A",
    "sw_4689_B",
    "sw_4694_A",
    "sw_4694_B",
    "sw_4776_A",
    "sw_4776_B",
    "sw_4824_A",
    "sw_4824_B",
    "sw_4854_A",
    "sw_4854_B",
    "sw_4910_A",
    "sw_4910_B",
]

WORKDIR = '/tmp/switchboard-benchmark'

VERBOSE = None  # HACK: use this as a global variable for logging ... see below.


def info(*args, **kwargs):
    if VERBOSE:
        print(*args, **kwargs)


def error(message):
    print(f"ERROR: {message}")
    exit(1)


def run(command, capture_output=True):
    try:
        return subprocess.run(command, capture_output=capture_output, shell=True, check=True)
    except subprocess.SubprocessError:
        error(f"Unable to run '{command}'.")


def install_reference(reference_filename, reference_url):
    if os.path.exists(reference_filename):
        info(f"Found a previously installed reference file: {reference_filename}")
    else:
        info(f"Did not find {reference_filename}; downloading from {reference_url} ...")
        try:
            with urllib.request.urlopen(os.path.join(reference_url, reference_filename)) as resp:
                with tempfile.NamedTemporaryFile(delete=False) as ntf:
                    shutil.copyfileobj(resp, ntf)
                    os.rename(ntf.name, reference_filename)
        except Exception:
            error(f"Unable to download {reference_filename} from {reference_url}.")


def convert_google_json_to_jsonl(json_filename, jsonl_filename):
    """
    Convert Google STT formatted JSON to ASR Engine formatted JSON lines.
    """
    with open(json_filename, 'r', encoding='utf-8') as f:
        response = json.load(f)
        if 'response' in response:
            # When retrieving a LongRunningRecognize "operation", the "response" is nested.
            response = response['response']

    # Assume the first result starts at time zero; see note below.
    start_time = 0.0

    with open(jsonl_filename, 'w', encoding='utf-8') as jsonl_file:
        for result in response['results']:
            reply = {
                'final': True,
                'transcript': result['alternatives'][0]['transcript']
            }

            reply['alternatives'] = [
                {'transcript': a['transcript']} for a in result['alternatives']
            ]

            end_time = float(result['resultEndTime'][:-1])  # strip the "s" suffix

            if 'words' in result['alternatives'][0]:
                words = result['alternatives'][0]['words']
                reply['words'] = [{'word': w['word']} for w in words]
                if words and 'confidence' in words[0]:
                    for i, w in enumerate(words):
                        reply['words'][i]['confidence'] = w['confidence']
                if words and 'startTime' in words[0]:
                    for i, w in enumerate(words):
                        reply['words'][i]['interval'] = [
                            float(w['startTime'][:-1]), float(w['endTime'][:-1])
                        ]

                    # HACK: if enableWordTimestamps was true, then we can infer the start/end times
                    # of the result as start/end times of first/last words in the first alternative.
                    start_time = float(words[0]['startTime'][:-1])
                    end_time = float(words[-1]['endTime'][:-1])

            if 'resultStartTime' in result:
                # This is not officially part of the Google STT REST API response format, but is
                # provided in Mod9's implementation as a compatible superset of functionality.
                # See the note below regarding the implications for NIST SCTK scoring tools.
                start_time = float(result['resultStartTime'][:-1])

            reply['interval'] = [start_time, end_time]

            if 'phrases' in result:
                reply['phrases'] = [
                    {
                        'phrase': p['phrase'],
                        'interval': [float(p['startTime'][:-1]), float(p['endTime'][:-1])],
                        'alternatives': p['alternatives'],
                    } for p in result['phrases']
                ]

            # NOTE: Google does not provide word-level alternatives, but Mod9's extension does.
            if 'words' in result:
                reply['words'] = [
                    {
                        'word': w['word'],
                        'interval': [float(w['startTime'][:-1]), float(w['endTime'][:-1])],
                        'alternatives': w['alternatives'],
                    } for w in result['words']
                ]

            jsonl_file.write(json.dumps(reply) + '\n')

            # Assume the next result's start time is the end time of this result.
            # Google's format does not provide a start time for each result (i.e. utterance).
            # Using the previous result's end time is an unfortunate workaround for the purpose
            # of producing a CTM, particularly if word-level timestamps are not available and are
            # then inferred as uniformly distributed over the utterance-level interval. This is
            # especially problematic for dual-channel telephony in which there are long stretches
            # of silence between speaker turns, during which words are mistimed.
            start_time = end_time


def convert_deepgram_json_to_jsonl(json_filename, jsonl_filename):
    """
    Convert Deepgram formatted JSON to ASR Engine formatted JSON lines.
    """
    response = json.load(open(json_filename, 'r', encoding='utf-8'))
    with open(jsonl_filename, 'w', encoding='utf-8') as jsonl_file:
        result = response['results']['channels'][0]
        reply = {'final': True}
        reply['transcript'] = result['alternatives'][0]['transcript']
        reply['words'] = [{'word': w['word'], 'interval': [w['start'], w['end']]}
                          for w in result['alternatives'][0]['words']]
        jsonl_file.write(json.dumps(reply) + '\n')


def convert_ibm_json_to_jsonl(json_filename, jsonl_filename):
    """
    Convert IBM Watson formatted JSON to ASR Engine formatted JSON lines.
    """
    response = json.load(open(json_filename, 'r', encoding='utf-8'))
    with open(jsonl_filename, 'w', encoding='utf-8') as jsonl_file:
        for result in response['results'][0]['results']:  # awkwardly nested
            reply = {'final': True}
            reply['transcript'] = result['alternatives'][0]['transcript'].strip()
            reply['words'] = [{'word': w, 'interval': [start, end]}
                              for w, start, end, in result['alternatives'][0]['timestamps']]
            jsonl_file.write(json.dumps(reply) + '\n')


def convert_msft_json_to_jsonl(json_filename, jsonl_filename):
    """
    Convert Microsoft formatted JSON to ASR Engine formatted JSON lines.
    """
    def convert_interval(offset_ticks, duration_ticks):
        start_time = offset_ticks / 10_000_000
        return [round(start_time, 2), round(start_time + duration_ticks / 10_000_000, 2)]

    response = json.load(open(json_filename, 'r', encoding='utf-8'))
    with open(jsonl_filename, 'w', encoding='utf-8') as jsonl_file:
        for result in response['recognizedPhrases']:
            reply = {'final': True}
            reply['transcript'] = result['nBest'][0]['lexical']

            reply['alternatives'] = [
                {'transcript': nbest['lexical']} for nbest in result['nBest']
            ]

            reply['interval'] = convert_interval(result['offsetInTicks'], result['durationInTicks'])
            reply['words'] = [{
                'word': w['word'],
                'interval': convert_interval(w['offsetInTicks'], w['durationInTicks'])
            } for w in result['nBest'][0]['words']]
            jsonl_file.write(json.dumps(reply) + '\n')


def convert_amazon_json_to_jsonl(json_filename, jsonl_filename):
    """
    Convert Amazon Transcribe formatted JSON to ASR Engine formatted JSON lines.
    This only handles results with 1-best transcripts, not transcript-level alternatives.
    """
    response = json.load(open(json_filename, 'r', encoding='utf-8'))
    with open(jsonl_filename, 'w', encoding='utf-8') as jsonl_file:
        result = response['results']
        reply = {
            'final': True,
            'transcript': result['transcripts'][0]['transcript']
        }
        reply['words'] = []
        for item in result['items']:
            if item['type'] == 'pronunciation':  # i.e. a "word"
                word = item['alternatives'][0]['content']
                reply['words'].append({
                    'confidence': item['alternatives'][0]['confidence'],
                    'interval': [float(item['start_time']), float(item['end_time'])],
                    'word': word})
        jsonl_file.write(json.dumps(reply) + '\n')


def convert_revai_json_to_jsonl(json_filename, jsonl_filename):
    """
    Convert Rev.ai formatted JSON to ASR Engine formatted JSON lines.
    """
    def multiword(word):
        """NIST SCTK does not like a "word" with spaces in it, so rewrite it with underscores."""
        return word.replace(' ', '_')

    response = json.load(open(json_filename, 'r', encoding='utf-8'))
    with open(jsonl_filename, 'w', encoding='utf-8') as jsonl_file:
        for segment in response['monologues']:
            reply = {'final': True}
            reply['transcript'] = ' '.join([multiword(x['value']) for x in segment['elements']
                                            if x['type'] == 'text'])
            reply['words'] = [{'word': multiword(w['value']), 'interval': [w['ts'], w['end_ts']]}
                              for w in segment['elements'] if w['type'] == 'text']
            jsonl_file.write(json.dumps(reply) + '\n')


def convert_jsonl_to_ctm(
        jsonl_filename, ctm_filename,
        key1, key2,
        alternatives=None,
        alternatives_max=None,
        omitted_words=[],
        split_initialisms=True,
):
    """
    Convert ASR Engine output (JSON lines) to a NIST-formatted CTM file.

    The keys are used to lookup the corresponding reference in the STM file.
    The first key is traditionally the basename of a 2-channel audio file.
    The second key is traditionally a channel identifier, e.g. "A" or "B".

    If the alternatives argument is specified (as "phrase", "transcript", or
    "word"), the resulting file will make use of the CTM format's ability to
    represent alternative hypotheses. This somewhat lesser-known feature of
    the NIST SCTK software is unfortunately a bit buggy, though, and will
    require further post-processing to be handled correctly after filtering
    with the csrfilt.sh tool. If the alternatives_max argument is specified,
    this will limit the number of alternatives considered.

    The list of omitted_words is used as a simple filter to remove words in
    ASR output that is not well matched to the reference conventions. For
    example, it is never helpful to transcribe non-speech noises, since the
    reference does not include these and they will become insertion errors.

    The split_initialisms argument is used to match a convention in the
    reference transcription: for example, Y M C A should be 4 single-letter
    words rather than one single word such as y._m._c._a.
    """
    with open(jsonl_filename, 'r') as jsonl_file, open(ctm_filename, 'w') as ctm_file:
        def write_ctm(key1, key2, begin_time, duration, word, confidence=None):
            # Some ASR systems have a convention of writing initialisms as "a._b._c."
            # But the Switchboard reference treats this as separate words "a", "b", "c".
            if split_initialisms and ('_' in word or (len(word) == 2 and word.endswith('.'))):
                words = word.split('_')
                duration /= len(words)
                for w in words:
                    if len(w) == 2 and w.endswith('.'):
                        w = w[0]
                    write_ctm(key1, key2, begin_time, duration, w, confidence)
                    begin_time += duration
            else:
                if word == '':
                    word = '@'  # Special NIST SCTK convention to indicate null word.
                ctm_file.write(f"{key1} {key2} {begin_time:0.3f} {duration:0.3f} {word}")
                if confidence:
                    ctm_file.write(f" {confidence}")
                ctm_file.write('\n')

        def write_words(key1, key2, begin_time, duration, words):
            duration /= len(words)
            for word in words:
                write_ctm(key1, key2, begin_time, duration, word)
                begin_time += duration

        for line in jsonl_file:
            reply = json.loads(line)
            if not reply.get('final'):
                # Skip replies that do not represent a finalized transcript result.
                continue
            if alternatives is None:
                if 'words' in reply:
                    for word_obj in reply['words']:
                        if 'interval' not in word_obj:
                            # TODO: we could try to infer these from the transcript-level interval.
                            error('Word-level intervals are expected.')
                        begin_time = word_obj['interval'][0]
                        duration = word_obj['interval'][1] - begin_time

                        word = word_obj['word']
                        if not word:
                            # This happens with the legacy Remeeting ASR API (IBM-compatible format)
                            # to represent situations where a top-ranked word alternative is silence.
                            continue

                        if word in omitted_words:
                            continue

                        confidence = word_obj.get('confidence')

                        write_ctm(key1, key2, begin_time, duration, word, confidence)
                else:
                    if 'interval' not in reply:
                        error('Transcript-level intervals are expected.')
                    begin_time = reply['interval'][0]
                    duration = reply['interval'][1] - begin_time
                    words = [w for w in reply['transcript'].split() if w not in omitted_words]
                    if words:
                        write_words(key1, key2, begin_time, duration, words)
            else:
                if alternatives == 'phrase':
                    if 'phrases' not in reply:
                        error('Phrase-level alternatives are expected.')
                    for phrase_obj in reply['phrases']:
                        if 'interval' not in phrase_obj:
                            error('Phrase-level intervals are expected.')
                        begin_time = phrase_obj['interval'][0]
                        duration = phrase_obj['interval'][1] - begin_time

                        if 'alternatives' not in phrase_obj:
                            error('Phrase-level alternatives are expected.')
                        alts = phrase_obj['alternatives'][:alternatives_max]

                        for alt in alts:
                            words = []
                            for w in alt['phrase'].split():
                                if w in omitted_words:
                                    w = '@'  # Special NIST SCTK convention for optional word.
                                words.append(w)
                            alt['phrase'] = ' '.join(words)
                            if alt['phrase'] == '':
                                alt['phrase'] = '@'

                        ctm_file.write(f"{key1} {key2} * * <ALT_BEGIN>\n")
                        for alt in alts[:-1]:
                            words = alt['phrase'].split()
                            write_words(key1, key2, begin_time, duration, words)
                            ctm_file.write(f"{key1} {key2} * * <ALT>\n")
                        words = alts[-1]['phrase'].split()
                        write_words(key1, key2, begin_time, duration, words)
                        ctm_file.write(f"{key1} {key2} * * <ALT_END>\n")
                elif alternatives == 'transcript':
                    if 'interval' not in reply:
                        error('Transcript-level intervals are expected.')
                    begin_time = reply['interval'][0]
                    duration = reply['interval'][1] - begin_time

                    if 'alternatives' not in reply:
                        error('Transcript-level alternatives are expected.')
                    alts = reply['alternatives'][:alternatives_max]

                    for alt in alts:
                        words = []
                        for w in alt['transcript'].split():
                            if w in omitted_words:
                                w = '@'  # Special NIST SCTK convention for optional word.
                            words.append(w)
                        alt['transcript'] = ' '.join(words)
                        if alt['transcript'] == '':
                            alt['transcript'] = '@'

                    ctm_file.write(f"{key1} {key2} * * <ALT_BEGIN>\n")
                    for alt in alts[:-1]:
                        words = alt['transcript'].split()
                        write_words(key1, key2, begin_time, duration, words)
                        ctm_file.write(f"{key1} {key2} * * <ALT>\n")
                    words = alts[-1]['transcript'].split()
                    write_words(key1, key2, begin_time, duration, words)
                    ctm_file.write(f"{key1} {key2} * * <ALT_END>\n")
                elif alternatives == 'word':
                    if 'words' not in reply:
                        error('Word-level alternatives are expected.')
                    for word_obj in reply['words']:
                        if 'interval' not in word_obj:
                            error('Word-level intervals are expected.')
                        begin_time = word_obj['interval'][0]
                        duration = word_obj['interval'][1] - begin_time

                        if 'alternatives' not in word_obj:
                            error('Word-level alternatives are expected.')
                        alts = word_obj['alternatives'][:alternatives_max]

                        for alt in alts:
                            if alt['word'] == '' or alt['word'] in omitted_words:
                                alt['word'] = '@'  # Special NIST SCTK convention for optional word.

                        # NOTE: in theory we should be able to use confidence for word alternatives.
                        # NIST BUG: cannot apply GLM to CTM with alternatives with confidence.
                        # TODO: report this to Jon Fiscus?
                        confidence = None

                        ctm_file.write(f"{key1} {key2} * * <ALT_BEGIN>\n")
                        for alt in alts[:-1]:
                            word = alt['word']
                            write_ctm(key1, key2, begin_time, duration, word, confidence)
                            ctm_file.write(f"{key1} {key2} * * <ALT>\n")
                        word = alts[-1]['word']
                        write_ctm(key1, key2, begin_time, duration, word, confidence)
                        ctm_file.write(f"{key1} {key2} * * <ALT_END>\n")
                else:
                    error('Unexpected alternatives level.')


def expand_alt_section(alt_section, max_expansions=None):
    """Helper function for refilter_ctm."""
    spans = [['']]
    alt_separator_line = ''
    for line in alt_section.strip().split('\n'):
        if '<ALT_BEGIN>' in line:
            spans.append([''])
        elif '<ALT_END>' in line:
            alt_separator_line = line.replace('<ALT_END>', '<ALT>').strip() + '\n'
            spans.append([''])
        elif '<ALT>' in line:
            spans[-1].append('')
        else:
            spans[-1][-1] += line + '\n'
    alts = list(itertools.product(*spans))
    if max_expansions and len(alts) > max_expansions:
        alts = alts[:max_expansions]
    expanded_alt_sections = [''.join(s) for s in alts]
    return alt_separator_line.join(expanded_alt_sections)


def unique_alt_sections(alt_sections):
    """Optimize downstream processing by removing duplicated (and empty) alt sections."""
    # NOTE: retain the order of alternatives so that tiebreakers with the same number of errors will
    # favor alternatives which occur first (i.e. are more likely in an N-best, or reflect the
    # surface form in a GLM expansion). If we wanted to game the scoring metric, we would sort these
    # in decreasing order of length so as to maximize the number of correct words.
    uniq_alt_sections = {}  # Python dicts iterate keys in insertion order.
    for alt_section in alt_sections:
        curr_alt_section = ''
        for line in alt_section.strip().split('\n'):
            key1, key2, begin_time, duration, word = line.split(None, 4)
            if word == '<ALT>':
                # This was an expanded alt section.
                uniq_alt_sections[curr_alt_section] = None
                curr_alt_section = ''
            else:
                curr_alt_section += line+'\n'
        uniq_alt_sections[curr_alt_section] = None
    return uniq_alt_sections.keys()


def refilter_ctm(ctm_in_filename, ctm_out_filename,
                 max_expansions=None, omit_backchannels=False, omit_hesitations=False):
    """
    Post-process a CTM file that was produced by running the NIST SCTK tool
    csrfilt.sh on an original input CTM including hypothesis alternations.

    These become doubly-nested, e.g.:
    sw_4390 A * * <ALT_BEGIN>
    ...1...
    sw_4390 A * * <ALT_BEGIN>
    ...2...
    sw_4390 A * * <ALT>
    ...3...
    sw_4390 A * * <ALT_END>
    ...4...
    sw_4390 A * * <ALT_END>

    The solution is to expand these into singly-nested alternations, as
    the Cartesian product of the alternated sections, e.g.:
    sw_4390 A * * <ALT_BEGIN>
    ...1...
    ...2...
    ...4...
    sw_4390 A * * <ALT>
    ...1...
    ...3...
    ...4...
    sw_4390 A * * <ALT_END>

    Unfortunately, this can create rather long alternations, particularly
    for transcript-level N-best alternatives. This can cause problems for
    the downstream NIST SCTK sclite software which may segfault due to low
    precision of indices (see https://github.com/usnistgov/SCTK/pull/34).
    To mitigate this, set max_expansions. This hack shouldn't be needed for
    word-level or phrase-level alternatives, or if the bugfix PR is merged.
    """
    in_alt = False
    in_nested_alt = False
    alt_sections = None
    alt_begin_line = None
    alt_end_line = None
    alt_separator_line = None

    with open(ctm_in_filename, 'r') as ctm_in_file, open(ctm_out_filename, 'w') as ctm_out_file:
        for line in ctm_in_file:
            if omit_backchannels:
                line = line.replace('%BCACK', '@')
                line = line.replace('%BCNACK', '@')
            if omit_hesitations:
                line = line.replace('%HESITATION', '@')
            if '<ALT_BEGIN>' in line:
                alt_begin_line = line
                if in_nested_alt:
                    error('Unexpected doubly-nested alternative.')
                elif in_alt:
                    in_nested_alt = True
                    alt_sections[-1] += line
                else:
                    in_alt = True
                    alt_sections = ['']
                continue
            elif '<ALT_END>' in line:
                alt_end_line = line
                alt_separator_line = alt_end_line.replace('<ALT_END>', '<ALT>').strip() + '\n'
                if in_nested_alt:
                    in_nested_alt = False
                    alt_sections[-1] += line
                elif in_alt:
                    in_alt = False
                    ctm_out_file.write(alt_begin_line)
                    alt_sections = [expand_alt_section(a, max_expansions) for a in alt_sections]
                    alt_sections = unique_alt_sections(alt_sections)
                    ctm_out_file.write(alt_separator_line.join(alt_sections))
                    ctm_out_file.write(alt_end_line)
                else:
                    error('Unexpected end of alternative.')
                continue
            elif '<ALT>' in line:
                if in_nested_alt:
                    alt_sections[-1] += line
                elif in_alt:
                    alt_sections.append('')
                continue

            if in_alt or in_nested_alt:
                alt_sections[-1] += line
            else:
                ctm_out_file.write(line)


def refilter_stm(stm_in_filename, stm_out_filename, optional_backchannels=False):
    """
    There is a bug in NIST SCTK w.r.t. placement of parentheses around curly braces and backslashes,
    i.e. when representing an optionally deleteable alternation.
    Also allow for optionally deletable backchannel mappings (i.e. %BCACK and %BCNACK in the GLM).
    """
    if optional_backchannels:
        optional_backchannels_filter = '| sed "s,%BCACK,(%BCACK),g; s,%BCNACK,(%BCNACK),g"'
    else:
        optional_backchannels_filter = ''
    run('sed "s,(/),/,g; s,({,{(,g; s,}),)},g"'
        f" < {stm_in_filename} {optional_backchannels_filter} > {stm_out_filename}")


def convert_stm_1seg(old_stm_filename, new_stm_filename):
    """
    Convert the STM into a single long segment, which can in some cases
    slightly improve the sclite alignment algorithm or minimize WER, e.g.
    in situations where a pair of insertion/deletion errors are across a
    segment boundary and could be consolidated as a single substitution.
    This also helps in situations where the reference segmentation is
    not used by the ASR system (i.e. a more fair evaluation than typical
    for academic research scenarios); slight differences in word-level
    timing may exist between the ASR system and the reference segments.
    """
    curr_key = None  # This function assumes the STM is speaker-specific.
    transcripts = []
    for line in open(old_stm_filename):
        if line.startswith(';;'):  # comment lines
            continue
        key1, key2, spkid, begin_time, end_time, labels, transcript = line.strip().split(None, 6)
        key = (key1, key2, spkid, labels)
        if curr_key and curr_key != key:
            # TODO: it's not that hard to generalize this to a multi-speaker STM file.
            error('Cannot collapse segmentation of a reference STM with multiple speakers.')
        else:
            curr_key = key
            transcripts.append(transcript)
    with open(new_stm_filename, 'w') as f:
        key1, key2, spkid, labels = curr_key
        f.write(f"{key1} {key2} {spkid} 0 999999999 {labels} {' '.join(transcripts)}\n")


def nist_report(switchboard_speaker_id, show_errors=False):
    """
    Read the .ctm, .pra, and .raw files produced by the NIST SCTK sclite tool.
    Produce a less verbose and somewhat more informative report than SCTK would.
    """
    if show_errors:
        errors = []
        print('\nSegments where reference and hypothesis are mis-aligned:')
        for line in open(f"{switchboard_speaker_id}.nist.pra"):
            if line.startswith('Scores:'):
                fields = line.strip().split()
                n_cor = int(fields[5])
                n_sub = int(fields[6])
                n_del = int(fields[7])
                n_ins = int(fields[8])
                n_err = n_sub + n_del + n_ins
                n_ref = n_err + n_cor
                wer = n_err / n_ref * 100
            elif line.startswith('REF:'):
                ref_line = line
            elif line.startswith('HYP:'):
                if n_err > 0:
                    errors.append((n_err, wer, ref_line, line))
        errors.sort(key=lambda x: x[0])  # or x[1] to sort by WER.
        for n_err, wer, ref_line, hyp_line in errors:
            print(ref_line, end='')
            print(hyp_line, end='')
        print()

    # Report the size of the CTM file, compressed.
    run(f"gzip -c {switchboard_speaker_id}.ctm > {switchboard_speaker_id}.ctm.gz")
    ctm_gz_kb = os.path.getsize(f"{switchboard_speaker_id}.ctm.gz") // 1000
    info(f"Size of *.ctm.gz:  {ctm_gz_kb} KB")

    # Report some statistics about the distribution of alternatives.
    depths = []
    widths = []
    curr_depth = 0
    curr_width = 0
    for line in open(f"{switchboard_speaker_id}.ctm"):
        if '<ALT_BEGIN>' in line:
            curr_depth += 1
        elif '<ALT>' in line:
            curr_depth += 1
            widths.append(curr_width)
            curr_width = 0
        elif '<ALT_END>' in line:
            depths.append(curr_depth)
            widths.append(curr_width)
            curr_depth = curr_width = 0
        else:
            curr_width += 1
    if depths and widths:
        N_max = max(depths)   # When requesting N-best, in practice the depth might be less than N.
        N_dec = sorted(depths)[-len(depths)//10]  # The top decile is more informative than the max.
        N_med = sorted(depths)[len(depths)//2]    # The median is more informative than the mean.
        S = len(depths)       # Number of spans/segments.
        W_avg = sum(widths) / len(widths)  # This relates to the average-case performance.
        W_max = max(widths)   # This relates to the worst-case performance, e.g. for sclite scoring.
        info('Alternatives:      '
             f"N_max={N_max} N_dec={N_dec} N_med={N_med} S={S} W_avg={W_avg:0.1f} W_max={W_max}")

    sum_line = run(f"grep Sum {switchboard_speaker_id}.nist.raw").stdout.decode()
    fields = sum_line.strip().split()
    n_ref = int(fields[4])
    n_cor = int(fields[6])
    n_sub = int(fields[7])
    n_del = int(fields[8])
    n_ins = int(fields[9])
    info(f"Sum (#C #S #D #I): {n_cor} {n_sub} {n_del} {n_ins}")

    # These are not standard metrics, but can be rather enlightening.
    precision = n_cor / (n_cor + n_sub + n_ins)
    recall = n_cor / (n_cor + n_sub + n_del)
    info(f"Precision/Recall:  {precision:0.3f} / {recall:0.3f}")

    # This is the standard ASR metric, which has a lot of drawbacks.
    wer = (n_sub + n_del + n_ins) / n_ref * 100
    print(f"{f'[{switchboard_speaker_id}]':<11} WER:  {f'{wer:.2f}':>6}%", flush=True)


def main():
    try:
        main_helper()
        return 0
    except KeyboardInterrupt:
        return 130


def main_helper():
    parser = argparse.ArgumentParser(description=DESCRIPTION,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        'switchboard_speaker_id',
        metavar='SW_SPEAKER_ID',
        nargs='?',
        help='Switchboard speaker identifier, including channel, such as "sw_4390_A" for example.',
        choices=SWITCHBOARD_SPEAKER_IDS,
    )
    parser.add_argument(
        '--alternatives',
        metavar='LEVEL',
        help='Word-, phrase-, or transcript-level alternatives used for oracle scoring.',
        choices=ALTERNATIVES_LEVELS,
    )
    parser.add_argument(
        '--alternatives-n',
        metavar='INT',
        type=int,
        help='Limit the number of alternatives (i.e. N-best) to be considered.',
    )
    parser.add_argument(
        '--max-expansions',
        metavar='INT',
        type=int,
        help='Mitigate SCTK bug by limiting nested alternative expansions, e.g. transcript-level.',
        default=0,
    )
    parser.add_argument(
        '--omit-hesitations',
        action='store_true',
        help='CTM should not include hesitations (optionally deletable in the STM).',
    )
    parser.add_argument(
        '--omit-backchannels',
        action='store_true',
        help='CTM should not include backchannels (cf. --optional-backchannels).',
    )
    parser.add_argument(
        '--omitted-words',
        metavar='LIST',
        default=','.join(OMITTED_WORDS),
        help='Comma-separated list of non-speech words to omit from the CTM.',
    )
    parser.add_argument(
        '--optional-backchannels',
        action='store_true',
        help='Backchannels can be optionally deleted in the STM.',
    )
    parser.add_argument(
        '--reference-glm',
        metavar='FILE',
        help='Name of NIST-formatted Global Language Mapping file.',
        default=REFERENCE_GLM,
    )
    parser.add_argument(
        '--reference-stm',
        metavar='FILE',
        help='Name of NIST-formatted Segment Time Mark file.',
        default=REFERENCE_STM,
    )
    parser.add_argument(
        '--reference-url',
        metavar='URL',
        help='Whence missing reference files may be downloaded.',
        default=REFERENCE_URL,
    )
    parser.add_argument(
        '--sclite',
        action='store_true',
        help='Run NIST sclite tool over all files.',
    )
    parser.add_argument(
        '--show-errors',
        action='store_true',
        help='Compare segment-level mis-alignment.',
    )
    parser.add_argument(
        '--single-segment-stm',
        action='store_true',
        help='Convert reference to a long segment.',
    )
    parser.add_argument(
        '--sum-overall',
        action='store_true',
        help='Report aggregate scores over corpus.',
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Print more verbose information.',
    )
    parser.add_argument(
        '--workdir',
        metavar='DIRECTORY',
        help='Where files will be saved for caching or debugging.',
        default=WORKDIR,
    )
    args = parser.parse_args()

    global VERBOSE
    VERBOSE = args.verbose

    # This tool will report on stdout, but these saved files might be helpful for debugging.
    os.makedirs(args.workdir, exist_ok=True)
    os.chdir(args.workdir)

    if args.sclite:
        run('cat *.ctm.refilt > overall.ctm.refilt_')
        run('cat *.stm.refilt > overall.stm.refilt_')
        run('sclite -h overall.ctm.refilt_ ctm -r overall.stm.refilt_ stm -n overall.nist '
            f"{SCLITE_OPTS} -o sum",
            capture_output=False)
        run("cat overall.nist.sys", capture_output=False)
        exit(0)
    elif args.sum_overall:
        run("""cat sw_*.nist.raw | grep Sum | awk '
{Snt+=$4;Wrd+=$5;Corr+=$7;Sub+=$8;Del+=$9;Ins+=$10} END \
{print" | Sum | "Snt"  "Wrd" | "Corr" "Sub" "Del" "Ins} \
' > overall.nist.raw""")
        run('cat *.ctm > overall.ctm')
        nist_report('overall')
        exit(0)
    elif not args.switchboard_speaker_id:
        error('Missing required arugment for Switchboard speaker ID.')

    info(f"Results will be saved in the work directory: {args.workdir}")

    # Check installed dependencies.
    install_reference(args.reference_glm, args.reference_url)
    install_reference(args.reference_stm, args.reference_url)
    for sctk_tool in SCTK_TOOLS:
        if not shutil.which(sctk_tool):
            error(f"Could not find {sctk_tool}; ensure that NIST SCTK is installed.")

    # Parse the Switchboard speaker identifier, renamed as `spkid` for convenience.
    spkid = args.switchboard_speaker_id
    filename_id, channel_id = spkid.rsplit('_', 1)

    lines = []
    info("Read Engine replies or other vendors' JSON on stdin: ...", flush=True)
    for line in sys.stdin:
        lines.append(line)

    if lines and lines[0].startswith('{"metadata"'):
        info(f"Save JSON (Deepgram formatted) from stdin: {spkid}.json")
        with open(spkid+'.json', 'w') as f:
            for line in lines:
                f.write(line)
        info(f"Convert JSON to Engine formatted JSON lines: {spkid}.jsonl")
        convert_deepgram_json_to_jsonl(spkid+'.json', spkid+'.jsonl')
    elif lines and lines[0] == '{\n' and ('"name"' in lines[1] or '"results"' in lines[1]):
        info(f"Save JSON (Google STT formatted) from stdin: {spkid}.json")
        with open(spkid+'.json', 'w') as f:
            for line in lines:
                f.write(line)
        info(f"Convert JSON to Engine formatted JSON lines: {spkid}.jsonl")
        convert_google_json_to_jsonl(spkid+'.json', spkid+'.jsonl')
    elif lines and lines[0] == '{\n' and '"created"' in lines[1]:
        info(f"Save JSON (IBM Watson formatted) from stdin: {spkid}.json")
        with open(spkid+'.json', 'w') as f:
            for line in lines:
                f.write(line)
        info(f"Convert JSON to Engine formatted JSON lines: {spkid}.jsonl")
        convert_ibm_json_to_jsonl(spkid+'.json', spkid+'.jsonl')
    elif lines and lines[0] == '{\n' and '"source"' in lines[1]:
        info(f"Save JSON (Microsoft formatted) from stdin: {spkid}.json")
        with open(spkid+'.json', 'w') as f:
            for line in lines:
                f.write(line)
        info(f"Convert JSON to Engine formatted JSON lines: {spkid}.jsonl")
        convert_msft_json_to_jsonl(spkid+'.json', spkid+'.jsonl')
    elif lines and lines[0].startswith('{"jobName"'):
        info(f"Save JSON (as Amazon Transcribe) from stdin: {spkid}.json")
        with open(spkid+'.json', 'w') as f:
            for line in lines:
                f.write(line)
        info(f"Convert JSON to Engine formatted JSON lines: {spkid}.jsonl")
        convert_amazon_json_to_jsonl(spkid+'.json', spkid+'.jsonl')
    elif lines and lines[0].startswith('{"monologues"'):
        info(f"Save JSON (as Rev.ai) from stdin: {spkid}.json")
        with open(spkid+'.json', 'w') as f:
            for line in lines:
                f.write(line)
        info(f"Convert JSON to Engine formatted JSON lines: {spkid}.jsonl")
        convert_revai_json_to_jsonl(spkid+'.json', spkid+'.jsonl')
    else:
        info(f"Save JSON lines (Engine replies) from stdin: {spkid}.jsonl")
        with open(spkid+'.jsonl', 'w') as f:
            for line in lines:
                f.write(line)

    info(f"Convert to NIST-formatted hypothesis format: {spkid}.ctm")
    convert_jsonl_to_ctm(spkid+'.jsonl', spkid+'.ctm',
                         filename_id, channel_id,
                         alternatives=args.alternatives if args.alternatives_n != 0 else None,
                         alternatives_max=args.alternatives_n,
                         omitted_words=args.omitted_words.split(','))

    info(f"Apply global mappings to make filtered file: {spkid}.ctm.filt")
    run(f"csrfilt.sh -i ctm -t hyp -dh {args.reference_glm} < {spkid}.ctm > {spkid}.ctm.filt")

    info(f"Fix SCTK bug, omit hesitations/backchannels: {spkid}.ctm.refilt")
    refilter_ctm(spkid+'.ctm.filt', spkid+'.ctm.refilt',
                 args.max_expansions, args.omit_backchannels, args.omit_hesitations)

    info(f"Extract reference transcription for speaker: {spkid}.stm")
    run(f"grep '^{filename_id} {channel_id} {spkid}' < {args.reference_stm} > {spkid}.stm")

    info(f"Apply global mappings to make filtered file: {spkid}.stm.filt")
    run(f"csrfilt.sh -i stm -t ref -dh {args.reference_glm} < {spkid}.stm > {spkid}.stm.filt")

    info(f"Fix SCTK bugs (optional alts, backchannels): {spkid}.stm.refilt")
    refilter_stm(spkid+'.stm.filt', spkid+'.stm.refilt', args.optional_backchannels)

    stm = f"{spkid}.stm.refilt"
    if args.single_segment_stm:
        info(f"Convert the reference into one long segment: {spkid}.stm.refilt.1seg")
        convert_stm_1seg(stm, f"{spkid}.stm.refilt.1seg")
        stm = f"{spkid}.stm.refilt.1seg"

    info(f"Run the NIST SCLITE tool to produce reports: {spkid}.nist.*", flush=True)
    run(f"sclite -h {spkid}.ctm.refilt ctm -r {stm} stm -n {spkid}.nist {SCLITE_OPTS}"
        " -o pralign rsum")

    nist_report(spkid, args.show_errors)


if __name__ == '__main__':
    main()
