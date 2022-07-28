#!/usr/bin/env python3

import argparse
import json
import logging
import sys
import uuid

from elasticsearch import Elasticsearch
from elasticsearch import RequestsHttpConnection
from elasticsearch.exceptions import RequestError

from mod9.reformat import utils

ES_HOSTS = ['localhost']
ES_PORT = 9200
ES_INDEX_CREATION_TIMEOUT_SECONDS = 120

TEMPLATE = {
    'mappings': {
        'date_detection': False,
        'properties': {
            'title': {
                'type': 'text',
            },
            'tracks': {
                'type': 'nested',
                'properties': {
                    'asr': {
                        'type': 'text',
                        'analyzer': 'analyzer-cats-index',
                        'search_analyzer': 'analyzer-query-simple',
                    },
                    'speaker': {
                        'type': 'keyword',
                        'ignore_above': 256,
                    },
                },
            },
        },
    },
    'settings': {
        'index': {
            'similarity': {
                'kaldi': {
                    'type': 'kaldi',
                    'mode': 'combined',
                },
            },
        },
        'analysis': {
            'char_filter': {
                'acronym_fixer': {
                    'type': 'pattern_replace',
                    'pattern': '(?<![a-zA-Z])([a-zA-Z])\\. (?=[a-zA-Z]\\.)',
                    'replacement': '$1.',
                },
            },
            'filter': {
                'payload_poslen': {
                    'type': 'payload-position-length',
                },
                'poslen_order': {
                    'type': 'position-length-order',
                },
                'english_stemmer': {
                    'type': 'stemmer',
                    'name': 'minimal_english',
                },
                'unique_stem': {
                    'type': 'unique',
                    'only_on_same_position': True,
                },
            },
            'analyzer': {
                'analyzer-cats-index': {
                    'type': 'custom',
                    'tokenizer': 'tokenizer-cats',
                    'filter': ['lowercase', 'poslen_order', 'payload_poslen'],
                },
                'analyzer-query': {
                    'type': 'custom',
                    'tokenizer': 'standard',
                    'filter': ['lowercase', 'keyword_repeat', 'english_stemmer', 'unique_stem'],
                },
                'analyzer-query-simple': {
                    'type': 'custom',
                    'tokenizer': 'whitespace',
                    'filter': ['lowercase'],
                },
            },
        },
    },
}


class ESWrapper:
    def __init__(
        self,
        hosts=ES_HOSTS,
        port=ES_PORT,
        use_ssl=False,
        verify_certs=False,
        api_key=None,
    ):
        self.logger = logging.getLogger(__name__)
        self.template = TEMPLATE

        # Elastic cloud cluster uses an API key stored on S3 to authenticate requests.
        if not api_key:
            api_key = {'id': 'foo', 'api_key': 'bar'}
        try:
            self.es = Elasticsearch(
                hosts,
                api_key=(api_key['id'], api_key['api_key']),
                port=port,
                use_ssl=use_ssl,
                verify_certs=verify_certs,
                connection_class=RequestsHttpConnection,
            )
        except Exception as e:
            self.logger.error("Failed to connect to Elasticsearch: %s", str(e))
            raise e

    def exists(self, index, doc_id=None):
        """Check if index exists. Checks if document exists if doc_id supplied."""
        if doc_id:
            return self.es.exists(index=index, id=doc_id)
        return self.es.indices.exists(index=index)

    def create(self, index):
        """Create index if necessary."""
        ic = self.es.indices
        if self.exists(index):
            return

        try:
            # Flush to give time for potentially time-consuming index creation.
            ic.create(
                index=index,
                body=self.template,
                request_timeout=ES_INDEX_CREATION_TIMEOUT_SECONDS,
            )
            return ic.flush(
                index=index,
                wait_if_ongoing=True,
                request_timeout=ES_INDEX_CREATION_TIMEOUT_SECONDS,
            )
        except RequestError as e:
            if getattr(e, 'error') == 'resource_already_exists_exception':
                return
            else:
                self.logger.error('Unexpected exception on index creation')
                raise e

    def index(self, index, doc_id, item):
        """Index resource item as document"""
        if not item:
            raise ValueError('You need to set an item to be indexed.')

        try:
            self.create(index)
            self.logger.info("Indexing %s/%s", index, doc_id)
            return self.es.index(index=index, id=doc_id, body=item)
        except Exception as e:
            self.logger.error("elasticsearch.index.index: %s", repr(e))
            raise e


def index_lines(json_lines_by_speaker, es, index, title):
    """Index speaker-separated reply messages passed in as JSON-formatted lines."""
    tracks = []
    for speaker, json_lines in json_lines_by_speaker.items():
        concats = []
        for json_line in json_lines:
            engine_reply = json.loads(json_line)
            if engine_reply.get('final'):
                concats += engine_reply.get('phrases', '')
        if len(concats) > 0:
            tracks.append({'asr': json.dumps(concats), 'speaker': speaker})

    if len(tracks) > 0:
        document = {'title': title, 'tracks': tracks}
        index_id = str(uuid.uuid4())
        es.index(index, index_id, document)
        logging.info("Indexed audio in index %s with ID %s.", index, index_id)
    else:
        logging.info('Not indexing; no "phrases" fields found.')


def get_speaker_names(speakers_string='', number_speaker_tracks=1):
    speakers = [] if speakers_string == '' else speakers_string.split(sep=',')
    if len(speakers) != number_speaker_tracks:
        if number_speaker_tracks == 1:
            speakers = ['Sole Speaker']
        else:
            speakers = [f"Speaker {i}" for i in range(number_speaker_tracks)]
        if speakers_string != '':
            logging.warning(
                "Expected %d speakers input, instead got '%s'. Falling back to %s",
                number_speaker_tracks,
                speakers_string,
                speakers,
            )
    return speakers


def search_index(terms, es, index_name, speakers=None, body=None):
    """Search for given terms, filtering by speakers if given."""
    if body is None:
        phrase_query = {
            'match_payload_phrase': {
                'tracks.asr': {
                    'query': terms,
                },
            },
        }
        and_query = {
            'match': {
                'tracks.asr': {
                    'operator': 'AND',
                    'query': terms,
                },
            },
        }
        or_query = {
            'match': {
                'tracks.asr': {
                    'operator': 'OR',
                    'query': terms,
                },
            },
        }
        if len(terms.split(sep=' ')) == 1:
            if not speakers:
                inner_query = phrase_query
            else:
                inner_query = {'bool': {'must': phrase_query}}
        else:
            inner_query = {
                'bool': {
                    'should': [
                        phrase_query,
                        and_query,
                        or_query,
                    ],
                },
            }
        if speakers:
            speakers_filter = {
                'filter': {'terms': {'tracks.speaker': [speaker for speaker in speakers]}},
                'minimum_should_match': 2,
            }
            inner_query['bool'].update(speakers_filter)
        nested_query = {
            'nested': {
                'path': 'tracks',
                'query': inner_query,
            },
        }
        body = {
            '_source': [
                'title',
                'tracks.speaker',
            ],
            'highlight': {
                'fields': {
                    'tracks.speaker': {},
                    'tracks.asr': {
                        'type': 'highlighter-cats',
                        'pre_tags': ['<em>'],
                        'post_tags': ['</em>'],
                    },
                },
            },
            'query': nested_query,
        }
    logging.info("POST request with query:\n%s", json.dumps(body, indent=2))
    return es.es.search(body=body, index=index_name)


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        'es_host',
        metavar='ES_HOST',
        help='Elasticsearch server host.',
        default='http://localhost',
    )
    parser.add_argument(
        'es_port',
        metavar='ES_PORT',
        help='Elasticsearch server port.',
        default='9200',
    )
    parser.add_argument(
        '--speakers',
        help='Name(s) of speaker(s) to add to index, or to filter for in search.',
        default='',
    )
    parser.add_argument(
        '--index-name',
        help='Name of index to add results to.',
        default='mod9-asr',
    )
    parser.add_argument(
        '--title',
        help='Title to apply to recording in index.',
        default='untitled',
    )
    parser.add_argument(
        '--api-key-id',
        help='API key ID for ES cluster.',
        default=None,
    )
    parser.add_argument(
        '--api-key-secret',
        help='API key secret for ES cluster.',
        default=None,
    )
    parser.add_argument(
        '--log-level',
        metavar='LEVEL',
        help='Verbosity of logging.',
        default='INFO',
    )
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument(
        '--index',
        help='Index from comma-delimited list of files.',
        default='',
    )
    actions.add_argument(
        '--search',
        help='Search given terms. If this flag is not set, index JSON lines from stdin.',
        default='',
    )
    args = parser.parse_args()

    args.log_level = args.log_level.upper()
    logging.basicConfig(
        format="[%(asctime)s] %(message)s",
        level=args.log_level,
    )

    # Use local time with ISO-8601 format for time output.
    logging.Formatter.formatTime = utils.format_log_time

    api_key = None
    if args.api_key_id is not None and args.api_key_secret is not None:
        api_key = {'id': args.api_key_id, 'api_key': args.api_key_secret}

    es = ESWrapper(
        hosts=args.es_host,
        port=args.es_port,
        api_key=api_key,
    )
    es.create(args.index_name)

    if not args.search:
        if args.index == '':
            logging.info('Indexing from stdin.')
            speaker = get_speaker_names(speakers_string=args.speakers)[0]
            index_lines({speaker: sys.stdin}, es, args.index_name, args.title)
        else:
            logging.info("Indexing from user input %s.", args.index)
            speakers = get_speaker_names(
                speakers_string=args.speakers,
                number_speaker_tracks=len(args.index.split(sep=',')),
            )
            json_lines_by_speaker = dict(
                zip(
                    speakers,
                    [open(input_file) for input_file in args.index.split(sep=',')],
                ),
            )

            index_lines(json_lines_by_speaker, es, args.index_name, args.title)

            for input_file in json_lines_by_speaker.values():
                input_file.close()
    else:
        speakers = [] if args.speakers == '' else args.speakers.split(sep=',')
        result = search_index(
            args.search,
            es,
            args.index_name,
            speakers=speakers,
        )
        print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
