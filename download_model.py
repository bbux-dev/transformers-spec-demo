#!/bin/env python
import os
import sys
import logging
from transformers import pipeline

# Change to logging.INFO for less verbose output from underlying libraries
_DEFAULT_LOG_LEVEL = logging.DEBUG
log = logging.getLogger(__name__)
logging.basicConfig(level=_DEFAULT_LOG_LEVEL,
                    format='%(levelname)s: %(message)s')


def main(output_dir, model_name):
    log.info('Loading %s pipeline...', model_name)
    nlp = pipeline(model_name)
    log.info('Saving %s to %s', model_name, output_dir)
    nlp.save_pretrained(output_dir)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Please specify output directory for model')
        exit(1)

    # First Arg is Output Directory
    outdir = sys.argv[1]
    if not os.path.exists(outdir):
        os.makedirs(os.path.expanduser(outdir))

    # Second optional arg is model name
    if len(sys.argv) > 2:
        model = sys.argv[2]
    else:
        model = 'fill-mask'

    main(outdir, model)
