"""
Create and save the model to use here

Example To Create Saved Model To Use:
>>> from transformers import pipeline
>>> nlp = pipeline('fill-mask')
>>> nlp.save_pretrained('/path/to/model/dir')

Then use this as the -d or --data-dir arg to commandline version of tool to specify this dir
Alternatively use the 'model-dir' config param for the field

Spec format
    {
        "field": {
            "type": "hf-fill-mask",
            "seed-ref": "SEED_REF"
        },
        "refs": {
            "SEED_REF": [
                "Go ahead, feel my shirt. It's made of __MASK__ material!",
                "If you were a Transformer you'd be __MASK__!",
                "Do you believe in love at __MASK__? Or should I walk past you again?",
                "I'm learning about important dates in __MASK__. Wanna be one of them?",
                "I seem to have lost my __MASK__. Can I have yours?",
                "Are you a __MASK__? Cause you've got fine written all over you!",
                "Did you invent the __MASK__? Because you seem Wright for me!",
                "I was wondering if you had an __MASK__. Because mine was just stolen!",
                "Can I follow you where you're going right now? Cause my __MASK__ always told me to follow my dreams!",
                "Are you Siri? Because you __MASK__ me!",
                "I hope you know __MASK__, because you are taking my breath away!",
                "If I had four quarters to give to the four __MASK__ women in the world, you would have a dollar!",
                "Are you a __MASK__? Because every time I look at you, I smile!",
                "Is there an __MASK__ nearby, or was that just my heart taking off?",
                "Are you a __MASK__ angle? Because you're a cutie!",
                "If nothing lasts __MASK__, will you be my nothing?",
                "If you were a phaser on __MASK__, you'd be set to stun!",
                "Do you have a __MASK__? Or can I call you mine?",
                "Is your name __MASK__? Because you have everything I've been searching for.",
                "Have you been covered in __MASK__ recently? I just assumed, because you look sweeter than honey."
            ]
        }
    }
"""

import json
import logging
import random
import dataspec
from dataspec import SpecException
from transformers import pipeline

log = logging.getLogger(__name__)


class HuggingFaceFillMaskSupplier:
    def __init__(self,
                 wrapped,
                 mask_token_placeholder,
                 pipeline_name,
                 token_only,
                 model_dir=None):
        self.wrapped = wrapped
        if _model_dir_is_valid(model_dir):
            log.debug('Loading %s pipeline from %s...', pipeline_name, model_dir)
            self.nlp = pipeline(pipeline_name, model=model_dir)
        else:
            log.debug('Loading %s pipeline from internets...', pipeline_name)
            self.nlp = pipeline(pipeline_name)
        log.debug('%s pipeline Loaded', pipeline_name)
        self.mask_token_placeholder = mask_token_placeholder
        self.mask_token = self.nlp.tokenizer.mask_token
        self.token_only = token_only

    def next(self, iteration):
        value = str(self.wrapped.next(iteration))
        if self.mask_token_placeholder not in value:
            raise SpecException(f'Mask token placeholder: {self.mask_token_placeholder} not found in generated data!')
        value = value.replace(self.mask_token_placeholder, self.mask_token)
        candidates = self.nlp(value)
        candidate = random.sample(candidates, 1)[0]
        if self.token_only:
            return candidate['token_str']
        return candidate['sequence']


def _model_dir_is_valid(model_dir):
    model_dir_str = str(model_dir).strip()
    return model_dir and model_dir_str not in ['', '[]']


@dataspec.registry.types('hf-fill-mask')
def configure_supplier(field_spec, loader):
    """
    Configures the supplier from the provided field spec using the huggingface fill-mask pipeline by default

    Config Params:
    :param mask-token-placeholder: place holder that should show up in the seed strings, default '__MASK__'
    :param pipeline: name of the transformers pipeline to use, default is 'fill-mask'
    :param model-dir: directory to load model from, default is loader.datadir
    :param token-only: if only the generated token should be output apart from the context, default is to output the full sequence
    """
    if 'seed-ref' not in field_spec:
        raise SpecException('seed-ref is required field for hf-fill-mask type: ' + json.dumps(field_spec))
    key = field_spec.get('seed-ref')
    seed_ref_spec = loader.refs.get(key)
    config = field_spec.get('config', {})
    mask_token_placeholder = config.get('mask-token-placeholder', '__MASK__')
    pipeline_name = config.get('pipeline', 'fill-mask')
    model_dir = config.get('model-dir', loader.datadir)
    token_only = config.get('token-only', False)
    wrapped = loader.get_from_spec(seed_ref_spec)
    return HuggingFaceFillMaskSupplier(wrapped, mask_token_placeholder, pipeline_name, token_only, model_dir)
