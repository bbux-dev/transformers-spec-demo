Data Spec Transformers Demo
========================
1. [Overview](#Overview)
1. [Build](#Build)
1. [Example](#Example)
1. [Download Model](#Download)

## <a name="Overview"></a>Overview

This is a more detailed demo for the [dataspec](https://github.com/bbux-dev/dataspec) library. This make use of the
[huggingface](https://huggingface.co/transformers/task_summary.html#masked-language-modeling) Masked Language Modeling
example to generate tokens from surrounding context.

## <a name="Build"></a>Build

To install the dataspec library

```shell
pip install git+https://github.com/bbux-dev/dataspec.git
```

The executable will be located in `dataspec` and should now be on your path


## <a name="Example"></a>Example

The custom_code.py module defines a `hf-fill-mask` type handler. This example code will load a huggingface 
`fill-mask` transformer pipeline. This pipeline is used to generate tokens given the surrounding context.
The handler uses the `__MASK__` token to denote where the token should be placed. To run the demo:

```shell
dataspec -s demo.json -i 20 -c custom_code.py -l debug
```

## <a name="Download"></a>Download Model

Every time you run the demo it will download the model from https://huggingface.co. The do this process once, use the
download_model.py script.

```shell
python download_model.py /path/to/model/dir 2>&1 | grep INFO
#INFO: Loading fill-mask pipeline...
#INFO: Saving fill-mask to /path/to/model/dir

# now specify the downloaded dir as the datadir
dataspec -s demo.json -i 20 -c custom_code.py -l debug -d /path/to/model/dir
```
