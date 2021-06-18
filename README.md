Data Spec Transformers Demo
========================
1. [Overview](#Overview)
1. [Build](#Build)
1. [Example](#Example)
1. [Download Model](#Download)
1. [CSV Example](#CSV_Example)

## <a name="Overview"></a>Overview

This is a more detailed demo for
the [dataspec](https://github.com/bbux-dev/dataspec) library. This make use of
the [huggingface](https://huggingface.co/transformers/task_summary.html#masked-language-modeling)
Masked Language Modeling example to generate tokens from surrounding context.

## <a name="Build"></a>Build

To install the dataspec library and the huggingface dependencies:

```shell
pip install git+https://github.com/bbux-dev/dataspec.git transformers torch
```

The `dataspec` executable should now be on your path


## <a name="Example"></a>Example

The custom_code.py module defines a `hf-fill-mask` type handler. This example
code will load a huggingface `fill-mask` transformer pipeline. This pipeline is
used to generate tokens given the surrounding context. The handler uses
the `__MASK__` token to denote where the token should be placed. To run the
demo:

```shell
dataspec -s demo.json -i 20 -c custom_code.py -l debug
```

## <a name="Download"></a>Download Model

By default, every time you run the demo it will download the model
from https://huggingface.co. To keep from doing this over and over, use the
download_model.py script.

```shell
python download_model.py /path/to/model/dir 2>&1 | grep INFO
#INFO: Loading fill-mask pipeline...
#INFO: Saving fill-mask to /path/to/model/dir

# now specify the downloaded dir as the datadir
dataspec -s demo.json -i 20 -c custom_code.py -l debug -d /path/to/model/dir
```

## <a name="CSV_Example"></a>CSV Example

Many times it is easier to externalize larger values lists into a csv file. This
can be referenced in a Data Spec using the `csv` Field Spec type.  The 
`demo-csv.json` spec does this with the lines.csv file. To use this example:

```shell
# copy csv file to same location as downloaded model:
cp lines.csv /path/to/model/dir
dataspec -s demo-csv.json -i 20 -c custom_code.py -l debug --data-dir /path/to/model/dir
```
