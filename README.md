Data Spec Transformers Demo
========================
1. [Overview](#Overview)
1. [Build](#Build)
1. [Example](#Example)
1. [Download Model](#Download)
1. [CSV Example](#CSV_Example)

## <a name="Overview"></a>Overview

This is a more detailed demo for
the [datacraft](https://github.com/bbux-dev/datacraft) library. This make use of
the [huggingface](https://huggingface.co/transformers/task_summary.html#masked-language-modeling)
Masked Language Modeling example to generate tokens from surrounding context.

## <a name="Build"></a>Build

To install the datacraft library and the huggingface dependencies:

```shell
pip install datacraft transformers torch
```

The `datacraft` executable should now be on your path


## <a name="Example"></a>Example

This example takes advantage of the datacraft custom code loading capability. This profides an extension point into 
the datacraft library to build and define custom types and handlers for those types. The custom_code.py module 
defines a `hf-fill-mask` type handler. This example code will load a huggingface `fill-mask` transformer pipeline. 
This pipeline is used to generate tokens given the surrounding context. The handler uses the `__MASK__` token to 
denote where the token should be placed. For example if we have the sentence:

`I seem to have lost my number. Can I have yours?`

We can put the `__MASK__` marker in various places to see what possible substitutions the huggingfaces model will 
provide:

`I seem to have lost my __MASK__. Can I have yours?`

To run the demo:

```shell
$ datacraft --spec demo.json -i 20 --code custom_code.py --log-level error
Go ahead, feel my shirt. It's made of recycled material!
If you were a Transformer you'd be thrilled!
Do you believe in love at first? Or should I walk past you again?
I'm learning about important dates in 2017. Wanna be one of them?
I seem to have lost my mind. Can I have yours?
...
Do you have a name? Or can I call you Bob?
```

## <a name="Download"></a>Download Model

By default, every time you run the demo it will download the model from https://huggingface.co. To keep from doing 
this over and over, use the download_model.py script.

```shell
python download_model.py /path/to/model/dir 2>&1 | grep INFO
#INFO: Loading fill-mask pipeline...
#INFO: Saving fill-mask to /path/to/model/dir

# now specify the downloaded dir as the datadir
datacraft -s demo.json -i 20 -c custom_code.py -l debug -d /path/to/model/dir
```

## <a name="CSV_Example"></a>CSV Example

Many times it is easier to externalize larger values lists into a csv file. This can be referenced in a Data Spec 
using the `csv` Field Spec type.  The `demo-csv.json` spec does this with the lines.csv file. To use this example:

```shell
# copy csv file to same location as downloaded model:
cp lines.csv /path/to/model/dir
datacraft -s demo-csv.json -i 20 -c custom_code.py -l debug --datadir /path/to/model/dir
```
