# Emergent In-Context Learning in Transformers

`eicl`:
Based on [Data Distributional Properties Drive Emergent In-Context Learning in Transformers](https://arxiv.org/abs/2205.05055).
Train a transformer on [Omniglot](https://github.com/brendenlake/omniglot).
Experimental design described in Sec 2 of the paper.

`jaxline`: c/p from the `jaxline` repo which seems unmaintained.
Minor changes.


## Handy commands

To install the necessary requirements:

```shell
uv venv .venv
source .venv/bin/activate
uv sync --all-extras
```

To run training:

```shell
$ python -m eicl --config $PATH_TO_CONFIG --jaxline_mode train --logtostderr
# (save checkpoints using Ctrl+C)
```

To evaluate a trained model, override `config.restore_path` with the
subdirectory of `config.checkpoint_dir` containing the relevant checkpoint
(`$CKPT_DIR` below).

To evaluate 
* in-weights learning (trained classes) use `eval_no_support_zipfian` 
* in-context learning (holdout classes) use `eval_fewshot_holdout`

```shell
# disable GPU if needed (easier parallel run train+eval)
CUDA_VISIBLE_DEVICES=-1 \
JAX_PLATFORM_NAME=cpu \
JAX_PLATFORMS=cpu \
python -m eicl \
    --config src/eicl/experiment/configs/images_all_exemplars.py \
    --config.restore_path /tmp/tk/models/latest/step_4770_2024-11-19T11:34:52/ \
    --config.one_off_evaluate \
    --jaxline_mode eval_fewshot_holdout \
    # --jaxline_post_mortem \
    # --jaxline_disable_pmap_jit \
    # ^ useful debug tips
```

<details>
<summary>Details</summary>

## Usage

### Default configs

Default experiment configurations are provided in `configs/`, and can be used
in `$PATH_TO_CONFIG` in the launch commands below.

*   `images_all_exemplars.py`: Each character class consists of 20 image
    examples (the original Omniglot problem).
*   `images_augmented.py`: We augment the total number of classes to 8x the
    original number, by applying transformations to each image class: flip left
    or right + rotate 0, 90, 180, or 270 degrees.
*   `images_identical.py`: Each character class consists only of a single image
    (the 1st of the 20 examples provided in the original Omniglot dataset)
*   `symbolic.py`: (relatively untested; not used in the paper)

Config files can be edited or forked as desired.

### Varieties of data sequences + Configurations for each

Omniglot sequences are generated in `datasets/data_generators.py`.

The image classes are divided into training and holdout. Training classes can be
"common" or "rare". The training classes can be uniformly or Zipf-distributed
(jointly over both common and rare classes). Related configurations are set in
`config.data.generator_config`.

There are few different types of data sequences:

*   `bursty` : These are the canonical bursty (and non-bursty) sequences used in
    training in the paper
*   `no_support_common`, `no_support_rare`, `non_support_zipfian` : These
    sequences enforce that the query class does not appear anywhere in the
    context, and are the sequences used for evaluating in-weights learning in
    the paper. They can consist entirely of common classes, rare classes, or be
    Zipf-distributed over all training classes.
*   `fewshot_common`, `fewshot_rare`, `fewshot_zipfian`, `fewshot_holdout` :
    These sequence are standard k-shot n-way fewshot sequences, and are used for
    evaluating in-context learning in the paper. They can exist of holdout
    classes, common classes, rare classes, or be Zipf-distributed over all
    training classes.
*   `mixed`: A mix of standard fewshot and iid randomly generated sequences.

Sequence types are specified in `config.data.train_seqs` and in
`config.eval_modes` (with an additional `eval_` prefix). You may specify a list
of eval modes, to evaluate the same learner on multiple sequence types.

See `experiment/experiment.py: _get_ds_seqs` and `datasets/data_generators.py:
SeqGenerator` for more details on settings, which are specified in
`config.data.seq_config`.

</details>