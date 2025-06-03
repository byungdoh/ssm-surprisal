# The Impact of Token Granularity on the Predictive Power of Language Model Surprisal

## Introduction
This is the code repository for the paper [The Impact of Token Granularity on the Predictive Power of Language Model Surprisal](https://arxiv.org/pdf/2412.11940v2), including trained [unigram language model (ULM) tokenizers](https://github.com/google/sentencepiece) and fairly minimal code for training [Mamba-2 models](https://github.com/state-spaces/mamba) and calculating surprisal from them.
Much of this repository is based on commit 4a8a2a2 of the official Mamba repository.

## Setup
Please refer to the [SentencePiece](https://github.com/google/sentencepiece) and [Mamba](https://github.com/state-spaces/mamba) repositories for the dependencies.
For reference, this repository contains an `environment.yml` file for the Mamba side of things.

## ULM Tokenizer
The files under the `tokenizers` directory can be used together with the [SentencePiece](https://github.com/google/sentencepiece) repository to tokenize text. For more details, please refer to the "Encode raw text into sentence pieces/ids" section in the SentencePiece readme.

## Mamba-2 LMs
The Mamba-2 LMs trained as part of this work are available on the [companion HuggingFace repository](https://huggingface.co/byungdoh/ssm-token-granularity).
Each model directory can be downloaded and used for surprisal calculation as outlined below.

## Mamba-2 LM Training
The command `python mamba_lm_train.py INI_FILE` (e.g. `python mamba_lm_train.py ini/wiki40b_en_48000_6_8_256.ini`) can be used to newly train Mamba-2 language models.
The input data file (e.g. `data/wiki40b_en_min25max1024.en_48000.first1000.ids`) should be a sequence of token indices, with one sequence on each line.

## Mamba-2 Surprisal Calculation
The command `python get_ssm_surprisal.py MAMBA_MODEL_DIR INPUT_FILE ULM_VOCAB_FILE > OUTPUT_FILE` (e.g. `python get_ssm_surprisal.py models/wiki40b_en_48000_6_8_256.ini_128 data/my_stimuli.en_48000.ids tokenizers/en_48000.vocab > my_stimuli.en_48000.surprisal`) can be used to calculate surprisal from Mamba-2 language models.
The input file should again be a sequence of token indices, with one sequence on each line.
The command `python postprocess_bi.py INITIAL_SURPRISAL > FINAL_SURPRISAL` (e.g. `python postprocess_bi.py my_stimuli.en_48000.surprisal > my_stimuli.en_48000_wt.surprisal`) can then be used to apply [whitespace-trailing decoding](https://aclanthology.org/2024.emnlp-main.202/) and correct word probabilities.

## Questions
For questions or concerns, please contact Byung-Doh Oh ([oh.b@nyu.edu](mailto:oh.b@nyu.edu)).
