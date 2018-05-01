# Language Generation

## Pre-Requisites

Tensorflow 1.4

## Instructions

1. Download pre-trained language models from [here](https://drive.google.com/file/d/1aYQzgcLdHehop2HK6Tv8GnCTCE_3BlLY/view?usp=sharing) and store the `save` and `save-best` folders under the root folder (`language-generation`).
2. To generate the experimental results, run the following command -  
`python train.py --job_id diversit --mode beam --diversity_type <type> --diversity_lambda <lambda> --diversity_beam_size <size> --diversity_prior <prior>`.
3. You can explore the other files to create and train your own language models.

## Configurations Used In Report

1. `<type>` - `"hamming"`, `"length"`
2. `<lambda>` - `1`, `5`
3. `<size>` - `10`
4. `<prior>` - `"why"`, `"how can i"`, `"he is a"`