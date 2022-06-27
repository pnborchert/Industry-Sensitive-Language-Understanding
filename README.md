# Industry-Sensitive Language Understanding
Presented at 32nd European Conference on Operation Research – July 2022

# Summary
With recent studies showcasing the added value of pretrained general-purpose language models like
Bidirectional Encoders from Transformers (BERT), they are widely adopted across domains.
By transferring the BERT architecture on domain specific text, related research achieved significant performance improvements in i.e. the
biomedical and legal domain. Due to its availability and immediate impact on decision-making,
processing textual information is particularly relevant in the financial and business domain. In this research project we investigate the impact of domain specific pretraining approaches on business language understanding. We perform industry classification (IC) based on earnings conference call transcripts ([Li et al., 2020](https://github.com/Earnings-Call-Dataset/MAEC-A-Multimodal-Aligned-Earnings-Conference-Call-Dataset-for-Financial-Risk-Prediction)) annotated with SIC labels ([www.sec.gov/edgar/searchedgar/companysearch](https://www.sec.gov/edgar/searchedgar/companysearch.html)) of the corresponding companies. We explain model prediction results in respect to industry-sensitivity using integrated gradients ([Sundarajan et al., 2017](https://arxiv.org/abs/1703.01365)). 

# Project page
The project page contains an step-by-step overview of the analysis with interactive examples of the results.  

> [Project Page](https://pnborchert.github.io/Industry-Sensitive-Language-Understanding/)

> [Dashboard](https://wandb.ai/pnborchert/EURO22-SIC2/reports/Industry-Classification-EURO-2022--VmlldzoyMjMwNzk3?accessToken=jsduua1xcr18mc76paah1lytel0ftpq8cknhtix7xwlg4dpjm2l0370wfufdra6v)

- [ ] BusinessBERT dependencies (Available soon)

# Code
We provide the code required to reproduce (1) the industry classification benchmark and (2) the model explainability results using the [Captum](https://github.com/pytorch/captum) library and integrated gradients.

## Industry Classification
```sh
PATH_TRAIN="./Data/train.parquet.gzip"
PATH_VALID="./Data/valid.parquet.gzip"
PATH_TEST="./Data/test.parquet.gzip"
BATCH_SIZE=4
ACCUM=8
PROJECT="PROJECT NAME"
WANDBDIR="CACHE DIRECTORY"
OUTPUTDIR="OUTPUT DIRECTORY"
MODEL_NAME="bert-base-uncased" # "roberta-base" "ProsusAI/finbert", "custom"
CUSTOM_PATH="" # file path to custom model

python run_EC.py \
--path_train=$PATH_TRAIN \
--path_valid=$PATH_VALID \
--path_test=$PATH_TEST \
--model_name=$MODEL_NAME \
--per_device_train_batch_size=$BATCH_SIZE \
--gradient_accumulation_steps=$ACCUM \
--wandb_project=$PROJECT \
--wandb_dir=$WANDBDIR \
--custom_path=$CUSTOM_PATH \
--output_dir=$OUTPUTDIR
```

## Integrated Gradients
```sh
PATH_TRAIN="./Data/train.parquet.gzip"
PATH_VALID="./Data/valid.parquet.gzip"
PATH_TEST="./Data/test.parquet.gzip"
BATCH_SIZE=3
INTERNAL_BATCH_SIZE=4
PATH_CKPT="./runs/sic2/bert-base-uncased_sic2/checkpoint-190" # Path to model checkpoint in OUTPUTDIR
MODEL_NAME="bert-base-uncased" # "roberta-base" "ProsusAI/finbert", "custom"
PATH_SAVE="FILE NAME OUTPUTS" # ".p" will be appended to the file path

python compute_IG.py \
--path_train=$PATH_TRAIN \
--path_valid=$PATH_VALID \
--path_test=$PATH_TEST \
--path_ckpt=$PATH_CKPT \
--path_save=$PATH_SAVE \
--model_name=$MODEL_NAME \
--batch_size=$BATCH_SIZE \
--internal_batch_size=$INTERNAL_BATCH_SIZE
```
The batch_size and gradient_accumulation parameters are selected for running the experiment on a NVIDIA RTX4000 (8GB) GPU.

# References

```bibtex
@misc{Borchert2022ISLU,
  author = {Borchert, Philipp},
  title = {Industry Sensitive Language Understanding},
  year = {2022},
  publisher = {GitHub},
  journal = {GitHub repository},
  url = {https://github.com/pnborchert/Industry-Sensitive-Language-Understanding}
}
```

```bibtex
@inproceedings{CIKM2020MAEC,
author = {Li, Jiazheng and Yang, Linyi and Smyth, Barry and Dong, Ruihai},
title = {MAEC: A Multimodal Aligned Earnings Conference Call Dataset for Financial Risk Prediction},
year = {2020},
isbn = {9781450368599},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3340531.3412879},
doi = {10.1145/3340531.3412879},
abstract = {In the area of natural language processing, various financial datasets have informed recent research and analysis including financial news, financial reports, social media, and audio data from earnings calls. We introduce a new, large-scale multi-modal, text-audio paired, earnings-call dataset named MAEC, based on S&amp;P 1500 companies. We describe the main features of MAEC, how it was collected and assembled, paying particular attention to the text-audio alignment process used. We present the approach used in this work as providing a suitable framework for processing similar forms of data in the future. The resulting dataset is more than six times larger than those currently available to the research community and we discuss its potential in terms of current and future research challenges and opportunities. All resources of this work are available at https://github.com/Earnings-Call-Dataset/},
booktitle = {Proceedings of the 29th ACM International Conference on Information &amp; Knowledge Management},
pages = {3063–3070},
numpages = {8},
keywords = {multimodal aligned datasets, earnings conference calls, financial risk prediction},
location = {Virtual Event, Ireland},
series = {CIKM '20}
}
```


# Terms Of Use

[![CC BY-SA 4.0][cc-by-sa-shield]][cc-by-sa]

This dataset and iterative forced alignment code is licensed under a
[Creative Commons Attribution-ShareAlike 4.0 International License][cc-by-sa].

[![CC BY-SA 4.0][cc-by-sa-image]][cc-by-sa]

[cc-by-sa]: http://creativecommons.org/licenses/by-sa/4.0/
[cc-by-sa-image]: https://licensebuttons.net/l/by-sa/4.0/88x31.png
[cc-by-sa-shield]: https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg
