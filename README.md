# Service Quality Monitoring in Confined Spaces Through Mining Twitter Data
The proposed method comprises of two main tasks, namely Aspect Extraction and Detecting Events of Interest.

![Flowchart](https://raw.githubusercontent.com/mmrahimi/sq_monitoring_ae/master/Image/flowchart.png)


# Aspect Extraction using fine-tuned BERT language model
## Introduction 
This is a pythonic implementation of Aspect Extraction in the context of service quality of public transport. This repository uses a pre-trained BERT language model to transform multi-label tweets into a vector of words. First, we fine-tune the model using a dataset of tweets. Then, using a binary classifier, tweets can be classified into semantically-related groups, i.e., service quality aspects in our application.

## Dataset
In this project, two major transport hubs are considered as the case studies due to their current importance on transferring a large number of people. First, a Twitter dataset comprising of more than 32 million tweets is collected. This data is obtained from the Australian Urban Research Infrastructure Network ([AURIN](www.aurin.org.au)). Keywords and spatial proximity to hubs are employed to detect relevant tweets.

Next, tweets are manually labelled and mapped to different aspects of SQ (``Safety, View, Information, Service Reliability, Comfort, Personnel, and Additional Services``). Those tweets that do not fall into any of these aspects are considered as irrelevant to the SQ of public transport and therefore, are discarded (Class -1). 
 
## Requirements
This code was tested on Python 3.7.4. Other requirements are as follows: 
- Tensorflow
- BERT
- Bert Tensorflow
- Scikit Learn
- Imbalanced Learn
- Numpy
- Pandas
- Jupyter

(See [requirements.txt](https://github.com/mmrahimi/aspect_extraction/blob/master/requirements.txt))

## Quick Start

1. Install Libraries:
```pip install -r requirements.txt```

2. Download [uncased BERT<sub>base</sub> with 12 layers](https://github.com/google-research/bert) and introduce it as ``bert_path``


## Baseline approaches
As discussed in the paper, Latent Dirichlet Allocation (LDA) and Skip-gram are two state-of-the-art baseline methods which are employed to evaluate the performance of the proposed method. Implementation of these two methods can be found in the "Baseline Methods" directory.

## Citation
```

```

## Author
Masoud Rahimi
- Email: <mmrahimi@student.unimelb.edu.au>
- Linkedin: [Here](https://www.linkedin.com/in/rahimimasoud/)


## Acknowledgments
This project incorporates code from the following repos:
- https://github.com/google-research/bert
- https://github.com/kaushaltrivedi/bert-toxic-comments-multilabel

