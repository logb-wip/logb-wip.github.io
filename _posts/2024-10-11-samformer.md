---
layout: distill
title: SAMformer
description: TODO # Deep Convolutional Representations in RKHS
tags: [transformers, deep learning, time series forecasting, maths, code]
giscus_comments: true
date: 2024-10-11
featured: false

authors:
  - name: Ambroise Odonnat
    url: "https://ambroiseodt.github.io/"
    affiliations:
      name: Huawei Noah's Ark Lab & Inria
  - name: Oussama Zekri
    url: "https://oussamazekri.fr"
    affiliations:
      name: ENS Paris-Saclay & Imperial

bibliography: 2024-10-11-samformer.bib

# Optionally, you can add a table of contents to your post.
# NOTES:
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - we may want to automate TOC generation in the future using
#     jekyll-toc plugin (https://github.com/toshimaru/jekyll-toc).
toc:
  - name: Goal 🚀 
    # if a section has subsections, you can add them as follows:
    # subsections:
    #   - name: Example Child Subsection 1
    #   - name: Example Child Subsection 2
  - name: Motivation 🔎
  - name: SAMformer ⚔️
  - name: Getting your hands dirty 🖥️
  - name: Acknowledgments 🙏🏾

# Below is an example of injecting additional post-specific styles.
# If you use this post as a template, delete this _styles block.
_styles: >
  .fake-img {
    background: #bbb;
    border: 1px solid rgba(0, 0, 0, 0.1);
    box-shadow: 0 0px 4px rgba(0, 0, 0, 0.1);
    margin-bottom: 12px;
  }
  .fake-img p {
    font-family: monospace;
    color: white;
    text-align: left;
    margin: 12px 0;
    text-align: center;
    font-size: 16px;
  }
---

## <a id="goal"></a>Goal 🚀
> Fear not, those who delved into the maths of the kernel trick, for its advent in deep learning is coming.

In this blog post, we focus on ***SAMformer***, a transformer-based architecture for time series forecasting proposed in [*SAMformer: Unlocking the Potential of Transformers in Time Series Forecasting*](https://arxiv.org/pdf/2402.10198) <d-cite key="mairal2016endtoend"></d-cite>, one of Ambroise's recent paper. SAMformer combines Sharpness-Aware Minimization (SAM) <d-cite key="mairal2016endtoend"></d-cite> and channel-wise attention to obtain a light-weight SOTA model with improved robustness and signal propagation compared to its competitors. This blog aims to provide a high-level view of the motivation behind SAMformer while explaining how to implement it. We invite the reader to read the original paper for more details.

## Motivation 🔎
Time series forecasting consists of analyzing time series data to predict future trends based on historical information. It has many applications in real-world scenarios such as forecasting ECG recordings to anticipate cardiac arrhythmia, predicting electricity consumption to match future demand, or predicting stock market prices (an exciting topic in times of inflation). Multivariate long-term forecasting is notoriously challenging due to feature correlations and long-term temporal dependencies in time series.

### Failure of Transformers

### Trainability Issues of the Attention

## SAMformer ⚔️

## Getting your hands dirty 🖥️
In this section, we discuss the implementation of SAMformer. 

### Overview
The original implementation of the SAMformer architecture makes use of modern deep learning frameworks such as `PyTorch` or `TensorFlow` and can be found [here](https://github.com/romilbert/samformer).

### Main Components
As can be seen below, SAMformer consists of 5 layers:

{% include figure.liquid path="assets/img/blog_samformer/samformer_arch.png" class="img-fluid rounded z-depth-0" zoomable=true %}

It leads to a shallow transformer with a single head and a single encoder that can be trained with SAM <d-cite key="mairal2016endtoend"></d-cite>. 

We provide a snippet of SAMformer (few) code lines below for the interested reader. 

{% details SAMformer Implementation %}
```python
import torch.nn as nn
import torch.nn.functional as F

class SAMFormerArchitecture(nn.Module):
    def __init__(self, num_channels, seq_len, hid_dim, pred_horizon):
        super().__init__()
        self.revin = RevIN(num_features=num_channels)
        self.compute_keys = nn.Linear(seq_len, hid_dim)
        self.compute_queries = nn.Linear(seq_len, hid_dim)
        self.compute_values = nn.Linear(seq_len, seq_len)
        self.linear_forecaster = nn.Linear(seq_len, pred_horizon)

    def forward(self, x):

        # RevIN Normalization
        x_norm = self.revin(x.transpose(1, 2), mode='norm').transpose(1, 2) # (n, D, L)

        # Channel-Wise Attention
        queries = self.compute_queries(x_norm) # (n, D, hid_dim)
        keys = self.compute_keys(x_norm) # (n, D, hid_dim)
        values = self.compute_values(x_norm) # (n, D, L)
        att_score = F.scaled_dot_product_attention(queries, keys, values) # (n, D, L)

        # Residual Connection
        out = x_norm + att_score # (n, D, L)

        # Linear Forecasting
        out = self.linear_forecaster(out) # (n, D, H)

        # RevIN Denormalization
        out = self.revin(out.transpose(1, 2), mode='denorm').transpose(1, 2) # (n, D, H)

        return out
```
{% enddetails%}

## <a id="acknowledgments"></a>Acknowledgments 🙏🏾

We thank TBD for taking the time to proofread this blog post. We thank Ambroise's co-authors: Romain Ilbert, Vasilii Feofanov, Aladin Virmaux, Giuseppe Paolo, Themis Palpanas, and Ievgen Redko. 

For any further questions, please feel free to leave a comment or contact us by mail!