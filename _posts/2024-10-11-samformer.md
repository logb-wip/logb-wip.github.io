---
layout: distill
title: SAMformer - Efficient Time Series Forecasting with Transformers
description: Improved attention and optimization for better performance
tags: [transformers, deep learning, time series forecasting, maths, code]
giscus_comments: true
date: 2024-10-11
featured: false

authors:
  - name: Ambroise Odonnat
    url: "https://ambroiseodt.github.io/"
    affiliations:
      name: Noah's Ark Lab, Inria
  - name: Oussama Zekri
    url: "https://oussamazekri.fr"
    affiliations:
      name: ENS Paris-Saclay 
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
> When the scientific method leads to an efficient implementation.

In this blog post, we focus on **SAMformer** proposed in [*SAMformer: Unlocking the Potential of Transformers in Time Series Forecasting*](https://arxiv.org/pdf/2402.10198) <d-cite key="ilbert2024samformer"></d-cite>. SAMformer combines Sharpness-Aware Minimization (SAM) <d-cite key="foret2021sharpnessaware"></d-cite> and channel-wise attention to obtain a light-weight SOTA model with improved robustness and signal propagation compared to its competitors. This blog aims to provide a high-level view of the motivation behind SAMformer while explaining how to implement it. For the reader interested in more details, the paper is on [arXiv](https://arxiv.org/pdf/2402.10198), and the code can be found on [github](https://github.com/romilbert/samformer).

1) Problem: transformers nuls en TS forecasting + very complicated and large-scale models --> hard to identify the failure.
2) We simplify transformer to only keep the key components
3) Problem identification: trainability issues
4) Possible solution: sigma reparam or SAM
5) SAM works --> putting evertyhing together
6) "On va droit au but, allez voir le papier pour plus de detail." (TO DO, something like "We'll keep it concise, refer to the paper for more details."). 

## Motivation 🔎
Time series forecasting has many applications in real-world scenarios, e.g., anticipating cardiac arrhythmia in ECG signals, predicting electricity consumption to match future demand, or forecasting stock market prices (an exciting topic in times of inflation). This is notoriously challenging, especially in multivariate and long-term settings, due to feature correlations and long-term temporal dependencies in time series. Since transformers are tailored to sequential data and given their impressive success in NLP <d-cite key="brown2020fewshot"></d-cite> and computer vision <d-cite key="foret2021sharpnessaware"></d-cite>, many recent variants of the original implementation have been proposed, with sparse attention to decrease the $N^2$ dependency or decomposition schemes to deal with the temporal dependencies. This led to a wide range of complex models with many parameters. However, it has been shown that linear models perform better than those SOTA Anything-formers <d-cite key="zeng2022transformerseffectivetimeseries"></d-cite>. This came as a surprise to us given the success of the Transformer architecture with other data modalities and motivated our study right up to the development of SAMformer.

{% include figure.liquid path="assets/img/blog_samformer/meme_dogs.png" class="img-fluid rounded z-depth-0" zoomable=true %}

## SAMformer ⚔️
We detail below the process that led to the SAMformer.
This motivated our study. 

Proposition : Traditional transformer models for time series forecasting are often complex and large, making it difficult to pinpoint and address their weaknesses. SAMformer addresses this by streamlining the architecture to include only essential components, enhancing simplicity without compromising performance. Trainability issues are identified during this simplification and are tackled using Sharpness-Aware Minimization (SAM), which proved highly effective. By integrating SAM with channel-wise attention, SAMformer achieves state-of-the-art performance with a lightweight and robust design, making it a superior choice for time series forecasting.

## Simplified Setting
The Anything-formers are often complex and large, making it difficult to pinpoint and address their weaknesses. To better understand what causes the low performance of those models, we  first simplify the transformer to only keep its key components. Given that linear models outperformed more complex transformer ones, we naturally considered a toy problem of linear regression. The question is whether a simple transformer *tailored* to solve this task manage to do it, at least as well as a linear model.

### Trainability Issues of the Attention
To identify the problem, we simplify the original Transformer <d-cite key="vaswani2017"></d-cite> to only keep the key components.

{% include figure.liquid path="assets/img/blog_samformer/sharpness_entropy_collapse_sam.png" class="img-fluid rounded z-depth-0" zoomable=true %}


### SAM to the rescue
There are two possible solutions:
- $\sigma$-reparam <d-cite key="zhai2023sigmareparam"></d-cite>:
- SAM <d-cite key="foret2021sharpnessaware"></d-cite>:

{% include figure.liquid path="assets/img/blog_samformer/toy_exp_losses_val_all_methods.png" class="img-fluid rounded z-depth-0" zoomable=true %}

### Putting Everything Together
Now it works on our toy example: congrats you can now solve linear regression tasks. Hum, what about true time series data? We are only one step away from the optimal architecture: add revin <d-cite key="kim2022reversible"></d-cite>

In the end, SAMformer consists of 5 layers: RevIN normalization, channel-wise attention, residual connection, linear forecasting, and RevIN denormalization. And we are SOTA: 

fig: add table and/or result figure (e.g. generalization plots with stars).

## Getting your hands dirty 🖥️
In this section, we discuss the implementation of SAMformer which makes use of modern deep learning frameworks such as `PyTorch` or `TensorFlow` and can be found [here](https://github.com/romilbert/samformer).

### Main Components
As can be seen below, SAMformer consists of 5 layers:

{% include figure.liquid path="assets/img/blog_samformer/samformer_arch.png" class="img-fluid rounded z-depth-0" zoomable=true %}

It leads to a shallow transformer with a single head and a single encoder that can be trained with SAM <d-cite key="foret2021sharpnessaware"></d-cite>. 

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
        x_norm = self.revin(x.transpose(1, 2), mode='norm') 
        x_norm = x_norm.transpose(1, 2) # (n, D, L)

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
        out = self.revin(out.transpose(1, 2), mode='denorm')
        out = out.transpose(1, 2) # (n, D, H)

        return out
```
{% enddetails%}

## Future Work
While studying the trainability issues of the Transformer, we stumbled over the fact that the entropy collapse appeared in tandem with a sharp loss. However, we observed that the entropy collapse was benign, i.e., solving it did not improve the performance that much. This is different from the conclusions on text and images found in <d-cite key="zhai2023sigmareparam"></d-cite>. Moreover, we saw that using $\sigma$-reparam <d-cite key="zhai2023sigmareparam"></d-cite> decreases the signal propagation a lot, up to the point of having almost uniform attention. This leads to rank collapse which is known to appear in attention-based models and to impact the generalization performance <d-cite key="anagnostidis2022signal"></d-cite> <d-cite key="dong2021attentionrank"></d-cite>. Finally, we manage to demonstrate that, indeed, $\sigma$-reparam induces a rank collapse. Formally, $\sigma$-reparam aims to minimize blabla, which in turn can be used to upper-bound the rank of the attention internal computations. We have

{% include figure.liquid path="assets/img/blog_samformer/nuclear_norm.png" class="img-fluid rounded z-depth-0" zoomable=true %}

Sigma reparam bla bla (citer Sinkformer <d-cite key="sander2022sinkformer"></d-cite> + rank and signal propagation work on attention (attention is not all u need + signal propagation in transformer).

## Conclusion
This blog post has presented Ambroise's recent work on transformers for time series forecasting. By integrating SAM with channel-wise attention, SAMformer achieves state-of-the-art performance with a lightweight and robust design, making it a superior choice for time series forecasting. While there are many models and now foundation models for time series forecasting, SAMformer has the benefit of simplicity and efficiency. Beyond the model itself, we felt it was important to share the research process that led to it in a world of rapid changes and extreme development of algorithms that are performance-oriented. While SAMformer outperforms SOTA models (at the time) for a fraction of the computational cost, our goal was primarily to better understand the failure modes by rigorous analysis. We hope this kind of methodology can inspire researchers and practitioners towards more efficient and reliable methods. 

Finally, we felt the need to recall in these troubled times the necessity of open-minded and open-source research for the community. In particular, this work would not have been possible without access to the code and papers of SAM[], $\sigma$-reparam[] or machine learning libraries such as PyTorch, TensorFlow, or Scikit-Learn. For the reader interested in more mathematical details or getting started with SAMformer, the paper is on [arXiv](https://arxiv.org/pdf/2402.10198), and the code can be found on [github](https://github.com/romilbert/samformer).

## <a id="acknowledgments"></a>Acknowledgments 🙏🏾
We thank TBD for taking the time to proofread this blog post. SAMformer <d-cite key="ilbert2024samformer"></d-cite> is the first published paper of Ambroise's thesis and he thanks all his co-authors and particularly his supervisor, Ievgen Redko, for the freedom and trust he provided during this project.

For any further questions, please feel free to leave a comment or contact us by mail!
