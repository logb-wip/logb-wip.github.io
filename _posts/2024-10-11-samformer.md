---
layout: distill
title: SAMformer
description: TODO # Deep Convolutional Representations in RKHS
tags: [transformers, deep learning, maths, code]
giscus_comments: true
date: 2024-10-11
featured: false

authors:
  - name: Ambroise Odonnat
    url: "https://ambroiseodt.github.io/"
    affiliations:
      name: Huawei Noah's Ark Lab
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
  - name: Trainability Issues due to the Attention 🔎
  - name: SAMformer in-depth ⚔️
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

In this blog post, we focus on the ***Convolutional Kernel Network*** (CKN) architecture proposed in [*End-to-End Kernel Learning with Supervised Convolutional Kernel Networks*](https://proceedings.neurips.cc/paper_files/paper/2016/file/fc8001f834f6a5f0561080d134d53d29-Paper.pdf) <d-cite key="mairal2016endtoend"></d-cite> and present its guiding principles and main components. TODO 

## Trainability Issues due to the Attention 🔎
Before diving into the CKN itself, we briefly recall what the *kernel trick* stands for. TODO

$$
\begin{equation*}
K(\mathbf{x}, \mathbf{x}') = \langle \varphi(\mathbf{x}), \varphi(\mathbf{x}')\rangle_{\mathcal{H}}
\end{equation*}
$$

## SAMformer in-depth ⚔️
Now that everyone has a clear head regarding the kernel trick, we are prepared to introduce the so-called ***Convolutional Kernel Network***. The main idea behind the CKN is to leverage the kernel trick to represent local neighborhoods of images. In this section, we explain how the authors build Convolutional Kernel layers that can be stacked into a multi-layer CKN.

### Motivation: representing local image neighborhoods
The CKN architecture is a type of convolutional neural network that couples the prediction task with representation learning.

## Getting your hands dirty 🖥️
In this section, we discuss the implementation of the CKN architecture and show how to reimplement it from scratch. 

### Modern Implementation
The original implementation of the CKN architecture makes use of modern deep learning frameworks such as `PyTorch`, `TensorFlow` or `JAX` and can be found [here](https://github.com/claying/CKN-Pytorch-image). We recommend using it if the performance is at stake.

#### Autodiff

Automatic differentiation (autodiff) is a well-known algorithm that is absolutely essential in deep learning.

<div style="display: flex; justify-content: center;"><blockquote class="twitter-tweet"><p lang="en" dir="ltr">Backprop in neural networks is reverse mode auto-diff applied to a simple linear computational graph. The simplicity of the resulting algorithm somehow overshadows the power and non-triviality of applying auto-diff to complex (e.g. recursive) graphs. <a href="https://t.co/5op8P7oYrE">https://t.co/5op8P7oYrE</a> <a href="https://t.co/ETafufgGqa">pic.twitter.com/ETafufgGqa</a></p>&mdash; Gabriel Peyré (@gabrielpeyre) <a href="https://twitter.com/gabrielpeyre/status/956932467092574213?ref_src=twsrc%5Etfw">January 26, 2018</a></blockquote> <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script></div>

For implementation from scratch details of autodiff, see this really simple [blog post](https://e-dorigatti.github.io/math/deep%20learning/2020/04/07/autodiff.html) from [Emilio Dorigatti](https://e-dorigatti.github.io/)'s website. But let's go back to our CKN now.

## <a id="acknowledgments"></a>Acknowledgments 🙏🏾

We would especially like to thank Prof. [Julien Mairal](https://lear.inrialpes.fr/people/mairal/) for taking the time to proofread this blog post. This is all the more important to us as he is the author of the [CKN paper](https://proceedings.neurips.cc/paper_files/paper/2016/file/fc8001f834f6a5f0561080d134d53d29-Paper.pdf). We are very grateful to the professors of the [Kernel Methods course](https://mva-kernel-methods.github.io/course-2023-2024/) of the [MVA Master](https://www.master-mva.com/): Prof. [Julien Mairal](https://lear.inrialpes.fr/people/mairal/), Prof. [Michel Arbel](https://michaelarbel.github.io/), Prof. [Jean-Philippe Vert](https://jpvert.github.io/) and Prof. [Alessandro Rudi](https://www.di.ens.fr/~rudi/) for introducing us to this field. 

For any further questions, please feel free to leave a comment or contact us by mail!
