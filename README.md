# HumanMAC

Code for "HumanMAC: Masked Motion Completion for Human Motion Prediction".

[Ling-Hao Chen](https://lhchen.top/)\*<sup>1</sup>, Jiawei Zhang\*<sup>2</sup>, [Yewen Li](https://scholar.google.com/citations?user=W5796yEAAAAJ)<sup>3</sup>, [Yiren Pang](https://www.linkedin.com/in/yrpang/)<sup>2</sup>, [Xiaobo Xia](https://xiaoboxia.github.io/)<sup>4</sup>, [Tongliang Liu](https://tongliang-liu.github.io/)<sup>4</sup>

<sup>1</sup>Tsinghua University, <sup>2</sup>Xidian University, <sup>3</sup>Nanyang Technological University, <sup>4</sup>The University of Sydney

[[Project Page](https://lhchen.top/Human-MAC/)] | [[Preprint]()] | [[中文文档]()]

> Human motion prediction is a classical problem in computer vision and computer graphics, which has a wide range of practical applications. Previous effects achieve great empirical performance based on an encoding-decoding style. The methods of this style work by first encoding previous motions to latent representations and then decoding the latent representations into predicted motions. However, in practice, they are still unsatisfactory due to several issues, including complicated loss constraints, cumbersome training processes, and scarce switch of different categories of motions in prediction. In this paper, to address the above issues, we jump out of the foregoing style and propose a novel framework from a new perspective. Specifically, our framework works in a denoising diffusion style. In the training stage, we learn a motion diffusion model that generates motions from random noise. In the inference stage, with a denoising procedure, we make motion prediction conditioning on observed motions to output more continuous and controllable predictions. The proposed framework enjoys promising algorithmic properties, which only needs one loss in optimization and is trained in an end-to-end manner. Additionally, it accomplishes the switch of different categories of motions effectively, which is significant in realistic tasks, e.g., the animation task. Comprehensive experiments on benchmarks confirm the superiority of the proposed framework. The project page is available at https://lhchen.top/HumanMAC.

⚡ The code is coming soon ...
