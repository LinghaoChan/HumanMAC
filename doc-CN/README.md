# HumanMAC

**为响应[开放共享科研记录行动倡议(DCOX)](https://mmcheng.net/docx/)，本工作将提供中文文档，为华人学者科研提供便利。**

"HumanMAC: 基于掩码图像补全的人体动作预测"开源代码。

[陈凌灏](https://lhchen.top/)\*<sup>1</sup>, 张嘉伟*<sup>2</sup>, [李晔文](https://scholar.google.com/citations?user=W5796yEAAAAJ)<sup>3</sup>, [庞义人](https://www.linkedin.com/in/yrpang/)<sup>2</sup>, [夏晓波](https://xiaoboxia.github.io/)<sup>4</sup>, [刘同亮](https://tongliang-liu.github.io/)<sup>4</sup>

<sup>1</sup>清华大学, <sup>2</sup>西安电子科技大学, <sup>3</sup>南洋理工大学, <sup>4</sup>悉尼大学

[[项目主页](https://lhchen.top/Human-MAC/)] | [[预印本](https://arxiv.org/abs/2302.03665)] | [[README-en](../README.md)]

> 人体动作预测是计算机视觉和计算机图形学中的经典问题，具有广泛的实际应用。 之前的工作基于编码-解码的样式实现了较好的预测性能。 这些方法首先将先前的动作编码为潜在表示，然后将潜在表示解码为预测动作。 然而，在实践中，由于损失函数约束复杂、训练过程繁琐、预测中不同类别动作的切换稀少等问题，它们仍然不尽如人意。 在本文中，针对上述问题，我们跳出上述的建模范式，从新的角度提出了一个新颖的框架。 具体来说，我们的框架以去噪扩散模型的方式工作。 在训练阶段，我们学习了一个人体动作扩散模型，该模型从随机噪声中生成人体动作。 在推理阶段，通过去噪过程，我们对观察到的动作进行动作预测调节，以输出更连续和可控的预测。 我们提出的框架具有较大发展潜力，它只需要优化一个损失函数，并且以端到端的方式进行训练。 此外，它有效地完成了不同类别动作的切换，这在现实任务中具有重要意义，例如动画任务。 基准测试的综合实验证实了所提出框架的优越性。 项目主页： https://lhchen.top/Human-MAC。

⚡ 代码即将开源...



联系方式：thu DOT lhchen AT gmail DOT cơm
