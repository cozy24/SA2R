# IEEE TCSS 审稿意见完整中文翻译

## 一、编辑决定

### 决定信

尊敬的 Chen Zhu 先生：

我们已经收到针对您所提交稿件的审稿报告以及副编辑的推荐意见，相关内容附于下方。审稿人认为，经过充分修改后，该稿件有可能达到正式论文的接收标准。因此，现要求您对稿件进行**大修（Major Revision）**，修改后的稿件将再次送审。

### 副编辑给作者的意见

作者在提交修改后的稿件之前，应认真处理并逐一回应所有审稿人的意见。

如果审稿人上传了 PDF 或文本形式的审稿附件，则需要从 Author Center 中下载，请在必要时检查 Author Center 获取更多信息。

在准备提交修改稿时，应在论文文件首页加入针对该论文编号的“作者对审稿意见的回复”，详细说明如何回应各位审稿人的意见。该回复应当作为修改稿文件的最前面几页。

期刊要求作者在收到该决定后的**一个月内**完成并提交修改稿，以便及时发表研究成果。如果修改稿超过这一时间提交，则可能被视为一篇新的投稿。如果确实需要更多时间，请联系编辑部。

---

# Reviewer 1

## 审稿建议

**Prepare A Major Revision**

即：**大修**

## 总体评价

本文提出了一种轻量且高效的自监督图异常检测方法，可用于检测社交网络和金融网络中的异常用户。

该方法被清晰地划分为三个模块。整体框架图简单且易于理解。论文还提供了详细的伪代码以及代码仓库链接，这使得该方法更加容易复现。

在八个真实世界数据集上，该方法仅在其中一个数据集上的性能略低于最佳基线方法，这表明该方法具有良好的泛化能力。

作者还进行了大量实验来证明该方法的有效性。

然而，以下问题仍需要解决：

### 意见 1

该方法各模块中使用的大多数技术已经得到过较为充分的研究。整体方法更像是对已有技术的组合，因此其创新性有限。

例如：

* Bernstein 谱滤波
* 属性重构
* 噪声重构
* 谱视图与属性视图对齐

这些技术都已经是相对成熟的方法。

### 意见 2

虽然作者提供了源代码，但代码仓库中的使用说明不够详细。

此外，代码仓库中没有提供 `requirements.txt` 文件。

论文虽然给出了一些参数设置，但没有描述软件运行环境，例如：

* Python 版本
* PyTorch 版本

这些问题增加了方法复现的难度。

### 意见 3

在 Algorithm 1 中，作者只使用了一种滤波器组来提取低频、中频和高频信息。

作者可以进一步与其他类型的滤波器进行比较。

### 意见 4

在方法部分，作者将图视为无向图，并将其特征值缩放到 ([0,1]) 范围。

然而，作者在数据集介绍中将 Elliptic 数据集描述为一个有向图。

论文没有解释在处理 Elliptic 数据集之前，是否将其转换为了无向图。

### 意见 5

Related Work 部分的 subsection (d) 中存在一个拼写错误。

单词 **“methods”** 拼写错误。

## 其他评价项

审稿人在以下项目中未填写具体内容：

* Summary of Evaluation：未填写
* Organization：未填写
* Clarity：未填写
* Length：未填写
* References：未填写
* Correctness：未填写
* Significance：未填写
* Originality：未填写
* Attachments：未填写
* If Survey Coverage：未填写
* Contribution：未填写

## 对“论文贡献是什么？”的回答

未填写。

## 对“论文还可以通过哪些方式改进？”的回答

未填写。

## 建议补充的参考文献

**NA**

即：没有额外建议的参考文献。

---

# Reviewer 2

## 审稿建议

**Prepare A Major Revision**

即：**大修**

## 意见 1

对齐模块使用负采样来构造对比样本对。

然而，论文并没有充分说明具体的采样策略。

例如，从彼此无关类别中采样节点可能会人为放大节点之间的差异，而从结构上相似的节点中进行采样，则可能提供更加有意义的对比关系。

## 意见 2

噪声重构模块向节点属性中注入扰动，并在图结构的指导下对这些扰动进行重构。

然而，由于存在虚假相关性（spurious correlations），异常节点是否也有可能很好地重构这些噪声？

## 意见 3

由于数据集的选择会显著影响异常检测的性能，因此作者应该提供一个更加全面的数据集总结，并解释为什么所选择的这些数据集能够代表真实世界中的异常检测场景。

## 意见 4

为了丰富论文内容，可以进一步综述更多相关工作，例如：

**Self-supervised semantic graph propagation for multi-view clustering**

DOI：

`10.1016/j.neunet.2026.108973`

## 意见 5

目前已经存在多种同时结合对比学习目标与重构目标的混合方法。

除了具体使用谱滤波这一点之外，SA2R 与这些已有方法相比，在本质上有什么不同？

## 意见 6

最终的异常分数结合了：

* 对齐差异
* 属性重构误差
* 噪声重构误差

那么，是否能够进一步区分一个异常究竟来源于：

* 结构不一致
* 恢复能力较差

这两种不同原因？

## 意见 7

论文中关于异配性（heterophily）的描述可以通过一个简单的示例图进行说明，从而使这一概念更加直观、具体。

## 意见 8

论文缺少以下方面的详细信息：

* 超参数
* 训练 epoch 数
* 优化设置

## 其他评价项

以下项目未填写：

* Summary of Evaluation
* Organization
* Clarity
* Length
* References
* Correctness
* Significance
* Originality
* Attachments
* If Survey Coverage
* Contribution

## 对“论文贡献是什么？”的回答

未填写。

## 对“论文还可以通过哪些方式改进？”的回答

未填写。

## 建议补充的参考文献

审稿人再次明确建议加入：

**Self-supervised semantic graph propagation for multi-view clustering**

DOI：

`10.1016/j.neunet.2026.108973`

---

# Reviewer 3

## 审稿建议

**Accept With Minor Changes**

即：**小修后接收**

## 总体评价

SA2R 引入了属性重构和噪声重构目标，以进行可恢复性学习。

在推理阶段，最终异常分数结合了：

* 跨视图不一致性
* 属性重构误差
* 噪声重构误差

在八个真实世界数据集上进行的大量实验表明，SA2R 在其中七个数据集上的性能优于当前先进的自监督图异常检测方法。

可以增加一些新的参考文献，以提高论文内容的全面性。

## 评分

* Summary of Evaluation：**Good**
* Organization：**4**
* Clarity：**4**
* Length：**4**
* References：**4**
* Correctness：**4**
* Significance：**4**
* Originality：**4**
* Attachments：**4**
* If Survey Coverage：**4**
* Contribution：**4**

## 对“论文贡献是什么？”的回答

作者提出了 SA2R，这是一种用于**无标签图异常检测（label-free GAD）**的自监督框架。

SA2R 的设计受到这样一个经验观察的启发：

在多个真实世界图中，异常节点往往表现出更高的节点级异配性（node-level heterophily）。

基于这一观察，SA2R 通过联合建模以下两个方面来识别异常实体：

1. 图结构与节点属性之间的不一致性
2. 节点级可恢复性

## 对“论文还可以通过哪些方式改进？”的回答

可以增加一些新的参考文献。

## 建议补充的参考文献

审稿人建议加入：

Muhammad Shoaib Khan, Chen Hongsong.

**Hybrid transformer deep neural architectures for enhanced misinformation detection on social media.**

Expert Systems With Applications, 2026, Volume 300, Article 130470.

DOI：

`10.1016/j.eswa.2025.130470`

---

# Reviewer 4

## 审稿建议

**Prepare A Major Revision**

即：**大修**

## 总体评价

论文组织较好，写作也比较清晰，但缺乏创新性。

作者需要处理所有这些意见。

我对该论文给出的评价介于：

**Weak Reject（弱拒稿）与 Major Revision（大修）之间。**

## 评分

* Summary of Evaluation：**Fair**
* Organization：**4**
* Clarity：**3**
* Length：**3**
* References：**4**
* Correctness：**3**
* Significance：**2**
* Originality：**1**
* Attachments：未填写
* If Survey Coverage：未填写
* Contribution：**2**

## 对“论文贡献是什么？”的回答

这篇论文没有显著的贡献。

它属于一种**增量式工作（incremental work）**，而不能算作一种新的方法。

## 对“论文还可以通过哪些方式改进？”的回答

未填写。

## 建议补充的参考文献

审稿人建议加入以下两篇文献。

### 文献 1

Wasim Khan, Nadhem Ebrahim.

**ANOGAT-Sparse-TL: A hybrid framework combining sparsification and graph attention for anomaly detection in attributed networks using the optimized loss function incorporating the Twersky loss for improved robustness.**

Knowledge-Based Systems, Volume 311, 2025, 113144.

ISSN 0950-7051.

DOI：

`10.1016/j.knosys.2025.113144`

### 文献 2

Khan, W., Ebrahim, N., Alsaadi, M. et al.

**Unified representation and scoring framework for anomaly detection in attributed networks with emphasis on structural consistency and attribute integrity.**

Scientific Reports, Volume 15, Article 35753, 2025.

DOI：

`10.1038/s41598-025-19650-y`

---

# Reviewer 5

## 审稿建议

**Prepare A Major Revision**

即：**大修**

## 总体意见

作者应该仔细检查论文的格式，尤其需要关注以下问题。

### 格式问题 1

作者姓名中出现了一个不合适的符号：

**“），”**

需要进行修正。

### 格式问题 2

正文中有很多地方出现了**单独一个单词占据一整行**的情况。

这些属于相对轻微的排版和写作问题。

然而，下面的问题明显影响了论文的可读性。

### 可读性问题 1

几乎所有公式后面都没有使用 **“where”** 对数学符号进行解释。

作者应该在每个公式之后立即定义其中出现的所有变量和符号。

### 可读性问题 2

Algorithm 1 的内容与正文之间的衔接不够充分。

作者应该清楚说明：

* Algorithm 1 对应于所提出方法的哪一个章节或模块
* Algorithm 1 在整体框架中承担什么作用

### 可读性问题 3

论文中包含过多的算法流程。

这些算法流程打断了论文的叙述逻辑，并降低了论文的可读性。

建议作者在适当的地方对这些算法流程进行合并或简化。

### 可读性问题 4

对于 Table II 中列出的基线方法，可以直接在每个方法名称之后给出：

* 对应的引用
* 发表年份

这样能够更加直观地体现所比较方法的新近程度，同时也可以取消目前单独用于说明方法来源的那一列。

### 可读性问题 5

作者应该详细解释为什么 **SmoothGNN** 在实验中的表现如此差。

## 评分

* Summary of Evaluation：**Good**
* Organization：**3**
* Clarity：**3**
* Length：**2**
* References：**3**
* Correctness：**3**
* Significance：**3**
* Originality：**3**
* Attachments：**3**
* If Survey Coverage：**3**
* Contribution：**4**

## 对“论文贡献是什么？”的回答

未填写。

## 对“论文还可以通过哪些方式改进？”的回答

未填写。

## 建议补充的参考文献

**NA**

即：没有额外建议的参考文献。

---

# Reviewer 6

## 审稿建议

**Prepare A Major Revision**

即：**大修**

## 意见 1

用于实验评估的数据集涵盖：

* 社交网络图
* 评论网络图
* 电子商务图
* 金融图

然而，这些数据集在论文中主要被当作通用异常检测基准数据集进行处理。

论文没有：

* 对某一种具体的社会机制进行建模
* 分析社区行为
* 分析协同行为（coordinated behavior）
* 讨论相关的社会影响或社会意义

为了充分证明该工作与 TCSS 期刊研究范围之间的契合度，需要增加一个详细的社交网络案例研究，并提供更深入的行为层面解释。

## 意见 2

对于 TCSS 而言，在：

* Weibo
* Yelp
* Amazon

等数据集上开展一个社交网络案例研究，会比完全依赖 **T-Finance** 这一金融数据集更加合适。

## 意见 3

论文没有清楚说明，在没有异常标签的情况下，是如何选择以下设置的：

* early stopping
* hidden dimensions
* score-balancing parameter
* loss weights
* 其他超参数

在作者所声称的**无标签设置（label-free setting）**下，目前论文对于超参数选择和模型选择协议的描述并不充分。

## 意见 4

SA²R 结合了以下技术：

* 多项式谱滤波
* 对比式跨视图对齐
* 属性重构
* 受扩散模型启发的噪声重构

目前所提出的框架看起来像是多个已有模块的组合。

由于这些技术分别都已经是已有技术，因此作者必须清楚解释：

1. 除了组合已有模块之外，SA²R 独特的技术贡献究竟是什么；
2. SA²R 与已有的、密切相关的**结构与属性不一致性方法（structure-attribute inconsistency methods）**相比有什么区别。

## 意见 5

此外，论文中还存在若干拼写、语法和一致性方面的问题，需要修正。

具体包括：

### Figure 2

“Edeg relation 2”

应修改为：

“Edge relation 2”

### Figure 6

论文对于 Figure 6 的讨论错误地声称：

(\alpha) 控制的是 **“contrastive objective”**。

然而 Equation 32 表明，(\alpha) 实际控制的是**推理阶段异常分数的组合**。

### 写作问题

论文中还存在：

* 多处表达不自然的句子
* 大小写使用不一致

因此需要对全文进行全面校对。

## 意见 6

Related Work 部分目前主要以描述性介绍为主，对于与本文最相关的以下研究方向讨论得还不够深入：

* 基于异配性的图异常检测方法
* 结构与属性不一致性方法
* 谱方法
* 扩散与去噪方法
* 多关系图异常检测方法

由于**谱滤波**是本文方法的核心组成部分，因此 Related Work 中应该专门讨论：

1. 谱图神经网络（spectral graph neural networks）
2. 异配图环境下的谱方法（spectral methods under heterophily）

## 意见 7

论文中包含大量描述标准操作的公式。

标准操作本身可以保留，但论文应该清楚地区分：

* 已有的标准公式或已有方法
* 本文真正提出的新组件

从而使读者能够清楚识别本文真正的技术创新。

## 意见 8

论文中的三个算法过长，并且重复描述了 Methodology 中已经给出的数学公式。

作者应该显著简化这些伪代码，并直接引用已有公式。

更推荐的方式是：

提供**一个完整的、覆盖整体训练和推理过程的算法**，而非针对每个阶段分别给出三个独立算法。

## 意见 9

当前用于比较的基线方法以及表格中列出的方法都只来自会议论文。

从期刊论文的标准来看，目前的基线选择还不够全面。

作者必须加入具有代表性的近期**期刊或 Transactions 方法**，尤其应该考虑以下方向：

* 自监督图异常检测
* 结构与属性不一致性
* 谱图学习
* 大规模图分析

如果某些相关方法没有加入比较，则作者需要清楚解释为什么将这些方法排除在实验比较之外。

## 意见 10

论文中存在多处段落之间衔接突然的问题。

## 意见 11

论文没有报告：

**mean ± standard deviation**

即：

**均值 ± 标准差**

然而，论文中明确说明每种方法都运行了十次。

仅报告平均值是不充分的，而且只能提供较为模糊的实验信息。

## 意见 12

T-Finance 数据集上的 AUPRC 提升非常大：

从：

**0.1448**

提高到：

**0.6300**

这一结果值得质疑，作者必须提供相应的支持性证据。

## 意见 13

### Figure 7

Figure 7 提供的信息不足以证明当前的效率比较是完全公平的。

### Figure 8

Figure 8 中存在符号不一致问题。

图注将：

(\alpha)

称为拟合得到的幂律指数（fitted power-law exponent）。

然而，图中和正文中使用的符号却是：

(\lambda)

需要统一这一符号表示。

## 意见 14

在 Equation 27 中，作者描述了如何生成 FiLM 的：

* scale 参数
* shift 参数

然而，论文中没有对 **FiLM** 本身进行介绍。

作者必须：

1. 说明 FiLM 是否属于一种已有且成熟的方法；
2. 在 FiLM 第一次出现时给出其完整名称；
3. 解释 FiLM 的基本原理；
4. 给出合适的参考文献。

## 评分

* Summary of Evaluation：**Fair**
* Organization：**2**
* Clarity：**2**
* Length：**3**
* References：**3**
* Correctness：**3**
* Significance：**4**
* Originality：**3**
* Attachments：**3**
* If Survey Coverage：**3**
* Contribution：**2**

## 对“论文贡献是什么？”的回答

本文提出了 SA2R，一个自监督图异常检测框架。

该框架通过：

* 谱对齐学习
* 重构学习

联合建模：

1. 图结构与节点属性之间的不一致性
2. 节点的可恢复性

从而识别异常节点。

## 对“论文还可以通过哪些方式改进？”的回答

作者需要认真处理审稿意见中提出的所有问题。

## 建议补充的参考文献或补充意见

审稿人在参考文献建议栏中再次强调了基线选择问题：

目前的比较基线和表格中只列出了会议论文。

从期刊论文的标准来看，基线方法选择不够全面。

作者必须加入具有代表性的近期期刊或 Transactions 方法，尤其应覆盖：

* 自监督图异常检测
* 结构与属性不一致性
* 谱图学习
* 大规模图分析

如果没有加入这些方法，则需要明确解释排除这些方法的原因。
