# Awesome Golang.AI

[![GitHub stars](https://img.shields.io/github/stars/promacanthus/awesome-golang-ai?style=social)](https://github.com/promacanthus/awesome-golang-ai)
[![GitHub forks](https://img.shields.io/github/forks/promacanthus/awesome-golang-ai?style=social)](https://github.com/promacanthus/awesome-golang-ai)
[![License](https://img.shields.io/github/license/promacanthus/awesome-golang-ai)](LICENSE)
[![GitHub last commit](https://img.shields.io/github/last-commit/promacanthus/awesome-golang-ai)](https://github.com/promacanthus/awesome-golang-ai/commits/main)

---

**Language** | **语言**  
[English](README.md) | [中文](README-zh.md)

---

## 概述

> 如需快速了解，请参阅 [Overview.md](Overview.md)。

本仓库是一个精选的资源列表，专注于使用 Go 编程语言进行人工智能（AI）和机器学习（ML）开发。它遵循流行的"awesome 列表"格式，将高质量、经过社区验证的工具、库、框架、基准测试和教育材料汇聚到一个有组织的参考资源中。

本仓库不包含可执行代码或软件项目。相反，它作为一个发现和参考工具，为对使用 Go 进行 AI 相关任务感兴趣的开发者、研究人员和工程师提供服务。该列表强调了 Go 在性能、并发性和系统级编程方面的优势，使其对构建可扩展的生产级 AI 应用程序特别有价值。

该仓库被划分为明确定义的类别，如基准测试、大语言模型（LLM）工具、检索增强生成（RAG）组件、通用机器学习库、神经网络和教育资源。每个条目都包含资源链接和功能或用途的简要描述。

## 收集范围

awesome-golang-ai 列表涵盖了 AI 和 ML 领域的广泛内容，特别强调实用工具和评估框架。主要类别包括：

- **[基准测试](docs/01-AI%20and%20ML%20Benchmarks.md)**：全面的评估套件，用于评估 LLM 在各个领域的能力，如代码生成、数学推理、多语言理解和现实世界软件工程任务。
- **[模型上下文协议实现](docs/02-Model%20Context%20Protocol%20Implementations.md)**：与 MCP 相关的资源，包括与主要 AI 平台（OpenAI、Google、Anthropic）交互的 SDK、用于本地模型执行的开发工具如 Ollama、代理框架和基于 Go 的 transformer 模型实现。
- **[大语言模型（LLMs）](docs/03-Large%20Language%20Model%20Tools.md)**：与 LLM 相关的资源，包括与主要 AI 平台（OpenAI、Google、Anthropic）交互的 SDK、用于本地模型执行的开发工具如 Ollama、代理框架和基于 Go 的 transformer 模型实现。
- **[RAG（检索增强生成）](docs/04-RAG%20Components.md)**：构建 RAG 管道的工具，包括用于将 PDF 和办公文件转换为结构化格式的文档解析器、嵌入模型，以及用于高效相似性搜索的向量数据库，如 Milvus 和 Weaviate。
- **[通用机器学习库](docs/05-Machine%20Learning%20Libraries.md)**：Go 中的基础 ML 库，支持回归、分类、聚类和数据操作等任务，包括 Gorgonia、Gonum 和 Golearn 等库。
- **[神经网络和深度学习](docs/05-Machine%20Learning%20Libraries.md)**：用于构建和训练神经网络的专业库，包括前馈网络、自组织映射和循环架构的实现。
- **[专业领域](docs/docs/05-Machine%20Learning%20Libraries.md)**：线性代数、概率分布、进化算法、图处理、异常检测和推荐系统的资源。
- **[教育材料](docs/06-Educational%20Resources.md)**：支持在 Go 中学习和实验 AI 的书籍、教程和数据集。

该列表还包括新兴标准，如**模型上下文协议（MCP）**，它支持 LLM 应用程序与外部工具之间的集成，突出了项目对实用、可互操作的 AI 开发的关注。

## 目标受众

本仓库的主要受众包括：

- **Go 开发者**：探索将 AI/ML 集成到应用程序中
- **机器学习工程师**：寻求用于 AI 系统的高性能、并发后端
- **研究人员**：评估 LLM 或在 Go 中构建实验管道
- **DevOps 和 MLOps 工程师**：对在生产环境中部署 AI 模型感兴趣，这些环境中 Go 的效率和可靠性具有优势
- **学生和学习者**：希望通过 Go 实现来学习 AI 概念

该列表旨在为具有不同技术专业水平的用户提供便利。虽然某些条目假设用户熟悉 AI 概念，但结构和分类允许初学者从基础库到高级框架逐步探索资源。

## 贡献指南

虽然仓库没有明确包含 `CONTRIBUTING.md` 文件或在 `README` 中详细的贡献说明，但它遵循 awesome 列表的标准做法。鼓励用户通过提交拉取请求来添加新的相关资源或改进现有条目。

理想的贡献包括：

- 高质量、积极维护的项目
- 文档完善的库和工具
- 有已发布结果或学术支持的基准测试
- 具有实用价值的教育资源

所有提交都应直接与 Go 中的 AI/ML 开发相关，并且必须提供清晰的描述和有效的链接。缺少正式指导方针表明贡献者在提出添加时应遵循列表的现有格式和结构。

## 如何使用此列表

用户可以通过多种方式利用 awesome-golang-ai 列表：

- **发现**：浏览类别以找到与特定需求相关的工具，例如用于 RAG 的向量数据库或用于调用 LLM API 的 SDK。
- **评估**：使用基准条目来比较模型性能或评估代码生成或数学推理等领域的能力。
- **学习**：访问书籍和教程以建立在 Go 中使用 AI 的基础知识。
- **开发**：将推荐的库和框架集成到项目中，以加速 AI 功能的实现。
- **研究**：利用标准化的基准测试和数据集进行可重现的实验。

分层结构允许用户从广泛的类别快速导航到特定工具，使用户无需事先了解生态系统即可轻松定位资源。

## 在 Go AI 生态系统中策展的价值

由于与 Python 等语言相比，专注于 AI 的库数量相对较少，策展在 Go AI 生态系统中起着关键作用。通过将分散的资源汇聚到一个组织良好的列表中，awesome-golang-ai 降低了对使用 Go 进行 AI 感兴趣的开发者的入门门槛。

它通过突出成熟、维护良好的项目来促进最佳实践，并通过展示创新工具和研究来鼓励社区增长。该列表还有助于识别生态系统中的差距，指导未来的开发工作。

此外，随着 Go 在后端 AI 服务、微服务和云原生应用程序中的采用率增加，拥有集中化的参考确保开发者可以高效地构建强大、可扩展的 AI 系统，而不会牺牲性能或可靠性。

## Star 历史

<!-- Copy-paste in your Readme.md file -->

<a href="https://next.ossinsight.io/widgets/official/analyze-repo-stars-history?repo_id=444070344" target="_blank" style="display: block" align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://next.ossinsight.io/widgets/official/analyze-repo-stars-history/thumbnail.png?repo_id=444070344&image_size=auto&color_scheme=dark" width="721" height="auto">
    <img alt="Star History of promacanthus/awesome-golang-ai" src="https://next.ossinsight.io/widgets/official/analyze-repo-stars-history/thumbnail.png?repo_id=444070344&image_size=auto&color_scheme=light" width="721" height="auto">
  </picture>
</a>

<!-- Made with [OSS Insight](https://ossinsight.io/) -->

## Star 地理分布

<!-- Copy-paste in your Readme.md file -->

<a href="https://next.ossinsight.io/widgets/official/analyze-repo-stars-map?repo_id=444070344&activity=stars" target="_blank" style="display: block" align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://next.ossinsight.io/widgets/official/analyze-repo-stars-map/thumbnail.png?repo_id=444070344&activity=stars&image_size=auto&color_scheme=dark" width="721" height="auto">
    <img alt="Star Geographical Distribution of promacanthus/awesome-golang-ai" src="https://next.ossinsight.io/widgets/official/analyze-repo-stars-map/thumbnail.png?repo_id=444070344&activity=stars&image_size=auto&color_scheme=light" width="721" height="auto">
  </picture>
</a>

<!-- Made with [OSS Insight](https://ossinsight.io/) -->