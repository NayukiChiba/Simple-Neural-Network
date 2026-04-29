import { defineConfig } from "vitepress"

export default defineConfig({
  base: "/Simple-Neural-Network/",
  lang: "zh-CN",
  title: "Simple Neural Network",
  description: "基于 NumPy 从零实现神经网络 — 从数学原理到代码实战",

  head: [
    [
      "link",
      { rel: "icon", href: "/Simple-Neural-Network/favicon.ico" },
    ],
  ],

  markdown: {
    math: true, // 内置 MathJax 3 渲染
  },

  themeConfig: {
    nav: [
      { text: "首页", link: "/" },
      { text: "快速开始", link: "/guide/getting-started" },
      {
        text: "数学基础",
        items: [
          { text: "感知机与线性变换", link: "/math/perceptron" },
          { text: "激活函数", link: "/math/activations" },
          { text: "前向传播", link: "/math/forward-propagation" },
          { text: "损失函数", link: "/math/loss-functions" },
          { text: "反向传播", link: "/math/backward-propagation" },
          { text: "梯度下降优化", link: "/math/gradient-descent" },
          { text: "Xavier 初始化", link: "/math/xavier-init" },
        ],
      },
      {
        text: "架构设计",
        items: [
          { text: "项目总览", link: "/architecture/overview" },
          { text: "BaseLayer 接口", link: "/architecture/base-layer" },
          { text: "数据流", link: "/architecture/data-flow" },
        ],
      },
      {
        text: "API 参考",
        items: [
          { text: "Layers", link: "/api/layers" },
          { text: "Losses", link: "/api/losses" },
          { text: "Models", link: "/api/models" },
          { text: "Optimizers", link: "/api/optimizers" },
          { text: "Training", link: "/api/training" },
          { text: "Metrics", link: "/api/metrics" },
          { text: "Data", link: "/api/data" },
          { text: "Persistence", link: "/api/persistence" },
          { text: "Config", link: "/api/config" },
        ],
      },
      { text: "示例", link: "/examples/xor-classification" },
    ],

    sidebar: {
      "/guide/": [
        { text: "快速开始", link: "/guide/getting-started" },
        { text: "快速示例", link: "/guide/quick-example" },
      ],
      "/math/": [
        { text: "感知机与线性变换", link: "/math/perceptron" },
        { text: "激活函数", link: "/math/activations" },
        { text: "前向传播", link: "/math/forward-propagation" },
        { text: "损失函数", link: "/math/loss-functions" },
        { text: "反向传播", link: "/math/backward-propagation" },
        { text: "梯度下降优化", link: "/math/gradient-descent" },
        { text: "Xavier 初始化", link: "/math/xavier-init" },
      ],
      "/architecture/": [
        { text: "项目总览", link: "/architecture/overview" },
        { text: "BaseLayer 接口", link: "/architecture/base-layer" },
        { text: "数据流", link: "/architecture/data-flow" },
      ],
      "/api/": [
        { text: "Layers — 网络层", link: "/api/layers" },
        { text: "Losses — 损失函数", link: "/api/losses" },
        { text: "Models — 模型", link: "/api/models" },
        { text: "Optimizers — 优化器", link: "/api/optimizers" },
        { text: "Training — 训练器", link: "/api/training" },
        { text: "Metrics — 评估指标", link: "/api/metrics" },
        { text: "Data — 数据模块", link: "/api/data" },
        { text: "Persistence — 持久化", link: "/api/persistence" },
        { text: "Config — 配置", link: "/api/config" },
      ],
      "/datasets/": [
        { text: "XOR 数据集", link: "/datasets/xor" },
        { text: "Spiral 数据集", link: "/datasets/spiral" },
        { text: "Sine 数据集", link: "/datasets/sine" },
      ],
      "/training-guide/": [
        { text: "训练循环详解", link: "/training-guide/training-loop" },
        { text: "评估指标", link: "/training-guide/metrics-and-evaluation" },
        { text: "模型持久化", link: "/training-guide/model-persistence" },
      ],
      "/examples/": [
        { text: "XOR 二分类", link: "/examples/xor-classification" },
        { text: "Spiral 三分类", link: "/examples/spiral-classification" },
        { text: "Sine 回归", link: "/examples/sine-regression" },
      ],
      "/development/": [
        { text: "测试", link: "/development/testing" },
        { text: "代码规范", link: "/development/code-style" },
        { text: "贡献指南", link: "/development/contributing" },
      ],
    },

    socialLinks: [
      {
        icon: "github",
        link: "https://github.com/NayukiChiba/Simple-Neural-Network",
      },
    ],

    search: {
      provider: "local",
    },

    outline: {
      level: [2, 3],
      label: "本页目录",
    },

    docFooter: {
      prev: "上一页",
      next: "下一页",
    },

    darkModeSwitchLabel: "主题",
    sidebarMenuLabel: "菜单",
    returnToTopLabel: "返回顶部",

    lastUpdated: {
      text: "最后更新",
    },
  },
})
