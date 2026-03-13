import { defineConfig } from 'vitepress'
// https://vitepress.dev/reference/site-config

// 1. 获取环境变量并判断
// 如果环境变量 EDGEONE 等于 '1'，说明在 EdgeOne 环境，使用根路径 '/'
// 否则默认是 GitHub Pages 环境，使用仓库子路径 '/easy-data-x-ai-for-devs/'
const isEdgeOne = process.env.EDGEONE === '1'
const baseConfig = isEdgeOne ? '/' : '/easy-data-x-ai-for-devs/'

export default defineConfig({
  lang: 'zh-CN',
  title: "Easy Data X AI for Devs",
  description: "开发者友好的数据与 AI 入门教程",
  base: baseConfig,
  markdown: {
    math: true
  },
  themeConfig: {
    // https://vitepress.dev/reference/default-theme-config
    logo: '/datawhale-logo.png',
    nav: [
      { text: 'PDF 版本下载', link: 'https://github.com/liboyang0730/easy-data-x-ai-for-devs/releases' },
    ],
    search: {
      provider: 'local',
      options: {
        translations: {
          button: {
            buttonText: '搜索文档',
            buttonAriaLabel: '搜索文档'
          },
          modal: {
            noResultsText: '无法找到相关结果',
            resetButtonTitle: '清除查询条件',
            footer: {
              selectText: '选择',
              navigateText: '切换'
            }
          }
        }
      }
    },
    sidebar: [
      { text: '《Easy Data X AI 课程介绍》', link: '/course-intro' },
      {
        text: '公共基础',
        collapsed: false,
        items: [
          { text: 'F1：AI 必知必会（一）—— 大模型的本质与边界', link: '/chapter1/F1：AI 必知必会（一） —— 大模型的本质与边界' },
          { text: 'F2：AI 必知必会（二）—— AI Agent 全景图', link: '/chapter1/F2：AI 必知必会（二） —— AI Agent 全景图' }
        ]
      },
      {
        text: '道篇',
        collapsed: false,
        items: [
          { text: 'P1：AI Agent 场景识别', link: '/daopian/P1' },
          { text: 'P2：Agentic RAG 产品设计', link: '/daopian/P2' },
          { text: 'P3：Agent 记忆系统设计', link: '/daopian/P3' },
          { text: 'P4：Skill 与知识管理', link: '/daopian/P4' },
          { text: 'P5：综合案例与度量', link: '/daopian/P5' }
        ]
      },
      {
        text: '术篇',
        collapsed: false,
        items: [
          { text: 'D1：大模型 API 基础', link: '/shupian/D1' },
          { text: 'D2：AI 应用的数据层', link: '/shupian/D2' },
          { text: 'D3：Agentic RAG 实战', link: '/shupian/D3' },
          { text: 'D4：Agent 开发与记忆系统', link: '/shupian/D4' },
          { text: 'D5：Skill、MCP 与综合项目', link: '/shupian/D5' }
        ]
      }
    ],

    socialLinks: [
      { icon: 'github', link: 'https://github.com/liboyang0730/easy-data-x-ai-for-devs' }
    ],

    editLink: {
      pattern: 'https://github.com/liboyang0730/easy-data-x-ai-for-devs/blob/main/docs/:path'
    },

    footer: {
      message: 'Built with VitePress',
      copyright: 'Licensed under CC BY-NC-SA 4.0'
    }
  }
})
