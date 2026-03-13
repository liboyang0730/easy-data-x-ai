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
      {
        text: '第 1 章：公共基础部分',
        collapsed: false,
        items: [
          { text: '章节概览', link: '/chapter1/' },
          { text: 'F1：AI 必知必会（一）—— 大模型的本质与边界', link: '/chapter1/F1：AI 必知必会（一） —— 大模型的本质与边界' },
          { text: 'F2：AI 必知必会（二）—— AI Agent 全景图', link: '/chapter1/F2：AI 必知必会（二） —— AI Agent 全景图' },
          { text: '第 2 章：第 2 章的标题',
            items: [
              { text: '第 2.1 节：第 2.1 节的标题', link: '/chapter2/chapter2_1' },
              { text: '第 2.2 节：第 2.2 节的标题', link: '/chapter2/chapter2_2' },
              { text: '第 2.3 节：第 2.3 节的标题', link: '/chapter2/chapter2_3' }
            ]
           }
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
