#!/usr/bin/env node
/**
 * 下载 F1、F2 课程图片到项目本地
 * 需在内网环境运行：node scripts/download-base-knowledge-images.mjs
 */
import { writeFileSync, mkdirSync, existsSync } from 'fs'
import { dirname } from 'path'
import { fileURLToPath } from 'url'

const __dirname = dirname(fileURLToPath(import.meta.url))
const OUTPUT_DIR = `${__dirname}/../docs/public/images/base_knowledge`

const IMAGES = [
  // F1
  { url: 'https://intranetproxy.alipay.com/skylark/lark/0/2026/png/132317/1773319080062-3965b637-81b9-4ff9-8f90-10afd2bac10b.png', file: 'F1-01.png' },
  { url: 'https://intranetproxy.alipay.com/skylark/lark/0/2026/png/132317/1773317900600-16a28242-df16-4200-aa24-f56eaed1e46d.png', file: 'F1-02.png' },
  { url: 'https://intranetproxy.alipay.com/skylark/lark/0/2026/png/132317/1773390923824-d7414660-bbe5-45c9-9568-692dabc89768.png', file: 'F1-03.png' },
  { url: 'https://intranetproxy.alipay.com/skylark/lark/0/2026/png/132317/1773318615446-950309c8-abc9-4803-8842-ef111c12e9e1.png', file: 'F1-04.png' },
  { url: 'https://intranetproxy.alipay.com/skylark/lark/0/2026/webp/132317/1773302317491-e6e1eac8-5ffd-4c3a-b630-3a070a3c7240.webp', file: 'F1-05.webp' },
  { url: 'https://intranetproxy.alipay.com/skylark/lark/0/2026/png/132317/1773391201779-86a6a48d-48df-46aa-b2e7-8ec7077c872c.png', file: 'F1-06.png' },
  // F2
  { url: 'https://intranetproxy.alipay.com/skylark/lark/0/2026/png/132317/1773383184491-4482c958-855e-4756-bf7b-aff631b2b5bf.png', file: 'F2-01.png' },
  { url: 'https://intranetproxy.alipay.com/skylark/lark/0/2026/png/132317/1773383212250-41a212ce-e697-4bf4-a7a8-836e39f386d5.png', file: 'F2-02.png' },
  { url: 'https://intranetproxy.alipay.com/skylark/lark/0/2026/png/132317/1773383305161-c21dce82-ec23-42ab-8493-205d26780462.png', file: 'F2-03.png' },
  { url: 'https://intranetproxy.alipay.com/skylark/lark/0/2026/png/132317/1773383329056-1838e195-1c2e-444c-b0cc-225b448f2314.png', file: 'F2-04.png' },
  { url: 'https://intranetproxy.alipay.com/skylark/lark/0/2026/png/132317/1773383351226-7d96d925-f3be-4517-aba3-4616658451fb.png', file: 'F2-05.png' },
  { url: 'https://intranetproxy.alipay.com/skylark/lark/0/2026/png/132317/1773373523817-b44373f4-dee7-408d-8d0e-ef5f196ff39c.png', file: 'F2-06.png' },
  { url: 'https://intranetproxy.alipay.com/skylark/lark/0/2026/png/132317/1773383399169-3705570b-8525-4ee9-97c6-df793d30d479.png', file: 'F2-07.png' },
  { url: 'https://intranetproxy.alipay.com/skylark/lark/0/2026/png/132317/1773383516284-a13e83ae-df0e-4a54-ba19-16ba3f6f2414.png', file: 'F2-08.png' },
  { url: 'https://intranetproxy.alipay.com/skylark/lark/0/2026/png/132317/1773383559409-8e665fe3-f933-4f64-a8a0-7e4cd81e7419.png', file: 'F2-09.png' },
  { url: 'https://intranetproxy.alipay.com/skylark/lark/0/2026/png/132317/1773382281855-7e23c7c6-2b81-4ef0-a941-db981795b06c.png', file: 'F2-10.png' },
  { url: 'https://intranetproxy.alipay.com/skylark/lark/0/2026/png/132317/1773382354037-76ccf5dc-a78c-426f-8188-fded158e6db9.png', file: 'F2-11.png' },
  { url: 'https://intranetproxy.alipay.com/skylark/lark/0/2026/png/132317/1773383631494-d4c40d7d-1198-41e6-a563-ba678e3cb60a.png', file: 'F2-12.png' },
  { url: 'https://intranetproxy.alipay.com/skylark/lark/0/2026/png/132317/1773382506361-5a88fccf-2d91-46e8-9cb4-fe9d79153925.png', file: 'F2-13.png' },
  { url: 'https://intranetproxy.alipay.com/skylark/lark/0/2026/png/132317/1773383968402-521c6f8b-f444-41fa-a10a-1c6aaa5c92b6.png', file: 'F2-14.png' },
]

async function download (url, filepath) {
  const res = await fetch(url)
  if (!res.ok) throw new Error(`HTTP ${res.status}: ${url}`)
  const buf = await res.arrayBuffer()
  writeFileSync(filepath, Buffer.from(buf))
}

async function main () {
  mkdirSync(OUTPUT_DIR, { recursive: true })
  for (const { url, file } of IMAGES) {
    const filepath = `${OUTPUT_DIR}/${file}`
    if (existsSync(filepath)) {
      console.log(`跳过（已存在）: ${file}`)
      continue
    }
    try {
      await download(url, filepath)
      console.log(`已下载: ${file}`)
    } catch (e) {
      console.error(`失败 ${file}: ${e.message}`)
    }
  }
  console.log('完成')
}

main().catch(console.error)
