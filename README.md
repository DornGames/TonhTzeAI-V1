# 《灌子集》LSTM 文言文仿写器

基于字符级 LSTM + Top‑p 采样的文言对话生成工具。  
自动从《灌子集校注》PDF 中提取人物对话（“灌子曰：...”等），训练一个轻量级 LSTM 模型，然后生成风格相似的文言新句子。

## ✨ 特性

- **自动提取对话**：从 PDF 中识别 `[人物]曰：“...内容...”` 并清洗注释。
- **字符级 LSTM**：使用 PyTorch 实现，小巧易训练（CPU 即可运行）。
- **Top‑p (nucleus) 采样**：生成时自动筛选合理候选字，避免混乱。
- **重复惩罚**：降低近期出现字符的概率，提升文本多样性。
- **多温度生成**：同一模型可生成保守（低温）或创造性（高温）的不同文本。

## 📦 依赖

- Python 3.8+
- PyTorch
- pdfplumber
- requests
- numpy

安装命令：

```bash
pip install torch pdfplumber requests numpy
```

## 🚀 快速开始

1. **准备 PDF 文件**  
   将《灌子集校注》PDF 文件放在与脚本相同的目录下，并确保文件名为：  
   `Tonh-Tze-Anthology-with-Correction-and-Annotation.pdf`  
   （你也可以修改 `LOCAL_PDF_PATH` 为其他路径）

2. **运行脚本**

```bash
python tonh.py
```

3. **观察输出**  
   - 脚本会先尝试从 PDF 中提取对话（优先使用本地文件）。  
   - 然后合并内置语料 + 提取的对话，开始训练 LSTM。  
   - 训练完成后，将生成两个温度（0.9 和 1.5）下的仿写文本。  
   - 可选保存训练好的模型（.pth 文件）。

## ⚙️ 主要参数说明

在 `main()` 函数中可以调整以下参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `LOCAL_PDF_PATH` | `"Tonh-Tze-Anthology-with-Correction-and-Annotation.pdf"` | 本地 PDF 路径（相对或绝对） |
| `SEQ_LEN` | 40 | 训练时使用的上下文长度（字符数） |
| `HIDDEN_SIZE` | 128 | LSTM 隐藏层维度 |
| `NUM_LAYERS` | 2 | LSTM 层数 |
| `EPOCHS` | 30 | 训练轮数 |
| `TEMPERATURES` | `[0.9, 1.5, 2.5]` | 生成时的温度值列表 |
| `TOP_P` | 0.92 | Top‑p 采样阈值 |
| `REP_PENALTY` | 1.15 | 重复惩罚因子（>1 惩罚重复） |
| `START_TEXT` | `"灌子曰："` | 生成文本的开头 |

## 📄 输出示例

训练完成后，你会看到类似下面的生成结果：

```
温度 0.9 | top-p 0.92 | rep_penalty 1.15:
灌子曰：“朕莫与铮子善！”剑子曰：“作业作毕需正之！”灌子问曰：“毕正抑毕后正？”剑子愣，后曰：“吾之 CPU 几近废矣！”

温度 1.5 | top-p 0.92 | rep_penalty 1.15:
灌子曰：“朕他妈是直的！大哥！”或论及篮球，灌子曰：“吾徒奉先得，兼容并蓄之道。”众人惊呼：“善哉！”
```

## 🗂️ 文件结构

```
.
├── guanzi_writer.py          # 主程序
├── Tonh-Tze-Anthology-with-Correction-and-Annotation.pdf   # 语料 PDF（需自行放置）
├── guanzi_lstm.pth           # 训练后保存的模型（可选）
└── README.md
```

## 📜 许可证

本项目采用 **MIT 许可证**。  
你可以自由使用、修改、分发本代码，但需保留版权声明。

```
MIT License

Copyright (c) 2025

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## ❓ 常见问题

**Q: 没有 PDF 文件怎么办？**  
A: 脚本会仅使用内置语料（约 11000 字符）进行训练，但效果可能不如加入全部对话。

**Q: 训练太慢？**  
A: 可以减小 `EPOCHS`、`HIDDEN_SIZE` 或 `REPEAT_CORPUS`。使用 GPU（设置 `DEVICE = 'cuda'`）也会显著加速。

**Q: 生成文本重复严重？**  
A: 可适当提高 `TEMPERATURES` 或降低 `REP_PENALTY`；也可以减小 `TOP_P`（如 0.85）让采样更集中。

**Q: 如何用自己训练的模型继续生成？**  
A: 取消注释保存模型的代码，下次运行时加载 `guanzi_lstm.pth` 并调用 `generate_top_p` 即可。

---

Enjoy generating classical-style dialogues with AI! 😊
