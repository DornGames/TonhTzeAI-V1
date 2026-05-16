#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
《灌子集》对话提取 + LSTM 仿写器
从 PDF 中自动提取人物对话，再使用字符级 LSTM 模仿其文言对话风格。
"""

import re
import random
import time
import requests
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

try:
    import pdfplumber
except ImportError:
    raise ImportError("请先安装 pdfplumber：pip install pdfplumber")

# ------------------------- 1. 从 PDF 提取对话 -------------------------
def extract_dialogues_from_pdf(pdf_url, local_pdf_path=None):
    """
    从 PDF 中提取《灌子集》的人物对话。
    如果提供了 local_pdf_path 且文件存在，则使用本地文件，否则尝试从网络下载。
    """
    # 优先使用本地文件
    if local_pdf_path and Path(local_pdf_path).exists():
        pdf_path = local_pdf_path
        print(f"使用本地 PDF：{pdf_path}")
    else:
        print(f"下载 PDF：{pdf_url}")
        # 若需绕过 SSL 证书验证，可加上 verify=False
        resp = requests.get(pdf_url, timeout=30, verify=False)
        resp.raise_for_status()
        pdf_path = Path("/tmp/temp_tze.pdf")
        pdf_path.write_bytes(resp.content)

    full_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                full_text += text + "\n"

    characters = [
        "灌子", "剑子", "小白子", "兰花子", "李博士", "盛云子",
        "云铮子", "玉铮子", "鱼", "铮子", "小通讯员", "鹏子",
        "李婆罗门", "汪子", "香樟子", "战鲁子", "梅花子", "武宗子",
        "升华子", "江城子", "淑惠子", "建鹏子", "大声江", "菠萝子",
        "主干子", "金瓶子", "甲鱼子", "史官陈", "史官高", "瑞丰子",
        "黄巾子", "晓江子", "佳婧子", "晴子", "士永子", "瑾子",
        "春子", "徐子", "坝子", "裴子"
    ]

    pattern = r"(" + "|".join(re.escape(c) for c in characters) + r")曰：“(.*?)”"
    matches = re.findall(pattern, full_text, re.DOTALL)

    dialogues = []
    for speaker, content in matches:
        cleaned = re.sub(r"【.*?】", "", content)
        cleaned = re.sub(r"\(.*?\)", "", cleaned)
        cleaned = re.sub(r"\s+", "", cleaned)
        if cleaned.strip():
            dialogues.append(f"{speaker}曰：“{cleaned}”")

    seen = set()
    unique = []
    for d in dialogues:
        if d not in seen:
            seen.add(d)
            unique.append(d)

    print(f"从 PDF 中共提取 {len(unique)} 条对话")
    return unique


# ------------------------- 2. 语料准备 -------------------------
DEFAULT_CORPUS = """
𣿅子曰：“朕莫与铮子善！”
剑子曰：“作业作毕需正之！”𣿅子问曰：“毕正抑毕后正？”剑子愣，后曰：“吾之 CPU 几近废矣！”
周一、三、五放学后需自习于班中。𣿅子忿，曰：“可之于操场而习否？”
小白子谓𣿅子曰：“此小童高俊兮。”𣿅子笑而不语。
𣿅子怒曰：“校之体育锻炼时已违教育局之规矣！”
或论及篮球，𣿅子曰：“吾徒奉先得,兼容并蓄之道。”
𣿅子怒曰：“解数学于化学课，汝为甚么 attitude？”众人惊呼。
𣿅子曰：“艺乃雅者也。”
小白子谓𣿅子曰：“污蔑打击同学。”𣿅子曰：“与朕何干？”
兰花子曰：“𣿅子不擦黑板，玩忽职守，甚恶也！”𣿅子曰：“汝盖有疾。”
𣿅子曰：“教者，国事也；国者，人事也；盖教者，人事也。”
𣿅子逾墙，人惊，问为何。𣿅子曰：“学校逾垣未触条。”
兰花子疯。𣿅子曰：“须朕取除颤仪之汝欤？”
𣿅子曰：“朕恒寐于高一数学课也。”
𣿅子呼：“众肃静！”众人惊呼。
𣿅子谓鱼曰：“朕与汝以𣿅国之进贡以食。”
某欲请𣿅子发作业，𣿅子怨曰：“发你马妈呢！朕犹食饭矣！”
𣿅子曰：“主席！请停食！遗我一啖之！”
𣿅子闻𣿅国夫人曰：“𣿅子实晦气者也！”𣿅子曰：“哈哈哈哈哈！”
𣿅子曰：“朕撰书于厕。”
𣿅子曰：“朕他妈是直的！大哥！”
𣿅子曰：“汝母闭嘴可否？父为父,子为子,守长幼序！{朕乃汝父！}”
𣿅子曰：“生辰会朕止请二人耳，其一为𣿅国夫人。”
盛云子问曰：“何为阴极？”𣿅子对曰：“氢氧化铁和氢气。”赞曰：“善哉！”
𣿅子耍教棒。鱼曰：“幼态！”
𣿅子未成作业。李博士曰：“汝明日可成否？”对曰：“善。”次日，𣿅子未成，博士请{𣿅子}舞。
鱼曰：“今之时日易使人眠。”𣿅子曰：“第一节课朕已寐矣。”
或曰：“汝为 M。”𣿅子辩曰：“老子是 S！”
或论及假制服，𣿅子曰：“废矣，朕不敢遗矣，或恐未归之险罢。”
𣿅子曰：“朕正服之衣裤异号也。”
云铮子谓𣿅子曰：“吾为汝父！”𣿅子曰：“朕为汝子！”
𣿅子于课中阅《科幻世界》，鱼取之。𣿅子大呼：“呜呼！师无虽阅之也可，毋弃之！”
𣿅子之优师，寐，唾淌千里，滴于师之平板。𣿅子醒，为众人笑。
李博士曰：“铁马为汝，冰河亦为汝。”𣿅子曰：“花火文学！”
李博士问曰：“但以彼拭黑板耶？”𣿅子曰：“可用手。”而后又曰：“而剑子以手拭之。”又曰：“酒精亦可罢。”
𣿅子曰：“朕已半月为成语文作业矣。”
鱼欲取𣿅子之扇。𣿅子怒曰：“朕放从信息教室归，汝岂敢取之耶？”
𣿅子曰：“师其勿忘熄屏也，恐有火灾之险。”
𣿅子惊曰：“她居然长按！”
玉铮子曰：“地理无需刷题。”史官许之。𣿅子曰：“生物无需刷题。”
𣿅子曰：“二大以巴狼反哉！”
鱼曰：“府上将巡诸此。”𣿅子曰：“扃牖而可寐。”
剑子查物理大本，观𣿅子之册，哂之，曰：“弊邪？”𣿅子辩曰：“否。”曰：“然则只见其果未见其因者，何哉？”曰：“为之正罢！”剑子请示。𣿅子试，未果。剑子轩然曰：“复辩否？”𣿅子遂靡且怨。
盛云子至，𣿅子未拭黑板。盛云子曰：“短二分而净黑板吼！”𣿅子谢之曰：“负师为耻！”
李博士谓小通讯员曰其名，曰“芃”者音“凡”。𣿅子闻之，大喜，遂谓博士曰：“此文音‘鹏’而非‘凡’，朕以为子戏也，其真不知也！”其言不让。博士哂，从容教之曰：“其真邪？吾诚不知也。”𣿅子乐，然其愚不可自治也。嗟乎！世人尝言：“小丑竟是我自己。”此乃𣿅子之拙态乎！
𣿅子过校门于自行车上。师当之，令之徒步推之。𣿅子曰：“吾推诸车上，安下而推之？”
𣿅子曰：“礼乐未知者慎言之！”
𣿅子曰：“朕请失面。”
李博士曰：“其成数学抑诵文者乎？”𣿅子对曰：“数学。”
尚记初中之时，有一生取花草用于饰而观之者一，持于怀中，之于厕，置于男便溺池中，乐而以为戏。俄而会师至，见草于池中，大怒，随循事之来龙去脉，得有一人为之。比及数日，此生为师所广诫而训于旌旗之下，为众人笑。此生者谁？盖帝王𣿅子者也。
𣿅子置一保温杯于五楼牖，不慎击之。杯落，坠一楼，人鹏子惊之。不日，𣿅子为师所广诫而训于旌旗之下。
𣿅子谓鱼曰：“先生勿击扇，其值三十块。”
李博士欲抽诵《燕歌行》，𣿅子隐名册。李博士寻之未果，问曰：“名册安在？”𣿅子曰：“脑中。”
兰花子请诵李婆罗门之诗，众人乐。𣿅子曰：“聒噪！”
或谓𣿅子曰：“汝诚噪也！”𣿅子曰：“盍不曰兰花子噪？”
李博士曰：“因 emo 闻乐，数年之后，闻之亦 emo。”𣿅子曰：“若《好运来》者何也？”
𣿅子曰：“终矣，今无见于汪子也！”问为何，𣿅子曰：“然朕以一刻如厕也。”
消防演练，𣿅子欲乘云梯。剑子以为𣿅子者操云梯。𣿅子曰：“若朕操之，云梯即可终矣！”
𣿅子集生物作业迟，师训之。𣿅子怒曰：“然朕可记之于旗下主席台否？不爽！彼者食其无集大本之言！”
""".strip()


def load_corpus(dialogues_from_pdf):
    lines = [line.strip() for line in DEFAULT_CORPUS.split('\n') if line.strip()]
    corpus = "\n".join(lines)
    if dialogues_from_pdf:
        corpus += "\n" + "\n".join(dialogues_from_pdf)
    return corpus


# ------------------------- 3. 数据集 -------------------------
class CharSeqDataset(Dataset):
    def __init__(self, ids, seq_len):
        if len(ids) <= seq_len:
            raise ValueError("语料太短，请减小 SEQ_LEN 或增加 REPEAT_CORPUS")
        self.ids = ids
        self.seq_len = seq_len

    def __len__(self):
        return len(self.ids) - self.seq_len

    def __getitem__(self, idx):
        x = self.ids[idx:idx + self.seq_len]
        y = self.ids[idx + 1:idx + self.seq_len + 1]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


# ------------------------- 4. LSTM 模型 -------------------------
class CharLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, num_layers=1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embed(x)
        out, hidden = self.lstm(x, hidden)
        logits = self.fc(out)
        return logits, hidden


# ------------------------- 5. 训练 -------------------------
def train(model, dataloader, epochs, lr, device, grad_clip=1.0):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    loss_history = []
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        total_tokens = 0
        for xb, yb in dataloader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits, _ = model(xb)
            loss = criterion(logits.view(-1, logits.size(-1)), yb.view(-1))
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            total_loss += loss.item() * yb.numel()
            total_tokens += yb.numel()
        avg_loss = total_loss / total_tokens
        loss_history.append(avg_loss)
        if epoch % max(1, epochs // 5) == 0 or epoch == epochs:
            print(f"Epoch {epoch:3d}/{epochs}   loss = {avg_loss:.4f}")
    return loss_history


# ------------------------- 6. 生成（Top‑p 采样） -------------------------
def generate_top_p(model, start_text, char_to_id, id_to_char, seq_length,
                   length=60, temperature=0.8, top_p=0.9, rep_penalty=1.2, device='cpu'):
    model.eval()
    valid_start = ''.join(ch for ch in start_text if ch in char_to_id)
    if not valid_start:
        valid_start = random.choice(list(char_to_id.keys()))
    ids = [char_to_id[ch] for ch in valid_start]
    recent_chars = []

    with torch.no_grad():
        for _ in range(length):
            context = ids[-seq_length:]
            x = torch.tensor([context], dtype=torch.long, device=device)
            logits, _ = model(x)
            next_logits = logits[0, -1] / temperature

            if rep_penalty != 1.0 and recent_chars:
                for cid in set(recent_chars[-10:]):
                    next_logits[cid] /= rep_penalty

            sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=0), dim=0)
            sorted_indices_to_keep = cumulative_probs <= top_p
            if sorted_indices_to_keep.sum() == 0:
                sorted_indices_to_keep[0] = True
            filtered_logits = sorted_logits.clone()
            filtered_logits[~sorted_indices_to_keep] = float('-inf')
            probs = torch.softmax(filtered_logits, dim=0)
            next_id = sorted_indices[torch.multinomial(probs, 1)].item()

            ids.append(next_id)
            recent_chars.append(next_id)

    return ''.join(id_to_char[i] for i in ids)


# ------------------------- 7. 主程序 -------------------------
def main():
    # ========== 可调整的参数 ==========
    PDF_URL = "在此处填写"
    # 修正：使用原始字符串（r"..."）或正斜杠
    LOCAL_PDF_PATH = r"Tonh-Tze-Anthology-with-Correction-and-Annotation.pdf"

    # LSTM 训练参数
    SEQ_LEN = 40
    EMBED_DIM = 64
    HIDDEN_SIZE = 128
    NUM_LAYERS = 2
    BATCH_SIZE = 64
    EPOCHS = 30
    LR = 0.002
    GRAD_CLIP = 1.0
    REPEAT_CORPUS = 8

    START_TEXT = "灌子曰："
    GENERATE_LEN = 200
    TEMPERATURES = [0.9, 1.5, 2.5]
    TOP_P = 0.92
    REP_PENALTY = 1.15
    DEVICE = 'cpu'
    # ================================

    if DEVICE == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"使用设备: {device}")

    # 提取对话（优先使用本地 PDF）
    try:
        extracted = extract_dialogues_from_pdf(PDF_URL, LOCAL_PDF_PATH)
        print(f"成功提取 {len(extracted)} 条对话")
        for d in extracted[:5]:
            print(f"  {d[:80]}...")
    except Exception as e:
        print(f"PDF 提取失败：{e}")
        print("将仅使用内置语料进行训练")
        extracted = []

    # 加载语料
    raw_corpus = load_corpus(extracted)
    corpus = (raw_corpus + "\n") * REPEAT_CORPUS
    print(f"总语料字符数: {len(corpus)}")

    chars = sorted(set(corpus))
    char_to_id = {ch: i for i, ch in enumerate(chars)}
    id_to_char = {i: ch for ch, i in char_to_id.items()}
    vocab_size = len(chars)
    print(f"词汇表大小: {vocab_size}")

    ids = [char_to_id[ch] for ch in corpus]
    dataset = CharSeqDataset(ids, SEQ_LEN)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    print(f"训练样本数: {len(dataset)}")

    model = CharLSTM(vocab_size, EMBED_DIM, HIDDEN_SIZE, NUM_LAYERS).to(device)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    print("\n开始训练...")
    start_time = time.time()
    train(model, loader, EPOCHS, LR, device, GRAD_CLIP)
    print(f"训练完成，耗时 {time.time() - start_time:.2f} 秒")

    print("\n" + "=" * 70)
    print("仿写结果 (Top‑p 采样 + 重复惩罚):")
    for temp in TEMPERATURES:
        gen = generate_top_p(model, START_TEXT, char_to_id, id_to_char,
                             SEQ_LEN, GENERATE_LEN, temp, TOP_P, REP_PENALTY, device)
        print(f"\n温度 {temp:.1f} | top-p {TOP_P} | rep_penalty {REP_PENALTY}:\n{gen}\n")
    print("=" * 70)

    save = input("\n是否保存模型？(y/n): ").strip().lower()
    if save == 'y':
        torch.save(model.state_dict(), "guanzi_lstm.pth")
        print("模型已保存为 guanzi_lstm.pth")


if __name__ == "__main__":
    # 可选：如果 SSL 证书问题仍然存在（例如仍需从网络下载），可以临时禁用警告
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    main()