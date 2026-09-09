"""Generate literature_bank.md rows from the downloaded corpus metadata."""
import json
import glob
import os

ROOT = ".pipeline/literature/cotguard-core/papers"
DATE = "2026-09-09"

# arxiv id (no version) -> (relevance note, accepted)
NOTES = {
    "2401.12242": ("主线A｜最近邻#1 BadChain：prompt层CoT后门，机制与本文同构，必须正面区分", "yes"),
    "2504.05605": ("主线A｜最近邻#2 ShadowCoT：白盒推理层后门，用黑盒/白盒划界", "yes"),
    "2511.10714": ("主线A｜最近邻#3 BadThink：保持答案只改轨迹，与本文设计最像", "yes"),
    "2305.01219": ("主线A｜prompt-as-trigger 谱系，说明注入方式本身不新", "yes"),
    "2404.02406": ("主线A｜多轮对话后门，多跳的单模型近似", "yes"),
    "2502.13141": ("主线A｜UniGuardian 统一触发检测器，可作检测端 baseline", "yes"),
    "2404.01245": ("主线B｜理论模板#1：pivot+密钥控FPR、闭式FNR、minimax最优规则", "yes"),
    "2501.02441": ("主线B｜理论模板#2：数据挪用假设检验，问题同构（低一层）", "yes"),
    "2411.13868": ("主线B｜Tr-GoF：证明加和型检测规则在编辑扰动下次优——直击本文Algorithm 4硬伤", "yes"),
    "2402.14904": ("主线B｜radioactivity：蒸馏型泄漏的可证明检测，威胁模型需区分此通道", "yes"),
    "2301.10226": ("主线B｜Kirchenbauer 水印奠基工作", "yes"),
    "2402.14883": ("主线B｜Double-I：LLM微调的模型版权主张", "yes"),
    "2406.12975": ("主线B｜SHIELD：版权合规评测与防御，潜在 baseline", "yes"),
    "2402.02333": ("主线B｜生成式AI版权保护技术综述，背景引用", "yes"),
    "2410.17552": ("主线B｜EaaS 水印，嵌入层版权主张", "yes"),
    "2411.19563": ("主线B｜集成水印，检测增强参考", "maybe"),
    "2501.12174": ("主线B｜BiMarker 双极水印检测增强", "maybe"),
    "2606.18430": ("主线B｜签名过滤，统计水印检测的轻量增强", "maybe"),
    "2403.16981": ("主线B｜二元假设检验样本复杂度：刻画『需观测多少推理步』", "maybe"),
    "2410.14259": ("主线B｜细粒度LLM生成文本检测", "maybe"),
    "2509.18862": ("主线B｜多层次特征的LLM生成文本检测", "maybe"),
    "2410.07283": ("主线C｜Prompt Infection：恶意信号跨agent自复制传播，本文的概念镜像", "yes"),
    "2510.25595": ("主线C｜信息不对称下agent通信与验证，多跳形式化参考", "yes"),
    "2504.20984": ("主线C｜ACE：LLM应用系统安全架构，部署侧威胁模型", "maybe"),
    "2603.17419": ("主线C｜零信任agent架构", "maybe"),
    "2402.06363": ("主线C｜StruQ：结构化查询防prompt injection，单agent baseline", "maybe"),
    "2410.05451": ("主线C｜SecAlign：偏好优化防prompt injection", "maybe"),
    "2601.20727": ("主线C｜LLM问责审计轨迹，支撑『agent审计原语』叙事", "yes"),
    "2510.19851": ("主线D｜CoT混淆压力测试：模型可混淆外部CoT逃避监控——本文最强自适应攻击", "yes"),
    "2510.27378": ("主线D｜faithfulness+verbosity 合成 monitorability 分数", "yes"),
    "2608.15392": ("主线D｜跨语言可见推理与间接注入可监控性", "maybe"),
    "2501.18617": ("主线A｜DarkMind：定制LLM中的潜伏CoT后门，ShadowCoT引用的推理层后门baseline", "yes"),
    "2406.05948": ("主线A｜Chain-of-Scrutiny：面向LLM后门的CoT检测器，检测端对照工作", "yes"),
    "2507.11473": ("主线D｜CoT可监控性立场论文（Korbak/Baker等）：safety叙事的权威锚点", "yes"),
    "2510.19476": ("主线D｜基于CoT监控构建safety case的路线图，可借其校准与论证范式", "yes"),
    "2503.09567": ("背景｜长CoT推理综述", "maybe"),
    "2410.03595": ("背景｜Hopfieldian视角理解CoT推理", "maybe"),
    "2309.02144": ("背景｜对齐提升LLM推理", "no"),
    "2312.06056": ("背景｜METAL变形测试框架", "no"),
    "2308.16684": ("噪声｜有损压缩作为后门，非CoT相关", "no"),
    "2305.18829": ("⚠️ 预印本[15]引用错误：该ID实际为UniScene自动驾驶论文，非CoDA", "no"),
}

DEFAULT = ("检索噪声，与本项目无直接关系", "no")

rows = []
for meta_path in sorted(glob.glob(os.path.join(ROOT, "*", "metadata.json"))):
    m = json.load(open(meta_path))
    pdir = os.path.dirname(meta_path)
    aid_full = m.get("arxiv_id") or ""
    aid = aid_full.split("v")[0]
    ocr = os.path.join(pdir, "ocr", "paper", "doc_0.md")
    ocr_path = ocr if os.path.exists(ocr) else "none"
    note, accepted = NOTES.get(aid, DEFAULT)
    rows.append({
        "url": m.get("landing_page") or f"https://arxiv.org/abs/{aid}",
        "title": (m.get("title") or "").replace("\n", " ").strip(),
        "year": m.get("year", ""),
        "venue": m.get("venue") or "arXiv",
        "relevance": note,
        "accepted": accepted,
        "ocr": ocr_path,
        "aid": aid,
    })

order = {"yes": 0, "maybe": 1, "no": 2}
rows.sort(key=lambda r: (order.get(r["accepted"], 3), r["aid"]))

lines = [
    "# Literature Bank",
    "",
    f"语料根目录：`.pipeline/literature/cotguard-core`　共 {len(rows)} 篇（真实 PDF + pdfminer OCR）",
    f"最近更新：{DATE}　gap 分析见 `.pipeline/docs/gap_matrix.md`",
    "",
    "`accepted` 含义：yes = 纳入 Related Work / 必读；maybe = 备用；no = 检索噪声。",
    "",
    "| URL | Title | Year | Venue | Relevance | accepted | Date | OCR路径 |",
    "|---|---|---|---|---|---|---|---|",
]
for r in rows:
    lines.append(
        f"| {r['url']} | {r['title']} | {r['year']} | {r['venue']} | {r['relevance']} "
        f"| {r['accepted']} | {DATE} | `{r['ocr']}` |"
    )

n_ocr = sum(1 for r in rows if r["ocr"] != "none")
lines += [
    "",
    f"统计：accepted=yes {sum(1 for r in rows if r['accepted']=='yes')} 篇，"
    f"maybe {sum(1 for r in rows if r['accepted']=='maybe')} 篇，"
    f"no {sum(1 for r in rows if r['accepted']=='no')} 篇；OCR 成功 {n_ocr}/{len(rows)}。",
]

open(".pipeline/memory/literature_bank.md", "w").write("\n".join(lines) + "\n")
print(f"wrote {len(rows)} rows, ocr={n_ocr}")
