const pptxgen = require("pptxgenjs");
const sharp = require("sharp");

const pptx = new pptxgen();
pptx.layout = "LAYOUT_WIDE";
pptx.author = "Nautilus Decision Admissibility";
pptx.company = "Nautilus";
pptx.subject = "WP8 final engineering and scientific evidence";
pptx.title = "Decision Admissibility - WP8 Final Evidence";
pptx.lang = "zh-CN";
pptx.theme = {
  headFontFace: "Aptos",
  bodyFontFace: "Aptos",
  lang: "zh-CN",
};

const W = 13.333;
const H = 7.5;
const C = {
  navy: "102A43",
  navy2: "173F5F",
  blue: "2F6BFF",
  blueLight: "EAF0FF",
  teal: "0F766E",
  tealLight: "E6F6F3",
  green: "16855B",
  greenLight: "E9F7F0",
  amber: "D97706",
  amberLight: "FFF4E5",
  red: "B42318",
  redLight: "FDECEC",
  purple: "6941C6",
  purpleLight: "F1ECFF",
  ink: "172B4D",
  text: "334E68",
  muted: "6B7C93",
  line: "D9E2EC",
  panel: "F7F9FC",
  white: "FFFFFF",
  gray: "E7ECF2",
  darkGray: "425466",
};

function addText(slide, text, x, y, w, h, options = {}) {
  slide.addText(text, {
    x, y, w, h,
    fontFace: options.fontFace || "Aptos",
    fontSize: options.fontSize || 18,
    color: options.color || C.text,
    bold: options.bold || false,
    align: options.align || "left",
    valign: options.valign || "mid",
    margin: options.margin === undefined ? 0 : options.margin,
    breakLine: false,
    fit: "shrink",
    ...options,
  });
}

function rect(slide, x, y, w, h, fill, radius = 0.08, line = fill) {
  slide.addShape(radius ? pptx.ShapeType.roundRect : pptx.ShapeType.rect, {
    x, y, w, h,
    rectRadius: radius,
    fill: { color: fill },
    line: { color: line, transparency: line === fill ? 100 : 0, width: 1 },
  });
}

function line(slide, x1, y1, x2, y2, color = C.line, width = 1.5, endArrowType) {
  slide.addShape(pptx.ShapeType.line, {
    x: x1, y: y1, w: x2 - x1, h: y2 - y1,
    line: { color, width, endArrowType },
  });
}

function escapeXml(text) {
  return String(text)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&apos;");
}

const zhImageCache = new Map();

async function zhImage(text, w, h, options = {}) {
  const fontSize = options.fontSize || 15;
  const color = options.color || C.text;
  const bold = options.bold || false;
  const align = options.align || "left";
  const valign = options.valign || "mid";
  const dpi = options.dpi || 240;
  const lines = String(text).split("\n");
  const pxW = Math.max(8, Math.round(w * dpi));
  const pxH = Math.max(8, Math.round(h * dpi));
  const fs = fontSize * dpi / 72;
  const lineHeight = fs * 1.18;
  const blockHeight = lines.length * lineHeight;
  const firstBaseline = valign === "top"
    ? fs * 0.92
    : valign === "bottom"
      ? pxH - blockHeight + fs * 0.92
      : (pxH - blockHeight) / 2 + fs * 0.92;
  const anchor = align === "center" ? "middle" : align === "right" ? "end" : "start";
  const x = align === "center" ? pxW / 2 : align === "right" ? pxW - 3 : 3;
  const tspans = lines.map((lineText, index) => (
    `<tspan x="${x}" y="${firstBaseline + index * lineHeight}">${escapeXml(lineText)}</tspan>`
  )).join("");
  const cacheKey = JSON.stringify({ text, pxW, pxH, fontSize, color, bold, align, valign });
  if (!zhImageCache.has(cacheKey)) {
    const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${pxW}" height="${pxH}" viewBox="0 0 ${pxW} ${pxH}"><text font-family="Hiragino Sans GB" font-size="${fs}" font-weight="${bold ? 600 : 400}" fill="#${color}" text-anchor="${anchor}">${tspans}</text></svg>`;
    const buffer = await sharp(Buffer.from(svg)).png({ compressionLevel: 9 }).toBuffer();
    zhImageCache.set(cacheKey, `data:image/png;base64,${buffer.toString("base64")}`);
  }
  return zhImageCache.get(cacheKey);
}

async function addZh(slide, text, x, y, w, h, options = {}) {
  const data = await zhImage(text, w, h, options);
  slide.addImage({ data, x, y, w, h, altText: text });
}

async function title(slide, kicker, heading, subheading, zh = {}) {
  addText(slide, kicker.toUpperCase(), 0.58, 0.28, 4.6, 0.25, {
    fontSize: 10, bold: true, color: C.blue, charSpacing: 1.5,
  });
  if (zh.kicker) {
    await addZh(slide, zh.kicker, 4.05, 0.26, 1.85, 0.25, {
      fontSize: 9.5, bold: true, color: C.blue,
    });
  }
  addText(slide, heading, 0.58, 0.58, 5.92, 0.52, {
    fontSize: 23, bold: true, color: C.navy,
  });
  await addZh(slide, zh.heading || "", 6.50, 0.58, 6.08, 0.52, {
    fontSize: 20, bold: true, color: C.navy, align: "right",
  });
  if (subheading) {
    addText(slide, subheading, 0.58, 1.08, 5.92, 0.33, {
      fontSize: 10.5, color: C.muted,
    });
    await addZh(slide, zh.subheading || "", 6.50, 1.08, 6.08, 0.33, {
      fontSize: 10.5, color: C.muted, align: "right",
    });
  }
  line(slide, 0.58, 1.47, 12.18, 1.47, C.line, 1);
}

function footer(slide, n, label = "Decision Admissibility | WP8 final evidence | 2026-07-24") {
  line(slide, 0.58, 7.13, 12.75, 7.13, C.line, 0.8);
  addText(slide, label, 0.58, 7.18, 10.7, 0.18, { fontSize: 8.5, color: C.muted });
  addText(slide, String(n).padStart(2, "0"), 12.15, 7.18, 0.6, 0.18, {
    fontSize: 8.5, bold: true, align: "right", color: C.navy,
  });
}

function pill(slide, text, x, y, w, fill, color) {
  rect(slide, x, y, w, 0.34, fill, 0.16);
  addText(slide, text, x + 0.08, y + 0.01, w - 0.16, 0.31, {
    fontSize: 10, bold: true, color, align: "center",
  });
}

function card(slide, x, y, w, h, heading, body, accent = C.blue, fill = C.white) {
  rect(slide, x, y, w, h, fill, 0.1, C.line);
  rect(slide, x, y, 0.08, h, accent, 0, accent);
  addText(slide, heading, x + 0.22, y + 0.16, w - 0.34, 0.35, {
    fontSize: 16, bold: true, color: C.navy,
  });
  addText(slide, body, x + 0.22, y + 0.56, w - 0.34, h - 0.7, {
    fontSize: 11.5, color: C.text, valign: "top", breakLine: true, margin: 0.02,
  });
}

async function cardBi(slide, x, y, w, h, heading, headingZh, body, bodyZh, accent = C.blue, fill = C.white) {
  rect(slide, x, y, w, h, fill, 0.1, C.line);
  rect(slide, x, y, 0.08, h, accent, 0, accent);
  addText(slide, heading, x + 0.22, y + 0.10, w * 0.52, 0.24, {
    fontSize: 11.5, bold: true, color: C.navy,
  });
  await addZh(slide, headingZh, x + w * 0.58, y + 0.10, w * 0.34, 0.24, {
    fontSize: 10.5, bold: true, color: C.navy, align: "right",
  });
  addText(slide, body, x + 0.22, y + 0.42, w - 0.34, (h - 0.50) * 0.48, {
    fontSize: 8.8, color: C.text, valign: "top", breakLine: true, margin: 0.01,
  });
  await addZh(slide, bodyZh, x + 0.22, y + 0.79, w - 0.34, (h - 0.50) * 0.44, {
    fontSize: 8.5, color: C.text, valign: "top",
  });
}

function metric(slide, x, y, w, h, value, label, fill, color) {
  rect(slide, x, y, w, h, fill, 0.1);
  addText(slide, value, x + 0.12, y + 0.12, w - 0.24, h * 0.48, {
    fontSize: 27, bold: true, color, align: "center",
  });
  addText(slide, label, x + 0.12, y + h * 0.58, w - 0.24, h * 0.26, {
    fontSize: 10.5, color, align: "center", valign: "top",
  });
}

async function metricBi(slide, x, y, w, h, value, label, labelZh, fill, color) {
  rect(slide, x, y, w, h, fill, 0.1);
  addText(slide, value, x + 0.12, y + 0.09, w - 0.24, h * 0.40, {
    fontSize: 25, bold: true, color, align: "center",
  });
  addText(slide, label, x + 0.12, y + h * 0.50, w - 0.24, h * 0.18, {
    fontSize: 8.8, color, align: "center", valign: "top",
  });
  await addZh(slide, labelZh, x + 0.12, y + h * 0.70, w - 0.24, h * 0.18, {
    fontSize: 8.7, color, align: "center",
  });
}

async function main() {

// Slide 1
{
  const s = pptx.addSlide();
  s.background = { color: C.navy };
  rect(s, 0, 0, W, H, C.navy, 0);
  rect(s, 0.64, 0.56, 1.65, 0.35, C.blue, 0.17);
  addText(s, "WP8 | FINAL EVIDENCE", 0.76, 0.58, 1.42, 0.29, {
    fontSize: 10, bold: true, color: C.white, align: "center", charSpacing: 1.1,
  });
  await addZh(s, "最终证据", 2.48, 0.58, 1.35, 0.29, {
    fontSize: 10, bold: true, color: "9FC5FF",
  });
  addText(s, "Decision Admissibility", 0.68, 1.24, 6.10, 0.62, {
    fontSize: 31, bold: true, color: C.white,
  });
  await addZh(s, "决策准入", 7.10, 1.27, 5.50, 0.55, {
    fontSize: 27, bold: true, color: C.white, align: "right",
  });
  addText(s, "Same-domain cross-task memory transfer", 0.68, 2.18, 6.10, 0.42, {
    fontSize: 20, bold: true, color: "DCE7F5",
  });
  await addZh(s, "同领域跨任务记忆迁移", 7.10, 2.20, 5.50, 0.38, {
    fontSize: 19, bold: true, color: "DCE7F5", align: "right",
  });
  addText(s, "Engineering closeout is complete. The formal performance headline is not supported.", 0.68, 3.02, 6.10, 0.55, {
    fontSize: 14.5, color: "B7C9DD", breakLine: true,
  });
  await addZh(s, "工程闭环已经完成；正式性能主张未获支持。", 7.10, 3.04, 5.50, 0.48, {
    fontSize: 14.5, color: "B7C9DD", align: "right",
  });
  rect(s, 0.68, 4.28, 11.95, 1.22, "173F5F", 0.12);
  addText(s, "PLAN_APPROVED", 0.95, 4.50, 2.55, 0.42, {
    fontSize: 23, bold: true, color: "7CE3BE",
  });
  addText(s, "Engineering", 3.56, 4.39, 1.64, 0.30, {
    fontSize: 14, bold: true, color: C.white, align: "center",
  });
  await addZh(s, "工程完成", 3.56, 4.76, 1.64, 0.28, {
    fontSize: 12.5, bold: true, color: C.white, align: "center",
  });
  addText(s, "≠", 5.28, 4.47, 0.55, 0.45, {
    fontSize: 25, bold: true, color: "FFCF8A", align: "center",
  });
  addText(s, "Superiority", 5.93, 4.39, 1.65, 0.30, {
    fontSize: 14, bold: true, color: "FFB4AB", align: "center",
  });
  await addZh(s, "性能优越", 5.93, 4.76, 1.65, 0.28, {
    fontSize: 12.5, bold: true, color: "FFB4AB", align: "center",
  });
  addText(s, "≠", 7.72, 4.47, 0.55, 0.45, {
    fontSize: 25, bold: true, color: "FFCF8A", align: "center",
  });
  addText(s, "Causality", 8.37, 4.39, 1.65, 0.30, {
    fontSize: 14, bold: true, color: "D9C2FF", align: "center",
  });
  await addZh(s, "经验因果", 8.37, 4.76, 1.65, 0.28, {
    fontSize: 12.5, bold: true, color: "D9C2FF", align: "center",
  });
  addText(s, "GLM-5.2[1m] independent read-only audit | 2026-07-24", 0.68, 6.55, 6.8, 0.3, {
    fontSize: 11, color: "8FAAC5",
  });
  await addZh(s, "GLM-5.2[1m] 独立只读审查", 0.68, 6.87, 4.20, 0.25, {
    fontSize: 9.5, color: "8FAAC5",
  });
  addText(s, "Formal headline: REJECTED", 9.15, 6.55, 3.45, 0.3, {
    fontSize: 11, bold: true, color: "FFB4AB", align: "right",
  });
  await addZh(s, "正式结论：拒绝", 9.15, 6.87, 3.45, 0.25, {
    fontSize: 10, bold: true, color: "FFB4AB", align: "right",
  });
}

// Slide 2
{
  const s = pptx.addSlide();
  s.background = { color: C.white };
  await title(s, "Architecture", "The governed unit is Claim × Operation", "Relevance proposes candidates; Authority decides influence before ranking and Prompt construction.", {
    kicker: "架构",
    heading: "治理单元是主张 × 操作",
    subheading: "相关性只提出候选；权限机制在排序与 Prompt 构造前决定其能否产生影响。",
  });
  const xs = [0.7, 3.0, 5.3, 7.6, 9.9];
  const labels = [
    ["RunForest / SOP", "运行森林 / SOP", "Evidence and actionable methods"],
    ["Stage Router", "阶段路由", "Draft / Improve / Debug granularity"],
    ["Authority Gate", "权限闸门", "Claim | Operation | Protocol | Task"],
    ["Agent Execution", "Agent 执行", "Generate and run target code"],
    ["Typed Writeback", "类型化回写", "Result / Adoption / Causal"],
  ];
  const colors = [C.purple, C.blue, C.amber, C.teal, C.green];
  for (let i = 0; i < xs.length; i++) {
    rect(s, xs[i], 1.9, 1.78, 1.45, i === 2 ? C.amberLight : C.panel, 0.12, C.line);
    rect(s, xs[i], 1.9, 1.78, 0.11, colors[i], 0, colors[i]);
    addText(s, String(i + 1), xs[i] + 0.12, 2.08, 0.34, 0.34, {
      fontSize: 12, bold: true, color: C.white, align: "center", fill: { color: colors[i] }, shape: pptx.ShapeType.ellipse,
    });
    addText(s, labels[i][0], xs[i] + 0.18, 2.39, 1.42, 0.25, {
      fontSize: 11.5, bold: true, color: C.navy, align: "center",
    });
    await addZh(s, labels[i][1], xs[i] + 0.18, 2.68, 1.42, 0.23, {
      fontSize: 10.5, bold: true, color: C.navy, align: "center",
    });
    addText(s, labels[i][2], xs[i] + 0.18, 2.96, 1.42, 0.24, {
      fontSize: 8.4, color: C.muted, align: "center",
    });
    if (i < xs.length - 1) line(s, xs[i] + 1.81, 2.62, xs[i] + 2.20, 2.62, C.blue, 2, "triangle");
  }
  rect(s, 0.72, 4.02, 11.84, 2.40, C.panel, 0.12);
  addText(s, "Three objects must remain separate", 0.98, 4.22, 3.4, 0.34, {
    fontSize: 17, bold: true, color: C.navy,
  });
  await addZh(s, "三个对象必须严格分离", 4.55, 4.22, 2.80, 0.34, {
    fontSize: 15, bold: true, color: C.navy,
  });
  const cols = [
    [0.98, "Result Fact", "结果事实", "The target node's executed code and legal evaluation", "No historical actuation required", C.green, C.greenLight],
    [4.55, "Adoption Edge", "采纳边", "The source experience appears in code and runtime", "Static + Runtime required (L2/L3)", C.blue, C.blueLight],
    [8.12, "Causal Edge", "因果边", "Removing memory changes action or code", "Counterfactual also required (L4)", C.purple, C.purpleLight],
  ];
  for (const [x, h, hZh, b, f, accent, fill] of cols) {
    rect(s, x, 4.75, 3.20, 1.28, fill, 0.1);
    addText(s, h, x + 0.16, 4.85, 1.48, 0.28, { fontSize: 13.5, bold: true, color: accent });
    await addZh(s, hZh, x + 1.66, 4.85, 1.38, 0.28, { fontSize: 12.5, bold: true, color: accent, align: "right" });
    addText(s, b, x + 0.16, 5.19, 2.88, 0.35, { fontSize: 9.6, color: C.text, valign: "top" });
    addText(s, f, x + 0.16, 5.62, 2.88, 0.22, { fontSize: 8.8, bold: true, color: accent });
  }
  footer(s, 2);
}

// Slide 3
{
  const s = pptx.addSlide();
  s.background = { color: C.white };
  await title(s, "Formal design", "Same-domain, cross-task, target-history excluded", "Three tasks, three agent seeds, five online systems per block, and one host-only Oracle.", {
    kicker: "正式设计",
    heading: "同领域、跨任务、排除目标历史",
    subheading: "3 个任务 × 3 个 Agent 种子；每个区块 5 个在线系统，并配 1 个仅主机可见的 Oracle。",
  });
  const domains = [
    [0.72, "IMAGE", "图像", "Aerial Cactus", "macro-F1", "Methods from other image tasks", "参考其他图像任务的方法", C.blue, C.blueLight],
    [4.50, "AUDIO", "音频", "MLSP Birds", "macro-F1", "Methods from other audio tasks", "参考其他音频任务的方法", C.purple, C.purpleLight],
    [8.28, "TABULAR", "表格", "NYC Taxi", "RMSE (lower is better)", "Methods from other tabular tasks", "参考其他表格任务的方法", C.teal, C.tealLight],
  ];
  for (const [x, domain, domainZh, task, metricName, source, sourceZh, accent, fill] of domains) {
    rect(s, x, 1.78, 3.30, 2.25, fill, 0.12);
    pill(s, domain, x + 0.18, 1.98, 0.82, C.white, accent);
    await addZh(s, domainZh, x + 1.12, 1.99, 0.72, 0.31, { fontSize: 10, bold: true, color: accent });
    addText(s, task, x + 0.18, 2.48, 2.92, 0.34, { fontSize: 17, bold: true, color: C.navy });
    addText(s, metricName, x + 0.18, 2.88, 2.92, 0.26, { fontSize: 12, bold: true, color: accent });
    addText(s, source, x + 0.18, 3.26, 2.92, 0.25, { fontSize: 9.3, color: C.text });
    await addZh(s, sourceZh, x + 0.18, 3.57, 2.92, 0.24, { fontSize: 9.2, color: C.text });
  }
  rect(s, 0.72, 4.35, 11.86, 1.55, C.panel, 0.12);
  const sys = ["No Memory", "Flat", "Global Bit", "Authority", "Full"];
  const sysZh = ["无记忆", "扁平检索", "全局可信位", "权限机制", "完整系统"];
  for (let i = 0; i < sys.length; i++) {
    rect(s, 1.02 + i * 2.05, 4.72, 1.75, 0.64, i === 4 ? C.navy : C.white, 0.08, C.line);
    addText(s, sys[i], 1.08 + i * 2.05, 4.76, 1.63, 0.22, {
      fontSize: 9.8, bold: true, align: "center", color: i === 4 ? C.white : C.navy,
    });
    await addZh(s, sysZh[i], 1.08 + i * 2.05, 5.04, 1.63, 0.22, {
      fontSize: 9.6, bold: true, align: "center", color: i === 4 ? C.white : C.navy,
    });
  }
  addText(s, "3 tasks × 3 seeds × 5 systems = 45 assigned online outcomes", 0.95, 6.09, 7.25, 0.27, {
    fontSize: 14.5, bold: true, color: C.navy,
  });
  await addZh(s, "3 个任务 × 3 个种子 × 5 个系统 = 45 个在线结果", 0.95, 6.40, 7.25, 0.24, {
    fontSize: 11, bold: true, color: C.navy,
  });
  pill(s, "9 Oracle", 8.62, 6.15, 1.32, C.amberLight, C.amber);
  pill(s, "0 target history", 10.12, 6.15, 2.02, C.redLight, C.red);
  addText(s, "Same-domain provisional methods may generate candidates; source scores never transfer; target execution is mandatory.", 0.95, 6.69, 11.2, 0.18, {
    fontSize: 8.8, color: C.muted,
  });
  await addZh(s, "同领域临时方法可生成候选；来源分数不可迁移；必须在目标任务重新执行。", 0.95, 6.91, 11.2, 0.15, {
    fontSize: 7.8, color: C.muted,
  });
  footer(s, 3);
}

// Slide 4
{
  const s = pptx.addSlide();
  s.background = { color: C.white };
  await title(s, "Formal outcome", "No Memory completes more assignments than Full", "Every failure is a formal outcome; the both-scored subset cannot replace intention-to-treat.", {
    kicker: "正式结果",
    heading: "无记忆的完成数高于完整系统",
    subheading: "每次失败都是正式结果；双方均有分数的子集不能替代意向性分析。",
  });
  rect(s, 0.72, 1.78, 7.15, 4.72, C.panel, 0.12);
  addText(s, "Protocol-legal completion", 1.02, 2.02, 2.80, 0.34, { fontSize: 15, bold: true, color: C.navy });
  await addZh(s, "协议合法完成数", 4.25, 2.02, 2.60, 0.34, { fontSize: 14, bold: true, color: C.navy, align: "right" });
  line(s, 1.35, 5.78, 6.98, 5.78, C.darkGray, 1);
  line(s, 1.35, 2.45, 1.35, 5.78, C.darkGray, 1);
  const maxH = 2.65;
  const bars = [
    [2.10, 4 / 9, "Full", C.blue, "4 / 9"],
    [4.55, 6 / 9, "No Memory", C.green, "6 / 9"],
  ];
  for (const [x, rate, label, color, val] of bars) {
    const h = maxH * rate;
    rect(s, x, 5.78 - h, 1.38, h, color, 0.06);
    addText(s, val, x, 5.36 - h, 1.38, 0.36, { fontSize: 20, bold: true, color, align: "center" });
    addText(s, label, x - 0.2, 5.96, 1.78, 0.30, { fontSize: 12, bold: true, color: C.navy, align: "center" });
    await addZh(s, label === "Full" ? "完整系统" : "无记忆", x - 0.2, 6.22, 1.78, 0.22, {
      fontSize: 9.5, bold: true, color: C.navy, align: "center",
    });
  }
  addText(s, "2/9 completion deficit", 5.90, 2.58, 1.48, 0.42, {
    fontSize: 11.5, bold: true, color: C.red, align: "center",
    fill: { color: C.redLight }, line: { color: C.redLight }, margin: 0.06,
  });
  await addZh(s, "完成数少 2/9", 5.90, 3.03, 1.48, 0.24, { fontSize: 9.5, bold: true, color: C.red, align: "center" });
  await metricBi(s, 8.20, 1.78, 2.05, 1.35, "22", "scored selected results", "有分数的入选结果", C.greenLight, C.green);
  await metricBi(s, 10.52, 1.78, 2.05, 1.35, "23", "retained failures", "保留的失败", C.redLight, C.red);
  await metricBi(s, 8.20, 3.42, 2.05, 1.35, "0", "score imputations", "分数插补", C.blueLight, C.blue);
  await metricBi(s, 10.52, 3.42, 2.05, 1.35, "0", "post-assignment exclusions", "分配后排除", C.purpleLight, C.purple);
  rect(s, 8.20, 5.05, 4.37, 1.45, C.amberLight, 0.1, C.line);
  rect(s, 8.20, 5.05, 0.08, 1.45, C.amber, 0, C.amber);
  addText(s, "Intention-to-treat", 8.42, 5.17, 2.00, 0.28, { fontSize: 13.5, bold: true, color: C.navy });
  await addZh(s, "意向性分析", 10.64, 5.17, 1.67, 0.28, { fontSize: 12.5, bold: true, color: C.navy, align: "right" });
  addText(s, "All 45 assigned outcomes remain; authority denials and runtime failures are system behavior.", 8.42, 5.53, 3.93, 0.34, {
    fontSize: 9.4, color: C.text, valign: "top",
  });
  await addZh(s, "45 个已分配结果全部保留；权限拒绝与运行失败属于系统行为。", 8.42, 5.94, 3.93, 0.30, {
    fontSize: 9.2, color: C.text,
  });
  footer(s, 4);
}

// Slide 5
{
  const s = pptx.addSlide();
  s.background = { color: C.white };
  await title(s, "Statistics", "All four available pairs are positive — diagnostic only", "The available-pair subset cannot replace the full intention-to-treat population.", {
    kicker: "统计",
    heading: "4 个可用配对均为正——仅作诊断",
    subheading: "可用配对子集不能替代完整的意向性分析总体。",
  });
  rect(s, 0.72, 1.82, 6.00, 4.92, C.panel, 0.12);
  addText(s, "9 assigned Full ↔ No Memory pairs", 1.02, 2.05, 4.6, 0.28, { fontSize: 15, bold: true, color: C.navy });
  await addZh(s, "9 个已分配的完整系统 ↔ 无记忆配对", 1.02, 2.38, 4.9, 0.27, { fontSize: 11.5, bold: true, color: C.navy });
  for (let i = 0; i < 9; i++) {
    const x = 1.08 + (i % 5) * 1.02;
    const y = 2.78 + Math.floor(i / 5) * 1.02;
    const available = i < 4;
    rect(s, x, y, 0.78, 0.78, available ? C.greenLight : C.gray, 0.12);
    addText(s, available ? "✓" : "—", x, y + 0.03, 0.78, 0.42, {
      fontSize: 20, bold: true, color: available ? C.green : C.muted, align: "center",
    });
    addText(s, available ? "Full +" : "missing", x, y + 0.44, 0.78, 0.18, {
      fontSize: 8.5, bold: available, color: available ? C.green : C.muted, align: "center",
    });
  }
  pill(s, "4 wins", 1.10, 5.00, 1.20, C.greenLight, C.green);
  pill(s, "0 ties", 2.46, 5.00, 1.15, C.panel, C.muted);
  pill(s, "0 losses", 3.76, 5.00, 1.25, C.panel, C.muted);
  pill(s, "5 unavailable", 5.17, 5.00, 1.25, C.redLight, C.red);
  await addZh(s, "4 胜", 1.10, 5.34, 1.20, 0.20, { fontSize: 8.5, bold: true, color: C.green, align: "center" });
  await addZh(s, "0 平", 2.46, 5.34, 1.15, 0.20, { fontSize: 8.5, color: C.muted, align: "center" });
  await addZh(s, "0 负", 3.76, 5.34, 1.25, 0.20, { fontSize: 8.5, color: C.muted, align: "center" });
  await addZh(s, "5 个不可用", 5.17, 5.34, 1.25, 0.20, { fontSize: 8.5, bold: true, color: C.red, align: "center" });
  addText(s, "Aerial: +0.014373 (3 pairs) · Birds: +0.059794 (1 pair) · Taxi: 0 pairs", 1.02, 5.63, 5.25, 0.56, {
    fontSize: 11, color: C.text, breakLine: true,
  });
  await addZh(s, "图像 3 对｜音频 1 对｜表格 0 对", 1.02, 6.17, 5.25, 0.24, {
    fontSize: 9.5, color: C.text,
  });
  const stats = [
    [7.08, 1.82, "raw p", "0.0625", C.amberLight, C.amber],
    [9.87, 1.82, "Holm p", "0.25", C.redLight, C.red],
    [7.08, 3.43, "task-macro CI", "not estimable", C.panel, C.darkGray],
    [9.87, 3.43, "mixed effects", "not estimable", C.panel, C.darkGray],
  ];
  for (const [x, y, label, value, fill, color] of stats) {
    rect(s, x, y, 2.48, 1.28, fill, 0.1);
    addText(s, label, x + 0.14, y + 0.14, 2.20, 0.24, { fontSize: 10, bold: true, color, align: "center" });
    addText(s, value, x + 0.14, y + 0.48, 2.20, 0.48, { fontSize: value.length > 8 ? 17 : 24, bold: true, color, align: "center" });
  }
  rect(s, 7.08, 5.10, 5.27, 1.62, C.redLight, 0.12);
  addText(s, "Frozen effect gate: REJECTED", 7.36, 5.22, 4.70, 0.30, {
    fontSize: 17, bold: true, color: C.red, align: "center",
  });
  await addZh(s, "冻结效应门禁：拒绝", 7.36, 5.55, 4.70, 0.28, {
    fontSize: 14.5, bold: true, color: C.red, align: "center",
  });
  addText(s, "Conditional utility = diagnostic only (not a trend)", 7.36, 5.89, 4.70, 0.25, {
    fontSize: 10.5, color: C.red, align: "center",
  });
  await addZh(s, "条件性效用仅作诊断，不得表述为“趋势”", 7.36, 6.20, 4.70, 0.24, {
    fontSize: 10.2, color: C.red, align: "center",
  });
  footer(s, 5);
}

// Slide 6
{
  const s = pptx.addSlide();
  s.background = { color: C.white };
  await title(s, "Evidence ledger", "Engineering completion and scientific claims are orthogonal", "The ledger preserves negative outcomes and prevents diagnostic evidence from escalating.", {
    kicker: "证据账本",
    heading: "工程完成与科学主张相互独立",
    subheading: "账本保留负结果，并阻止诊断性证据越级升级为正式结论。",
  });
  const summary = [
    [0.72, "4", "supported", "已支持", C.green, C.greenLight],
    [3.05, "1", "diagnostic", "仅诊断", C.amber, C.amberLight],
    [5.38, "1", "rejected", "已拒绝", C.red, C.redLight],
    [7.71, "1", "pending", "待证实", C.purple, C.purpleLight],
  ];
  for (const [x, val, label, labelZh, accent, fill] of summary) {
    await metricBi(s, x, 1.78, 2.05, 1.22, val, label, labelZh, fill, accent);
  }
  rect(s, 10.10, 1.78, 2.47, 1.22, C.navy, 0.1);
  addText(s, "WP8", 10.24, 1.93, 2.19, 0.34, { fontSize: 24, bold: true, color: C.white, align: "center" });
  addText(s, "engineering complete", 10.24, 2.38, 2.19, 0.24, { fontSize: 10, color: "DCE7F5", align: "center" });
  await addZh(s, "工程完成", 10.24, 2.65, 2.19, 0.20, { fontSize: 9, color: "DCE7F5", align: "center" });
  const rows = [
    ["C1", "Formal execution", "正式执行", "supported", "45 outcomes + 9 Oracles traceable", C.green],
    ["C2", "Result writeback", "结果回写", "supported", "22/22 success → Result Fact", C.green],
    ["C3", "Full superiority", "完整系统优越性", "rejected", "Completion, coverage, multiplicity fail", C.red],
    ["C4", "Conditional utility", "条件性效用", "diagnostic", "Four available pairs only", C.amber],
    ["C5", "No imputation", "无插补", "supported", "23 failures retained", C.green],
    ["C6", "Experience causality", "经验因果性", "pending", "Formal gains lack L4", C.purple],
    ["C7", "Prior kill gates", "既有终止门禁", "supported", "Mechanism safety cannot override results", C.green],
  ];
  let y = 3.40;
  for (const [id, claim, claimZh, status, note, accent] of rows) {
    rect(s, 0.72, y, 11.86, 0.46, y % 1 > 0.5 ? C.white : C.panel, 0.04, C.line);
    addText(s, id, 0.88, y + 0.06, 0.62, 0.28, { fontSize: 10.5, bold: true, color: accent });
    addText(s, claim, 1.56, y + 0.02, 1.38, 0.18, { fontSize: 8.8, bold: true, color: C.navy });
    await addZh(s, claimZh, 2.96, y + 0.02, 1.30, 0.18, { fontSize: 8.3, bold: true, color: C.navy, align: "right" });
    pill(s, status, 4.42, y + 0.06, 1.35, accent === C.green ? C.greenLight : accent === C.red ? C.redLight : accent === C.amber ? C.amberLight : C.purpleLight, accent);
    addText(s, note, 5.98, y + 0.06, 6.34, 0.28, { fontSize: 10.5, color: C.text });
    y += 0.50;
  }
  footer(s, 6);
}

// Slide 7
{
  const s = pptx.addSlide();
  s.background = { color: C.white };
  await title(s, "Engineering closeout", "The Stop Gate proves implementation and evidence integrity", "Tests, argv, environment, JUnit, source closure, and failure-to-fix chains are frozen.", {
    kicker: "工程收口",
    heading: "Stop Gate 证明实现与证据完整性",
    subheading: "测试、argv、环境、JUnit、源码闭包及失败到修复链均已冻结。",
  });
  await metricBi(s, 0.72, 1.78, 2.22, 1.35, "1,454", "tests across 7 green runs", "7 次绿色运行的测试总数", C.blueLight, C.blue);
  await metricBi(s, 3.18, 1.78, 2.22, 1.35, "760", "full suite", "完整测试套件", C.tealLight, C.teal);
  await metricBi(s, 5.64, 1.78, 2.22, 1.35, "0", "failure / error / skip", "失败 / 错误 / 跳过", C.greenLight, C.green);
  await metricBi(s, 8.10, 1.78, 2.22, 1.35, "579", "source/test dependencies", "源码 / 测试依赖", C.purpleLight, C.purple);
  await metricBi(s, 10.56, 1.78, 2.02, 1.35, "TRUE", "source unchanged", "源码未变化", C.greenLight, C.green);
  const gates = [
    [0.78, "20 / 20", "prerequisites", "先决条件"],
    [3.25, "6 / 6", "kill gates", "终止门禁"],
    [5.72, "47 / 47", "acceptance", "验收项"],
    [8.19, "2", "failure→fix chains", "失败到修复链"],
    [10.66, "0555", "immutable roots", "不可变根目录"],
  ];
  for (const [x, value, label, labelZh] of gates) {
    rect(s, x, 3.62, 1.90, 1.05, C.panel, 0.10, C.line);
    addText(s, value, x + 0.10, 3.76, 1.70, 0.36, { fontSize: 20, bold: true, color: C.navy, align: "center" });
    addText(s, label, x + 0.10, 4.14, 1.70, 0.18, { fontSize: 8.5, color: C.muted, align: "center" });
    await addZh(s, labelZh, x + 0.10, 4.36, 1.70, 0.18, { fontSize: 8.2, color: C.muted, align: "center" });
  }
  rect(s, 0.78, 5.05, 11.80, 1.38, C.panel, 0.12);
  const chain = [
    ["r1", "partial", "部分结果", C.muted],
    ["r2", "source changed", "源码变化", C.red],
    ["r3", "concurrent edit", "并发编辑", C.amber],
    ["r4", "authoritative", "权威版本", C.green],
  ];
  for (let i = 0; i < chain.length; i++) {
    const x = 1.16 + i * 2.78;
    rect(s, x, 5.36, 2.10, 0.70, chain[i][0] === "r4" ? C.greenLight : C.white, 0.08, C.line);
    addText(s, chain[i][0], x + 0.12, 5.45, 0.46, 0.24, { fontSize: 12, bold: true, color: chain[i][3] });
    addText(s, chain[i][1], x + 0.60, 5.35, 1.34, 0.20, { fontSize: 8.8, color: C.text, align: "center" });
    await addZh(s, chain[i][2], x + 0.60, 5.63, 1.34, 0.20, { fontSize: 8.5, color: C.text, align: "center" });
    if (i < chain.length - 1) line(s, x + 2.12, 5.71, x + 2.63, 5.71, C.muted, 1.5, "triangle");
  }
  addText(s, "Final Gate report hash: de73a72b… · independent verification: dcf2e62c…", 0.88, 6.62, 11.45, 0.26, {
    fontSize: 10.5, color: C.muted, align: "center",
  });
  footer(s, 7);
}

// Slide 8
{
  const s = pptx.addSlide();
  s.background = { color: C.white };
  await title(s, "Writeback semantics", "The memory subject is the newly executed target node", "Actuation only authorizes claims about historical influence; it must not block independent success.", {
    kicker: "回写语义",
    heading: "记忆主体是本次真实执行的目标节点",
    subheading: "Actuation 只授权“历史经验产生影响”的主张；不能阻止独立成功节点入库。",
  });
  const boxes = [
    [0.72, "1", "Result Fact", "结果事实", "Current training code\n+ host execution\n+ protocol / evaluator\n+ target metric", "本次训练代码\n+ 主机执行\n+ 协议 / 评估器\n+ 目标指标", "PROMOTE_RESULT", C.green, C.greenLight],
    [4.50, "2", "Adoption Edge", "采纳边", "Historical Claim to target\n+ static actuation\n+ runtime actuation", "历史主张 → 目标节点\n+ 静态生效证据\n+ 运行时生效证据", "PUBLISH_ADOPTION", C.blue, C.blueLight],
    [8.28, "3", "Causal Edge", "因果边", "Existing Adoption\n+ memory-on/off\n+ counterfactual change", "既有采纳关系\n+ 记忆开 / 关\n+ 反事实变化", "PUBLISH_CAUSAL", C.purple, C.purpleLight],
  ];
  for (const [x, n, h, hZh, body, bodyZh, op, accent, fill] of boxes) {
    rect(s, x, 1.86, 3.30, 3.75, fill, 0.14);
    addText(s, n, x + 0.18, 2.08, 0.46, 0.46, {
      fontSize: 17, bold: true, color: C.white, align: "center", fill: { color: accent }, shape: pptx.ShapeType.ellipse,
    });
    addText(s, h, x + 0.78, 2.02, 1.36, 0.30, { fontSize: 15, bold: true, color: accent });
    await addZh(s, hZh, x + 2.10, 2.03, 0.92, 0.30, { fontSize: 13.5, bold: true, color: accent, align: "right" });
    addText(s, body, x + 0.30, 2.67, 2.70, 0.98, { fontSize: 11.5, color: C.text, breakLine: true, valign: "top", margin: 0.02 });
    await addZh(s, bodyZh, x + 0.30, 3.69, 2.70, 0.74, { fontSize: 10.5, color: C.text, valign: "top" });
    pill(s, op, x + 0.34, 4.75, 2.62, C.white, accent);
  }
  line(s, 4.08, 3.62, 4.42, 3.62, C.blue, 2, "triangle");
  line(s, 7.86, 3.62, 8.20, 3.62, C.purple, 2, "triangle");
  rect(s, 0.72, 6.02, 11.86, 0.68, C.navy, 0.10);
  addText(s, "Cold-start success can publish a Result Fact without historical adoption: derived_from_refs = [].", 0.98, 6.08, 11.34, 0.24, {
    fontSize: 11.5, bold: true, color: C.white, align: "center",
  });
  await addZh(s, "冷启动成功节点无需历史采纳关系即可发布结果事实：derived_from_refs = []。", 0.98, 6.38, 11.34, 0.23, {
    fontSize: 10.5, bold: true, color: C.white, align: "center",
  });
  footer(s, 8);
}

// Slide 9
{
  const s = pptx.addSlide();
  s.background = { color: C.white };
  await title(s, "Independent audit", "GLM-5.2: engineering complete, claim boundaries preserved", "Claudeagent MCP | strict read-only | 48 turns | 47 Read/Glob/Grep calls | 0 web/write", {
    kicker: "独立审查",
    heading: "GLM-5.2：工程完成，主张边界保持",
    subheading: "Claudeagent MCP｜严格只读｜48 轮｜47 次 Read/Glob/Grep｜0 次网络或写入",
  });
  rect(s, 0.72, 1.84, 4.05, 4.92, C.navy, 0.14);
  addText(s, "PLAN_APPROVED", 1.02, 2.23, 3.45, 0.58, {
    fontSize: 28, bold: true, color: "7CE3BE", align: "center",
  });
  addText(s, "WP8 engineering completion", 1.02, 2.88, 3.45, 0.24, {
    fontSize: 11.5, color: C.white, align: "center",
  });
  await addZh(s, "WP8 工程完成", 1.02, 3.16, 3.45, 0.24, { fontSize: 10.5, color: C.white, align: "center" });
  line(s, 1.20, 3.48, 4.29, 3.48, "537A9B", 1);
  const auditStats = [
    ["Model", "glm-5.2[1m]"],
    ["Session", "b0d62720…"],
    ["Tools", "Read · Glob · Grep"],
    ["Blocking", "0"],
  ];
  let ay = 3.75;
  for (const [k, v] of auditStats) {
    addText(s, k, 1.15, ay, 0.90, 0.28, { fontSize: 10.5, color: "9FB8CE" });
    addText(s, v, 2.10, ay, 2.14, 0.28, { fontSize: 11.5, bold: true, color: C.white, align: "right" });
    ay += 0.47;
  }
  await cardBi(s, 5.10, 1.84, 3.45, 1.36, "Established", "已证实", "Authority, visibility, writeback, replay, and immutable evidence operate under contract.", "权限、可见性、回写、重放\n和不可变证据均按契约运行。", C.green, C.greenLight);
  await cardBi(s, 8.88, 1.84, 3.45, 1.36, "Not established", "未证实", "Full memory outperforms No Memory; formal experience-level causality.", "未证明完整记忆优于无记忆，\n也未证明正式经验因果性。", C.red, C.redLight);
  await cardBi(s, 5.10, 3.55, 3.45, 1.46, "Diagnostic only", "仅作诊断", "All four both-scored pairs favor Full, but this is neither a trend nor a headline.", "4 个双方均有分数的配对偏向完整系统，\n但既非趋势也非主结论。", C.amber, C.amberLight);
  await cardBi(s, 8.88, 3.55, 3.45, 1.46, "Next scientific step", "下一步", "Add same-domain tasks/seeds, reduce denials/runtime failures, and bind L2/L3/L4.", "增加同领域任务与种子，减少拒绝和运行失败，\n并绑定 L2/L3/L4。", C.purple, C.purpleLight);
  rect(s, 5.10, 5.43, 7.23, 1.33, C.panel, 0.12);
  addText(s, "Bottom line", 5.36, 5.55, 1.35, 0.22, { fontSize: 10.5, bold: true, color: C.blue });
  await addZh(s, "结论", 5.36, 5.83, 1.35, 0.22, { fontSize: 10.5, bold: true, color: C.blue });
  addText(s, "Knowledge governance is closed out; the hypothesis that memory improves training was not supported here.", 6.72, 5.52, 5.28, 0.42, {
    fontSize: 12, bold: true, color: C.navy, breakLine: true,
  });
  await addZh(s, "知识治理已经闭环；本次实验不支持“记忆提升训练表现”的假设。", 6.72, 6.02, 5.28, 0.36, {
    fontSize: 11.5, bold: true, color: C.navy,
  });
  footer(s, 9);
}

const out = process.argv[2] || "outputs/nautilus_decision_admissibility_wp8_final_bilingual_20260724_r3.pptx";
await pptx.writeFile({ fileName: out });
console.log(`Wrote bilingual presentation: ${out}`);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
