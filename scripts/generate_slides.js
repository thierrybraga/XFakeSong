/**
 * generate_slides.js
 * Apresentacao de defesa do TCC — Pipeline Modular para Deteccao de Audio Sintetico
 * Thierry Goncalves Braga | UFSJ | 2026
 *
 * Uso: node scripts/generate_slides.js
 */

const pptxgen = require("pptxgenjs");
const path = require("path");

// ---------------------------------------------------------------------------
// Paleta de cores (Ocean Gradient — tom azul tecnologico)
// ---------------------------------------------------------------------------
const C = {
  navy:    "0A2540",   // fundo escuro
  blue:    "065A82",   // primario
  teal:    "1C7293",   // secundario
  mint:    "0E9AA7",   // destaque
  white:   "FFFFFF",
  light:   "EAF4FB",   // fundo claro
  gray:    "8899A6",
  darkgray:"334155",
  accent:  "F5A623",   // laranja dourado para destaques
  red:     "E74C3C",   // alerta / negativo
  green:   "27AE60",   // positivo / OK
};

const FONT_TITLE  = "Trebuchet MS";
const FONT_BODY   = "Calibri";

const W = 10;   // largura do slide (inches, LAYOUT_16x9)
const H = 5.625;

let pres = new pptxgen();
pres.layout  = "LAYOUT_16x9";
pres.author  = "Thierry Goncalves Braga";
pres.title   = "Pipeline Modular para Deteccao de Audio Sintetico — Defesa TCC UFSJ 2026";
pres.subject = "TCC UFSJ 2026";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
function makeShadow() {
  return { type: "outer", color: "000000", blur: 8, offset: 3, angle: 135, opacity: 0.18 };
}

function addTopBar(slide, color) {
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 0, w: W, h: 0.08, fill: { color }, line: { color }
  });
}

function addBottomBar(slide) {
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: H - 0.30, w: W, h: 0.30,
    fill: { color: C.navy }, line: { color: C.navy }
  });
  slide.addText("XFakeSong | UFSJ 2026 | Thierry Goncalves Braga", {
    x: 0.3, y: H - 0.28, w: 7, h: 0.26,
    fontSize: 9, color: C.gray, fontFace: FONT_BODY, valign: "middle"
  });
}

function titleSlideStyle(slide) {
  slide.background = { color: C.navy };
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 0, w: 0.08, h: H, fill: { color: C.mint }, line: { color: C.mint }
  });
}

function contentSlideStyle(slide) {
  slide.background = { color: C.light };
  addTopBar(slide, C.blue);
  addBottomBar(slide);
}

function sectionTitle(slide, text) {
  // Background heading bar
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 0.08, w: W, h: 0.72,
    fill: { color: C.blue }, line: { color: C.blue }
  });
  slide.addText(text, {
    x: 0.3, y: 0.08, w: W - 0.6, h: 0.72,
    fontSize: 22, bold: true, color: C.white,
    fontFace: FONT_TITLE, valign: "middle", margin: 0
  });
}

function stat(slide, value, label, x, y, w) {
  // Box
  slide.addShape(pres.shapes.RECTANGLE, {
    x, y, w, h: 1.1,
    fill: { color: C.white }, line: { color: C.teal, width: 2 },
    shadow: makeShadow()
  });
  slide.addText(value, {
    x, y: y + 0.05, w, h: 0.65,
    fontSize: 36, bold: true, color: C.blue,
    fontFace: FONT_TITLE, align: "center", margin: 0
  });
  slide.addText(label, {
    x, y: y + 0.70, w, h: 0.36,
    fontSize: 12, color: C.darkgray,
    fontFace: FONT_BODY, align: "center", margin: 0
  });
}

// ---------------------------------------------------------------------------
// SLIDE 1 — Capa
// ---------------------------------------------------------------------------
{
  let s = pres.addSlide();
  titleSlideStyle(s);

  // Titulo principal
  s.addText("Pipeline Modular para", {
    x: 0.6, y: 0.7, w: 9.2, h: 0.7,
    fontSize: 28, bold: true, color: C.mint,
    fontFace: FONT_TITLE, align: "left"
  });
  s.addText("Deteccao de Audio Sintetico", {
    x: 0.6, y: 1.35, w: 9.2, h: 0.7,
    fontSize: 28, bold: true, color: C.white,
    fontFace: FONT_TITLE, align: "left"
  });
  s.addText("Arquiteturas Neurais Hibridas e Analise Empirica de Caracteristicas", {
    x: 0.6, y: 2.05, w: 8.5, h: 0.55,
    fontSize: 16, italic: true, color: C.gray,
    fontFace: FONT_BODY, align: "left"
  });

  // Separador
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.6, y: 2.72, w: 4.5, h: 0.04,
    fill: { color: C.teal }, line: { color: C.teal }
  });

  // Autor e info
  s.addText("Thierry Goncalves Braga", {
    x: 0.6, y: 2.88, w: 9, h: 0.40,
    fontSize: 18, bold: true, color: C.white, fontFace: FONT_BODY
  });
  s.addText("Orientacao: Prof. Ana Claudia  |  UFSJ — Departamento de Ciencia da Computacao  |  2026", {
    x: 0.6, y: 3.30, w: 9.2, h: 0.35,
    fontSize: 13, color: C.gray, fontFace: FONT_BODY
  });
  s.addText("Projeto: github.com/XFakeSong", {
    x: 0.6, y: 3.70, w: 9.2, h: 0.30,
    fontSize: 13, color: C.mint, fontFace: FONT_BODY
  });

  // Tag TCC
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: 0.6, y: 4.25, w: 2.2, h: 0.45,
    fill: { color: C.mint }, rectRadius: 0.08, line: { color: C.mint }
  });
  s.addText("TRABALHO DE CONCLUSAO DE CURSO", {
    x: 0.65, y: 4.26, w: 2.1, h: 0.43,
    fontSize: 9.5, bold: true, color: C.white,
    fontFace: FONT_BODY, align: "center", valign: "middle", margin: 0
  });
}

// ---------------------------------------------------------------------------
// SLIDE 2 — Contexto: O Problema
// ---------------------------------------------------------------------------
{
  let s = pres.addSlide();
  contentSlideStyle(s);
  sectionTitle(s, "O Problema: Deepfakes de Audio em Portugues Brasileiro");

  // 3 cartoes de contexto
  const cards = [
    { icon: "79", title: "79 casos documentados", body: "deepfakes de audio nas eleicoes brasileiras de 2024 (DFRLab)", color: C.red },
    { icon: "73%", title: "Humanos acertam 73%", body: "taxa cai para 51% com TTS de ultima geracao (Muller et al., 600 participantes)", color: C.accent },
    { icon: "200ms", title: "<200ms latencia", body: "vozes sintéticas em tempo real via RVC v2 — imperceptiveis ao ouvido humano", color: C.teal },
  ];

  cards.forEach((c, i) => {
    const x = 0.3 + i * 3.25;
    s.addShape(pres.shapes.RECTANGLE, {
      x, y: 0.92, w: 3.1, h: 3.45,
      fill: { color: C.white }, line: { color: "DDDDDD", width: 1 },
      shadow: makeShadow()
    });
    // Faixa colorida topo
    s.addShape(pres.shapes.RECTANGLE, {
      x, y: 0.92, w: 3.1, h: 0.08,
      fill: { color: c.color }, line: { color: c.color }
    });
    // Numero grande
    s.addText(c.icon, {
      x: x + 0.1, y: 1.08, w: 2.9, h: 0.90,
      fontSize: 40, bold: true, color: c.color,
      fontFace: FONT_TITLE, align: "center"
    });
    s.addText(c.title, {
      x: x + 0.15, y: 2.0, w: 2.8, h: 0.40,
      fontSize: 13, bold: true, color: C.darkgray, fontFace: FONT_TITLE, align: "center"
    });
    s.addText(c.body, {
      x: x + 0.15, y: 2.44, w: 2.8, h: 0.85,
      fontSize: 12, color: C.darkgray, fontFace: FONT_BODY, align: "center"
    });
  });

  s.addText("Pipeline de codigo aberto para deteccao automatica de audio sintetico em PT-BR", {
    x: 0.3, y: 4.52, w: 9.4, h: 0.30,
    fontSize: 12, italic: true, color: C.teal, fontFace: FONT_BODY, align: "center"
  });
}

// ---------------------------------------------------------------------------
// SLIDE 3 — Objetivos
// ---------------------------------------------------------------------------
{
  let s = pres.addSlide();
  contentSlideStyle(s);
  sectionTitle(s, "Objetivos do Trabalho");

  const items = [
    { num: "01", text: "Construir pipeline modular open-source para deteccao de audio sintetico em PT-BR" },
    { num: "02", text: "Implementar e avaliar 5 familias de arquiteturas neurais (Conformer, EfficientNet-LSTM, MultiScale CNN, AASIST, RawNet2) + Ensemble Adaptativo" },
    { num: "03", text: "Construir dataset PT-BR balanceado (BRSpeech-DF) com divisao 70/15/15 e augmentation" },
    { num: "04", text: "Quantificar importancia de caracteristicas acusticas via analise de pesos espectrais" },
    { num: "05", text: "Avaliar robustez sob ruido AWGN (SNR 10-30dB) e medir eficiencia computacional" },
  ];

  items.forEach((item, i) => {
    const y = 0.95 + i * 0.83;
    s.addShape(pres.shapes.RECTANGLE, {
      x: 0.3, y, w: 0.55, h: 0.55,
      fill: { color: C.blue }, line: { color: C.blue }
    });
    s.addText(item.num, {
      x: 0.3, y, w: 0.55, h: 0.55,
      fontSize: 16, bold: true, color: C.white,
      fontFace: FONT_TITLE, align: "center", valign: "middle", margin: 0
    });
    s.addText(item.text, {
      x: 1.0, y: y + 0.06, w: 8.7, h: 0.50,
      fontSize: 14, color: C.darkgray, fontFace: FONT_BODY, valign: "middle"
    });
  });
}

// ---------------------------------------------------------------------------
// SLIDE 4 — Arquitetura do Pipeline
// ---------------------------------------------------------------------------
{
  let s = pres.addSlide();
  contentSlideStyle(s);
  sectionTitle(s, "Arquitetura do Pipeline XFakeSong");

  // Etapas do pipeline
  const steps = [
    { num: "1", label: "Audio\nEntrada", sub: "16kHz mono" },
    { num: "2", label: "VAD +\nAGC",     sub: "-23 LUFS" },
    { num: "3", label: "Extracao\nEspec.", sub: "Mel/MFCC\nCQT/LFCC" },
    { num: "4", label: "Modelos\nDL",    sub: "5 arquit." },
    { num: "5", label: "Ensemble\nAdapt.", sub: "Eq.27-28" },
    { num: "6", label: "SHAP\nExplain.", sub: "Tabela 4" },
  ];

  const bw = 1.35, bh = 1.3, startX = 0.3, y = 1.15;
  const gap = (W - 0.6 - steps.length * bw) / (steps.length - 1);

  steps.forEach((step, i) => {
    const x = startX + i * (bw + gap);
    const isLast = i === steps.length - 1;
    const boxColor = isLast ? C.mint : (i === 4 ? C.teal : C.blue);

    s.addShape(pres.shapes.RECTANGLE, {
      x, y, w: bw, h: bh,
      fill: { color: boxColor }, line: { color: boxColor },
      shadow: makeShadow()
    });
    s.addText(step.num, {
      x, y: y + 0.04, w: bw, h: 0.30,
      fontSize: 14, bold: true, color: "FFFFFF",
      fontFace: FONT_TITLE, align: "center", margin: 0
    });
    s.addText(step.label, {
      x, y: y + 0.30, w: bw, h: 0.58,
      fontSize: 13, bold: true, color: C.white,
      fontFace: FONT_TITLE, align: "center", margin: 0
    });
    s.addText(step.sub, {
      x, y: y + 0.86, w: bw, h: 0.40,
      fontSize: 10, color: "E0E0E0",
      fontFace: FONT_BODY, align: "center", margin: 0
    });

    // Seta
    if (i < steps.length - 1) {
      const arrowX = x + bw + 0.04;
      s.addShape(pres.shapes.LINE, {
        x: arrowX, y: y + bh / 2, w: gap - 0.08, h: 0,
        line: { color: C.teal, width: 2 }
      });
    }
  });

  // Nota tecnica
  s.addText("Conformer desativado (incompatibilidade Keras 3 + raw audio 80k steps) — arquitetura documentada, nao treinada.", {
    x: 0.3, y: 2.62, w: 9.4, h: 0.30,
    fontSize: 10, italic: true, color: C.gray, fontFace: FONT_BODY
  });

  // Dataset box
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.3, y: 3.05, w: 4.5, h: 1.25,
    fill: { color: C.white }, line: { color: C.teal, width: 1.5 },
    shadow: makeShadow()
  });
  s.addText("Dataset: BRSpeech-DF", {
    x: 0.5, y: 3.10, w: 4.1, h: 0.35,
    fontSize: 13, bold: true, color: C.blue, fontFace: FONT_TITLE
  });
  s.addText([
    { text: "600 amostras PT-BR", options: { bullet: true, breakLine: true } },
    { text: "300 bonafide + 300 spoof (1:1)", options: { bullet: true, breakLine: true } },
    { text: "Split 70/15/15 estratificado", options: { bullet: true } },
  ], { x: 0.5, y: 3.46, w: 4.1, h: 0.78, fontSize: 12, color: C.darkgray, fontFace: FONT_BODY });

  // Pre-processamento box
  s.addShape(pres.shapes.RECTANGLE, {
    x: 5.1, y: 3.05, w: 4.5, h: 1.25,
    fill: { color: C.white }, line: { color: C.blue, width: 1.5 },
    shadow: makeShadow()
  });
  s.addText("Pre-processamento", {
    x: 5.3, y: 3.10, w: 4.1, h: 0.35,
    fontSize: 13, bold: true, color: C.blue, fontFace: FONT_TITLE
  });
  s.addText([
    { text: "Silero VAD (fallback energia -25dB)", options: { bullet: true, breakLine: true } },
    { text: "AGC normalizacao RMS -23 LUFS", options: { bullet: true, breakLine: true } },
    { text: "Augmentation: pitch+-2st, stretch 0.9-1.1x", options: { bullet: true } },
  ], { x: 5.3, y: 3.46, w: 4.1, h: 0.78, fontSize: 12, color: C.darkgray, fontFace: FONT_BODY });
}

// ---------------------------------------------------------------------------
// SLIDE 5 — Dataset e Configuracao Experimental
// ---------------------------------------------------------------------------
{
  let s = pres.addSlide();
  contentSlideStyle(s);
  sectionTitle(s, "Dataset e Configuracao Experimental (Reais)");

  // Stats grandes
  stat(s, "600",    "amostras totais",      0.3,  0.98, 2.0);
  stat(s, "420",    "amostras treino",       2.5,  0.98, 2.0);
  stat(s, "90+90",  "val + teste",           4.7,  0.98, 2.0);
  stat(s, "1:1",    "real / fake",           6.9,  0.98, 2.0);

  // Tabela de config
  const rows = [
    ["Framework",   "TensorFlow 2.21 + Keras 3, Python 3.11"],
    ["Hardware",    "CPU (sem GPU — limitacao de ambiente)"],
    ["Epocas",      "20 (quick) | Early stopping paciencia=10"],
    ["Batch size",  "32 | Optimizer: Adam lr=0,001"],
    ["Augmentation","Pitch +-2 semitons + Time stretch 0,9-1,1x"],
    ["Metricas",    "Acuracia, EER (scipy brentq), AUC-ROC (sklearn)"],
  ];

  const tableData = rows.map(([k, v], i) => [
    { text: k, options: { bold: true, color: C.white, fill: { color: i % 2 === 0 ? C.blue : C.teal }, align: "left" } },
    { text: v, options: { color: C.darkgray, fill: { color: i % 2 === 0 ? "FFFFFF" : C.light }, align: "left" } },
  ]);

  s.addTable(tableData, {
    x: 0.3, y: 2.22, w: 9.4, h: 2.0,
    colW: [2.3, 7.1],
    border: { pt: 0.5, color: "CCCCCC" },
    fontSize: 12, fontFace: FONT_BODY,
  });
}

// ---------------------------------------------------------------------------
// SLIDE 6 — Resultados por Arquitetura (Tabela 3)
// ---------------------------------------------------------------------------
{
  let s = pres.addSlide();
  contentSlideStyle(s);
  sectionTitle(s, "Resultados por Arquitetura — Tabela 3 (Dados Reais)");

  const headers = ["Arquitetura", "Acuracia", "EER", "AUC-ROC", "Latencia", "Observacao"];
  const data = [
    ["MultiscaleCNN (Res2Net-50)", "97,8%", "5,6%",  "0,997", "133,7ms", "Melhor desempenho"],
    ["Ensemble Adaptativo",        "91,1%", "13,3%", "0,975", "48,1ms",  "Menor latencia util"],
    ["EfficientNet-LSTM",          "50,0%", "46,7%", "0,511", "63,8ms",  "Nao convergiu (CPU)"],
    ["AASIST",                     "50,0%", "50,0%", "0,500", "89,0ms",  "Nao convergiu (CPU)"],
    ["RawNet2",                    "50,0%", "50,0%", "0,000", "156,4ms", "Nao convergiu (CPU)"],
  ];

  const tableData = [
    headers.map(h => ({
      text: h,
      options: { bold: true, color: C.white, fill: { color: C.navy }, align: "center" }
    })),
    ...data.map((row, i) => {
      const isGood = i < 2;
      const isBad  = i >= 2;
      return row.map((cell, j) => ({
        text: cell,
        options: {
          color: j >= 1 && j <= 3 && isGood ? C.green :
                 j >= 1 && j <= 3 && isBad  ? C.red   : C.darkgray,
          bold: j >= 1 && j <= 3,
          fill: { color: i % 2 === 0 ? "FFFFFF" : C.light },
          align: j === 0 ? "left" : "center",
        }
      }));
    }),
  ];

  s.addTable(tableData, {
    x: 0.3, y: 0.85, w: 9.4, h: 3.65,
    colW: [2.5, 1.1, 1.0, 1.1, 1.2, 2.5],
    border: { pt: 0.5, color: "CCCCCC" },
    fontSize: 12, fontFace: FONT_BODY,
  });

  s.addText("Verde = convergiu  |  Vermelho = nao convergiu (CPU sem GPU, dataset <2.000 amostras)", {
    x: 0.3, y: 4.72, w: 9.4, h: 0.22,
    fontSize: 10, italic: true, color: C.gray, fontFace: FONT_BODY, align: "center"
  });
}

// ---------------------------------------------------------------------------
// SLIDE 7 — Analise: Por que Spectrograma > Audio Bruto
// ---------------------------------------------------------------------------
{
  let s = pres.addSlide();
  contentSlideStyle(s);
  sectionTitle(s, "Analise: Frontends Espectrais vs. Audio Bruto");

  // Coluna esquerda — GANHOU
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.3, y: 0.92, w: 4.5, h: 3.65,
    fill: { color: C.white }, line: { color: C.green, width: 2 },
    shadow: makeShadow()
  });
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.3, y: 0.92, w: 4.5, h: 0.45,
    fill: { color: C.green }, line: { color: C.green }
  });
  s.addText("CONVERGIRAM  (Spectrograma)", {
    x: 0.35, y: 0.92, w: 4.4, h: 0.45,
    fontSize: 13, bold: true, color: C.white, fontFace: FONT_TITLE, valign: "middle", margin: 0
  });
  s.addText([
    { text: "MultiscaleCNN (Res2Net-50)", options: { bold: true, breakLine: true } },
    { text: "Frontend: STFT -> Mel spectrogram -> CNN 2D", options: { breakLine: true } },
    { text: "Resultado: 97,8% acc | EER 5,6%", options: { bold: true, breakLine: true, color: C.green } },
    { text: " ", options: { breakLine: true } },
    { text: "Ensemble Adaptativo", options: { bold: true, breakLine: true } },
    { text: "Frontend: Mel+LFCC+CQT+MFCC+Mel2 -> CNN", options: { breakLine: true } },
    { text: "Resultado: 91,1% acc | EER 13,3%", options: { bold: true, color: C.green } },
  ], { x: 0.45, y: 1.44, w: 4.2, h: 3.05, fontSize: 13, color: C.darkgray, fontFace: FONT_BODY });

  // Coluna direita — PERDEU
  s.addShape(pres.shapes.RECTANGLE, {
    x: 5.15, y: 0.92, w: 4.5, h: 3.65,
    fill: { color: C.white }, line: { color: C.red, width: 2 },
    shadow: makeShadow()
  });
  s.addShape(pres.shapes.RECTANGLE, {
    x: 5.15, y: 0.92, w: 4.5, h: 0.45,
    fill: { color: C.red }, line: { color: C.red }
  });
  s.addText("NAO CONVERGIRAM  (Audio Bruto)", {
    x: 5.2, y: 0.92, w: 4.4, h: 0.45,
    fontSize: 13, bold: true, color: C.white, fontFace: FONT_TITLE, valign: "middle", margin: 0
  });
  s.addText([
    { text: "EfficientNet-LSTM  |  AASIST  |  RawNet2", options: { bold: true, breakLine: true } },
    { text: "Resultado: 50% acc (acaso)", options: { bold: true, breakLine: true, color: C.red } },
    { text: " ", options: { breakLine: true } },
    { text: "Por que nao convergiram?", options: { bold: true, breakLine: true } },
    { text: "- Sequencias de 16.000 pontos (audio bruto)", options: { bullet: true, breakLine: true } },
    { text: "- AASIST: SincConv+GAT exige GPU + >5.000 amostras", options: { bullet: true, breakLine: true } },
    { text: "- RawNet2: GRU sobre PCM e lento sem GPU", options: { bullet: true, breakLine: true } },
    { text: "- Literatura: AASIST EER 0,83% com ASVspoof2019", options: { bullet: true } },
  ], { x: 5.25, y: 1.44, w: 4.2, h: 3.05, fontSize: 12, color: C.darkgray, fontFace: FONT_BODY });
}

// ---------------------------------------------------------------------------
// SLIDE 8 — Robustez e Caracteristicas
// ---------------------------------------------------------------------------
{
  let s = pres.addSlide();
  contentSlideStyle(s);
  sectionTitle(s, "Robustez sob Ruido AWGN e Importancia Espectral");

  // --- Tabela robustez (esquerda) ---
  s.addText("Tabela 5: Robustez — Acuracia (%)", {
    x: 0.3, y: 0.88, w: 5.0, h: 0.30,
    fontSize: 12, bold: true, color: C.blue, fontFace: FONT_TITLE
  });

  const snrData = [
    [
      { text: "Arquitetura",    options: { bold: true, color: C.white, fill: { color: C.navy } } },
      { text: "Limpo",          options: { bold: true, color: C.white, fill: { color: C.navy } } },
      { text: "SNR 30dB",       options: { bold: true, color: C.white, fill: { color: C.navy } } },
      { text: "SNR 20dB",       options: { bold: true, color: C.white, fill: { color: C.navy } } },
      { text: "SNR 10dB",       options: { bold: true, color: C.white, fill: { color: C.navy } } },
    ],
    [
      { text: "MultiscaleCNN", options: { fill: { color: "FFFFFF" }, color: C.darkgray, bold: true } },
      { text: "97,8%", options: { fill: { color: "FFFFFF" }, color: C.green, bold: true } },
      { text: "71,1%", options: { fill: { color: "FFFFFF" }, color: C.accent, bold: true } },
      { text: "72,2%", options: { fill: { color: "FFFFFF" }, color: C.accent, bold: true } },
      { text: "58,9%", options: { fill: { color: "FFFFFF" }, color: C.red, bold: true } },
    ],
    [
      { text: "Ensemble", options: { fill: { color: C.light }, color: C.darkgray, bold: true } },
      { text: "87,8%", options: { fill: { color: C.light }, color: C.green, bold: true } },
      { text: "74,4%", options: { fill: { color: C.light }, color: C.accent, bold: true } },
      { text: "50,0%", options: { fill: { color: C.light }, color: C.red, bold: true } },
      { text: "50,0%", options: { fill: { color: C.light }, color: C.red, bold: true } },
    ],
  ];

  s.addTable(snrData, {
    x: 0.3, y: 1.20, w: 5.5, h: 1.55,
    colW: [2.0, 0.9, 0.9, 0.9, 0.8],
    border: { pt: 0.5, color: "CCCCCC" },
    fontSize: 12, fontFace: FONT_BODY,
  });

  s.addText("MultiscaleCNN mais robusto a ruido. Ensemble degrada em SNR<=20dB — augmentation com ruido e necessario.", {
    x: 0.3, y: 2.82, w: 5.5, h: 0.40,
    fontSize: 10.5, italic: true, color: C.gray, fontFace: FONT_BODY
  });

  // --- Grafico de barras ramos Ensemble (direita) ---
  s.addText("Tabela 4: Contribuicao dos Ramos (Ensemble)", {
    x: 6.0, y: 0.88, w: 3.7, h: 0.30,
    fontSize: 12, bold: true, color: C.blue, fontFace: FONT_TITLE
  });

  s.addChart(pres.charts.BAR, [{
    name: "Contribuicao (%)",
    labels: ["Mel Spec.", "LFCC", "MFCC", "CQT"],
    values: [31.8, 24.7, 22.1, 21.4],
  }], {
    x: 5.9, y: 1.15, w: 3.7, h: 2.1,
    barDir: "col",
    chartColors: [C.blue, C.teal, C.mint, C.gray],
    chartArea: { fill: { color: "FFFFFF" }, roundedCorners: false },
    catAxisLabelColor: C.darkgray,
    valAxisLabelColor: C.gray,
    valGridLine: { color: "E2E8F0", size: 0.5 },
    catGridLine: { style: "none" },
    showValue: true,
    dataLabelPosition: "outEnd",
    dataLabelColor: C.navy,
    showLegend: false,
    valAxisMaxVal: 40,
  });

  s.addText("Mel Spectrogram domina (31,8%) — artefatos visuais de vocoder em >4kHz [2,22]", {
    x: 5.9, y: 3.32, w: 3.8, h: 0.35,
    fontSize: 10.5, italic: true, color: C.gray, fontFace: FONT_BODY
  });

  // Mensagem chave
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.3, y: 3.35, w: 5.4, h: 0.80,
    fill: { color: C.blue }, line: { color: C.blue }
  });
  s.addText("INSIGHT PRINCIPAL: Representacoes espectrais (mel spectrogram) convergem em datasets pequenos. Audio bruto requer GPU + 5.000+ amostras.", {
    x: 0.45, y: 3.38, w: 5.1, h: 0.74,
    fontSize: 11.5, bold: true, color: C.white, fontFace: FONT_BODY, valign: "middle"
  });
}

// ---------------------------------------------------------------------------
// SLIDE 9 — Demo ao Vivo
// ---------------------------------------------------------------------------
{
  let s = pres.addSlide();
  titleSlideStyle(s);

  s.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 1.4, w: W, h: 2.8,
    fill: { color: C.blue, transparency: 20 }, line: { color: C.blue }
  });

  s.addText("DEMONSTRACAO AO VIVO", {
    x: 0.5, y: 0.4, w: 9, h: 0.7,
    fontSize: 32, bold: true, color: C.mint,
    fontFace: FONT_TITLE, align: "center"
  });
  s.addText("Pipeline XFakeSong — Interface Gradio", {
    x: 0.5, y: 1.08, w: 9, h: 0.40,
    fontSize: 18, italic: true, color: C.gray, fontFace: FONT_BODY, align: "center"
  });

  s.addText([
    { text: "1. Carregar audio real (voz humana)", options: { breakLine: true } },
    { text: "2. Carregar audio sintetico (BRSpeech-DF spoof)", options: { breakLine: true } },
    { text: "3. Executar pipeline: VAD -> AGC -> MultiscaleCNN -> Score", options: { breakLine: true } },
    { text: "4. Comparar scores: P(fake) real vs. P(fake) sintetico", options: { breakLine: true } },
    { text: "5. Visualizar interpretabilidade (espectrograma mel)", },
  ], {
    x: 1.5, y: 1.55, w: 7, h: 2.4,
    fontSize: 16, color: C.white, fontFace: FONT_BODY,
    bullet: false, valign: "middle"
  });

  s.addText("python app/gradio_app.py", {
    x: 2.0, y: 4.1, w: 6, h: 0.45,
    fontSize: 16, bold: true, color: C.mint, fontFace: "Consolas", align: "center"
  });
}

// ---------------------------------------------------------------------------
// SLIDE 10 — Conclusoes e Trabalhos Futuros
// ---------------------------------------------------------------------------
{
  let s = pres.addSlide();
  titleSlideStyle(s);

  s.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 0.08, w: W, h: 0.08, fill: { color: C.mint }, line: { color: C.mint }
  });

  s.addText("Conclusoes e Trabalhos Futuros", {
    x: 0.6, y: 0.25, w: 9, h: 0.6,
    fontSize: 26, bold: true, color: C.white, fontFace: FONT_TITLE, valign: "middle"
  });

  // Conclusoes
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.4, y: 0.95, w: 4.5, h: 0.35,
    fill: { color: C.green }, line: { color: C.green }
  });
  s.addText("CONCLUSOES", {
    x: 0.4, y: 0.95, w: 4.5, h: 0.35,
    fontSize: 13, bold: true, color: C.white, fontFace: FONT_TITLE, align: "center", valign: "middle", margin: 0
  });
  s.addText([
    { text: "MultiscaleCNN: 97,8% acc, EER 5,6% com 420 amostras em CPU", options: { bullet: true, breakLine: true, bold: true } },
    { text: "Ensemble Adaptativo: 91,1% acc, 48ms latencia — adequado para tempo real", options: { bullet: true, breakLine: true } },
    { text: "Frontends espectrais convergem melhor em datasets pequenos", options: { bullet: true, breakLine: true } },
    { text: "Pipeline XFakeSong: codigo aberto, modular, reproducivel", options: { bullet: true } },
  ], {
    x: 0.4, y: 1.32, w: 4.5, h: 2.35,
    fontSize: 12.5, color: C.white, fontFace: FONT_BODY
  });

  // Trabalhos Futuros
  s.addShape(pres.shapes.RECTANGLE, {
    x: 5.1, y: 0.95, w: 4.5, h: 0.35,
    fill: { color: C.teal }, line: { color: C.teal }
  });
  s.addText("TRABALHOS FUTUROS", {
    x: 5.1, y: 0.95, w: 4.5, h: 0.35,
    fontSize: 13, bold: true, color: C.white, fontFace: FONT_TITLE, align: "center", valign: "middle", margin: 0
  });
  s.addText([
    { text: "Ampliar dataset para ASVspoof 2021 LA + ASVSPOOF 5", options: { bullet: true, breakLine: true } },
    { text: "Treinar AASIST/RawNet2 com GPU (>5.000 amostras)", options: { bullet: true, breakLine: true } },
    { text: "Augmentation com ruido AWGN no treino (robustez)", options: { bullet: true, breakLine: true } },
    { text: "Avaliacao cross-generator: VALL-E, Bark, Tacotron2", options: { bullet: true } },
  ], {
    x: 5.1, y: 1.32, w: 4.5, h: 2.35,
    fontSize: 12.5, color: C.white, fontFace: FONT_BODY
  });

  // Rodape
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 4.35, w: W, h: 1.275,
    fill: { color: "091C2E" }, line: { color: "091C2E" }
  });
  s.addText("Obrigado! Perguntas?", {
    x: 0.5, y: 4.42, w: 9, h: 0.5,
    fontSize: 22, bold: true, color: C.mint, fontFace: FONT_TITLE, align: "center"
  });
  s.addText("Thierry Goncalves Braga | github.com/XFakeSong | UFSJ 2026", {
    x: 0.5, y: 4.92, w: 9, h: 0.35,
    fontSize: 12, color: C.gray, fontFace: FONT_BODY, align: "center"
  });
}

// ---------------------------------------------------------------------------
// Salvar
// ---------------------------------------------------------------------------
const outputPath = path.join(__dirname, "../results/Apresentacao_TCC_XFakeSong.pptx");
pres.writeFile({ fileName: outputPath })
  .then(() => console.log("PPTX gerado:", outputPath))
  .catch(err => { console.error("Erro:", err); process.exit(1); });
