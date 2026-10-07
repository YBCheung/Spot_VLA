#!/usr/bin/env python3
"""Generate the MSc thesis seminar deck (~22-min talk for a 30-min slot incl. a short Q&A)."""
import os
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from PIL import Image

FIG = "figures"
OUT = "thesis_seminar.pptx"

# ---- palette ----
INK      = RGBColor(0x1A, 0x1A, 0x1A)
GREY     = RGBColor(0x55, 0x55, 0x55)
LIGHT    = RGBColor(0x8A, 0x8A, 0x8A)
AALTO    = RGBColor(0xFF, 0xD1, 0x00)   # Aalto yellow
BLUE     = RGBColor(0x00, 0x6F, 0xB9)
GREEN    = RGBColor(0x1B, 0x8A, 0x4A)
RED      = RGBColor(0xC0, 0x2A, 0x2A)
PAPER    = RGBColor(0xFF, 0xFF, 0xFF)
BAND     = RGBColor(0xF4, 0xF4, 0xF2)
CARD     = RGBColor(0xF7, 0xF6, 0xF2)

prs = Presentation()
prs.slide_width  = Inches(13.333)
prs.slide_height = Inches(7.5)
SW, SH = prs.slide_width, prs.slide_height
BLANK = prs.slide_layouts[6]

def slide():
    return prs.slides.add_slide(BLANK)

def rect(s, x, y, w, h, color, line=None):
    from pptx.enum.shapes import MSO_SHAPE
    sp = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, x, y, w, h)
    sp.fill.solid(); sp.fill.fore_color.rgb = color
    if line is None:
        sp.line.fill.background()
    else:
        sp.line.color.rgb = line; sp.line.width = Pt(1)
    sp.shadow.inherit = False
    return sp

def txt(s, x, y, w, h, runs, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP,
        space_after=6, line_spacing=1.06, wrap=True):
    """runs: list of paragraphs; each paragraph a list of (text,size,bold,color,italic)."""
    tb = s.shapes.add_textbox(x, y, w, h); tf = tb.text_frame
    tf.word_wrap = wrap; tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Pt(0)
    tf.margin_top = tf.margin_bottom = Pt(0)
    for i, para in enumerate(runs):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align; p.space_after = Pt(space_after); p.space_before = Pt(0)
        p.line_spacing = line_spacing
        for (t, sz, b, c, *rest) in para:
            it = rest[0] if rest else False
            r = p.add_run(); r.text = t
            r.font.size = Pt(sz); r.font.bold = b; r.font.color.rgb = c
            r.font.italic = it; r.font.name = "Calibri"
    return tb

def bullets(s, x, y, w, h, items, size=17, gap=9, color=INK, lead=AALTO, lh=1.08):
    tb = s.shapes.add_textbox(x, y, w, h); tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = Pt(0); tf.margin_top = tf.margin_bottom = Pt(0)
    for i, it in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(gap); p.space_before = Pt(0); p.line_spacing = lh
        # bullet dash
        rd = p.add_run(); rd.text = "— "
        rd.font.size = Pt(size); rd.font.bold = True; rd.font.color.rgb = lead
        rd.font.name = "Calibri"
        if isinstance(it, tuple):
            head, rest = it
            rh = p.add_run(); rh.text = head
            rh.font.size = Pt(size); rh.font.bold = True; rh.font.color.rgb = color
            rh.font.name = "Calibri"
            rr = p.add_run(); rr.text = rest
            rr.font.size = Pt(size); rr.font.bold = False; rr.font.color.rgb = color
            rr.font.name = "Calibri"
        else:
            r = p.add_run(); r.text = it
            r.font.size = Pt(size); r.font.color.rgb = color; r.font.name = "Calibri"
    return tb

def notes(s, text):
    s.notes_slide.notes_text_frame.text = text

def pic_fit(s, path, x, y, w, h, align="center", valign="middle"):
    """Fit image inside box (x,y,w,h) preserving aspect ratio."""
    iw, ih = Image.open(path).size
    ar = iw/ih; box_ar = w/h
    if ar > box_ar:
        nw = w; nh = int(w/ar)
    else:
        nh = h; nw = int(h*ar)
    if align == "center":  px = x + (w-nw)//2
    elif align == "left":  px = x
    else:                  px = x + (w-nw)
    if valign == "middle": py = y + (h-nh)//2
    elif valign == "top":  py = y
    else:                  py = y + (h-nh)
    return s.shapes.add_picture(path, px, py, nw, nh)

def header(s, kicker, title, num):
    rect(s, 0, 0, SW, Inches(1.18), PAPER)
    rect(s, Inches(0.55), Inches(0.30), Inches(0.11), Inches(0.62), AALTO)
    txt(s, Inches(0.80), Inches(0.22), Inches(10.8), Inches(0.34),
        [[(kicker, 12.5, True, LIGHT)]], space_after=0)
    txt(s, Inches(0.80), Inches(0.50), Inches(11.6), Inches(0.6),
        [[(title, 25, True, INK)]], space_after=0)
    rect(s, 0, Inches(1.18), SW, Pt(2.2), AALTO)
    # page number
    txt(s, Inches(12.5), Inches(7.02), Inches(0.7), Inches(0.32),
        [[(str(num), 11, False, LIGHT)]], align=PP_ALIGN.RIGHT, space_after=0)

def footer(s, label):
    txt(s, Inches(0.55), Inches(7.02), Inches(9), Inches(0.32),
        [[("Yibo Zhang · Offline Trajectory Metrics for Early Stopping in VLA Fine-Tuning", 10, False, LIGHT)]],
        space_after=0)

N = 0
def num():
    global N; N += 1; return N

# ============================================================= 1 TITLE
s = slide()
rect(s, 0, 0, SW, SH, PAPER)
rect(s, 0, 0, Inches(0.32), SH, AALTO)
rect(s, 0, Inches(4.98), SW, Pt(2.2), AALTO)
txt(s, Inches(0.95), Inches(0.7), Inches(11.4), Inches(0.4),
    [[("MSc THESIS SEMINAR  ·  AALTO UNIVERSITY", 13.5, True, LIGHT)]], space_after=0)
txt(s, Inches(0.9), Inches(1.55), Inches(11.7), Inches(2.6),
    [[("Offline Trajectory Metrics for Early Stopping", 40, True, INK)],
     [("in Vision-Language-Action Model Fine-Tuning", 40, True, INK)]],
    space_after=6, line_spacing=1.02)
txt(s, Inches(0.95), Inches(3.75), Inches(11.4), Inches(0.6),
    [[("When should you stop fine-tuning a robot policy — ", 18, False, GREY),
      ("without running it?", 18, True, BLUE)]], space_after=0)
txt(s, Inches(0.95), Inches(5.25), Inches(11.4), Inches(1.6),
    [[("Yibo Zhang", 20, True, INK)],
     [("Supervisor:  Prof. Joni Pajarinen        Advisor:  Dr. Wenyan Yang", 14.5, False, GREY)],
     [("Department of Electrical Engineering and Automation  ·  Aalto Robot Learning Lab", 13, False, LIGHT)],
     [("Espoo, 21 July 2026", 13, False, LIGHT)]], space_after=7)
notes(s, "Good morning. My thesis asks a very practical question in robot learning: when fine-tuning a "
          "vision-language-action model, when do we stop? The honest answer today is 'after a fixed number "
          "of steps' — and I'll show why that is often suboptimal, and what cheap offline signal we can use "
          "instead. [~40s] Target: ~22 minutes of talk within the 30-minute slot, leaving time for questions.")

# ============================================================= 2 BACKGROUND / PRIMER
# (added for the mixed audience; the slides below were renumbered 3..18 to match.)
s = slide(); header(s, "BACKGROUND", "A one-minute primer on the terms used today", num()); footer(s,"")
prim=[
    ("Robot policy (VLA)", "A single neural network that takes a camera image plus a plain-language "
     "instruction (“put the carrot on the plate”) and outputs the robot’s next movements.", BLUE),
    ("Fine-tuning (LoRA)", "Adapting a large pretrained model to a new task with only a little data and "
     "compute — here by training a small set of add-on parameters (LoRA).", GREEN),
    ("Checkpoint", "A snapshot of the model saved periodically during training. One run produces many "
     "snapshots; in the end we keep just one.", RGBColor(0x8A,0x62,0x00)),
    ("Early stopping", "Choosing which snapshot to keep. Done well, it needs a signal that predicts real "
     "robot performance — the question this thesis studies.", RGBColor(0x6A,0x3D,0x9A)),
]
x0=Inches(0.8); cw=Inches(5.75); gap=Inches(0.25); rh=Inches(2.1); ty=Inches(1.6)
for i,(h,b,c) in enumerate(prim):
    cx=Emu(int(x0)+(i%2)*(int(cw)+int(gap)))
    cy=Emu(int(ty)+(i//2)*(int(rh)+int(Inches(0.3))))
    rect(s, cx, cy, cw, rh, CARD)
    rect(s, cx, cy, cw, Inches(0.12), c)
    txt(s, Emu(int(cx)+Inches(0.3)), Emu(int(cy)+Inches(0.3)), Inches(5.2), Inches(0.6),
        [[(h, 17.5, True, c)]], space_after=0, line_spacing=1.0)
    txt(s, Emu(int(cx)+Inches(0.3)), Emu(int(cy)+Inches(0.98)), Inches(5.2), Inches(1.0),
        [[(b, 13.8, False, GREY)]], space_after=0, line_spacing=1.12)
txt(s, Inches(0.8), Inches(6.35), Inches(11.8), Inches(0.6),
    [[("In one line:  ", 15, True, INK),
      ("can we predict how well the robot will perform — without the cost of actually running it?", 15, False, GREY)]],
    space_after=0, line_spacing=1.1)
notes(s, "A quick primer, since this room spans several fields. A VLA is one network that maps a camera image "
          "and a language instruction to robot actions. We do not train it from scratch — we fine-tune a "
          "pretrained model with LoRA, which is cheap. During training we periodically save checkpoints, and "
          "the practical question is simply which checkpoint to keep. Choosing the best one is early stopping, "
          "and doing it well needs a signal that predicts real robot performance — ideally without running "
          "the robot every time. That is what the rest of the talk is about. [~55s]")

# ============================================================= 3 MOTIVATION
s = slide(); header(s, "MOTIVATION", "Fine-tuning a VLA: powerful, but when do we stop?", num()); footer(s,"")
bullets(s, Inches(0.8), Inches(1.55), Inches(6.15), Inches(4.4), [
    ("VLAs are the foundation-model recipe for robots. ", "One model maps camera + language instruction to robot actions."),
    ("Deployment means fine-tuning. ", "A pretrained VLA is adapted with LoRA to a new task, robot, or scene."),
    ("Training is stopped by a fixed budget. ", "Many current VLAs are trained for a preset number of steps, with no validation-based stopping rule."),
    ("But the optimum is not fixed. ", "It shifts with dataset size, learning rate, and batch size: too long overfits, too short underfits."),
], size=16.5, gap=15)
# right card
rect(s, Inches(7.35), Inches(1.7), Inches(5.25), Inches(3.9), CARD)
rect(s, Inches(7.35), Inches(1.7), Inches(0.10), Inches(3.9), AALTO)
txt(s, Inches(7.65), Inches(1.95), Inches(4.7), Inches(0.5),
    [[("The core tension", 17, True, INK)]], space_after=0)
bullets(s, Inches(7.65), Inches(2.55), Inches(4.75), Inches(2.9), [
    ("Online success = what we ultimately want, ", "but a rollout per checkpoint is slow, high-variance, and can wear on hardware."),
    ("Training loss = cheap, ", "but a poor proxy for closed-loop task success."),
    ("We need a middle ground: ", "an offline signal, computed every checkpoint, that predicts success."),
], size=15, gap=13, color=GREY)
notes(s, "VLAs are the robot version of foundation models. In practice we don't train from scratch — we "
          "fine-tune with LoRA. The problem: everyone trains for a fixed number of steps, with no principled "
          "stopping criterion. The optimum isn't fixed — it depends on data size, LR, batch size. The tension "
          "on the right frames the whole thesis: online success is the truth but too expensive to measure "
          "every checkpoint; training loss is cheap but a bad proxy. We want something in between. [~1:15]")

# ============================================================= 4 THE GAP
s = slide(); header(s, "THE PROBLEM", "Why training loss is a limited stopping signal", num()); footer(s,"")
cols = [
    ("Teacher-forced,\nper-step", "Loss is computed on ground-truth context, one timestep at a time. It never sees a rolled-out trajectory, so it can miss compounding error.", BLUE),
    ("Weakly aligned\nwith success", "In behaviour cloning, validation loss often correlates only weakly with closed-loop success, so the loss-minimising checkpoint need not be the success-maximising one.", RED),
    ("Averaging hides\nstructure", "A scalar mean of per-step errors does not distinguish a smooth-but-lagged motion from a jittery-but-timed one — errors of shape vs. phase.", GREEN),
]
x0 = Inches(0.8); cw = Inches(3.86); gap = Inches(0.30); top = Inches(1.75); ch = Inches(3.7)
for i,(h,b,c) in enumerate(cols):
    x = Emu(int(x0) + i*(int(cw)+int(gap)))
    rect(s, x, top, cw, ch, CARD)
    rect(s, x, top, cw, Inches(0.14), c)
    txt(s, Emu(int(x)+Inches(0.25)), Emu(int(top)+Inches(0.35)), Emu(int(cw)-Inches(0.5)), Inches(1.0),
        [[(h.replace("\n"," "), 19, True, c)]], space_after=0)
    txt(s, Emu(int(x)+Inches(0.25)), Emu(int(top)+Inches(1.35)), Emu(int(cw)-Inches(0.5)), Inches(2.2),
        [[(b, 14.5, False, GREY)]], space_after=0, line_spacing=1.1)
txt(s, Inches(0.8), Inches(5.85), Inches(11.8), Inches(1.0),
    [[("Consequence  ", 16, True, INK),
      ("— early stopping on loss can pick a checkpoint that is worse in the sense that matters most at "
       "deployment: closed-loop success. This thesis measures that cost directly as ", 16, False, GREY),
      ("selection regret.", 16, True, BLUE)]], space_after=0, line_spacing=1.1)
notes(s, "Three structural reasons loss fails. One: it's teacher-forced and per-step — never a rollout, so "
          "blind to compounding error. Two: empirically it correlates weakly with success in BC. Three: it's "
          "an average, so it can't tell a lagged-but-smooth motion from a jittery one. The consequence, "
          "bottom, is that stopping on loss forgoes real success — I quantify that as selection regret. [~1:10]")

# ============================================================= 5 IDEA
s = slide(); header(s, "PROPOSED IDEA", "Compare whole trajectories, not single steps", num()); footer(s,"")
txt(s, Inches(0.8), Inches(1.45), Inches(11.7), Inches(0.75),
    [[("Chunking VLAs predict a short ", 17, False, INK),
      ("sequence", 17, True, BLUE),
      (" of future actions. That makes the unit of comparison a ", 17, False, INK),
      ("trajectory segment", 17, True, BLUE),
      (" — so geometric, sequence-level similarity to the expert becomes measurable.", 17, False, INK)]],
    space_after=0, line_spacing=1.12)
cards = [
    ("DTW", "Dynamic Time Warping", "Aligns two sequences in time. Tolerant of pace / phase lag — forgives a correct shape run slightly fast or slow.", BLUE),
    ("OT", "Optimal Transport", "Matches sequences as distributions of waypoints (entropic Sinkhorn). Geometry-aware and a true metric.", GREEN),
    ("COS", "Cosine distance", "Per-step direction agreement, magnitude-blind. Cheap; used alongside, never alone.", RGBColor(0x8A,0x62,0x00)),
    ("NLL", "Likelihood surrogate", "Token NLL for discrete policies; the flow-matching / L1–L2 loss stands in for continuous ones.", RED),
]
x0=Inches(0.8); cw=Inches(2.86); gap=Inches(0.19); top=Inches(2.65); ch=Inches(2.95)
for i,(tag,name,desc,c) in enumerate(cards):
    x=Emu(int(x0)+i*(int(cw)+int(gap)))
    rect(s, x, top, cw, ch, CARD)
    rect(s, x, top, cw, Inches(0.62), c)
    txt(s, x, Emu(int(top)+Inches(0.09)), cw, Inches(0.45),
        [[(tag, 21, True, PAPER)]], align=PP_ALIGN.CENTER, space_after=0)
    txt(s, Emu(int(x)+Inches(0.2)), Emu(int(top)+Inches(0.78)), Emu(int(cw)-Inches(0.4)), Inches(0.5),
        [[(name, 13.5, True, INK)]], space_after=0, align=PP_ALIGN.CENTER)
    txt(s, Emu(int(x)+Inches(0.2)), Emu(int(top)+Inches(1.3)), Emu(int(cw)-Inches(0.4)), Inches(1.6),
        [[(desc, 12.5, False, GREY)]], space_after=0, line_spacing=1.08, align=PP_ALIGN.CENTER)
txt(s, Inches(0.8), Inches(5.95), Inches(11.8), Inches(0.8),
    [[("All metrics normalised per action dimension, ", 15, False, GREY),
      ("lower = better", 15, True, INK),
      (". L1 / L2 per-step distances serve as the baseline against which the geometric metrics are tested.", 15, False, GREY)]],
    space_after=0, line_spacing=1.1)
notes(s, "The key idea: chunking VLAs output a sequence, so we can compare trajectory segments, not steps. "
          "Four candidate metrics. DTW aligns in time — forgives pace. OT matches waypoint distributions, a "
          "true metric. Cosine is cheap directional agreement, magnitude-blind, so only ever a companion. NLL "
          "is the probabilistic one, but the surrogate is architecture-dependent — remember that, it comes "
          "back in the results. L1/L2 are the baseline the geometric metrics must beat. [~1:20]")

# ============================================================= 6 METHOD PIPELINE
s = slide(); header(s, "METHOD", "The pipeline: score → select → judge", num()); footer(s,"")
steps = [
    ("1", "Score every checkpoint", "On held-out expert data, compute the trajectory metric  m̄(k)  for each saved checkpoint.", BLUE),
    ("2", "Select one checkpoint", "Early-stopping rule: earliest checkpoint within 1% of the metric's best value  →  k*_m.", GREEN),
    ("3", "Judge the metric", "Against true success Q(k):  Spearman ρ (rank agreement, +1 to −1) and selection regret (success forgone).", RED),
]
top=Inches(1.7); bh=Inches(1.28); x=Inches(0.8); w=Inches(11.7)
for i,(n,h,b,c) in enumerate(steps):
    y=Emu(int(top)+i*(int(bh)+int(Inches(0.28))))
    rect(s, x, y, w, bh, CARD)
    rect(s, x, y, Inches(1.28), bh, c)
    txt(s, x, Emu(int(y)+Inches(0.30)), Inches(1.28), Inches(0.7),
        [[(n, 34, True, PAPER)]], align=PP_ALIGN.CENTER, space_after=0)
    txt(s, Emu(int(x)+Inches(1.6)), Emu(int(y)+Inches(0.20)), Inches(9.7), Inches(0.5),
        [[(h, 19, True, INK)]], space_after=0)
    txt(s, Emu(int(x)+Inches(1.6)), Emu(int(y)+Inches(0.68)), Inches(9.7), Inches(0.5),
        [[(b, 14.5, False, GREY)]], space_after=0)
txt(s, Inches(0.8), Inches(6.35), Inches(11.8), Inches(0.7),
    [[("Selection rule:   ", 14, True, INK),
      ("k*_m = min { k : m̄(k) ≤ min m̄ + η },   η = 0.01 · |m̄(0) − min m̄|", 15, False, BLUE, True),
      ("      Regret(m) = max_k Q(k) − Q(k*_m)", 15, False, GREY, True)]],
    space_after=0)
notes(s, "The method is three steps applied identically to every checkpoint. One: score each checkpoint with "
          "the metric on held-out data. Two: the early-stopping rule turns that curve into a single choice — "
          "the earliest checkpoint within 1% of the best value, so we don't chase one noisy minimum. Three: "
          "judge each metric against true success by two numbers — Spearman rho for ranking and regret for "
          "the success we gave up. High rho, low regret is a good metric. [~1:10]")

# ============================================================= 7 RESEARCH QUESTIONS
s = slide(); header(s, "RESEARCH QUESTIONS", "Four questions, each a check on the last", num()); footer(s,"")
rqs = [
    ("RQ1", "Do the metrics carry any information at all?", "Model-independent precondition: can a metric tell same-task from different-task expert trajectories? If not, it is disqualified.", BLUE),
    ("RQ2", "Do they correlate with closed-loop success?", "The central question — across 3 architectures and 2 data sizes, does the offline signal predict online success Q(k)?", GREEN),
    ("RQ3", "Is metric-based early stopping effective?", "Does the signal survive the hard, converged plateau — and can it catch genuine overtraining when it happens?", RED),
    ("RQ4", "Which differences are statistically real?", "With as few as 11 evaluations per run, which rankings survive bootstrap resampling rather than sampling noise?", RGBColor(0x6A,0x3D,0x9A)),
]
x0=Inches(0.8); cw=Inches(5.75); gap=Inches(0.2); rh=Inches(2.15); ty=Inches(1.65)
for i,(tag,q,d,c) in enumerate(rqs):
    cx = Emu(int(x0)+(i%2)*(int(cw)+int(gap)))
    cy = Emu(int(ty)+(i//2)*(int(rh)+int(Inches(0.28))))
    rect(s, cx, cy, cw, rh, CARD)
    rect(s, cx, cy, Inches(0.12), rh, c)
    txt(s, Emu(int(cx)+Inches(0.32)), Emu(int(cy)+Inches(0.22)), Inches(5.1), Inches(0.45),
        [[(tag+"   ", 18, True, c),(q, 16.5, True, INK)]], space_after=0)
    txt(s, Emu(int(cx)+Inches(0.32)), Emu(int(cy)+Inches(0.86)), Inches(5.15), Inches(1.1),
        [[(d, 13.8, False, GREY)]], space_after=0, line_spacing=1.1)
notes(s, "I organised the study as four research questions, each a precondition for the next. RQ1: do the "
          "metrics carry any task information at all — a sanity check before anything else. RQ2, the heart: "
          "do they predict success, across architectures and data sizes. RQ3: do they still work in the hard "
          "converged regime, and can they catch overtraining. RQ4: with tiny evaluation budgets, which "
          "differences are statistically real. I'll answer them in order. [~1:00]")

# ============================================================= 8 EXPERIMENTAL SETUP
s = slide(); header(s, "EXPERIMENTAL SETUP", "Three architectures, five LoRA runs, one benchmark", num()); footer(s,"")
# left: models + benchmark
bullets(s, Inches(0.8), Inches(1.55), Inches(5.4), Inches(4.6), [
    ("OpenVLA (7B) ", "— discrete-token, autoregressive.  The likelihood is a genuine token NLL."),
    ("SmolVLA (450M) ", "— flow-matching, edge-scale."),
    ("π₀.₅ (4B) ", "— flow-matching, Physical Intelligence."),
    ("Benchmark: LIBERO-Goal ", "— 10 tasks, shared scene, varying instruction: the LIBERO suite with the most instruction-driven action diversity."),
    ("Ground truth Q(k) ", "— 100-episode simulated success, ±5% noise; sparse by design (11–35 evals per run)."),
], size=15.5, gap=14)
# right: runs table
rect(s, Inches(6.55), Inches(1.55), Inches(6.05), Inches(4.35), CARD)
txt(s, Inches(6.8), Inches(1.7), Inches(5.6), Inches(0.4),
    [[("The five fine-tuning runs", 15.5, True, INK)]], space_after=0)
rows = [
    ("Run","Arch.","Train","Peak Q", True),
    ("SmolVLA-150","FM 450M","150","0.68", False),
    ("OpenVLA-150","tok 7B","150","0.83", False),
    ("OpenVLA-full","tok 7B","342","0.78", False),
    ("π₀.₅-full","FM 4B","342","0.18†", False),
    ("π₀.₅-150","FM 4B","150","0.38†", False),
]
ry=Inches(2.2); rhh=Inches(0.5)
cxs=[Inches(6.8),Inches(9.1),Inches(10.5),Inches(11.5)]
cws=[Inches(2.3),Inches(1.4),Inches(1.0),Inches(1.0)]
for j,(a,b,c,d,hd) in enumerate(rows):
    yy=Emu(int(ry)+j*int(rhh))
    if hd:
        rect(s, Inches(6.8), yy, Inches(5.55), rhh, INK)
    elif j%2==0:
        rect(s, Inches(6.8), yy, Inches(5.55), rhh, BAND)
    col = PAPER if hd else INK
    for k,val in enumerate((a,b,c,d)):
        al = PP_ALIGN.LEFT if k==0 else PP_ALIGN.CENTER
        txt(s, cxs[k], Emu(int(yy)+Inches(0.08)), cws[k], Inches(0.4),
            [[(val, 12.5, hd, col)]], align=al, space_after=0)
txt(s, Inches(6.8), Inches(5.42), Inches(5.6), Inches(0.5),
    [[("† π₀.₅ rows use different replan horizons — only relative metric ranking is meaningful.", 10.5, False, LIGHT)]],
    space_after=0, line_spacing=1.05)
notes(s, "Setup. Three architectures spanning both paradigms: OpenVLA is discrete-token — importantly its "
          "NLL is a real likelihood. SmolVLA and pi-0.5 are flow-matching. Five runs cross architecture with "
          "data size, plus a deliberate pi-0.5 overtraining stress run. Benchmark is LIBERO-Goal, chosen "
          "because varying only the instruction gives the most action diversity. Truth is 100-episode "
          "success, but sparse and noisy — that sparsity drives RQ4. The dagger note: the two pi-0.5 rows use "
          "different horizons, so compare rankings within a row, not absolute levels. [~1:20]")

# ============================================================= 9 RQ1
s = slide(); header(s, "RESULT · RQ1", "Do the metrics carry task information? At the trajectory level, yes.", num()); footer(s,"")
pic_fit(s, f"{FIG}/task_discriminative_validity_traj.png",
        Inches(0.7), Inches(1.45), Inches(7.35), Inches(5.15), align="center", valign="top")
rect(s, Inches(8.35), Inches(1.6), Inches(4.35), Inches(4.75), CARD)
rect(s, Inches(8.35), Inches(1.6), Inches(0.10), Inches(4.75), BLUE)
txt(s, Inches(8.6), Inches(1.8), Inches(3.9), Inches(0.5),
    [[("Test", 14, True, LIGHT)]], space_after=0)
txt(s, Inches(8.6), Inches(2.15), Inches(3.9), Inches(1.0),
    [[("Same-task trajectories should score closer than different-task ones — measured by AUROC (0.5 = chance, 1.0 = perfect separation).", 14.5, False, INK)]],
    space_after=0, line_spacing=1.12)
bullets(s, Inches(8.6), Inches(3.35), Inches(3.9), Inches(2.6), [
    ("Single chunk: ", "AUROC 0.65–0.69 — tasks share a common motion vocabulary."),
    ("Trajectory-level: ", "AUROC ≥ 0.90 for every metric — Cohen's d roughly triples."),
    ("Phase alignment, not averaging, ", "drives the gain (DTW intra-distance 7.06 → 1.84)."),
], size=13.8, gap=11, color=GREY)
txt(s, Inches(0.7), Inches(6.7), Inches(7.4), Inches(0.4),
    [[("Verdict: all five metrics pass the precondition — none is disqualified.", 13.5, True, GREEN)]],
    space_after=0)
notes(s, "RQ1, the sanity check. Can a metric tell two demos of the same task from different tasks, on "
          "ground truth alone? On a single 50-step chunk, barely — AUROC 0.65-0.69, because all LIBERO tasks "
          "reach, grasp, transport. But aggregate a whole trajectory phase-aligned, and every metric jumps "
          "above 0.90. Crucially it's the phase alignment, not just averaging away noise — DTW's within-task "
          "distance collapses while across-task barely moves. All five pass. This is why I use the "
          "trajectory-level estimator everywhere after. [~1:10]")

# ============================================================= 10 RQ2 headline
s = slide(); header(s, "RESULT · RQ2", "Trajectory metrics track success across all five runs", num()); footer(s,"")
pic_fit(s, f"{FIG}/crossmodel_early_stopping_shared5.png",
        Inches(0.7), Inches(1.45), Inches(11.9), Inches(3.55), align="center", valign="top")
rect(s, Inches(0.8), Inches(5.25), Inches(11.75), Inches(1.55), CARD)
rect(s, Inches(0.8), Inches(5.25), Inches(0.10), Inches(1.55), GREEN)
txt(s, Inches(1.05), Inches(5.42), Inches(11.3), Inches(1.3),
    [[("DTW is positive in all five runs — ", 16.5, True, INK),
      ("ρ = 0.72 / 0.75 / 0.85 / 0.90 / 0.72", 16.5, False, BLUE, True),
      (" — and its selection regret never exceeds 0.17.", 16.5, True, INK)],
     [("Trajectory (DTW/OT/cosine) and regression (L1/L2) metrics are positive across both architecture "
       "families (discrete-token and flow-matching) and both data sizes. The likelihood family (right-hand "
       "bars) is the one that appears to split by architecture.",
       14, False, GREY)]], space_after=8, line_spacing=1.12)
notes(s, "RQ2, the headline. Left: Spearman rho of every metric with success, per run. Right: the success "
          "curves. The trajectory metrics — DTW, OT, cosine — and the L1/L2 baselines are positive in every "
          "single run, across discrete-token and flow-matching, across 150 and full data. DTW specifically: "
          "0.72 to 0.90, never negative, regret never above 0.17 — the most consistent default. The one "
          "family that behaves differently is likelihood — hold that thought for RQ4. [~1:05]")

# ============================================================= 11 RQ2 per-model
s = slide(); header(s, "RESULT · RQ2", "Per-run rankings: a family of comparable metrics", num()); footer(s,"")
def minitable(x, title, rows, c):
    rect(s, x, Inches(1.6), Inches(3.75), Inches(3.95), CARD)
    rect(s, x, Inches(1.6), Inches(3.75), Inches(0.5), c)
    txt(s, x, Inches(1.68), Inches(3.75), Inches(0.4),
        [[(title, 14.5, True, PAPER)]], align=PP_ALIGN.CENTER, space_after=0)
    hy=Inches(2.2)
    txt(s, Emu(int(x)+Inches(0.25)), hy, Inches(1.6), Inches(0.35),[[("Metric",11.5,True,LIGHT)]],space_after=0)
    txt(s, Emu(int(x)+Inches(2.0)), hy, Inches(0.9), Inches(0.35),[[("ρ",11.5,True,LIGHT)]],align=PP_ALIGN.CENTER,space_after=0)
    txt(s, Emu(int(x)+Inches(2.85)), hy, Inches(0.85), Inches(0.35),[[("Regret",11.5,True,LIGHT)]],align=PP_ALIGN.CENTER,space_after=0)
    for j,(m,r,g,hot) in enumerate(rows):
        yy=Emu(int(Inches(2.6))+j*int(Inches(0.42)))
        # No per-row highlight: the trajectory/regression metrics form a comparable
        # family, so shading individual rows implied a meaning that was not consistent
        # across tables. Only a negative ρ (anti-correlated with success) is flagged red.
        cm = RED if r.startswith("−") else INK
        bold = False
        txt(s, Emu(int(x)+Inches(0.25)), Emu(int(yy)+Inches(0.03)), Inches(1.7), Inches(0.35),[[(m,12.5,bold,cm)]],space_after=0)
        txt(s, Emu(int(x)+Inches(2.0)), Emu(int(yy)+Inches(0.03)), Inches(0.9), Inches(0.35),[[(r,12.5,bold,cm)]],align=PP_ALIGN.CENTER,space_after=0)
        txt(s, Emu(int(x)+Inches(2.85)), Emu(int(yy)+Inches(0.03)), Inches(0.85), Inches(0.35),[[(g,12.5,bold,cm)]],align=PP_ALIGN.CENTER,space_after=0)
minitable(Inches(0.8), "OpenVLA-full  (discrete-token)", [
    ("Cosine","+0.89","0.05",False),("DTW","+0.85","0.00",True),("L₂","+0.85","0.00",True),
    ("L₁","+0.85","0.05",False),("CE / Acc / NLL","+0.83","0.04",False),("OT","+0.66","0.04",False),
], BLUE)
minitable(Inches(4.79), "π₀.₅-150  (flow-matching)", [
    ("L₁","+0.78","0.05",False),("Cosine","+0.77","0.04",False),("L₂","+0.73","0.08",False),
    ("DTW","+0.72","0.07",False),("OT","+0.68","0.17",False),("NLL / FM","−0.16","0.20",False),
], GREEN)
minitable(Inches(8.78), "SmolVLA-150  (flow-matching)", [
    ("L₁","+0.77","0.15",False),("DTW","+0.72","0.17",False),("L₂","+0.70","0.14",False),
    ("Cosine","+0.68","0.16",False),("OT","+0.62","0.20",False),("NLL / FM","−0.35","0.14",False),
], RGBColor(0x6A,0x3D,0x9A))
txt(s, Inches(0.8), Inches(5.8), Inches(11.8), Inches(1.2),
    [[("Two patterns:  ", 15, True, INK),
      ("(1) on OpenVLA-full, DTW and L₂ pick the ", 15, False, GREY),
      ("identical", 15, True, INK),
      (" checkpoint (21499) with identical ρ and zero regret;  (2) the NLL/FM surrogate drops sharply on "
       "flow-matching runs (green, purple) while staying strong on discrete-token (blue).", 15, False, GREY)]],
    space_after=0, line_spacing=1.12)
notes(s, "Zooming in per run. Two things to see. First, the trajectory and regression metrics are "
          "interchangeable — on OpenVLA-full, DTW and L2 literally select the same checkpoint, 21499, same "
          "rho, zero regret. So this is a family of comparable choices, not a strict winner. Second, look at "
          "the highlighted NLL/FM row: strong on the blue discrete-token table, but drops to 0.46 and "
          "even negative on the flow-matching tables. That architecture split is the RQ4 story. [~1:15]")

# ============================================================= 12 RQ3 plateau
s = slide(); header(s, "RESULT · RQ3", "Effective for the coarse decision; harder on the plateau", num()); footer(s,"")
pic_fit(s, f"{FIG}/plateau_restriction.png",
        Inches(0.7), Inches(1.5), Inches(7.2), Inches(3.4), align="center", valign="top")
rect(s, Inches(8.15), Inches(1.55), Inches(4.5), Inches(3.5), CARD)
txt(s, Inches(8.4), Inches(1.72), Inches(4.05), Inches(0.45),
    [[("Full range  vs.  converged plateau", 14, True, INK)]], space_after=0)
prows=[("Run","ρ full","ρ plat",True),
       ("SmolVLA-150","+0.72","+0.55",False),
       ("OpenVLA-150","+0.75","+0.59",False),
       ("OpenVLA-full","+0.86","+0.73",False),
       ("π₀.₅-full","+0.91","+0.80",False),
       ("π₀.₅-150","+0.72","+0.43",False)]
for j,(a,b,c,hd) in enumerate(prows):
    yy=Emu(int(Inches(2.25))+j*int(Inches(0.42)))
    if hd: rect(s, Inches(8.35), yy, Inches(4.1), Inches(0.4), INK)
    elif j==5: rect(s, Inches(8.35), yy, Inches(4.1), Inches(0.4), RGBColor(0xFF,0xE3,0xE0))
    elif j%2==0: rect(s, Inches(8.35), yy, Inches(4.1), Inches(0.4), BAND)
    col = PAPER if hd else (RED if j==5 else INK)
    txt(s, Inches(8.5), Emu(int(yy)+Inches(0.03)), Inches(2.0), Inches(0.35),[[(a,12.5,hd or j==5,col)]],space_after=0)
    txt(s, Inches(10.4), Emu(int(yy)+Inches(0.03)), Inches(0.95), Inches(0.35),[[(b,12.5,hd,col)]],align=PP_ALIGN.CENTER,space_after=0)
    txt(s, Inches(11.4), Emu(int(yy)+Inches(0.03)), Inches(0.95), Inches(0.35),[[(c,12.5,hd or j==5,col)]],align=PP_ALIGN.CENTER,space_after=0)
bullets(s, Inches(0.8), Inches(5.15), Inches(11.8), Inches(1.9), [
    ("Coarse decision is robust: ", "“do not stop before the knee” holds at ρ ≈ 0.8–0.9 over the full descent in all five runs."),
    ("Fine-ranking the plateau is harder ", "but stays usable (ρ ≥ 0.55) for four of the five runs."),
    ("π₀.₅-150 is the exception, and it appears to be an artifact: ", "the apparent 31%→23% dip recovers to 38% once training is extended — consistent with a noisy low ceiling rather than genuine overtraining."),
], size=14, gap=9)
notes(s, "RQ3: is it actually usable for stopping? Split each run at the knee — where DTW has done 90% of "
          "its descent — and recompute rho before vs after. Full-range rho is 0.8-0.9 everywhere, so the "
          "coarse decision 'don't stop before the knee' is robust. Inside the plateau it weakens but "
          "stays usable, 0.55+, for four of five. The exception, pi-0.5-150 in red, looked like overtraining "
          "— success fell 31 to 23% — but extending training recovered it to 38%. So it was restriction of "
          "range on a noisy low ceiling, not a real decline. [~1:15]")

# ============================================================= 13 RQ3 overtraining
s = slide(); header(s, "RESULT · RQ3", "The overtraining we aimed to induce did not appear", num()); footer(s,"")
pic_fit(s, f"{FIG}/success_vs_val_metrics_timeseries_pi_150.png",
        Inches(0.75), Inches(1.4), Inches(4.4), Inches(5.3), align="center", valign="top")
txt(s, Inches(5.5), Inches(1.6), Inches(7.1), Inches(0.6),
    [[("Three runs were designed to overtrain", 19, True, INK)]], space_after=0)
bullets(s, Inches(5.5), Inches(2.35), Inches(7.1), Inches(3.0), [
    ("SmolVLA-150 & OpenVLA-150 ", "— small data, pushed well past convergence."),
    ("π₀.₅-150 ", "— aggressive LR (6×), a non-decaying schedule, 40k steps."),
    ("Outcome: ", "every run settled into a noisy plateau in both the offline metrics and Q(k) — no monotone late decline."),
], size=15.5, gap=13)
rect(s, Inches(5.5), Inches(5.35), Inches(7.1), Inches(1.35), CARD)
rect(s, Inches(5.5), Inches(5.35), Inches(0.1), Inches(1.35), RED)
txt(s, Inches(5.75), Inches(5.5), Inches(6.7), Inches(1.1),
    [[("Honest limitation. ", 15.5, True, RED),
      ("Because no run genuinely overtrained, I cannot show the metrics catching a real decline — only that "
       "they reliably track the rise and plateau. Proving the preventive value of early stopping needs a run "
       "that actually degrades.", 14.5, False, GREY)]], space_after=0, line_spacing=1.12)
notes(s, "A point of intellectual honesty. Three runs were built to force overtraining — tiny data, and for "
          "pi-0.5-150 an aggressive 6x learning rate held near peak for 40k steps. It still didn't overtrain: "
          "everything settled into a noisy plateau. So I can show the metrics track the rise and plateau, but "
          "I cannot show them catching a genuine late decline, because I never produced one. That's a real "
          "limitation and I'll return to it. [~1:00]")

# ============================================================= 14 RQ4 bootstrap
s = slide(); header(s, "RESULT · RQ4", "How much of the ranking is real? Power is the limit", num()); footer(s,"")
pic_fit(s, f"{FIG}/crossmodel_bootstrap_ci.png",
        Inches(0.7), Inches(1.4), Inches(11.9), Inches(2.55), align="center", valign="top")
c1=Inches(0.8); c2=Inches(6.75); cw=Inches(5.8); cy=Inches(4.2); chh=Inches(2.5)
rect(s, c1, cy, cw, chh, CARD); rect(s, c1, cy, Inches(0.1), chh, GREEN)
txt(s, Emu(int(c1)+Inches(0.28)), Emu(int(cy)+Inches(0.18)), Inches(5.2), Inches(0.4),
    [[("What is statistically resolved", 15.5, True, GREEN)]], space_after=0)
bullets(s, Emu(int(c1)+Inches(0.28)), Emu(int(cy)+Inches(0.7)), Inches(5.3), Inches(1.7), [
    ("NLL architecture split ", "— significant: token likelihood tracks success; the FM surrogate does not (now negative on both flow-matching runs)."),
    ("Enough evaluations resolve them ", "— π₀.₅-full (n=33) and now SmolVLA-150 (n=35, densified) lift every trajectory metric’s CI above zero."),
], size=13.8, gap=10, color=GREY)
rect(s, c2, cy, cw, chh, CARD); rect(s, c2, cy, Inches(0.1), chh, RED)
txt(s, Emu(int(c2)+Inches(0.28)), Emu(int(cy)+Inches(0.18)), Inches(5.2), Inches(0.4),
    [[("What is NOT resolved", 15.5, True, RED)]], space_after=0)
bullets(s, Emu(int(c2)+Inches(0.28)), Emu(int(cy)+Inches(0.7)), Inches(5.3), Inches(1.7), [
    ("DTW vs. L1/L2/OT/cosine ", "— intervals overlap; on OpenVLA-full DTW ≡ L₂ exactly."),
    ("Fine within-family ranking ", "— even at n=35 DTW isn’t separable from L₁/L₂ (paired P ≈ 0.2–0.7)."),
], size=13.8, gap=10, color=GREY)
notes(s, "RQ4 keeps me honest about what the numbers support. Point estimates come from as few as 11 "
          "evaluations, so I bootstrap a 95% CI on every rho and on paired differences. What survives, left: "
          "the architecture split of NLL is real, and with enough evaluations — pi-0.5-full at 33, and "
          "SmolVLA-150 now at 35 after densifying the rise — every trajectory metric clears zero. What "
          "doesn't, right: DTW is still not statistically separable from L1, L2, OT or cosine — they're one "
          "family, even at n=35. The binding constraint is usually statistical power, not the choice of "
          "metric. [~1:10]")

# ============================================================= 15 ANSWERS
s = slide(); header(s, "SYNTHESIS", "The four questions, answered", num()); footer(s,"")
ans=[
    ("RQ1","Yes — every metric is task-discriminative (AUROC ≥ 0.90) at the trajectory level.", GREEN),
    ("RQ2","Yes — DTW/OT/cosine (and L1/L2) correlate positively with success in all five runs.", GREEN),
    ("RQ3","Partly — robust for the coarse “don’t stop early” call; genuine overtraining never occurred.", RGBColor(0xC7,0x8A,0x00)),
    ("RQ4","Qualified — the architecture split is real; finer metric orderings need more evaluations.", RGBColor(0xC7,0x8A,0x00)),
]
top=Inches(1.6); bh=Inches(1.15)
for i,(t,a,c) in enumerate(ans):
    y=Emu(int(top)+i*(int(bh)+int(Inches(0.14))))
    rect(s, Inches(0.8), y, Inches(11.75), bh, CARD)
    rect(s, Inches(0.8), y, Inches(1.4), bh, c)
    txt(s, Inches(0.8), Emu(int(y)+Inches(0.32)), Inches(1.4), Inches(0.6),
        [[(t, 24, True, PAPER)]], align=PP_ALIGN.CENTER, space_after=0)
    txt(s, Inches(2.45), Emu(int(y)+Inches(0.30)), Inches(9.9), Inches(0.7),
        [[(a, 16.5, False, INK)]], space_after=0, anchor=MSO_ANCHOR.MIDDLE, line_spacing=1.08)
notes(s, "So, the four answers. RQ1 yes, cleanly. RQ2 yes — the core positive result, architecture-general. "
          "RQ3 a qualified yes: excellent for the coarse decision, but I couldn't test the overtraining case. "
          "RQ4 qualified: the one hard, significant claim is the NLL architecture split; the rest is a family "
          "of comparable metrics limited by evaluation budget. Two green, two amber — I think that's an "
          "honest scorecard. [~1:00]")

# ============================================================= 16 TAKEAWAYS
s = slide(); header(s, "TAKEAWAYS", "Practical guidance for VLA practitioners", num()); footer(s,"")
tk=[
    ("Use a trajectory-similarity metric.", "DTW is a reliable default: positive in all five runs, regret ≤ 0.17.", BLUE),
    ("Trust the coarse decision.", "“Do not stop before the knee” is reliable; fine-ranking a converged plateau is much less so.", GREEN),
    ("Match the surrogate to the architecture.", "Token NLL is fine for discrete-token policies; for flow-matching, prefer a trajectory metric over the FM loss.", RED),
    ("Report uncertainty, not point ranks.", "With 10–30 rollouts, report bootstrap CIs — many single-metric orderings may be noise.", RGBColor(0x6A,0x3D,0x9A)),
]
x0=Inches(0.8); cw=Inches(5.75); gap=Inches(0.25); rh=Inches(2.1); ty=Inches(1.65)
for i,(h,b,c) in enumerate(tk):
    cx=Emu(int(x0)+(i%2)*(int(cw)+int(gap)))
    cy=Emu(int(ty)+(i//2)*(int(rh)+int(Inches(0.3))))
    rect(s, cx, cy, cw, rh, CARD)
    rect(s, cx, cy, cw, Inches(0.12), c)
    txt(s, Emu(int(cx)+Inches(0.3)), Emu(int(cy)+Inches(0.3)), Inches(5.2), Inches(0.7),
        [[(f"{i+1}   ", 20, True, c),(h, 17.5, True, INK)]], space_after=0, line_spacing=1.0)
    txt(s, Emu(int(cx)+Inches(0.3)), Emu(int(cy)+Inches(1.02)), Inches(5.2), Inches(0.95),
        [[(b, 14.5, False, GREY)]], space_after=0, line_spacing=1.1)
notes(s, "If you take four things away as a practitioner. One: use a trajectory metric — DTW is the safe "
          "default. Two: trust the coarse call, don't stop before the knee, but don't over-trust fine ranking "
          "on the plateau. Three: match the surrogate to the architecture — token NLL for discrete, a "
          "trajectory metric for flow-matching. Four: with tiny rollout budgets, report confidence intervals, "
          "because most single-metric orderings are noise. [~1:00]")

# ============================================================= 17 LIMITATIONS / FUTURE
s = slide(); header(s, "LIMITATIONS & FUTURE WORK", "What this study cannot yet claim", num()); footer(s,"")
cols=[
    ("Open-loop only", "Metrics score teacher-forced action chunks, not rollouts. A learned world model would test whether they also predict compounding closed-loop error.", BLUE),
    ("No overtraining regime", "No run genuinely degraded, so the preventive value of early stopping is untested. Longer runs at higher LR are needed to force a decline.", RED),
    ("Sparse, confounded Q(k)", "11–35 evals per run, and two π₀.₅ runs at mismatched horizons. Densifying SmolVLA-150 (18→35) sharpened its correlations but left DTW and L₁/L₂ statistically tied.", GREEN),
]
x0=Inches(0.8); cw=Inches(3.86); gap=Inches(0.30); top=Inches(1.8); ch=Inches(4.2)
for i,(h,b,c) in enumerate(cols):
    x=Emu(int(x0)+i*(int(cw)+int(gap)))
    rect(s, x, top, cw, ch, CARD)
    rect(s, x, top, cw, Inches(0.14), c)
    txt(s, Emu(int(x)+Inches(0.25)), Emu(int(top)+Inches(0.4)), Emu(int(cw)-Inches(0.5)), Inches(0.9),
        [[(h, 18.5, True, c)]], space_after=0, line_spacing=1.0)
    txt(s, Emu(int(x)+Inches(0.25)), Emu(int(top)+Inches(1.45)), Emu(int(cw)-Inches(0.5)), Inches(2.6),
        [[(b, 14.5, False, GREY)]], space_after=0, line_spacing=1.14)
txt(s, Inches(0.8), Inches(6.35), Inches(11.8), Inches(0.6),
    [[("None of these overturns the central finding ", 15, True, INK),
      ("— they mark where the current evidence stops and the next study begins.", 15, False, GREY)]], space_after=0)
notes(s, "Three limitations, stated plainly. One: everything here is open-loop, teacher-forced chunks — a "
          "world-model rollout would test compounding error directly. Two, the big one: no run overtrained, "
          "so I can't claim early stopping prevents degradation, only that it tracks the plateau. Three: "
          "success was measured sparsely and, for pi-0.5, at mismatched horizons — denser matched evaluation "
          "would finally separate DTW from L1/L2. None of these breaks the main result; they mark where the "
          "next study starts. [~1:05]")

# ============================================================= 18 CONCLUSION / THANKS
s = slide()
rect(s, 0, 0, SW, SH, PAPER)
rect(s, 0, 0, Inches(0.32), SH, AALTO)
rect(s, 0, Inches(5.35), SW, Pt(2.2), AALTO)
txt(s, Inches(0.95), Inches(0.75), Inches(11.5), Inches(0.4),
    [[("CONCLUSION", 14, True, LIGHT)]], space_after=0)
txt(s, Inches(0.9), Inches(1.35), Inches(11.6), Inches(1.7),
    [[("An offline trajectory-similarity metric, computed once", 30, True, INK)],
     [("per checkpoint, can serve as a practical, largely", 30, True, INK)],
     [("architecture-agnostic signal for when to stop fine-tuning a VLA.", 26, True, INK)]],
    space_after=4, line_spacing=1.04)
bullets(s, Inches(0.95), Inches(3.6), Inches(11.4), Inches(1.6), [
    ("Evaluated ", "across 3 architectures, 5 runs, and a 10-task benchmark — DTW a reliable default."),
    ("Honest about scope ", "— coarse stopping is well-supported; overtraining and closed-loop rollout remain open."),
], size=16, gap=11)
txt(s, Inches(0.95), Inches(5.65), Inches(11.4), Inches(1.5),
    [[("Thank you — I welcome your questions.", 22, True, INK)],
     [("Yibo Zhang    ·    yibo.zhang@aalto.fi", 14.5, False, GREY)],
     [("Supervisor: Prof. Joni Pajarinen    ·    Advisor: Dr. Wenyan Yang    ·    Aalto Robot Learning Lab", 12.5, False, LIGHT)]],
    space_after=6)
notes(s, "To conclude: a single offline trajectory metric, computed once per checkpoint, can be a practical, "
          "largely architecture-agnostic answer to when to stop fine-tuning a VLA. Evaluated across three "
          "architectures and five runs, with DTW as a reliable default — and honest about what's still open: "
          "the overtraining case and true closed-loop rollout. Thank you, I'm happy to take questions. "
          "[Total ~22 min in the 30-min slot, leaving time for Q&A.] Backup slides / thesis tables ready for "
          "detail questions.")

prs.save(OUT)
print("Saved", OUT, "with", len(prs.slides.__iter__.__self__._sldIdLst), "slides")
