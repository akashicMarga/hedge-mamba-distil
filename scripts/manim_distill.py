"""
Manim animation — HedgeMamba two-stage distillation.
No LaTeX required (Text + Unicode only).

Render:
    manim -pql --format=gif scripts/manim_distill.py Stage1
    manim -pql --format=gif scripts/manim_distill.py Stage2
    manim -pqh --format=gif scripts/manim_distill.py Stage1   # high quality
"""
from manim import *

TEACHER_C = "#4FC3F7"
STUDENT_C = "#81C784"
LOSS_C    = "#FF7043"
DIM_C     = "#90A4AE"

# ── tiny helpers ──────────────────────────────────────────────────────────────

def label(txt, size=28, color=WHITE, **kw):
    return Text(txt, font_size=size, color=color, **kw)

def eq(txt, size=34, color=WHITE, **kw):
    return Text(txt, font_size=size, color=color, font="monospace", **kw)


# ─────────────────────────────────────────────────────────────────────────────
class Stage1(Scene):
    def construct(self):
        self._header()
        self._surgery()
        self._hedgehog()
        self._ssm()
        self._loss()

    # helpers
    def _header(self):
        t = label("Stage 1 — Cosine Distillation", 38, weight=BOLD).to_edge(UP, buff=0.25)
        s = label("warm-init SSM from teacher attention weights", 21, DIM_C).next_to(t, DOWN, buff=0.1)
        self._hdr = VGroup(t, s)
        self.play(FadeIn(self._hdr))

    def _step_title(self, txt):
        t = label(txt, 26, YELLOW).move_to(UP * 1.85)
        self.play(Write(t), run_time=0.45)
        return t

    def _fade(self, *mobs):
        self.play(*[FadeOut(m) for m in mobs], run_time=0.4)

    # ── Parameter surgery ─────────────────────────────────────────────────────
    def _surgery(self):
        step = self._step_title("① Parameter surgery  (Appendix B)")

        tc = label("Teacher", 20, TEACHER_C).move_to(LEFT*3.3 + UP*0.9)
        wk = label("W_k", 36, TEACHER_C).next_to(tc, DOWN, buff=0.4)
        wq = label("W_q", 36, TEACHER_C).next_to(wk, DOWN, buff=0.5)

        sc = label("Student SSM", 20, STUDENT_C).move_to(RIGHT*3.3 + UP*0.9)
        wb = label("W_B", 36, STUDENT_C).next_to(sc, DOWN, buff=0.4)
        wc = label("W_C", 36, STUDENT_C).next_to(wb, DOWN, buff=0.5)

        a1 = Arrow(wk.get_right(), wb.get_left(), color=YELLOW, buff=0.15, stroke_width=3)
        a2 = Arrow(wq.get_right(), wc.get_left(), color=YELLOW, buff=0.15, stroke_width=3)
        l1 = label("W_B ← W_k", 22, YELLOW).next_to(a1, UP, buff=0.06)
        l2 = label("W_C ← W_q", 22, YELLOW).next_to(a2, DOWN, buff=0.06)
        note = label("key/query projections become B/C in the scan", 18, DIM_C).move_to(DOWN*1.8)

        self.play(FadeIn(tc), FadeIn(wk), FadeIn(wq))
        self.play(FadeIn(sc), FadeIn(wb), FadeIn(wc))
        self.play(GrowArrow(a1), GrowArrow(a2))
        self.play(Write(l1), Write(l2))
        self.play(FadeIn(note))
        self.wait(1.5)
        self._fade(step, tc, wk, wq, sc, wb, wc, a1, a2, l1, l2, note)

    # ── Hedgehog projection ───────────────────────────────────────────────────
    def _hedgehog(self):
        step = self._step_title("② Hedgehog feature map  φ(x)")

        phi  = eq("φ(x) = softmax( [Wx,  −Wx] )", 36, STUDENT_C).move_to(UP*0.7)
        box  = SurroundingRectangle(phi, color=STUDENT_C, buff=0.2, corner_radius=0.1)
        dim  = label("x ∈ ℝᴰ  ──φ──▶  ℝ²ᴺ   (state size doubles)", 28, DIM_C).next_to(phi, DOWN, buff=0.5)
        why  = label("approximates softmax attention as a linear inner product", 19, DIM_C).next_to(dim, DOWN, buff=0.3)

        self.play(Write(phi))
        self.play(Create(box))
        self.play(FadeIn(dim))
        self.play(FadeIn(why))
        self.wait(1.5)
        self._fade(step, phi, box, dim, why)

    # ── SSM recurrence ────────────────────────────────────────────────────────
    def _ssm(self):
        step = self._step_title("③ Selective SSM scan")

        r1 = eq("h_t  =  Ā · h_(t-1)  +  B̄ · x_t", 36, STUDENT_C).move_to(UP*0.9)
        r2 = eq("y_t  =  C · h_t",                   36, STUDENT_C).next_to(r1, DOWN, buff=0.4)

        brace = Brace(VGroup(r1, r2), LEFT, color=STUDENT_C)
        blbl  = label("O(1) per step\nat inference", 17, STUDENT_C).next_to(brace, LEFT, buff=0.15)

        dt  = label("Δ_t = softplus(W_Δ · x_t)   ← input-dependent step size", 25, DIM_C).next_to(r2, DOWN, buff=0.5)
        zoh = label("Ā = exp(Δ·A)     B̄ = (exp(Δ·A) − I) A⁻¹B     (ZOH discretization)", 21, DIM_C).next_to(dt, DOWN, buff=0.3)

        self.play(Write(r1))
        self.play(Write(r2))
        self.play(FadeIn(brace), FadeIn(blbl))
        self.play(FadeIn(dt))
        self.play(FadeIn(zoh))
        self.wait(1.5)
        self._fade(step, r1, r2, brace, blbl, dt, zoh)

    # ── Cosine loss ───────────────────────────────────────────────────────────
    def _loss(self):
        step = self._step_title("④ Cosine distillation loss")

        ht  = label("h_ℓᵀ", 44, TEACHER_C).move_to(LEFT*2.8 + UP*0.5)
        hs  = label("h_ℓˢ", 44, STUDENT_C).move_to(RIGHT*2.8 + UP*0.5)
        bt  = SurroundingRectangle(ht, color=TEACHER_C, buff=0.18, corner_radius=0.08)
        bs  = SurroundingRectangle(hs, color=STUDENT_C, buff=0.18, corner_radius=0.08)
        arr = DoubleArrow(bt.get_right(), bs.get_left(), color=LOSS_C, buff=0.12, stroke_width=3)
        sim = label("cos ↑", 20, LOSS_C).next_to(arr, UP, buff=0.08)

        loss = eq(
            "ℒ₁ = (1/L) Σ_ℓ  [ 1  −  (h_ℓᵀ · h_ℓˢ) / (‖h_ℓᵀ‖ · ‖h_ℓˢ‖) ]",
            30, LOSS_C
        ).move_to(DOWN*1.1)

        frozen = label(
            "encoder · cross-attn · FFN  frozen     |     SSM weights only trained",
            18, DIM_C
        ).next_to(loss, DOWN, buff=0.35)

        self.play(FadeIn(ht), Create(bt))
        self.play(FadeIn(hs), Create(bs))
        self.play(GrowArrow(arr), Write(sim))
        self.play(Write(loss))
        self.play(FadeIn(frozen))
        self.wait(2.5)


# ─────────────────────────────────────────────────────────────────────────────
class Stage2(Scene):
    def construct(self):
        self._header()
        self._scheduled_sampling()
        self._ce_loss()

    def _header(self):
        t = label("Stage 2 — ASR Fine-tuning", 38, weight=BOLD).to_edge(UP, buff=0.25)
        s = label("scheduled sampling closes the teacher-forcing gap", 21, DIM_C).next_to(t, DOWN, buff=0.1)
        self._hdr = VGroup(t, s)
        self.play(FadeIn(self._hdr))

    def _step_title(self, txt):
        t = label(txt, 26, YELLOW).move_to(UP * 1.85)
        self.play(Write(t), run_time=0.45)
        return t

    def _fade(self, *mobs):
        self.play(*[FadeOut(m) for m in mobs], run_time=0.4)

    def _token(self, txt, fill, outline, pos):
        box = RoundedRectangle(
            width=0.88, height=0.60, corner_radius=0.08,
            fill_color=fill, fill_opacity=0.85,
            stroke_color=outline, stroke_width=2,
        ).move_to(pos)
        lbl = label(txt, 20, WHITE).move_to(box.get_center())
        return VGroup(box, lbl)

    # ── Scheduled sampling ────────────────────────────────────────────────────
    def _scheduled_sampling(self):
        step = self._step_title("① Scheduled sampling")

        n, w, gap = 6, 0.88, 0.16
        xs = [(i - n/2 + 0.5) * (w + gap) for i in range(n)]
        y_row = UP * 0.45

        gt = VGroup(*[
            self._token(f"y{i+1}", "#1565C0", TEACHER_C, RIGHT*x + y_row)
            for i, x in enumerate(xs)
        ])
        gt_lbl = label("Ground-truth decoder input", 19, TEACHER_C).next_to(gt, UP, buff=0.2)

        self.play(FadeIn(gt_lbl), *[FadeIn(t) for t in gt])

        # p counter animates
        p_tracker = ValueTracker(0.0)
        p_display = always_redraw(lambda: label(
            f"p = {p_tracker.get_value():.1f}",
            32, LOSS_C
        ).move_to(DOWN*0.55))
        self.play(FadeIn(p_display))
        self.wait(0.3)

        # replace tokens 1, 3, 5 to simulate p = 0.5
        replace_idx = [1, 3, 5]
        new_tokens = [
            self._token(f"ŷ{i+1}", "#1B5E20", STUDENT_C, gt[i].get_center())
            for i in replace_idx
        ]
        self.play(p_tracker.animate.set_value(0.5), run_time=0.9)
        self.play(*[Transform(gt[i], nt) for i, nt in zip(replace_idx, new_tokens)], run_time=0.6)
        self.wait(0.4)

        # legend
        def swatch(fill, outline, txt):
            box = RoundedRectangle(width=0.28, height=0.20, corner_radius=0.05,
                                   fill_color=fill, fill_opacity=0.85, stroke_color=outline)
            return VGroup(box, label(txt, 17, outline)).arrange(RIGHT, buff=0.1)
        leg = VGroup(
            swatch("#1565C0", TEACHER_C, "ground truth"),
            swatch("#1B5E20", STUDENT_C, "student prediction"),
        ).arrange(RIGHT, buff=0.6).move_to(DOWN*1.3)
        self.play(FadeIn(leg))

        sched = label("p(t) = min( 2t/T,  0.5 )   — ramps 0→0.5 over first half of training",
                      22, DIM_C).move_to(DOWN*2.05)
        self.play(Write(sched))
        self.wait(1.5)

        self._fade(step, gt, gt_lbl, p_display, leg, sched)

    # ── CE loss ───────────────────────────────────────────────────────────────
    def _ce_loss(self):
        step = self._step_title("② Cross-entropy loss")

        ce  = eq("ℒ₂ = −(1/|Y|) Σ_t  log P_θ( y_t | x, y_<t )", 34, LOSS_C).move_to(UP*0.65)
        box = SurroundingRectangle(ce, color=LOSS_C, buff=0.2, corner_radius=0.1)

        trains = label(
            "trains:    SSM weights  +  cross-attn  +  FFN  +  layer norms",
            24, STUDENT_C,
        ).next_to(ce, DOWN, buff=0.5)

        frozen = label(
            "frozen:   encoder  +  token embedding",
            24, DIM_C,
        ).next_to(trains, DOWN, buff=0.3)

        self.play(Write(ce))
        self.play(Create(box))
        self.play(FadeIn(trains))
        self.play(FadeIn(frozen))
        self.wait(2.5)
