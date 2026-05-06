"""
PDF Term Sheet generator for the Autocall MC Pricer.
"""

import io
import numpy as np
from datetime import datetime
from fpdf import FPDF, XPos, YPos

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


NAVY   = (15,  23,  42)
INDIGO = (99, 102, 241)
GREEN  = (34, 197,  94)
RED    = (239,  68,  68)
AMBER  = (245, 158,  11)
LGRAY  = (241, 245, 249)
WHITE  = (255, 255, 255)
DARK   = ( 30,  41,  59)
MUTED  = (100, 116, 139)


def _fig_to_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    return buf.read()


def _section(pdf, title: str):
    pdf.set_fill_color(*INDIGO)
    pdf.set_text_color(*WHITE)
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 7, f"  {title}", new_x=XPos.LMARGIN, new_y=YPos.NEXT, fill=True)
    pdf.ln(2)
    pdf.set_text_color(*DARK)


def _row(pdf, label: str, value: str, fill: bool = False):
    pdf.set_font("Helvetica", "", 9)
    pdf.set_fill_color(*LGRAY)
    pdf.cell(80, 6, f"  {label}", border=0, fill=fill)
    pdf.set_font("Helvetica", "B", 9)
    pdf.cell(0,  6, f"  {value}", border=0, fill=fill,
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)


def _chart_call_prob(prob_by_obs: dict, title: str) -> bytes:
    dates = [f"T={t:.2f}y" for t in prob_by_obs]
    probs = [v * 100 for v in prob_by_obs.values()]
    cum   = np.cumsum(probs)

    fig, ax = plt.subplots(figsize=(10, 3.5))
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#1e293b")
    ax.bar(dates, probs, color="#6366f1", alpha=0.85, label="P(call at date)")
    ax2 = ax.twinx()
    ax2.plot(dates, cum, color="#f59e0b", lw=2, marker="o", ms=5,
             label="Cumulative")
    ax2.set_facecolor("#1e293b")
    ax2.tick_params(colors="white", labelsize=7)
    ax2.set_ylabel("Cumulative (%)", color="white", fontsize=8)
    for sp in ax2.spines.values(): sp.set_color("#334155")

    ax.set_title(title, color="white", fontsize=9)
    ax.tick_params(colors="white", labelsize=7)
    ax.set_ylabel("P(call) %", color="white", fontsize=8)
    for sp in ax.spines.values(): sp.set_color("#334155")

    lines1, lbl1 = ax.get_legend_handles_labels()
    lines2, lbl2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, lbl1 + lbl2, fontsize=7,
              facecolor="#1e293b", labelcolor="white")

    plt.tight_layout()
    b = _fig_to_bytes(fig)
    plt.close(fig)
    return b


def _chart_payoff_dist(payoffs: np.ndarray, called_mask, pdi_mask, price: float) -> bytes:
    ok_mask = ~called_mask & ~pdi_mask
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
    fig.patch.set_facecolor("#0f172a")

    for ax in axes:
        ax.set_facecolor("#1e293b")
        ax.tick_params(colors="white", labelsize=7)
        for sp in ax.spines.values(): sp.set_color("#334155")

    for mask, nm, col in [
        (called_mask, "Called",  "#22c55e"),
        (pdi_mask,    "PDI",     "#ef4444"),
        (ok_mask,     "Mat. OK", "#38bdf8"),
    ]:
        if mask.any():
            axes[0].hist(payoffs[mask], bins=50, color=col, alpha=0.65,
                         density=True, label=nm)
    axes[0].axvline(price, color="white", lw=1.5, ls="--", label=f"FV={price:.2f}")
    axes[0].axvline(100,   color=AMBER,   lw=1,   ls=":",  label="Par=100", alpha=0.7)
    axes[0].set_title("Payoff Distribution", color="white", fontsize=9)
    axes[0].set_xlabel("Discounted Payoff", color="white", fontsize=8)
    axes[0].legend(fontsize=7, facecolor="#1e293b", labelcolor="white")

    sorted_pf = np.sort(payoffs)
    cdf = np.arange(1, len(sorted_pf) + 1) / len(sorted_pf)
    axes[1].plot(sorted_pf, cdf * 100, color="#6366f1", lw=2)
    axes[1].axvline(price, color="white", lw=1.5, ls="--")
    axes[1].axhline(50, color=AMBER, lw=0.8, ls=":", alpha=0.6)
    axes[1].set_title("CDF", color="white", fontsize=9)
    axes[1].set_xlabel("Payoff", color="white", fontsize=8)
    axes[1].set_ylabel("Percentile (%)", color="white", fontsize=8)

    plt.tight_layout()
    b = _fig_to_bytes(fig)
    plt.close(fig)
    return b


def _chart_paths(worst_paths: np.ndarray | None,
                 T: float, barrier_ac: float, barrier_pdi: float,
                 called_mask, pdi_mask, n_show: int = 100) -> bytes:
    if worst_paths is None:
        return None

    n_paths = worst_paths.shape[0]
    times   = np.linspace(0, T, worst_paths.shape[1])

    fig, ax = plt.subplots(figsize=(11, 4))
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#1e293b")

    # Percentile fan
    for lo, hi, al in [(5, 95, 0.12), (25, 75, 0.22)]:
        ax.fill_between(times,
                         np.percentile(worst_paths, lo, axis=0) * 100,
                         np.percentile(worst_paths, hi, axis=0) * 100,
                         alpha=al, color="#6366f1")

    # Sample paths
    idx_sample = np.random.default_rng(0).choice(n_paths, min(n_show, n_paths), replace=False)
    for i in idx_sample:
        cl = "#22c55e" if called_mask[i] else ("#ef4444" if pdi_mask[i] else "#38bdf8")
        ax.plot(times, worst_paths[i] * 100, color=cl, lw=0.5, alpha=0.5)

    ax.axhline(barrier_ac  * 100, color="#22c55e", lw=1.5, ls="--",
               label=f"Recall {barrier_ac*100:.0f}%")
    ax.axhline(barrier_pdi * 100, color="#ef4444", lw=1.5, ls="--",
               label=f"PDI {barrier_pdi*100:.0f}%")

    ax.set_xlabel("Time (years)", color="white", fontsize=8)
    ax.set_ylabel("Performance (%)", color="white", fontsize=8)
    ax.set_title("Simulated Paths -- Worst Performer", color="white", fontsize=9)
    ax.tick_params(colors="white", labelsize=7)
    for sp in ax.spines.values(): sp.set_color("#334155")
    ax.legend(fontsize=7, facecolor="#1e293b", labelcolor="white")

    plt.tight_layout()
    b = _fig_to_bytes(fig)
    plt.close(fig)
    return b


def generate_autocall_pdf(
    # Product params
    S0, r, sigma, q, notional,
    T, obs_freq, coupon_pa,
    barrier_ac, barrier_pdi,
    barrier_cpn, memory,
    n_paths,
    # Results
    price, std_error, ci_lo, ci_hi,
    prob_called, prob_pdi, prob_mat_ok,
    avg_call_date,
    prob_by_obs: dict,
    payoffs: np.ndarray,
    called_mask, pdi_mask,
    worst_paths=None,
    ticker: str = "",
    structure: str = "Standard Autocall",
) -> bytes:

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # ?? Header ????????????????????????????????????????????????????????????????
    pdf.set_fill_color(*NAVY)
    pdf.rect(0, 0, 210, 32, "F")
    pdf.set_text_color(*WHITE)
    pdf.set_font("Helvetica", "B", 15)
    pdf.set_y(5)
    pdf.cell(0, 8, "AUTOCALL STRUCTURED NOTE -- TERM SHEET",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 6, structure,
             new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
    pdf.set_font("Helvetica", "", 9)
    ticker_str = f" | Underlying: {ticker}" if ticker else ""
    pdf.cell(0, 6,
             f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}{ticker_str}",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
    pdf.ln(8)
    pdf.set_text_color(*DARK)

    # ?? Two-column layout: Product + Market ???????????????????????????????????
    obs_label = {0.25: "Quarterly", 0.5: "Semi-annual", 1.0: "Annual"}.get(obs_freq, f"{obs_freq}y")

    left_items = [
        ("Notional",         f"{notional:.0f}"),
        ("Maturity",         f"{T:.1f} years"),
        ("Observations",     obs_label),
        ("Annual coupon",    f"{coupon_pa*100:.2f}%"),
        ("Recall barrier",   f"{barrier_ac*100:.0f}% of S0  ({barrier_ac*S0:.2f})"),
        ("PDI barrier",      f"{barrier_pdi*100:.0f}% of S0  ({barrier_pdi*S0:.2f})"),
    ]
    if barrier_cpn is not None:
        left_items.append(("Coupon barrier", f"{barrier_cpn*100:.0f}% of S0"))
        left_items.append(("Memory coupon",  "Yes" if memory else "No"))

    right_items = [
        ("Spot  S0",       f"{S0:.4f}"),
        ("Risk-free  r",   f"{r*100:.2f}%"),
        ("Volatility  ?",  f"{sigma*100:.2f}%"),
        ("Dividend  q",    f"{q*100:.2f}%"),
        ("MC Paths",       f"{n_paths:,}"),
    ]

    _section(pdf, "PRODUCT PARAMETERS")
    col_w = 90
    y_start = pdf.get_y()
    for i, (lbl, val) in enumerate(left_items):
        fill = i % 2 == 0
        pdf.set_fill_color(*LGRAY)
        pdf.set_font("Helvetica", "", 8.5)
        pdf.cell(35, 6, f"  {lbl}", fill=fill)
        pdf.set_font("Helvetica", "B", 8.5)
        pdf.cell(col_w - 35, 6, f"  {val}", fill=fill,
                 new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(2)

    _section(pdf, "MARKET PARAMETERS")
    for i, (lbl, val) in enumerate(right_items):
        _row(pdf, lbl, val, fill=i % 2 == 0)
    pdf.ln(4)

    # ?? MC Results ????????????????????????????????????????????????????????????
    _section(pdf, "MONTE CARLO PRICING RESULTS")

    # Big price highlight
    pdf.set_fill_color(*INDIGO)
    pdf.set_text_color(*WHITE)
    pdf.set_font("Helvetica", "B", 20)
    vs_par = price - notional
    sign   = "+" if vs_par >= 0 else ""
    pdf.cell(0, 14, f"Fair Value: {price:.4f}  ({sign}{vs_par:.2f} vs par)",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT, fill=True, align="C")
    pdf.ln(2)
    pdf.set_text_color(*DARK)

    for i, (lbl, val) in enumerate([
        ("Std Error (MC)",     f"{std_error:.4f}"),
        ("95% CI",             f"[ {ci_lo:.4f} , {ci_hi:.4f} ]"),
        ("CI Width",           f"{ci_hi - ci_lo:.4f}"),
    ]):
        _row(pdf, lbl, val, fill=i % 2 == 0)
    pdf.ln(4)

    _section(pdf, "SCENARIO PROBABILITIES")
    for i, (lbl, val, col) in enumerate([
        ("P(Recalled early)",     f"{prob_called*100:.2f}%",  GREEN),
        ("P(PDI at maturity)",    f"{prob_pdi*100:.2f}%",     RED),
        ("P(Maturity protected)", f"{prob_mat_ok*100:.2f}%",  AMBER),
    ]):
        pdf.set_font("Helvetica", "", 9)
        pdf.set_fill_color(*LGRAY)
        fill = i % 2 == 0
        pdf.cell(80, 6, f"  {lbl}", fill=fill)
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_text_color(*col)
        pdf.cell(0, 6, f"  {val}", fill=fill,
                 new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_text_color(*DARK)

    if not np.isnan(avg_call_date):
        pdf.ln(2)
        pdf.set_font("Helvetica", "I", 8.5)
        pdf.set_text_color(*MUTED)
        pdf.cell(0, 5, f"  Average call date (conditional on recall): {avg_call_date:.3f}y",
                 new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_text_color(*DARK)
    pdf.ln(4)

    # ?? Charts page ???????????????????????????????????????????????????????????
    pdf.add_page()
    _section(pdf, "CALL PROBABILITY BY OBSERVATION DATE")
    chart_cp = _chart_call_prob(prob_by_obs, f"P(Recalled) by Date -- {structure}")
    pdf.image(io.BytesIO(chart_cp), x=10, w=190)
    pdf.ln(4)

    _section(pdf, "PAYOFF DISTRIBUTION")
    chart_pd = _chart_payoff_dist(payoffs, called_mask, pdi_mask, price)
    pdf.image(io.BytesIO(chart_pd), x=10, w=190)

    # ?? Paths page (if available) ?????????????????????????????????????????????
    if worst_paths is not None:
        pdf.add_page()
        _section(pdf, "SIMULATED PATHS -- WORST PERFORMER")
        chart_paths = _chart_paths(worst_paths, T, barrier_ac, barrier_pdi,
                                    called_mask, pdi_mask)
        if chart_paths:
            pdf.image(io.BytesIO(chart_paths), x=10, w=190)

    # ?? Footer ????????????????????????????????????????????????????????????????
    pdf.set_y(-15)
    pdf.set_font("Helvetica", "I", 7)
    pdf.set_text_color(*MUTED)
    pdf.cell(0, 5,
             "Autocall MC Pricer | For informational purposes only -- not financial advice.",
             align="C")

    return bytes(pdf.output())
