# -*- coding: utf-8 -*-
"""
Анализ рынка квартир Хамовников: описательная статистика, точка окупаемости, графики.

Вход : data/processed/flats.csv
Выход: figures/*.png, data/processed/stats_summary.csv, payback.csv
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, MaxNLocator

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
FIG = ROOT / "figures"

ORDER = ["1-комнатные", "2-комнатные", "3+ комнатные"]

# Палитра проверена скриптом validate_palette.js (навык dataviz): все проверки
# PASS - светлота, насыщенность, различимость при цветовой слепоте (dE 9.2
# при дейтеранопии), контраст к фону.
C_RENT, C_SALE = "#2a78d6", "#eb6834"          # аренда / продажа
C_SUB = {"1-комнатные": "#2a78d6", "2-комнатные": "#eb6834", "3+ комнатные": "#1baf7a"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#dedcd6"

# Допущения "реалистичного" сценария окупаемости
OCCUPANCY = 11 / 12      # 11 занятых месяцев из 12 (простой между арендаторами)
TAX = 0.13               # НДФЛ с арендного дохода


def setup_style() -> None:
    mpl.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "figure.facecolor": "#ffffff", "axes.facecolor": "#ffffff",
        "axes.edgecolor": GRID, "axes.linewidth": 0.8,
        "axes.labelcolor": INK2, "axes.titlesize": 12, "axes.titleweight": "bold",
        "axes.titlecolor": INK, "axes.grid": True, "axes.axisbelow": True,
        "grid.color": GRID, "grid.linewidth": 0.6,
        "xtick.color": INK2, "ytick.color": INK2,
        "xtick.labelsize": 9, "ytick.labelsize": 9,
        "legend.frameon": False, "figure.dpi": 110, "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "axes.spines.top": False, "axes.spines.right": False,
    })


def money(v, _=None) -> str:
    """
    Компактные подписи осей: 5 200 000 в «5.2 млн», 95 000 в «95 тыс».
    Разряд «млрд» обязателен: цены продажи в Хамовниках доходят до 2.6 млрд,
    и без него подпись превращалась в нечитаемое «2500.0 млн».
    """
    a = abs(v)
    if a >= 1e9:
        return f"{v / 1e9:.1f} млрд"
    if a >= 1e8:                       # от 100 млн дробная часть только мешает
        return f"{v / 1e6:.0f} млн"
    if a >= 1e6:
        return f"{v / 1e6:.1f} млн"
    if a >= 1e3:
        return f"{v / 1e3:.0f} тыс"
    return f"{v:.0f}"


def save(fig, name: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.png")
    plt.close(fig)
    print(f"  figures/{name}.png")


# --------------------------------------------------------------------------- #
#  Расчёты
# --------------------------------------------------------------------------- #
def descriptive(df: pd.DataFrame) -> pd.DataFrame:
    """Описательная статистика по каждой из 6 подкатегорий."""
    rows = []
    for (deal, sub), g in df.groupby(["deal_label", "subcat_label"], observed=True):
        p = g["price_rub"]
        rows.append({
            "Категория": deal, "Подкатегория": sub, "N": len(g),
            "Средняя цена": p.mean(), "Медиана": p.median(),
            "Ст. отклонение": p.std(), "Мин": p.min(), "Макс": p.max(),
            "Q1": p.quantile(.25), "Q3": p.quantile(.75),
            "Коэф. вариации, %": p.std() / p.mean() * 100,
            "Средняя цена за м2": g["price_per_m2"].mean(),
            "Медиана цены за м2": g["price_per_m2"].median(),
            "Средняя площадь, м2": g["total_area"].mean(),
            "Медианная площадь, м2": g["total_area"].median(),
        })
    out = pd.DataFrame(rows)
    out["Подкатегория"] = pd.Categorical(out["Подкатегория"], ORDER, ordered=True)
    return out.sort_values(["Категория", "Подкатегория"]).reset_index(drop=True)


def payback(df: pd.DataFrame) -> pd.DataFrame:
    """
    Точка окупаемости: за сколько месяцев аренда покрывает стоимость покупки.

    Считаем тремя способами, потому что у каждого свои ограничения:
      1) по средним     - прямой ответ на вопрос задания;
      2) по медианам    - устойчив к дорогим квартирам, тянущим среднее вверх;
      3) по цене за м2  - снимает искажение от того, что средняя площадь
                          продаваемых и сдаваемых квартир различается.
    """
    rows = []
    for sub in ORDER:
        r = df[(df.deal_type == "rent") & (df.subcat_label == sub)]
        s = df[(df.deal_type == "sale") & (df.subcat_label == sub)]
        if r.empty or s.empty:
            continue
        rm, sm = r.price_rub.mean(), s.price_rub.mean()
        rmd, smd = r.price_rub.median(), s.price_rub.median()
        rm2, sm2 = r.price_per_m2.mean(), s.price_per_m2.mean()
        net = rm * OCCUPANCY * (1 - TAX)          # доход с простоем и НДФЛ
        rows.append({
            "Подкатегория": sub,
            "N аренда": len(r), "N продажа": len(s),
            "Ср. аренда, руб/мес": rm, "Ср. продажа, руб": sm,
            "Окупаемость по средним, мес": sm / rm,
            "Окупаемость по средним, лет": sm / rm / 12,
            "Окупаемость по медианам, мес": smd / rmd,
            "Окупаемость по медианам, лет": smd / rmd / 12,
            "Окупаемость по цене за м2, мес": sm2 / rm2,
            "Окупаемость по цене за м2, лет": sm2 / rm2 / 12,
            "Реалистичный сценарий, лет": sm / net / 12,
            "Доходность годовая, %": rm * 12 / sm * 100,
            "Доходность реалистичная, %": net * 12 / sm * 100,
        })
    return pd.DataFrame(rows)


def segment_bias(df: pd.DataFrame) -> pd.DataFrame:
    """
    Численная оценка расхождения выборок аренды и продажи по составу.

    Обнаруженная проблема: в предложении о продаже много элитных новостроек,
    а в аренде их нет вовсе. Из-за этого «средняя цена продажи / средняя цена
    аренды» сравнивает разные сегменты рынка и завышает срок окупаемости.
    """
    rows = []
    for sub in ORDER:
        r = df[(df.deal_type == "rent") & (df.subcat_label == sub)]
        s = df[(df.deal_type == "sale") & (df.subcat_label == sub)]
        rows.append({
            "Подкатегория": sub,
            "Медианная площадь, аренда": r.total_area.median(),
            "Медианная площадь, продажа": s.total_area.median(),
            "Разница площади, %": (s.total_area.median() / r.total_area.median() - 1) * 100,
            "Доля новостроек, аренда, %": r.is_newbuild.mean() * 100,
            "Доля новостроек, продажа, %": s.is_newbuild.mean() * 100,
            "Доля в ЖК, аренда, %": r.jk_name.notna().mean() * 100,
            "Доля в ЖК, продажа, %": s.jk_name.notna().mean() * 100,
        })
    return pd.DataFrame(rows)


N_BANDS = 6          # число интервалов площади при стратификации
MIN_N = 5            # минимум объявлений в интервале с каждой стороны


def _matched_pair(ready: pd.DataFrame, sub: str, trim: bool = True):
    """Пары «аренда/продажа» одной подкатегории на общей области определения."""
    r = ready[(ready.deal_type == "rent") & (ready.subcat_label == sub)]
    s = ready[(ready.deal_type == "sale") & (ready.subcat_label == sub)]
    if trim and not r.empty and not s.empty:
        # общая область определения по площади: интервал, где есть и то и другое
        lo = max(r.total_area.quantile(.05), s.total_area.quantile(.05))
        hi = min(r.total_area.quantile(.95), s.total_area.quantile(.95))
        r, s = r[r.total_area.between(lo, hi)], s[s.total_area.between(lo, hi)]
    return r, s


def payback_matched(df: pd.DataFrame, n_bands: int = N_BANDS,
                    min_n: int = MIN_N) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Сопоставимая оценка окупаемости: устраняем перекос выборок по составу.

    Три исправления:
      1) Из продажи убираем строящееся жильё (ДДУ). Такую квартиру нельзя
         сдать в аренду, поэтому в расчёте окупаемости ей не место.
      2) Обрезаем обе выборки до общей области определения по площади
         (5–95 перцентили): сравниваем только тот диапазон, где реально
         есть и предложение аренды, и предложение продажи.
      3) Стратификация по площади: внутри подкатегории делим объявления на
         интервалы по квантилям арендной выборки и считаем окупаемость
         в каждом интервале отдельно — так крупная дорогая квартира из
         продажи сравнивается с такой же крупной из аренды, а не со средней.

    Итог по подкатегории — взвешенное отношение, где веса берутся из выборки
    аренды: именно её структура отражает то, что реально можно сдать.
    """
    detail, summary = [], []
    ready = df[~df.is_newbuild.fillna(False)]

    for sub in ORDER:
        r_all, s_all = _matched_pair(ready, sub)
        if r_all.empty or s_all.empty:
            continue

        qs = np.linspace(0, 1, n_bands + 1)
        edges = sorted(set(np.round(r_all.total_area.quantile(qs)).tolist()))

        num = den = 0.0
        for lo, hi in zip(edges, edges[1:]):
            r = r_all[r_all.total_area.between(lo, hi)]
            s = s_all[s_all.total_area.between(lo, hi)]
            if len(r) < min_n or len(s) < min_n:
                continue
            w = len(r)                       # вес интервала = объём арендной выборки
            num += w * s.price_rub.median()
            den += w * r.price_rub.median()
            detail.append({
                "Подкатегория": sub, "Интервал площади, м2": f"{lo:.0f}–{hi:.0f}",
                "N аренда": len(r), "N продажа": len(s),
                "Медиана аренды, руб/мес": r.price_rub.median(),
                "Медиана продажи, руб": s.price_rub.median(),
                "Окупаемость, лет": s.price_rub.median() / r.price_rub.median() / 12,
            })
        if den:
            months = num / den
            summary.append({
                "Подкатегория": sub,
                "N аренда (готовое)": len(r_all), "N продажа (готовое)": len(s_all),
                "Окупаемость сопоставимая, мес": months,
                "Окупаемость сопоставимая, лет": months / 12,
                "Доходность сопоставимая, %": 100 * 12 / months,
            })
    return pd.DataFrame(detail), pd.DataFrame(summary)


def robustness(df: pd.DataFrame) -> pd.DataFrame:
    """
    Проверка устойчивости: меняем число интервалов площади и смотрим,
    насколько «плывёт» оценка. Если результат мало зависит от разбиения,
    значит он определяется данными, а не выбором параметра.
    """
    rows = []
    for n in (4, 6, 8, 10):
        _, s = payback_matched(df, n_bands=n)
        row = {"Число интервалов": n}
        row.update({r["Подкатегория"]: round(r["Окупаемость сопоставимая, лет"], 1)
                    for _, r in s.iterrows()})
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
#  Графики
# --------------------------------------------------------------------------- #
def fig_hist(df: pd.DataFrame) -> None:
    """Гистограммы распределения цен: 6 панелей = 6 подзапросов."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5))
    for i, (deal, dl, color) in enumerate([("rent", "Аренда", C_RENT),
                                           ("sale", "Продажа", C_SALE)]):
        for j, sub in enumerate(ORDER):
            ax = axes[i, j]
            g = df[(df.deal_type == deal) & (df.subcat_label == sub)]["price_rub"]
            ax.hist(g, bins=18, color=color, edgecolor="white", linewidth=0.8)
            ax.axvline(g.mean(), color=INK, lw=1.6,
                       label=f"среднее {money(g.mean())}")
            ax.axvline(g.median(), color=INK, lw=1.4, ls="--",
                       label=f"медиана {money(g.median())}")
            ax.set_title(f"{dl}, {sub}   (N={len(g)})")
            ax.xaxis.set_major_formatter(FuncFormatter(money))
            # ограничиваем число делений, иначе подписи цен налезают друг на друга
            ax.xaxis.set_major_locator(MaxNLocator(nbins=5, prune="both"))
            ax.set_xlabel("руб/мес" if deal == "rent" else "руб")
            ax.set_ylabel("число объявлений")
            ax.legend(fontsize=8, loc="upper right")
            ax.grid(axis="x", visible=False)
    fig.suptitle("Распределение цен по подкатегориям, район Хамовники",
                 fontsize=14, fontweight="bold", color=INK, y=1.0)
    fig.tight_layout()
    save(fig, "01_histograms_prices")


def fig_payback(pb: pd.DataFrame) -> None:
    """Ключевой график: точка окупаемости по подкатегориям."""
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 5.4))
    x = np.arange(len(pb))
    w = 0.26

    series = [("Окупаемость по средним, лет", "по средним", C_RENT),
              ("Окупаемость по медианам, лет", "по медианам", C_SALE),
              ("Реалистичный сценарий, лет", "реалистичный*", "#1baf7a")]
    for k, (col, label, color) in enumerate(series):
        bars = ax.bar(x + (k - 1) * w, pb[col], w, color=color, label=label)
        # прямые подписи значений (relief rule для аквамаринового слота)
        ax.bar_label(bars, fmt="%.0f", fontsize=8.5, color=INK, padding=2)
    ax.set_xticks(x, pb["Подкатегория"])
    ax.set_ylabel("лет до окупаемости")
    ax.set_title("Точка окупаемости покупки за счёт аренды")
    ax.legend(fontsize=9)
    ax.grid(axis="x", visible=False)
    ax.set_ylim(0, pb["Реалистичный сценарий, лет"].max() * 1.18)

    bars = ax2.bar(pb["Подкатегория"], pb["Доходность годовая, %"],
                   color=[C_SUB[s] for s in pb["Подкатегория"]], width=0.55)
    ax2.bar_label(bars, fmt="%.2f%%", fontsize=9.5, color=INK, padding=2)
    ax2.set_ylabel("% годовых")
    ax2.set_title("Валовая арендная доходность")
    ax2.grid(axis="x", visible=False)
    ax2.set_ylim(0, pb["Доходность годовая, %"].max() * 1.2)

    fig.suptitle("Окупаемость и доходность, район Хамовники", fontsize=14,
                 fontweight="bold", color=INK)
    fig.text(0.01, -0.03, "* реалистичный сценарий: 11 занятых месяцев из 12 "
             "и НДФЛ 13% с арендного дохода", fontsize=8.5, color=INK2)
    fig.tight_layout()
    save(fig, "02_payback_point")


def fig_means(st: pd.DataFrame) -> None:
    """Средние цены. Аренда и продажа - разные шкалы, поэтому два графика."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    for ax, deal, color, unit in [(a1, "Аренда", C_RENT, "руб/мес"),
                                  (a2, "Продажа", C_SALE, "руб")]:
        d = st[st["Категория"] == deal]
        x = np.arange(len(d))
        b1 = ax.bar(x - 0.2, d["Средняя цена"], 0.38, color=color, label="среднее")
        b2 = ax.bar(x + 0.2, d["Медиана"], 0.38, color=color, alpha=0.5,
                    label="медиана")
        for b in (b1, b2):
            ax.bar_label(b, labels=[money(v) for v in b.datavalues],
                         fontsize=8, color=INK, padding=2)
        ax.set_xticks(x, d["Подкатегория"])
        ax.set_title(f"{deal}, {unit}")
        ax.yaxis.set_major_formatter(FuncFormatter(money))
        ax.legend(fontsize=9)
        ax.grid(axis="x", visible=False)
        ax.set_ylim(0, d["Средняя цена"].max() * 1.15)
    fig.suptitle("Средняя и медианная цена по подкатегориям",
                 fontsize=14, fontweight="bold", color=INK)
    fig.tight_layout()
    save(fig, "03_mean_prices")


def fig_box(df: pd.DataFrame) -> None:
    """Ящики с усами: разброс цены за м2 - видно асимметрию и хвосты."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, deal, dl, color in [(axes[0], "rent", "Аренда, руб/м2 в месяц", C_RENT),
                                (axes[1], "sale", "Продажа, руб/м2", C_SALE)]:
        data = [df[(df.deal_type == deal) & (df.subcat_label == s)]["price_per_m2"]
                for s in ORDER]
        bp = ax.boxplot(data, tick_labels=ORDER, patch_artist=True, widths=0.5,
                        medianprops=dict(color=INK, lw=1.8),
                        whiskerprops=dict(color=INK2, lw=1),
                        capprops=dict(color=INK2, lw=1),
                        flierprops=dict(marker="o", ms=3.5, mfc=color,
                                        mec="white", mew=0.6, alpha=0.7))
        for patch in bp["boxes"]:
            patch.set(facecolor=color, alpha=0.45, edgecolor=color, lw=1.2)
        for i, d in enumerate(data, 1):
            ax.text(i + 0.28, d.median(), f"{d.median():,.0f}".replace(",", " "),
                    va="center", ha="left", fontsize=8.5, color=INK)
        ax.set_title(dl)
        ax.yaxis.set_major_formatter(FuncFormatter(money))
        ax.grid(axis="x", visible=False)
    fig.suptitle("Разброс цены за квадратный метр", fontsize=14,
                 fontweight="bold", color=INK)
    fig.tight_layout()
    save(fig, "04_boxplot_price_per_m2")


def fig_scatter(df: pd.DataFrame) -> None:
    """Связь площади и цены плюс линия регрессии по каждой подкатегории."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for ax, deal, dl in [(axes[0], "rent", "Аренда, руб/мес"),
                         (axes[1], "sale", "Продажа, руб")]:
        for sub in ORDER:
            g = df[(df.deal_type == deal) & (df.subcat_label == sub)]
            if len(g) < 3:
                continue
            ax.scatter(g.total_area, g.price_rub, s=22, color=C_SUB[sub],
                       alpha=0.55, edgecolor="white", linewidth=0.5, label=sub)
            k, b = np.polyfit(g.total_area, g.price_rub, 1)
            xs = np.linspace(g.total_area.min(), g.total_area.max(), 50)
            ax.plot(xs, k * xs + b, color=C_SUB[sub], lw=2)
        r = df[df.deal_type == deal]
        corr = r.total_area.corr(r.price_rub)
        ax.set_title(f"{dl}   (r = {corr:.2f})")
        ax.set_xlabel("общая площадь, м2")
        ax.yaxis.set_major_formatter(FuncFormatter(money))
        ax.legend(fontsize=9)
    fig.suptitle("Зависимость цены от площади", fontsize=14,
                 fontweight="bold", color=INK)
    fig.tight_layout()
    save(fig, "05_area_vs_price")


def fig_structure(df: pd.DataFrame) -> None:
    """Круговые диаграммы структуры выборки и состав жилого фонда."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))

    for ax, deal, dl in [(axes[0], "rent", "Предложение аренды"),
                         (axes[1], "sale", "Предложение продажи")]:
        cnt = df[df.deal_type == deal]["subcat_label"].value_counts().reindex(ORDER)
        ax.pie(cnt, labels=[f"{s}\n{v} шт." for s, v in cnt.items()],
               colors=[C_SUB[s] for s in ORDER], autopct="%1.0f%%",
               startangle=90, wedgeprops=dict(edgecolor="white", linewidth=2),
               textprops=dict(fontsize=9, color=INK))
        ax.set_title(f"{dl}\nN = {int(cnt.sum())}")

    names = {"brick": "кирпич", "monolith": "монолит", "panel": "панель",
             "monolithBrick": "монолит-кирпич", "block": "блочный",
             "stalin": "сталинский", "old": "старый фонд", "wood": "деревянный"}
    mat = df["material"].value_counts().head(5)
    ax = axes[2]
    bars = ax.barh([names.get(i, i) for i in mat.index][::-1], mat.values[::-1],
                   color=C_RENT, height=0.6)
    ax.bar_label(bars, fontsize=9, color=INK, padding=3)
    ax.set_title("Материал стен дома")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, mat.max() * 1.18)

    fig.suptitle("Структура выборки", fontsize=14, fontweight="bold", color=INK)
    fig.tight_layout()
    save(fig, "06_sample_structure")


def fig_floor_year(df: pd.DataFrame) -> None:
    """Влияние этажа и года постройки на цену за м2."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))

    pos = df.copy()
    pos["Этаж"] = np.where(pos.is_first_floor, "первый",
                           np.where(pos.is_last_floor, "последний", "средние"))
    order = ["первый", "средние", "последний"]
    x = np.arange(len(order))
    for k, (deal, dl, color) in enumerate([("rent", "аренда", C_RENT),
                                           ("sale", "продажа", C_SALE)]):
        vals = [pos[(pos.deal_type == deal) & (pos["Этаж"] == o)]["price_per_m2"].median()
                for o in order]
        base = vals[1]                       # нормируем к "средним этажам" = 100%
        rel = [v / base * 100 for v in vals]
        bars = a1.bar(x + (k - 0.5) * 0.38, rel, 0.36, color=color, label=dl)
        a1.bar_label(bars, fmt="%.0f%%", fontsize=8.5, color=INK, padding=2)
    a1.axhline(100, color=INK2, lw=1, ls="--")
    a1.set_xticks(x, order)
    a1.set_ylabel("медиана цены за м2, % к средним этажам")
    a1.set_title("Влияние этажа на цену")
    a1.legend(fontsize=9)
    a1.grid(axis="x", visible=False)
    a1.set_ylim(0, 135)

    bins = [0, 1950, 1980, 2000, 2015, 2030]
    labels = ["до 1950", "1950-79", "1980-99", "2000-14", "2015+"]
    d = df[df.build_year.notna() & (df.deal_type == "sale")].copy()
    d["Эпоха"] = pd.cut(d.build_year, bins, labels=labels)
    med = d.groupby("Эпоха", observed=True)["price_per_m2"].median()
    n = d.groupby("Эпоха", observed=True).size()
    bars = a2.bar(med.index.astype(str), med.values, color=C_SALE, width=0.6)
    a2.bar_label(bars, labels=[money(v) for v in med.values], fontsize=9,
                 color=INK, padding=2)
    for i, v in enumerate(n.values):
        a2.text(i, med.values[i] / 2, f"N={v}", ha="center", fontsize=8,
                color="white", fontweight="bold")
    a2.set_ylabel("медиана цены за м2, руб")
    a2.set_title("Цена продажи по году постройки")
    a2.yaxis.set_major_formatter(FuncFormatter(money))
    a2.grid(axis="x", visible=False)
    a2.set_ylim(0, med.max() * 1.18)

    fig.suptitle("Влияние характеристик дома на цену", fontsize=14,
                 fontweight="bold", color=INK)
    fig.tight_layout()
    save(fig, "07_floor_and_year")


def fig_corr(df: pd.DataFrame) -> None:
    """Тепловая карта корреляций числовых признаков, отдельно аренда и продажа."""
    cols = {"price_rub": "цена", "price_per_m2": "цена за м2",
            "total_area": "общая площ.", "living_area": "жилая площ.",
            "kitchen_area": "кухня", "floor": "этаж",
            "floors_total": "этажность", "build_year": "год постройки",
            "rooms": "комнат"}
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6))
    for ax, deal, dl in [(axes[0], "rent", "Аренда"), (axes[1], "sale", "Продажа")]:
        m = df[df.deal_type == deal][list(cols)].rename(columns=cols).corr()
        im = ax.imshow(m, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(m)), m.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(m)), m.columns)
        for i in range(len(m)):
            for j in range(len(m)):
                v = m.iloc[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.5,
                        color="white" if abs(v) > 0.55 else INK)
        ax.set_title(dl)
        ax.grid(False)
    fig.colorbar(im, ax=axes, shrink=0.75, label="коэффициент корреляции Пирсона")
    fig.suptitle("Корреляция признаков", fontsize=14, fontweight="bold", color=INK)
    save(fig, "08_correlations")


def fig_payback_explained(pb: pd.DataFrame) -> None:
    """Накопленный арендный доход против цены покупки: где пересекаются."""
    fig, ax = plt.subplots(figsize=(9.8, 5.6))
    # горизонт берём с запасом: наивная окупаемость доходит до 75 лет,
    # иначе точки пересечения для 1к и 2к не попадают на график
    years = np.arange(0, 86)
    for sub in ORDER:
        row = pb[pb["Подкатегория"] == sub].iloc[0]
        rent, price = row["Ср. аренда, руб/мес"], row["Ср. продажа, руб"]
        ax.plot(years, rent * 12 * years, color=C_SUB[sub], lw=2, label=sub)
        ax.axhline(price, color=C_SUB[sub], lw=1.1, ls="--", alpha=0.75)
        yr = row["Окупаемость по средним, лет"]
        if yr <= years.max():
            ax.plot(yr, price, "o", ms=8, color=C_SUB[sub], mec="white", mew=1.5)
            ax.annotate(f"{yr:.0f} лет", (yr, price), textcoords="offset points",
                        xytext=(9, -15), fontsize=9.5, color=INK, fontweight="bold")
        ax.text(years.max() + 1, price, money(price), va="center", fontsize=8.5,
                color=C_SUB[sub])
    ax.set_xlabel("лет сдачи в аренду")
    ax.set_ylabel("накопленный арендный доход, руб")
    ax.set_title("Когда аренда окупает покупку\n"
                 "сплошная линия - накопленный доход, пунктир - цена покупки")
    ax.yaxis.set_major_formatter(FuncFormatter(money))
    ax.set_xlim(0, years.max() + 8)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    save(fig, "09_payback_accumulation")


def fig_bias(df: pd.DataFrame, bias: pd.DataFrame) -> None:
    """Диагностика: чем выборка продажи отличается от выборки аренды."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))

    x = np.arange(len(ORDER))
    for k, (col, lbl, color) in enumerate([
            ("Доля в ЖК, аренда, %", "аренда", C_RENT),
            ("Доля в ЖК, продажа, %", "продажа", C_SALE)]):
        bars = a1.bar(x + (k - 0.5) * 0.38, bias[col], 0.36, color=color, label=lbl)
        a1.bar_label(bars, fmt="%.0f%%", fontsize=8.5, color=INK, padding=2)
    a1.set_xticks(x, ORDER)
    a1.set_ylabel("% объявлений в жилых комплексах")
    a1.set_title("Доля предложений в ЖК")
    a1.legend(fontsize=9)
    a1.grid(axis="x", visible=False)
    a1.set_ylim(0, 100)

    for k, (deal, lbl, color) in enumerate([("rent", "аренда", C_RENT),
                                            ("sale", "продажа", C_SALE)]):
        vals = [df[(df.deal_type == deal) & (df.subcat_label == s)].total_area.median()
                for s in ORDER]
        bars = a2.bar(x + (k - 0.5) * 0.38, vals, 0.36, color=color, label=lbl)
        a2.bar_label(bars, fmt="%.0f", fontsize=8.5, color=INK, padding=2)
    a2.set_xticks(x, ORDER)
    a2.set_ylabel("медианная площадь, м2")
    a2.set_title("Медианная площадь объекта")
    a2.legend(fontsize=9)
    a2.grid(axis="x", visible=False)

    fig.suptitle("Почему нельзя сравнивать средние напрямую:\n"
                 "выборки аренды и продажи различаются по составу",
                 fontsize=13, fontweight="bold", color=INK)
    fig.tight_layout()
    save(fig, "10_sample_bias")


def fig_payback_compare(pb: pd.DataFrame, pbm: pd.DataFrame) -> None:
    """Наивная оценка окупаемости против сопоставимой."""
    m = pb.merge(pbm, on="Подкатегория")
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    x = np.arange(len(m))
    series = [("Окупаемость по средним, лет", "наивная: средняя продажа / средняя аренда", C_SALE),
              ("Окупаемость по медианам, лет", "по медианам", "#1baf7a"),
              ("Окупаемость сопоставимая, лет", "сопоставимая: без ДДУ + по площади", C_RENT)]
    for k, (col, lbl, color) in enumerate(series):
        bars = ax.bar(x + (k - 1) * 0.26, m[col], 0.25, color=color, label=lbl)
        ax.bar_label(bars, fmt="%.0f", fontsize=9, color=INK, padding=2)
    ax.set_xticks(x, m["Подкатегория"])
    ax.set_ylabel("лет до окупаемости")
    ax.set_title("Как способ расчёта меняет ответ")
    ax.legend(fontsize=9)
    ax.grid(axis="x", visible=False)
    ax.set_ylim(0, m[[s[0] for s in series]].values.max() * 1.18)
    fig.tight_layout()
    save(fig, "11_payback_naive_vs_matched")


def main() -> None:
    setup_style()
    df = pd.read_csv(PROC / "flats.csv")
    df["subcat_label"] = pd.Categorical(df["subcat_label"], ORDER, ordered=True)

    st, pb = descriptive(df), payback(df)
    bias = segment_bias(df)
    detail, pbm = payback_matched(df)
    rob = robustness(df)

    st.to_csv(PROC / "stats_summary.csv", index=False, encoding="utf-8-sig")
    pb.to_csv(PROC / "payback.csv", index=False, encoding="utf-8-sig")
    bias.to_csv(PROC / "segment_bias.csv", index=False, encoding="utf-8-sig")
    detail.to_csv(PROC / "payback_by_area.csv", index=False, encoding="utf-8-sig")
    pbm.to_csv(PROC / "payback_matched.csv", index=False, encoding="utf-8-sig")
    rob.to_csv(PROC / "robustness.csv", index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(PROC / "analysis.xlsx") as xl:
        st.to_excel(xl, sheet_name="Описательная статистика", index=False)
        pb.to_excel(xl, sheet_name="Окупаемость наивная", index=False)
        pbm.to_excel(xl, sheet_name="Окупаемость сопоставимая", index=False)
        detail.to_excel(xl, sheet_name="Окупаемость по площади", index=False)
        bias.to_excel(xl, sheet_name="Смещение выборок", index=False)
        rob.to_excel(xl, sheet_name="Устойчивость", index=False)

    print("\n=== ОПИСАТЕЛЬНАЯ СТАТИСТИКА ===")
    print(st.round(1).to_string(index=False))
    print("\n=== СМЕЩЕНИЕ ВЫБОРОК ===")
    print(bias.round(1).to_string(index=False))
    print("\n=== ТОЧКА ОКУПАЕМОСТИ (наивная) ===")
    print(pb.round(2).to_string(index=False))
    print("\n=== ОКУПАЕМОСТЬ ПО ИНТЕРВАЛАМ ПЛОЩАДИ ===")
    print(detail.round(1).to_string(index=False))
    print("\n=== ОКУПАЕМОСТЬ СОПОСТАВИМАЯ ===")
    print(pbm.round(2).to_string(index=False))
    print("\n=== УСТОЙЧИВОСТЬ ОЦЕНКИ К ЧИСЛУ ИНТЕРВАЛОВ, лет ===")
    print(rob.to_string(index=False))

    print("\nГрафики:")
    fig_hist(df)
    fig_payback(pb)
    fig_means(st)
    fig_box(df)
    fig_scatter(df)
    fig_structure(df)
    fig_floor_year(df)
    fig_corr(df)
    fig_payback_explained(pb)
    fig_bias(df, bias)
    fig_payback_compare(pb, pbm)
    print("\nГотово.")


if __name__ == "__main__":
    main()
