# -*- coding: utf-8 -*-
"""
Очистка сырых выгрузок и сведение их в одну таблицу.

Вход : data/raw/*.csv        (результаты 6 подзапросов)
Выход: data/processed/flats.csv, flats.xlsx, cleaning_report.csv
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"

SUBCAT_LABEL = {"1": "1-комнатные", "2": "2-комнатные", "3plus": "3+ комнатные"}
DEAL_LABEL = {"rent": "Аренда", "sale": "Продажа"}

IQR_K = 3.0          # коэффициент межквартильного размаха при отсеве выбросов

# Грубые границы правдоподобия: отсекают опечатки и «цена по запросу»
SANE = {
    "rent": (15_000, 3_000_000),          # ₽/мес
    "sale": (3_000_000, 3_000_000_000),   # ₽
}


def load_raw() -> pd.DataFrame:
    files = sorted(RAW.glob("*_21.csv"))
    if not files:
        raise SystemExit("нет файлов в data/raw — сначала запустите src/parse.py")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df["subcategory"] = df["subcategory"].astype(str)
    print(f"загружено {len(files)} файлов, {len(df)} строк")
    return df


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Возвращает очищенную таблицу и отчёт о том, сколько чего отброшено."""
    steps: list[dict] = []

    def drop(mask: pd.Series, reason: str, frame: pd.DataFrame) -> pd.DataFrame:
        removed = int((~mask).sum())
        steps.append({"шаг": reason, "удалено": removed, "осталось": int(mask.sum())})
        return frame[mask].copy()

    steps.append({"шаг": "исходно собрано", "удалено": 0, "осталось": len(df)})

    # 1. Дубликаты между подзапросами (одно объявление могло попасть в оба файла)
    df = drop(~df.duplicated(subset="cian_id", keep="first"),
              "дубликаты по cian_id", df)

    # 2. Только целевой район
    df = drop(df["district"].eq("Хамовники"), "не Хамовники", df)

    # 3. Только долгосрочная аренда (продажи проходят без условия).
    #    for_day="!1" убирает лишь посуточную, но не аренду «на несколько месяцев».
    df = drop(df["deal_type"].eq("sale") | df["lease_term"].eq("longTerm"),
              "аренда не долгосрочная (fewMonths)", df)

    # 4. Обязательные поля
    df = drop(df["price_rub"].notna() & df["total_area"].notna()
              & df["total_area"].gt(0), "нет цены или площади", df)

    # 5. Правдоподобие цены
    sane = pd.Series(False, index=df.index)
    for deal, (lo, hi) in SANE.items():
        sane |= df["deal_type"].eq(deal) & df["price_rub"].between(lo, hi)
    df = drop(sane, "цена вне правдоподобных границ", df)

    # 6. Правдоподобие площади
    df = drop(df["total_area"].between(10, 500), "площадь вне 10–500 м²", df)

    # 7. Этаж не больше этажности дома
    bad_floor = df["floor"].notna() & df["floors_total"].notna() \
        & (df["floor"] > df["floors_total"])
    df = drop(~bad_floor, "этаж выше этажности дома", df)

    # 8. Статистические выбросы по цене за м² внутри каждой подкатегории.
    #    Межквартильный размах с коэффициентом 3 — «внешняя ограда» Тьюки,
    #    она отсекает именно экстремальные значения. Обычная «внутренняя
    #    ограда» (1.5·IQR) для цен на недвижимость слишком строга: распределение
    #    скошено вправо, и длинный правый хвост здесь не ошибка данных, а
    #    реальные дорогие лоты. Сравнение порогов — в outlier_sensitivity.csv.
    #    Считаем по цене за м², чтобы не наказывать крупные квартиры.
    keep = pd.Series(True, index=df.index)
    for _, g in df.groupby(["deal_type", "subcategory"], observed=True):
        q1, q3 = g["price_per_m2"].quantile([0.25, 0.75])
        iqr = q3 - q1
        keep.loc[g.index] = g["price_per_m2"].between(q1 - IQR_K * iqr,
                                                      q3 + IQR_K * iqr)
    df = drop(keep, f"выбросы по цене за м² ({IQR_K:g}·IQR)", df)

    # Производные признаки для анализа
    df["subcat_label"] = df["subcategory"].map(SUBCAT_LABEL)
    df["deal_label"] = df["deal_type"].map(DEAL_LABEL)
    df["is_first_floor"] = df["floor"].eq(1)
    df["is_last_floor"] = df["floor"].eq(df["floors_total"])
    df["age"] = 2026 - df["build_year"]

    return df.reset_index(drop=True), pd.DataFrame(steps)


def sensitivity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Сколько объявлений осталось бы в каждой подкатегории при разных порогах
    отсева выбросов. Нужна, чтобы показать: порог выбран осознанно, а не
    подогнан под результат, и требование «не менее 50» выдерживается.
    """
    rows = []
    for (deal, sub), g in df.groupby(["deal_type", "subcategory"], observed=True):
        q1, q3 = g["price_per_m2"].quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = g["price_per_m2"].quantile([0.01, 0.99])
        rows.append({
            "Подзапрос": f"{deal} / {sub}",
            "Без отсева": len(g),
            "1.5·IQR": int(g["price_per_m2"].between(q1 - 1.5 * iqr,
                                                     q3 + 1.5 * iqr).sum()),
            "3·IQR (выбран)": int(g["price_per_m2"].between(q1 - 3 * iqr,
                                                           q3 + 3 * iqr).sum()),
            "перцентиль 1–99": int(g["price_per_m2"].between(lo, hi).sum()),
        })
    return pd.DataFrame(rows)


def _before_outliers(raw: pd.DataFrame) -> pd.DataFrame:
    """Данные после всех шагов очистки, кроме отсева выбросов."""
    d = raw.drop_duplicates("cian_id")
    d = d[d["district"].eq("Хамовники")]
    d = d[d["deal_type"].eq("sale") | d["lease_term"].eq("longTerm")]
    d = d[d["price_rub"].notna() & d["total_area"].between(10, 500)]
    sane = pd.Series(False, index=d.index)
    for deal, (lo, hi) in SANE.items():
        sane |= d["deal_type"].eq(deal) & d["price_rub"].between(lo, hi)
    return d[sane]


def main() -> None:
    raw = load_raw()
    df, report = clean(raw)
    PROC.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROC / "flats.csv", index=False, encoding="utf-8-sig")
    df.to_excel(PROC / "flats.xlsx", index=False)
    report.to_csv(PROC / "cleaning_report.csv", index=False, encoding="utf-8-sig")

    sens = sensitivity(_before_outliers(raw))
    sens.to_csv(PROC / "outlier_sensitivity.csv", index=False, encoding="utf-8-sig")

    print("\nОтчёт об очистке")
    print(report.to_string(index=False))

    print("\nЧувствительность к порогу отсева выбросов (требование задания — 50)")
    print(sens.to_string(index=False))

    print("\nИтоговая выборка по подкатегориям")
    pivot = df.pivot_table(index="subcat_label", columns="deal_label",
                           values="cian_id", aggfunc="count")
    pivot["порог 50"] = ["да" if r.min() >= 50 else "НЕТ" for _, r in pivot.iterrows()]
    print(pivot.to_string())
    print(f"\nВсего строк: {len(df)} -> data/processed/flats.csv")


if __name__ == "__main__":
    main()
