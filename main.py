# -*- coding: utf-8 -*-
"""
Единая точка входа: сбор данных с ЦИАН, очистка, анализ и построение графиков.

    python main.py                # весь цикл: сбор -> очистка -> анализ
    python main.py --no-parse     # без обращения к ЦИАН, на сохранённых данных
    python main.py --rooms 1      # собрать только однокомнатные

Запуск без сбора (--no-parse) нужен, когда ЦИАН ограничивает частоту запросов:
сырые ответы сохранены в data/raw/*.json, и все таблицы и графики
пересобираются из них без единого обращения к сайту.

Этапы:
    1. src/parse.py      сбор объявлений (6 подзапросов) -> data/raw/*.csv
    2. src/reflatten.py  пересборка таблиц из сохранённых JSON
    3. src/clean.py      очистка и сведение -> data/processed/flats.csv
    4. src/analyze.py    расчёты и графики -> figures/*.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SRC))


def banner(step: int, title: str) -> None:
    print(f"\n{'=' * 62}\n  ЭТАП {step}. {title}\n{'=' * 62}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Парсинг ЦИАН и анализ рынка квартир района Хамовники")
    ap.add_argument("--no-parse", action="store_true",
                    help="пропустить сбор, работать на сохранённых данных")
    ap.add_argument("--rooms", choices=["1", "2", "3plus", "all"], default="all",
                    help="комнатность для сбора (по умолчанию все)")
    ap.add_argument("--delay", type=float, default=1.2,
                    help="пауза между запросами к ЦИАН, сек")
    args = ap.parse_args()

    import analyze
    import clean
    import parse
    import reflatten

    if args.no_parse:
        banner(1, "СБОР ДАННЫХ — пропущен (--no-parse)")
    else:
        banner(1, "СБОР ДАННЫХ С ЦИАН")
        rooms = list(parse.ROOM_GROUPS) if args.rooms == "all" else [args.rooms]
        for rooms_key in rooms:
            for deal in ("rent", "sale"):
                parse.collect(deal, rooms_key, parse.DEFAULT_DISTRICT, args.delay)

    banner(2, "ПЕРЕСБОРКА ТАБЛИЦ ИЗ СЫРЫХ ОТВЕТОВ")
    reflatten.main()

    banner(3, "ОЧИСТКА И СВЕДЕНИЕ В ОДНУ ТАБЛИЦУ")
    clean.main()

    banner(4, "РАСЧЁТЫ И ГРАФИКИ")
    analyze.main()

    print(f"\n{'=' * 62}\n  ГОТОВО\n{'=' * 62}")
    print("  data/processed/flats.csv     итоговая таблица")
    print("  data/processed/analysis.xlsx все расчёты")
    print("  figures/                     графики")


if __name__ == "__main__":
    main()
