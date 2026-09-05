# -*- coding: utf-8 -*-
"""
Пересборка CSV из уже сохранённых сырых JSON-ответов ЦИАН.

Нужна, когда в разбор объявления добавили новые поля: сырые ответы лежат
в data/raw/*.json, поэтому обращаться к сайту повторно не требуется.
Это одно из практических преимуществ сохранения сырых ответов «как есть».

    python src/reflatten.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from parse import flatten  # noqa: E402

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"


def main() -> None:
    for path in sorted(RAW.glob("*_21.json")):
        deal, rooms, _ = path.stem.split("_", 2)
        offers = json.loads(path.read_text(encoding="utf-8"))

        df = pd.DataFrame([flatten(o) for o in offers])
        df["subcategory"] = rooms
        df["collected_at"] = datetime.fromtimestamp(
            path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        df.to_csv(path.with_suffix(".csv"), index=False, encoding="utf-8-sig")
        print(f"  {path.stem}: {len(df)} строк, {df.shape[1]} полей")


if __name__ == "__main__":
    main()
