# -*- coding: utf-8 -*-
"""
Сбор объявлений с ЦИАН и приведение их к плоской таблице.

Работа в команде (3 человека, у каждого своя комнатность):
    python src/parse.py --rooms 1        # участник 1: однокомнатные
    python src/parse.py --rooms 2        # участник 2: двухкомнатные
    python src/parse.py --rooms 3plus    # участник 3: трёх- и более комнатные

Каждый запуск делает 2 подзапроса (аренда + продажа), итого на команду
6 подзапросов - как и требует задание.

Результат: data/raw/<deal>_<rooms>_<district>.csv (+ .json с сырым ответом).
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cian_api import ROOM_GROUPS, count_offers, iter_offers  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"

DISTRICTS = {21: "Хамовники"}          # основной район исследования
DEFAULT_DISTRICT = 21


# --------------------------------------------------------------------------- #
#  Разбор одного объявления в плоскую строку таблицы
# --------------------------------------------------------------------------- #
def _num(value):
    """'105.0' -> 105.0, None -> None (ЦИАН отдаёт площади строками)."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def flatten(offer: dict) -> dict:
    geo = offer.get("geo") or {}
    terms = offer.get("bargainTerms") or {}
    building = offer.get("building") or {}
    utilities = terms.get("utilitiesTerms") or {}

    districts = geo.get("districts") or []
    raion = next((d["name"] for d in districts if d.get("type") == "raion"), None)
    mikro = next((d["name"] for d in districts if d.get("type") == "mikroraion"), None)
    coords = geo.get("coordinates") or {}
    jk = geo.get("jk") or {}

    # Ближайшее метро. ЦИАН отдаёт список станций, отсортированный по удалённости;
    # берём первую - помеченную isDefault, а если пометки нет, то самую близкую.
    metros = geo.get("undergrounds") or []
    metro = next((m for m in metros if m.get("isDefault")), metros[0] if metros else {})
    walk = metro.get("transportType") == "walk"

    rooms = offer.get("roomsCount")
    total_area = _num(offer.get("totalArea"))
    price = terms.get("priceRur")
    added = offer.get("addedTimestamp")

    return {
        "cian_id": offer.get("cianId") or offer.get("id"),
        "deal_type": offer.get("dealType"),                 # rent / sale
        "rooms": rooms,
        "is_studio": bool(offer.get("isStudio")),
        # для аренды это ставка за месяц, для продажи - полная стоимость
        "price_rub": price,
        "price_per_m2": round(price / total_area, 2) if price and total_area else None,
        "total_area": total_area,
        "living_area": _num(offer.get("livingArea")),
        "kitchen_area": _num(offer.get("kitchenArea")),
        "floor": offer.get("floorNumber"),
        "floors_total": building.get("floorsCount"),
        "build_year": building.get("buildYear"),
        "material": building.get("materialType"),
        "district": raion,
        "microdistrict": mikro,
        # Признаки сегмента рынка. Оказались критичными: выборка продажи сильно
        # смещена в элитные новостройки, которых в аренде нет вовсе,
        # поэтому «в лоб» сравнивать средние по продаже и аренде нельзя.
        "sale_type": terms.get("saleType"),
        # fz214 = договор долевого участия, то есть строящееся жильё:
        # сдать такую квартиру в аренду нельзя, она не готова
        "is_newbuild": terms.get("saleType") == "fz214",
        "jk_name": jk.get("name"),
        "address": geo.get("userInput"),
        # Поле «Адрес/Метро» из задания
        "metro": metro.get("name"),
        "metro_time": metro.get("time"),                        # минут до метро
        "metro_transport": ("пешком" if walk else "транспортом") if metro else None,
        "lat": coords.get("lat"),
        "lng": coords.get("lng"),
        # условия аренды (для продажи будут пустыми)
        "deposit": terms.get("deposit"),
        "utilities_included": utilities.get("includedInPrice"),
        "client_fee_pct": terms.get("clientFee"),
        "lease_term": terms.get("leaseTermType"),
        "added": datetime.fromtimestamp(added, timezone.utc).date().isoformat()
                 if added else None,
        "url": (offer.get("fullUrl") or "").split("?")[0],
    }


# --------------------------------------------------------------------------- #
#  Сбор одного подзапроса
# --------------------------------------------------------------------------- #
def collect(deal: str, rooms_key: str, district: int = DEFAULT_DISTRICT,
            delay: float = 1.2) -> pd.DataFrame:
    rooms = ROOM_GROUPS[rooms_key]
    name = DISTRICTS.get(district, str(district))
    print(f"[{deal} / {rooms_key} / {name}] старт")

    raw = list(iter_offers(deal, rooms, district, delay=delay,
                           log=lambda m: print(m, flush=True)))

    # ЦИАН может отдать одно объявление на нескольких страницах - дедуплицируем
    seen, unique = set(), []
    for offer in raw:
        key = offer.get("cianId") or offer.get("id")
        if key not in seen:
            seen.add(key)
            unique.append(offer)

    stem = f"{deal}_{rooms_key}_{district}"
    RAW.mkdir(parents=True, exist_ok=True)
    (RAW / f"{stem}.json").write_text(
        json.dumps(unique, ensure_ascii=False), encoding="utf-8")

    df = pd.DataFrame([flatten(o) for o in unique])
    df["subcategory"] = rooms_key
    df["collected_at"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    df.to_csv(RAW / f"{stem}.csv", index=False, encoding="utf-8-sig")

    print(f"[{deal} / {rooms_key}] собрано {len(raw)}, уникальных {len(df)}"
          f" -> data/raw/{stem}.csv\n")
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description="Парсер ЦИАН (аренда + продажа)")
    ap.add_argument("--rooms", choices=[*ROOM_GROUPS, "all"], default="all",
                    help="комнатность: 1 | 2 | 3plus | all")
    ap.add_argument("--deal", choices=["rent", "sale", "both"], default="both")
    ap.add_argument("--district", type=int, default=DEFAULT_DISTRICT,
                    help="id района ЦИАН (Хамовники = 21)")
    ap.add_argument("--delay", type=float, default=1.2,
                    help="пауза между страницами, сек")
    args = ap.parse_args()

    rooms_keys = list(ROOM_GROUPS) if args.rooms == "all" else [args.rooms]
    deals = ["rent", "sale"] if args.deal == "both" else [args.deal]

    report = []
    for rooms_key in rooms_keys:
        for deal in deals:
            df = collect(deal, rooms_key, args.district, args.delay)
            report.append((deal, rooms_key, count_offers(
                deal, ROOM_GROUPS[rooms_key], args.district), len(df)))

    print("=" * 58)
    print(f"{'подзапрос':<22}{'нашлось':>10}{'собрано':>10}{'порог 50':>12}")
    for deal, rooms_key, found, got in report:
        flag = "да" if got >= 50 else "НЕТ"
        print(f"{deal + ' / ' + rooms_key:<22}{found:>10}{got:>10}{flag:>12}")


if __name__ == "__main__":
    main()
