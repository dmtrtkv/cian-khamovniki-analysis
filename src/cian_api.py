# -*- coding: utf-8 -*-
"""
Низкоуровневый клиент неофициального JSON-API ЦИАН.

Эндпоинт: POST https://api.cian.ru/search-offers/v2/search-offers-desktop/
Тело запроса: {"jsonQuery": {...}} - тот же формат, который использует сам сайт.

Почему не HTML-парсинг (cianparser):
  * ответ приходит уже структурированным JSON - не нужно разбирать вёрстку,
    которая ломается при каждом редизайне;
  * в одном ответе 28 объявлений со ВСЕМИ полями (площади, этаж, год постройки,
    материал стен, координаты, депозит), а не только те, что видны в карточке;
  * скорость: ~28 объектов за 1 запрос против 1 запроса на карточку.

Защита ЦИАН от ботов проверяет TLS-отпечаток клиента, поэтому обычный requests
получает 403. Используем curl_cffi с impersonate="chrome" - он воспроизводит
TLS/JA3-отпечаток настоящего браузера.
"""
from __future__ import annotations

import random
import time
from typing import Any, Iterator

from curl_cffi import requests

SEARCH_URL = "https://api.cian.ru/search-offers/v2/search-offers-desktop/"
DISTRICTS_URL = "https://www.cian.ru/api/geo/get-districts-tree/?locationId={loc}"

HEADERS = {
    "Content-Type": "application/json",
    "Accept": "*/*",
    "Origin": "https://www.cian.ru",
    "Referer": "https://www.cian.ru/",
}

OFFERS_PER_PAGE = 28
MAX_PAGE = 54          # жёсткий предел пагинации ЦИАН: 54 * 28 = 1512 объявлений
ROOM_GROUPS = {        # подкатегории по комнатности
    "1": [1],
    "2": [2],
    "3plus": [3, 4, 5, 6],
}


class CianError(RuntimeError):
    pass


def _post(payload: dict, attempts: int = 6, timeout: int = 40) -> dict:
    """POST с экспоненциальным backoff: у curl_cffi бывают плавающие TLS-сбои."""
    last = None
    for attempt in range(attempts):
        try:
            resp = requests.post(
                SEARCH_URL, json=payload, headers=HEADERS,
                impersonate="chrome", timeout=timeout,
            )
            if resp.status_code != 200:
                raise CianError(f"HTTP {resp.status_code}")
            return resp.json()["data"]
        except Exception as exc:              # TLS-сбой, таймаут, 429, битый JSON
            last = exc
            time.sleep(2 ** attempt + random.uniform(0, 1.5))
    raise CianError(f"не удалось выполнить запрос после {attempts} попыток: {last}")


def build_query(deal: str, rooms: list[int], geo_id: int, page: int = 1,
                geo_type: str = "district",
                price_range: tuple[int, int] | None = None) -> dict:
    """
    Собирает jsonQuery.

    deal        : "rent" (аренда) | "sale" (продажа)
    rooms       : список комнатности, напр. [1] или [3,4,5,6]
    geo_id      : id района ЦИАН (Хамовники = 21)
    price_range : (от, до) в рублях - нужен, чтобы разбить выборку,
                  которая не влезает в 54 страницы
    """
    query: dict[str, Any] = {
        "_type": f"flat{deal}",
        "engine_version": {"type": "term", "value": 2},
        "room": {"type": "terms", "value": rooms},
        "page": {"type": "term", "value": page},
        # ВАЖНО: ключ "district" ЦИАН молча игнорирует, фильтр по району
        # применяется только через "geo".
        "geo": {"type": "geo", "value": [{"type": geo_type, "id": geo_id}]},
    }
    if deal == "rent":
        # "!1" = исключить посуточную аренду, оставить только длительную
        query["for_day"] = {"type": "term", "value": "!1"}
    if price_range:
        query["price"] = {"type": "range",
                          "value": {"gte": price_range[0], "lte": price_range[1]}}
    return {"jsonQuery": query}


def count_offers(deal: str, rooms: list[int], geo_id: int, **kw) -> int:
    """Сколько всего объявлений находит ЦИАН по подзапросу (без выкачивания)."""
    return _post(build_query(deal, rooms, geo_id, page=1, **kw))["offerCount"]


def iter_offers(deal: str, rooms: list[int], geo_id: int,
                price_range: tuple[int, int] | None = None,
                delay: float = 1.2, log=print) -> Iterator[dict]:
    """
    Постранично отдаёт сырые объявления одного подзапроса.

    Если объявлений больше 1512 (предел пагинации), выборка автоматически
    режется по диапазонам цены и каждый кусок выкачивается отдельно.
    """
    total = count_offers(deal, rooms, geo_id, price_range=price_range)
    log(f"    найдено {total} объявлений"
        + (f" в диапазоне цены {price_range}" if price_range else ""))

    if total > MAX_PAGE * OFFERS_PER_PAGE and price_range is None:
        log("    > предел пагинации, режу выборку по цене")
        yield from _iter_split_by_price(deal, rooms, geo_id, delay, log)
        return

    for page in range(1, min(MAX_PAGE, -(-total // OFFERS_PER_PAGE)) + 1):
        data = _post(build_query(deal, rooms, geo_id, page, price_range=price_range))
        offers = data.get("offersSerialized") or []
        if not offers:
            break
        yield from offers
        log(f"    стр. {page}: +{len(offers)}")
        time.sleep(delay + random.uniform(0, 0.6))


def _iter_split_by_price(deal, rooms, geo_id, delay, log):
    """Рекурсивное деление выборки пополам по цене, пока кусок не влезет в 1512."""
    hi = 3_000_000 if deal == "rent" else 3_000_000_000
    stack = [(0, hi)]
    while stack:
        lo, hi = stack.pop()
        n = count_offers(deal, rooms, geo_id, price_range=(lo, hi))
        if n > MAX_PAGE * OFFERS_PER_PAGE and hi - lo > 1000:
            mid = (lo + hi) // 2
            stack += [(lo, mid), (mid + 1, hi)]
            continue
        if n:
            yield from iter_offers(deal, rooms, geo_id, (lo, hi), delay, log)


def get_districts(location_id: int = 1) -> list[dict]:
    """Справочник районов города (Москва = 1, Екатеринбург = 4743)."""
    for attempt in range(5):
        try:
            r = requests.get(DISTRICTS_URL.format(loc=location_id),
                             impersonate="chrome", timeout=30)
            return r.json()
        except Exception:
            time.sleep(2 ** attempt)
    raise CianError("не удалось получить справочник районов")
