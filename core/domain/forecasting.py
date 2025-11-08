import json
import os
import random
import time
from datetime import datetime
from functools import reduce
from typing import Literal

import requests
import tqdm

from core.domain.schema import BinaryProblem, Problem, ProblemDomain
from utils.io_utils import dump_file, load_file, logger


class Forecasting(ProblemDomain):
    """Forecasting as a problem domain, with access to ground truth."""

    def __init__(
        self,
        dataset_files: list[str] = [
            "metaculus_resolved_binary.json",
            "polymarket_resolved_binary.json",
        ],  # noqa: B006
        train_size: float = 0.75,
        auto_fetch: bool = True,
        max_questions_per_source: int = 15000,
        date_cutoff: str = "2024-09-20T00:00:00Z",
    ):
        """Instantiate a forecasting problem set.

        :param dataset_files: dataset filepaths relative to `data/questions/`, defaults to ["metaculus_resolved_binary.json"]
        :type dataset_files: list[str], optional
        :param train_size: the portion of samples to serve as training samples, defaults to 0.8
        :type train_size: float, optional
        """
        super().__init__()
        self.train_size = train_size
        self.auto_fetch = auto_fetch
        self.max_questions_per_source = max_questions_per_source
        self.date_cutoff = date_cutoff

        # Access debating data
        self.dataset_paths = [os.path.join("data", "questions", dataset_file) for dataset_file in dataset_files]

        # Try to load existing data or fetch if missing
        self.dataset_content = []
        for dataset_file, dataset_path in zip(dataset_files, self.dataset_paths, strict=False):
            if os.path.exists(dataset_path):
                logger.minor(f"Loading existing {dataset_file}...")
                self.dataset_content.extend(load_file(dataset_path))
            elif self.auto_fetch:
                logger.major(f"File {dataset_path} not found. Fetching data...")
                if "metaculus" in dataset_file:
                    fetched_data = self._fetch_metaculus_resolved_binary()
                    if fetched_data:
                        os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
                        dump_file(dataset_path, fetched_data, indent=2)
                        self.dataset_content.extend(fetched_data)
                        logger.major(f"Saved {len(fetched_data)} Metaculus questions to {dataset_path}")
                elif "polymarket" in dataset_file:
                    fetched_data = self._fetch_polymarket_resolved_binary()
                    if fetched_data:
                        os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
                        dump_file(dataset_path, fetched_data, indent=2)
                        self.dataset_content.extend(fetched_data)
                        logger.major(f"Saved {len(fetched_data)} PolyMarket questions to {dataset_path}")
            else:
                logger.major(f"File {dataset_path} not found and auto_fetch is disabled")

        # Parse questions
        self.questions_all = [
            BinaryProblem(
                id=q["id"],
                question=(
                    q["description"].replace("\n\n", "\n")
                    + " "
                    + q["title"]
                    + " Provider your best guess as a probability estimate."
                    if "description" in q and q["description"]
                    else q["title"]
                ),
                options=q["outcomes"],
                correct_option=q["outcomes"].index(q["resolution"]),
                aux_info=q | {"domain_name": "forecasting"},
            )
            for q_id, q in enumerate(self.dataset_content)
            if (
                q["marketType"] in ["binary", "normal", None]
                and q["resolution"] in q["outcomes"]
                and len(q["outcomes"]) == 2
                and ("Short" not in q["outcomes"] or "Long" not in q["outcomes"])  # Rule out scalar markets
                and q["resolution"] is not None
                and q["endTime"] is not None
                and datetime.fromisoformat(q["endTime"]) >= datetime.fromisoformat(self.date_cutoff)
            )
        ]

        self.make_questions_splits(self.train_size)

    def postprocess_sample(self, sample: BinaryProblem) -> BinaryProblem:
        """Postprocess a sample after it is sampled. Sample count must not change at this stage."""
        return sample.shuffle_options(self._rng)

    def _fetch_polymarket_resolved_binary(self):
        """Fetch resolved binary markets from PolyMarket with comprehensive strategies."""
        url = "https://gamma-api.polymarket.com/markets"
        batch_size = 500
        all_markets = []
        seen_ids = set()  # Track unique market IDs to avoid duplicates

        logger.minor(f"Fetching PolyMarket markets (target: {self.max_questions_per_source})...")

        # Try multiple strategies to ensure comprehensive coverage
        strategies = [
            {"order": "endDate", "ascending": False},  # Newest first
            {"order": "endDate", "ascending": True},  # Oldest first
            {"order": "volume", "ascending": False},  # Highest volume first
        ]

        for strategy in strategies:
            if len(all_markets) >= self.max_questions_per_source:
                break

            offset = 0
            consecutive_empty = 0
            max_consecutive_empty = 5

            with tqdm.tqdm(desc=f"PolyMarket ({strategy['order']})", unit="market") as pbar:
                while consecutive_empty < max_consecutive_empty and len(all_markets) < self.max_questions_per_source:
                    params = {"closed": "true", "limit": batch_size, "offset": offset, **strategy}

                    # Add date cutoff if specified
                    if self.date_cutoff:
                        params["end_date_min"] = self.date_cutoff + "T00:00:00Z"

                    try:
                        response = requests.get(url, params=params, timeout=30)
                        response.raise_for_status()
                        batch = response.json()

                        if not batch:
                            consecutive_empty += 1
                            offset += batch_size
                            continue

                        consecutive_empty = 0

                        # Deduplicate markets
                        new_markets = []
                        for market in batch:
                            market_id = market.get("id")
                            if market_id and market_id not in seen_ids:
                                seen_ids.add(market_id)
                                new_markets.append(market)

                        if new_markets:
                            all_markets.extend(new_markets)
                            pbar.update(len(new_markets))
                            pbar.set_postfix({"unique": len(all_markets), "offset": offset})

                        offset += len(batch)

                        # Rate limiting - be respectful
                        if len(all_markets) % 10000 == 0 and len(all_markets) > 0:
                            time.sleep(1)

                    except requests.exceptions.RequestException as e:
                        logger.minor(f"Failed to fetch batch at offset {offset}: {e}")
                        consecutive_empty += 1
                        offset += batch_size
                        time.sleep(2)

        # Limit to max_questions_per_source
        if len(all_markets) > self.max_questions_per_source:
            all_markets = all_markets[: self.max_questions_per_source]

        # Process markets into our format
        data = []
        for m in all_markets:
            try:
                # Extract outcomes
                outcomes_raw = m.get("outcomes", [])
                if isinstance(outcomes_raw, str):
                    outcomes = eval(outcomes_raw)
                else:
                    outcomes = outcomes_raw

                if len(outcomes) != 2:
                    continue

                # Extract prices
                prices_raw = m.get("outcomePrices", [])
                if isinstance(prices_raw, str):
                    outcomePrices = eval(prices_raw)
                else:
                    outcomePrices = prices_raw
                outcomePrices = [float(price) for price in outcomePrices]

                # Check if resolved (>= 99% confidence)
                max_price = max(outcomePrices)
                if max_price < 0.99:
                    continue

                resolution_index = outcomePrices.index(max_price)
                resolution = outcomes[resolution_index]

                # Apply date cutoff if specified
                if self.date_cutoff:
                    try:
                        end_date = datetime.fromisoformat(m.get("endDate", "").replace("Z", "+00:00"))
                        cutoff_date = datetime.fromisoformat(self.date_cutoff)
                        if end_date < cutoff_date:
                            continue
                    except (ValueError, TypeError, AttributeError):
                        pass

                market_data = {
                    "id": f"polymarket-{m['id']}",
                    "source": "polymarket",
                    "title": m.get("question", m.get("title", "")),
                    "description": m.get("description", ""),
                    "category": m.get("category"),
                    "endTime": m.get("endDate"),
                    "volume": float(m.get("volume", 0)) if m.get("volume") else 0,
                    "liquidity": float(m.get("liquidity", 0)) if m.get("liquidity") else 0,
                    "outcomes": outcomes,
                    "outcomePrices": outcomePrices,
                    "resolution": resolution,
                    "marketType": m.get("marketType", "binary"),
                }
                data.append(market_data)

            except (ValueError, TypeError, SyntaxError, IndexError):
                continue

        logger.minor(f"Fetched {len(data)} resolved binary markets from PolyMarket")
        return data

    def _fetch_metaculus_resolved_binary(self):
        """Fetch resolved binary questions from Metaculus with comprehensive strategies."""
        url = "https://www.metaculus.com/api/posts/"
        all_questions = []
        seen_ids = set()  # Track unique question IDs to avoid duplicates

        logger.minor(f"Fetching Metaculus questions (target: {self.max_questions_per_source})...")

        # Try multiple parameter combinations to maximize coverage
        param_sets = [
            {"statuses": "resolved", "forecast_type": "binary", "order_by": "-vote_score"},
            {"statuses": "resolved", "forecast_type": "binary", "order_by": "vote_score"},
            {"statuses": "resolved", "forecast_type": "binary", "order_by": "-created_at"},
            {"statuses": "resolved", "forecast_type": "binary", "order_by": "created_at"},
            {"statuses": "resolved", "forecast_type": "binary", "order_by": "-scheduled_resolve_time"},
        ]

        # Try different date ranges if no specific cutoff is provided
        date_ranges = [self.date_cutoff] if self.date_cutoff else [None, "2020-01-01", "2018-01-01", "2015-01-01"]

        for params in param_sets:
            if len(all_questions) >= self.max_questions_per_source:
                break

            for date_cutoff in date_ranges:
                if len(all_questions) >= self.max_questions_per_source:
                    break

                if date_cutoff:
                    params["scheduled_resolve_time__gt"] = date_cutoff
                elif "scheduled_resolve_time__gt" in params:
                    del params["scheduled_resolve_time__gt"]

                with tqdm.tqdm(desc=f"Metaculus ({params.get('order_by', 'default')})", unit="question") as pbar:
                    try:
                        response = requests.get(url, params=params, timeout=30)

                        # Handle rate limiting
                        if response.status_code == 429:
                            time.sleep(30)
                            continue

                        response.raise_for_status()
                        result = response.json()

                        next_page = result.get("next")
                        questions = result.get("results", [])

                        # Deduplicate questions
                        for q in questions:
                            q_id = q.get("id")
                            if q_id and q_id not in seen_ids:
                                seen_ids.add(q_id)
                                all_questions.append(q)
                                pbar.update(1)

                        # Follow pagination
                        while next_page and len(all_questions) < self.max_questions_per_source:
                            try:
                                response = requests.get(next_page, timeout=30)

                                # Handle rate limiting
                                if response.status_code == 429:
                                    time.sleep(30)
                                    break

                                response.raise_for_status()
                                result = response.json()

                                next_page = result.get("next")
                                questions = result.get("results", [])

                                for q in questions:
                                    q_id = q.get("id")
                                    if q_id and q_id not in seen_ids:
                                        seen_ids.add(q_id)
                                        all_questions.append(q)
                                        pbar.update(1)

                                # Rate limiting - be respectful
                                if len(all_questions) % 500 == 0:
                                    time.sleep(2)

                            except requests.exceptions.RequestException as e:
                                logger.minor(f"Failed to fetch page: {e}")
                                break

                    except requests.exceptions.RequestException as e:
                        logger.minor(f"Failed initial request: {e}")

        # Limit to max_questions_per_source
        if len(all_questions) > self.max_questions_per_source:
            all_questions = all_questions[: self.max_questions_per_source]

        # Process questions into our format
        data = []
        for q in all_questions:
            # Skip if missing required fields
            if "question" not in q or not q["question"]:
                continue

            # Skip annulled questions
            if q["question"].get("resolution") == "annulled":
                continue

            # Apply date cutoff
            if self.date_cutoff and q.get("scheduled_resolve_time"):
                try:
                    resolve_time = datetime.fromisoformat(q["scheduled_resolve_time"].replace("Z", "+00:00"))
                    cutoff_date = datetime.fromisoformat(self.date_cutoff)
                    if resolve_time < cutoff_date:
                        continue
                except (ValueError, TypeError):
                    pass

            resolution = q["question"].get("resolution")
            if resolution not in ["yes", "no"]:
                continue

            question_data = {
                "id": f"metaculus-{q['id']}",
                "source": "metaculus",
                "title": q["question"]["title"],
                "description": q["question"].get("description", ""),
                "category": q.get("topic_name"),
                "endTime": q.get("scheduled_resolve_time"),
                "outcomes": ["Yes", "No"],
                "outcomePrices": [1.0, 0.0] if resolution == "yes" else [0.0, 1.0],
                "resolution": "Yes" if resolution == "yes" else "No",
                "marketType": "binary",
                "vote_score": q.get("vote_score", 0),
                "comment_count": q.get("comment_count", 0),
                "forecaster_count": q.get("forecaster_count", 0),
            }
            data.append(question_data)

        logger.minor(f"Fetched {len(data)} resolved binary questions from Metaculus")
        return data
