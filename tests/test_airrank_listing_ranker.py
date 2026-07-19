from pathlib import Path
import os
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault(
    "AIRRANK_OUTPUT_DIR",
    str(Path(tempfile.gettempdir()) / "airrank-tests"),
)
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from airrank_listing_ranker import (  # noqa: E402
    aggregate_listing_scores,
    build_vocab,
    clean_text,
    encode_text,
    split_by_listing,
)


class AirRankCoreTests(unittest.TestCase):
    def test_clean_text_and_encoding_are_deterministic(self) -> None:
        cleaned = clean_text(" Great <b>stay</b>! https://example.com ")
        self.assertEqual(cleaned, "great stay")

        vocab = build_vocab(["great stay", "great host"], min_freq=1)
        ids, mask = encode_text("great unknown", vocab, max_len=4)
        self.assertEqual(
            ids.tolist(),
            [vocab["great"], vocab["<unk>"], vocab["<pad>"], vocab["<pad>"]],
        )
        self.assertEqual(mask.tolist(), [1, 1, 0, 0])

    def test_split_by_listing_prevents_listing_leakage(self) -> None:
        frame = pd.DataFrame(
            {
                "listing_id": np.repeat([f"listing-{index}" for index in range(20)], 2),
                "review_id": range(40),
            }
        )
        train, validation, test = split_by_listing(frame, random_state=7)
        train_ids = set(train["listing_id"])
        validation_ids = set(validation["listing_id"])
        test_ids = set(test["listing_id"])

        self.assertTrue(train_ids.isdisjoint(validation_ids))
        self.assertTrue(train_ids.isdisjoint(test_ids))
        self.assertTrue(validation_ids.isdisjoint(test_ids))
        self.assertEqual(train_ids | validation_ids | test_ids, set(frame["listing_id"]))

    def test_listing_aggregation_orders_the_stronger_listing_first(self) -> None:
        reviews = pd.DataFrame(
            [
                {"listing_id": "strong", "review_id": "1", "true_rating": 5.0, "pred_rating": 4.8, "review_score": 0.95},
                {"listing_id": "strong", "review_id": "2", "true_rating": 5.0, "pred_rating": 4.9, "review_score": 0.98},
                {"listing_id": "weak", "review_id": "3", "true_rating": 3.0, "pred_rating": 3.1, "review_score": 0.52},
                {"listing_id": "weak", "review_id": "4", "true_rating": 3.0, "pred_rating": 2.9, "review_score": 0.48},
            ]
        )

        ranked = aggregate_listing_scores(reviews)

        self.assertEqual(ranked.iloc[0]["listing_id"], "strong")
        self.assertTrue(ranked["final_score"].is_monotonic_decreasing)


if __name__ == "__main__":
    unittest.main()
