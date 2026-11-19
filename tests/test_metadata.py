import io

import pandas as pd
import pytest

pytest.importorskip("openpyxl")
pytest.importorskip("sklearn")

from avocado_ai.data.metadata import (  # noqa: E402
    TARGET_COLUMN,
    add_regression_target,
    load_raw_metadata,
    split_by_group,
)


def spreadsheet(rows) -> io.BytesIO:
    frame = pd.DataFrame(
        rows,
        columns=[
            "File Name",
            "Time Stamp",
            "Storage Group",
            "Sample",
            "Day of Experiment",
            "Ripening Index Classification",
        ],
    )
    buffer = io.BytesIO()
    frame.to_excel(buffer, index=False)
    buffer.seek(0)
    return buffer


def test_target_is_days_until_first_overripe_day_clipped_at_minus_two():
    rows = [
        (f"T20_d{d:02d}_001_a_1", "2022-04-04 10:00:00", "T20", 1, d, stage)
        for d, stage in [(1, 1), (3, 2), (5, 3), (6, 4), (9, 5), (12, 5)]
    ]
    rows.append(("T10_d01_002_a_1", "2022-04-04 10:00:00", "T10", 2, 1, 2))  # never overripe
    df = add_regression_target(load_raw_metadata(spreadsheet(rows)))

    sample_1 = df[df["group_id"] == "T20_1"].set_index("day")[TARGET_COLUMN]
    assert sample_1.to_dict() == {1: 8, 3: 6, 5: 4, 6: 3, 9: 0, 12: -2}  # 9-12=-3 -> -2
    assert df.loc[df["group_id"] == "T10_2", TARGET_COLUMN].isna().all()


def test_split_keeps_each_fruit_in_one_partition():
    df = pd.DataFrame(
        {
            "group_id": [f"g{i // 4}" for i in range(400)],  # 100 fruits x 4 photos
            "stage": [1 + (i % 5) for i in range(400)],
        }
    )
    train, val, test = split_by_group(df, seed=0)
    assert len(train) + len(val) + len(test) == len(df)
    groups = [set(part["group_id"]) for part in (train, val, test)]
    assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
    assert 0.10 < len(test) / len(df) < 0.20
