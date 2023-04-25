import copy
import dataclasses
import typing as tp

import pytest
from pytest import approx

from compgraph import operations as ops


class _Key:
    def __init__(self, *args: str) -> None:
        self._items = args

    def __call__(self, d: tp.Mapping[str, tp.Any]) -> tuple[str, ...]:
        return tuple(str(d.get(key)) for key in self._items)


@dataclasses.dataclass
class MapCase:
    mapper: ops.Mapper
    data: list[ops.TRow]
    ground_truth: list[ops.TRow]
    cmp_keys: tuple[str, ...]
    mapper_item: int = 0
    mapper_ground_truth_items: tuple[int, ...] = (0,)


MAP_CASES = [
    MapCase(
        mapper=ops.LogTransform(column_numerator='x_col', column_denominator='y_col', result_column='log'),
        data=[
            {'test_id': 1, 'x_col': 5, 'y_col': 5},
            {'test_id': 2, 'x_col': 2, 'y_col': 1},
            {'test_id': 3, 'x_col': 1, 'y_col': 10}
        ],
        ground_truth=[
            {'test_id': 1, 'x_col': 5, 'y_col': 5, 'log': approx(0, abs=0.001)},
            {'test_id': 2, 'x_col': 2, 'y_col': 1, 'log': approx(0.6931, abs=0.001)},
            {'test_id': 3, 'x_col': 1, 'y_col': 10, 'log': approx(-2.3025, abs=0.001)}
        ],
        cmp_keys=('test_id', 'x_col', 'y_col', 'log')
    ),
    MapCase(
        mapper=ops.Divide(column_numerator='x_col', column_denominator='y_col', result_column='divide'),
        data=[
            {'test_id': 1, 'x_col': 5, 'y_col': 5},
            {'test_id': 2, 'x_col': 2, 'y_col': 1},
            {'test_id': 3, 'x_col': 1, 'y_col': 10}
        ],
        ground_truth=[
            {'test_id': 1, 'x_col': 5, 'y_col': 5, 'divide': 1},
            {'test_id': 2, 'x_col': 2, 'y_col': 1, 'divide': 2},
            {'test_id': 3, 'x_col': 1, 'y_col': 10, 'divide': 1 / 10}
        ],
        cmp_keys=('test_id', 'x_col', 'y_col', 'divide')
    ),
    MapCase(
        mapper=ops.LongerThanN(column='text', n=3),
        data=[
            {'test_id': 1, 'text': 'hello'},
            {'test_id': 2, 'text': 'hell'},
            {'test_id': 3, 'text': 'hel'},
            {'test_id': 4, 'text': 'no'},
            {'test_id': 5, 'text': 'yes'},
            {'test_id': 6, 'text': '...'},
            {'test_id': 7, 'text': '....'}
        ],
        ground_truth=[
            {'test_id': 1, 'text': 'hello'},
            {'test_id': 2, 'text': 'hell'},
            {'test_id': 7, 'text': '....'}
        ],
        cmp_keys=('test_id', 'text')
    ),
    MapCase(
        mapper=ops.AtLeastNTimes(column='count', n=2),
        data=[
            {'test_id': 1, 'count': 2},
            {'test_id': 2, 'count': 1},
            {'test_id': 3, 'count': 0},
            {'test_id': 4, 'count': 3},
            {'test_id': 5, 'count': 4},
            {'test_id': 6, 'count': 4},
            {'test_id': 7, 'count': 1}
        ],
        ground_truth=[
            {'test_id': 1, 'count': 2},
            {'test_id': 4, 'count': 3},
            {'test_id': 5, 'count': 4},
            {'test_id': 6, 'count': 4},
        ],
        cmp_keys=('test_id', 'count')
    ),
    MapCase(
        mapper=ops.HourWeekday(column='datetime', weekday_column='weekday', hour_column='hour'),
        data=[
            {'test_id': 1, 'datetime': '20221120T112255.10'},
            {'test_id': 2, 'datetime': '20221120T000000'},
            {'test_id': 3, 'datetime': '20221001T102155.10'},
            {'test_id': 4, 'datetime': '20200229T102051.10'},
            {'test_id': 5, 'datetime': '20221124T100000.33'}
        ],
        ground_truth=[
            {'test_id': 1, 'datetime': '20221120T112255.10', 'weekday': 'Sun', 'hour': 11},
            {'test_id': 2, 'datetime': '20221120T000000', 'weekday': 'Sun', 'hour': 0},
            {'test_id': 3, 'datetime': '20221001T102155.10', 'weekday': 'Sat', 'hour': 10},
            {'test_id': 4, 'datetime': '20200229T102051.10', 'weekday': 'Sat', 'hour': 10},
            {'test_id': 5, 'datetime': '20221124T100000.33', 'weekday': 'Thu', 'hour': 10}
        ],
        cmp_keys=('test_id', 'datetime', 'weekday', 'hour')
    ),
    MapCase(
        mapper=ops.Haversine(column='distance', first_point='start', second_point='end'),
        data=[
            {'test_id': 1, 'start': [10.32, 55.11], 'end': [44.8, 52.2]},
            {'test_id': 2, 'start': [05.12, 51.15], 'end': [32.7, 61.61]},
            {'test_id': 3, 'start': [10.32, 32.66], 'end': [27.47, 32.75]},
        ],
        ground_truth=[
            {'test_id': 1, 'start': [10.32, 55.11], 'end': [44.8, 52.2], 'distance': approx(2272.02, abs=0.01)},
            {'test_id': 2, 'start': [05.12, 51.15], 'end': [32.7, 61.61], 'distance': approx(2034.20, abs=0.01)},
            {'test_id': 3, 'start': [10.32, 32.66], 'end': [27.47, 32.75], 'distance': approx(1603.44, abs=0.01)}
        ],
        cmp_keys=('test_id', 'datetime', 'start', 'end')
    ),
    MapCase(
        mapper=ops.TimeDiff(column='timediff', start_time='start', end_time='end'),
        data=[
            {'test_id': 1, 'start': '20221120T122255.10', 'end': '20221120T122256.10'},
            {'test_id': 2, 'start': '20221120T122255', 'end': '20221120T132255'},
            {'test_id': 3, 'start': '20221120T122355.10', 'end': '20221120T132555.10'},
        ],
        ground_truth=[
            {'test_id': 1, 'start': '20221120T122255.10', 'end': '20221120T122256.10', 'timediff': 1 / 3600},
            {'test_id': 2, 'start': '20221120T122255', 'end': '20221120T132255', 'timediff': 1},
            {'test_id': 3, 'start': '20221120T122355.10', 'end': '20221120T132555.10',
             'timediff': approx(1 / 0.9677, abs=0.0001)},
        ],
        cmp_keys=('test_id', 'datetime', 'start', 'end')
    ),
]


@pytest.mark.parametrize('case', MAP_CASES)
def test_mapper(case: MapCase) -> None:
    mapper_data_row = copy.deepcopy(case.data[case.mapper_item])
    mapper_ground_truth_rows = [copy.deepcopy(case.ground_truth[i]) for i in case.mapper_ground_truth_items]

    key_func = _Key(*case.cmp_keys)

    mapper_result = case.mapper(mapper_data_row)
    assert isinstance(mapper_result, tp.Iterator)
    assert sorted(mapper_result, key=key_func) == sorted(mapper_ground_truth_rows, key=key_func)

    result = ops.Map(case.mapper)(iter(case.data))
    assert isinstance(result, tp.Iterator)
    assert sorted(result, key=key_func) == sorted(case.ground_truth, key=key_func)


@dataclasses.dataclass
class ReduceCase:
    reducer: ops.Reducer
    reducer_keys: tuple[str, ...]
    data: list[ops.TRow]
    ground_truth: list[ops.TRow]
    cmp_keys: tuple[str, ...]
    reduce_data_items: tuple[int, ...] = (0,)
    reduce_ground_truth_items: tuple[int, ...] = (0,)


REDUCE_CASES = [
    ReduceCase(
        reducer=ops.TermFrequencyFromCounts(words_column='text', count_column='count'),
        reducer_keys=('doc_id',),
        data=[
            {'doc_id': 1, 'text': 'hello', 'count': 1},
            {'doc_id': 1, 'text': 'little', 'count': 1},
            {'doc_id': 1, 'text': 'world', 'count': 1},

            {'doc_id': 2, 'text': 'little', 'count': 1},

            {'doc_id': 3, 'text': 'little', 'count': 3},

            {'doc_id': 4, 'text': 'little', 'count': 2},
            {'doc_id': 4, 'text': 'hello', 'count': 1},
            {'doc_id': 4, 'text': 'world', 'count': 1},

            {'doc_id': 5, 'text': 'hello', 'count': 2},
            {'doc_id': 5, 'text': 'world', 'count': 1},

            {'doc_id': 6, 'text': 'world', 'count': 4},
            {'doc_id': 6, 'text': 'hello', 'count': 1}
        ],
        ground_truth=[
            {'doc_id': 1, 'text': 'hello', 'tf': approx(0.3333, abs=0.001)},
            {'doc_id': 1, 'text': 'little', 'tf': approx(0.3333, abs=0.001)},
            {'doc_id': 1, 'text': 'world', 'tf': approx(0.3333, abs=0.001)},

            {'doc_id': 2, 'text': 'little', 'tf': approx(1.0)},

            {'doc_id': 3, 'text': 'little', 'tf': approx(1.0)},

            {'doc_id': 4, 'text': 'hello', 'tf': approx(0.25)},
            {'doc_id': 4, 'text': 'little', 'tf': approx(0.5)},
            {'doc_id': 4, 'text': 'world', 'tf': approx(0.25)},

            {'doc_id': 5, 'text': 'hello', 'tf': approx(0.666, abs=0.001)},
            {'doc_id': 5, 'text': 'world', 'tf': approx(0.333, abs=0.001)},

            {'doc_id': 6, 'text': 'hello', 'tf': approx(0.2)},
            {'doc_id': 6, 'text': 'world', 'tf': approx(0.8)}
        ],
        cmp_keys=('doc_id', 'text', 'tf'),
        reduce_data_items=(0, 1, 2),
        reduce_ground_truth_items=(0, 1, 2)
    ),
    ReduceCase(
        reducer=ops.CountRows(column='count'),
        reducer_keys=('doc_id',),
        data=[
            {'doc_id': 1, 'text': 'a'},
            {'doc_id': 1, 'text': 'ab'},
            {'doc_id': 1, 'text': 'abc'},
            {'doc_id': 1, 'text': 'abcd'},
            {'doc_id': 1, 'text': 'b'},
            {'doc_id': 1, 'text': 'c'},
            {'doc_id': 1, 'text': 'd'},
            {'doc_id': 1, 'text': 'bcde'},
        ],
        ground_truth=[
            {'count': 8}
        ],
        cmp_keys=('count',),
        reduce_data_items=(0, 1, 2, 3, 4, 5, 6, 7),
        reduce_ground_truth_items=(0,)
    ),
    ReduceCase(
        reducer=ops.Mean(column='score'),
        reducer_keys=('match_id',),
        data=[
            {'match_id': 1, 'player_id': 1, 'score': 42},
            {'match_id': 1, 'player_id': 2, 'score': 7},
            {'match_id': 1, 'player_id': 3, 'score': 0},
            {'match_id': 1, 'player_id': 4, 'score': 39},

            {'match_id': 2, 'player_id': 5, 'score': 15},
            {'match_id': 2, 'player_id': 6, 'score': 39},
            {'match_id': 2, 'player_id': 7, 'score': 27},
            {'match_id': 2, 'player_id': 8, 'score': 7}
        ],
        ground_truth=[
            {'match_id': 1, 'score': 22},
            {'match_id': 2, 'score': 22}
        ],
        cmp_keys=('test_id', 'text'),
        reduce_data_items=(0, 1, 2, 3),
        reduce_ground_truth_items=(0,)
    )
]


@pytest.mark.parametrize('case', REDUCE_CASES)
def test_reducer(case: ReduceCase) -> None:
    reducer_data_rows = [copy.deepcopy(case.data[i]) for i in case.reduce_data_items]
    reducer_ground_truth_rows = [copy.deepcopy(case.ground_truth[i]) for i in case.reduce_ground_truth_items]

    key_func = _Key(*case.cmp_keys)

    reducer_result = case.reducer(case.reducer_keys, iter(reducer_data_rows))
    assert isinstance(reducer_result, tp.Iterator)
    assert sorted(reducer_result, key=key_func) == sorted(reducer_ground_truth_rows, key=key_func)

    result = ops.Reduce(case.reducer, case.reducer_keys)(iter(case.data))
    assert isinstance(result, tp.Iterator)
    assert sorted(result, key=key_func) == sorted(case.ground_truth, key=key_func)
