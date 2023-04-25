from abc import abstractmethod, ABC
from collections.abc import Callable, Sequence
from collections import defaultdict
import calendar
import dateutil.parser
import heapq
import itertools
import math
import string
import re
import typing as tp
from operator import itemgetter

TRow = dict[str, tp.Any]
TRowsIterable = tp.Iterable[TRow]
TRowsGenerator = tp.Generator[TRow, None, None]


class Operation(ABC):  # pragma: no cover
    @abstractmethod
    def __call__(self, rows: TRowsIterable, *args: tp.Any, **kwargs: tp.Any) -> TRowsGenerator:
        pass


class Read(Operation):
    def __init__(self, filename: str, parser: Callable[[str], TRow]) -> None:
        self.filename = filename
        self.parser = parser

    def __call__(self, *args: tp.Any, **kwargs: tp.Any) -> TRowsGenerator:
        with open(self.filename) as f:
            for line in f:
                yield self.parser(line)


class ReadIterGenerator(Operation):
    def __init__(self, name: str) -> None:
        self.name = name

    def __call__(self, *args: tp.Any, **kwargs: tp.Any) -> TRowsGenerator:
        for row in kwargs[self.name]():
            yield row


# Operations


class Mapper(ABC):  # pragma: no cover
    """Base class for mappers"""

    @abstractmethod
    def __call__(self, row: TRow) -> TRowsGenerator:
        """
        :param row: one table row
        """
        pass


class Map(Operation):
    def __init__(self, mapper: Mapper) -> None:
        self.mapper = mapper

    def __call__(self, rows: TRowsIterable, *args: tp.Any, **kwargs: tp.Any) -> TRowsGenerator:
        for row in rows:
            yield from self.mapper(row)


class Reducer(ABC):  # pragma: no cover
    """Base class for reducers"""

    @abstractmethod
    def __call__(self, group_key: tuple[str, ...], rows: TRowsIterable) -> TRowsGenerator:
        """
        :param rows: table rows
        """
        pass


def _safe_groupby(rows: TRowsIterable, keys: Sequence[str]) -> \
        tp.Generator[tuple[tp.Any, tp.Iterable[dict[str, tp.Any]]], None, None]:
    if keys:
        groups = itertools.groupby(rows, itemgetter(*keys))
        prev_keys, group = next(groups)
        yield prev_keys, group

        for keys, group in groups:
            if prev_keys > keys:
                raise ValueError('Stream is not sorted by keys')
            prev_keys = keys
            yield keys, group
    else:
        yield keys, rows


class Reduce(Operation):
    def __init__(self, reducer: Reducer, keys: Sequence[str]) -> None:
        self.reducer = reducer
        self.keys = keys

    def __call__(self, rows: TRowsIterable, *args: tp.Any, **kwargs: tp.Any) -> TRowsGenerator:
        for _, group in _safe_groupby(rows, self.keys):
            yield from self.reducer(tuple(self.keys), group)


class Joiner(ABC):
    """Base class for joiners"""

    def __init__(self, suffix_a: str = '_1', suffix_b: str = '_2') -> None:
        self._a_suffix = suffix_a
        self._b_suffix = suffix_b

    @abstractmethod
    def __call__(self, keys: Sequence[str], rows_a: TRowsIterable, rows_b: TRowsIterable) -> TRowsGenerator:
        """
        :param keys: join keys
        :param rows_a: left table rows
        :param rows_b: right table rows
        """
        pass

    def common_join_part(self, keys: Sequence[str], left_rows: TRowsIterable, right_rows: list[TRow],
                         join_type: str = 'any') -> TRowsGenerator:
        if join_type == 'right':
            relevant_suffix_a, relevant_suffix_b = self._b_suffix, self._a_suffix
        else:
            relevant_suffix_a, relevant_suffix_b = self._a_suffix, self._b_suffix

        for left_row in left_rows:
            for right_row in right_rows:
                new_row = right_row.copy()
                for key, value in left_row.items():
                    if key not in new_row:
                        new_row.update({key: value})
                    else:
                        if key not in keys:
                            old_right_value = new_row[key]
                            del new_row[key]
                            new_row.update({key + relevant_suffix_a: value})
                            new_row.update({key + relevant_suffix_b: old_right_value})
                yield new_row


class Join(Operation):
    def __init__(self, joiner: Joiner, keys: Sequence[str]):
        self.keys = keys
        self.joiner = joiner

    def __call__(self, rows: TRowsIterable, *args: tp.Any, **kwargs: tp.Any) -> TRowsGenerator:
        rows_left = _safe_groupby(rows, self.keys)
        rows_right = _safe_groupby(args[0], self.keys)

        keys_left, rows_left_group = next(rows_left, (None, None))
        keys_right, rows_right_group = next(rows_right, (None, None))

        while True:
            if (rows_left_group is not None and rows_right_group is not None
                    and keys_left is not None and keys_right is not None):
                if keys_left == keys_right:
                    yield from self.joiner(self.keys, rows_left_group, rows_right_group)
                    keys_left, rows_left_group = next(rows_left, (None, None))
                    keys_right, rows_right_group = next(rows_right, (None, None))
                elif keys_left > keys_right:
                    yield from self.joiner(self.keys, [], rows_right_group)
                    keys_right, rows_right_group = next(rows_right, (None, None))
                elif keys_left < keys_right:
                    yield from self.joiner(self.keys, rows_left_group, [])
                    keys_left, rows_left_group = next(rows_left, (None, None))

            elif rows_left_group is not None:
                yield from self.joiner(self.keys, rows_left_group, [])
                keys_left, rows_left_group = next(rows_left, (None, None))

            elif rows_right_group is not None:
                yield from self.joiner(self.keys, [], rows_right_group)
                keys_right, rows_right_group = next(rows_right, (None, None))

            else:
                break


# Dummy operators


class DummyMapper(Mapper):
    """Yield exactly the row passed"""

    def __call__(self, row: TRow) -> TRowsGenerator:
        yield row


class FirstReducer(Reducer):
    """Yield only first row from passed ones"""

    def __call__(self, group_key: tuple[str, ...], rows: TRowsIterable) -> TRowsGenerator:
        for row in rows:
            yield row
            break


# Mappers


class FilterPunctuation(Mapper):
    """Left only non-punctuation symbols"""

    def __init__(self, column: str):
        """
        :param column: name of column to process
        """
        self.column = column
        self.maketrans = str.maketrans('', '', string.punctuation)

    def __call__(self, row: TRow) -> TRowsGenerator:
        row[self.column] = row[self.column].translate(self.maketrans)
        yield row


class LowerCase(Mapper):
    """Replace column value with value in lower case"""

    def __init__(self, column: str):
        """
        :param column: name of column to process
        """
        self.column = column

    @staticmethod
    def _lower_case(txt: str) -> str:
        return txt.lower()

    def __call__(self, row: TRow) -> TRowsGenerator:
        row[self.column] = self._lower_case(row[self.column])
        yield row


class Split(Mapper):
    """Split row on multiple rows by separator"""

    def __init__(self, column: str, separator: str | None = None) -> None:
        """
        :param column: name of column to split
        :param separator: string to separate by
        """
        self.column = column
        self.split_regex = f'[^{separator}]*{separator}' if separator is not None else '(\S*)\s*'  # noqa

    def __call__(self, row: TRow) -> TRowsGenerator:
        for part in re.finditer(self.split_regex, row[self.column]):
            value = part.group().strip()
            if value:
                new_row = row.copy()
                new_row[self.column] = value
                yield new_row


class Product(Mapper):
    """Calculates product of multiple columns"""

    def __init__(self, columns: Sequence[str], result_column: str = 'product') -> None:
        """
        :param columns: column names to product
        :param result_column: column name to save product in
        """
        self.columns = columns
        self.result_column = result_column

    def __call__(self, row: TRow) -> TRowsGenerator:
        row[self.result_column] = 1
        for column in self.columns:
            row[self.result_column] *= row[column]
        yield row


class Divide(Mapper):
    """Calculates product of multiple columns"""

    def __init__(self, column_numerator: str, column_denominator: str, result_column: str = 'divide') -> None:
        """
        :param column_numerator: name of column with numerator
        :param column_denominator: name of column with denominator
        :param result_column: column name to save product in
        """
        self.column_numerator = column_numerator
        self.column_denominator = column_denominator
        self.result_column = result_column

    def __call__(self, row: TRow) -> TRowsGenerator:
        try:
            row[self.result_column] = row[self.column_numerator] / row[self.column_denominator]
        except ZeroDivisionError:
            raise ValueError('Denominator column contains zero value')
        yield row


class Filter(Mapper):
    """Remove records that don't satisfy some condition"""

    def __init__(self, condition: Callable[[TRow], bool]) -> None:
        """
        :param condition: if condition is not true - remove record
        """
        self.condition = condition

    def __call__(self, row: TRow) -> TRowsGenerator:
        if self.condition(row):
            yield row


class Project(Mapper):
    """Leave only mentioned columns"""

    def __init__(self, columns: Sequence[str]) -> None:
        """
        :param columns: names of columns
        """
        self.columns = columns

    def __call__(self, row: TRow) -> TRowsGenerator:
        yield {column: row[column] for column in self.columns}


class LogTransform(Mapper):
    """Maps the point (x, y) -> log(x / y) = log(x) - log(y)"""

    def __init__(self, column_numerator: str, column_denominator: str, result_column: str) -> None:
        """
        :param column_numerator: name of column with numerator
        :param column_denominator: name of column with denominator
        :param result_column: names of resulting column
        """
        self.column_numerator = column_numerator
        self.column_denominator = column_denominator
        self.result_column = result_column

    def __call__(self, row: TRow) -> TRowsGenerator:
        row[self.result_column] = math.log(row[self.column_numerator]) - math.log(row[self.column_denominator])
        yield row


class LongerThanN(Mapper):
    """Leaves only strings that contains more than n chars"""

    def __init__(self, column: str, n: int) -> None:
        """
        :param column: name of column to process
        :param n: threshold for string length
        """
        self.column = column
        self.n = n

    def __call__(self, row: TRow) -> TRowsGenerator:
        if len(row[self.column]) > self.n:
            yield row


class AtLeastNTimes(Mapper):
    """Leaves only strings that occur more than n times"""

    def __init__(self, column: str, n: int) -> None:
        """
        :param column: name of column with counts
        :param n: threshold for frequency
        """
        self.column = column
        self.n = n

    def __call__(self, row: TRow) -> TRowsGenerator:
        if row[self.column] >= self.n:
            yield row


class Haversine(Mapper):
    """Calculate the great circle distance in kilometers between two points
    on the earth, with the earth radius set to 6373 km"""

    EARTH_RADIUS_KM = 6373

    def __init__(self, column: str, first_point: str, second_point: str) -> None:
        """
        :param column: name of resulting column
        :param first_point: name of column with the coordinates of the first point
        :param second_point: name of column with the coordinates of the second point
        """
        self.column = column
        self.first_point = first_point
        self.second_point = second_point

    def __call__(self, row: TRow) -> TRowsGenerator:
        lon1, lat1 = row[self.first_point]
        lon2, lat2 = row[self.second_point]
        lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])

        arg = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
        haversine_dist = 2 * math.asin(math.sqrt(arg)) * self.EARTH_RADIUS_KM

        row[self.column] = haversine_dist
        yield row


class HourWeekday(Mapper):
    """Splits datetime to two different columns with weekday and hour"""

    def __init__(self, column: str, weekday_column: str, hour_column: str) -> None:
        """
        :param column: name of column to process
        :param weekday_column: name for column with weekday
        :param hour_column: name for column with hour
        """
        self.column = column
        self.weekday_column = weekday_column
        self.hour_column = hour_column

    def __call__(self, row: TRow) -> TRowsGenerator:
        dt = dateutil.parser.isoparse(row[self.column])
        row[self.weekday_column] = calendar.day_abbr[dt.weekday()]
        row[self.hour_column] = dt.hour
        yield row


class TimeDiff(Mapper):
    """Calculate the inverse of the difference between two datetimes (in hours)"""

    def __init__(self, column: str, start_time: str, end_time: str) -> None:
        """
        :param column: name of resulting column
        :param start_time: name for column with start time
        :param end_time: name for column with end time
        """
        self.column = column
        self.start_time = start_time
        self.end_time = end_time

    def __call__(self, row: TRow) -> TRowsGenerator:
        dt_start = dateutil.parser.isoparse(row[self.start_time])
        dt_end = dateutil.parser.isoparse(row[self.end_time])

        diff = (dt_end - dt_start).total_seconds() / 3600
        row[self.column] = diff
        yield row


# Reducers


class TopN(Reducer):
    """Calculate top N by value"""

    def __init__(self, column: str, n: int) -> None:
        """
        :param column: column name to get top by
        :param n: number of top values to extract
        """
        self.column_max = column
        self.n = n

    def __call__(self, group_key: tuple[str, ...], rows: TRowsIterable) -> TRowsGenerator:
        yield from heapq.nlargest(self.n, rows, key=itemgetter(self.column_max))


class TermFrequency(Reducer):
    """Calculate frequency of values in column"""

    def __init__(self, words_column: str, result_column: str = 'tf') -> None:
        """
        :param words_column: name for column with words
        :param result_column: name for result column
        """
        self.words_column = words_column
        self.result_column = result_column

    def __call__(self, group_key: tuple[str, ...], rows: TRowsIterable) -> TRowsGenerator:
        new_row: dict[str, tp.Any] = defaultdict(int)

        row = next(iter(rows))
        common_dict: dict[str, tp.Any] = {key: row[key] for key in group_key}

        new_row[row[self.words_column]] += 1
        stream_size = 1

        for row in rows:
            new_row[row[self.words_column]] += 1
            stream_size += 1

        yield from (dict(common_dict, **{self.words_column: column, self.result_column: value / stream_size})
                    for column, value in new_row.items())


class TermFrequencyFromCounts(Reducer):
    """Calculate frequency of values in column having the counts of each word"""

    def __init__(self, words_column: str, count_column: str = 'ctr', result_column: str = 'tf') -> None:
        """
        :param words_column: name for column with words
        :param result_column: name for column with counts
        :param result_column: name for result column
        """
        self.words_column = words_column
        self.count_column = count_column
        self.result_column = result_column

    def __call__(self, group_key: tuple[str, ...], rows: TRowsIterable) -> TRowsGenerator:
        new_row: dict[str, tp.Any] = defaultdict(int)

        row = next(iter(rows))
        common_dict: dict[str, tp.Any] = {key: row[key] for key in group_key}

        new_row[row[self.words_column]] += row[self.count_column]
        stream_size = row[self.count_column]

        for row in rows:
            new_row[row[self.words_column]] += row[self.count_column]
            stream_size += row[self.count_column]

        yield from (dict(common_dict, **{self.words_column: column, self.result_column: value / stream_size})
                    for column, value in new_row.items())


class CountRows(Reducer):
    """
    Count all records
    Example for group_key=(, ) and column='d'
    """

    def __init__(self, column: str) -> None:
        """
        :param column: name for result column
        """
        self.column = column

    def __call__(self, group_key: tuple[str, ...], rows: TRowsIterable) -> TRowsGenerator:
        rows_ctr = sum(1 for _ in rows)
        yield {self.column: rows_ctr}


class Count(Reducer):
    """
    Count records by key
    Example for group_key=('a',) and column='d'
        {'a': 1, 'b': 5, 'c': 2}
        {'a': 1, 'b': 6, 'c': 1}
        =>
        {'a': 1, 'd': 2}
    """

    def __init__(self, column: str) -> None:
        """
        :param column: name for result column
        """
        self.column = column

    def __call__(self, group_key: tuple[str, ...], rows: TRowsIterable) -> TRowsGenerator:
        row = next(iter(rows))

        new_row: dict[str, tp.Any] = {key: row[key] for key in group_key}
        new_row[self.column] = 1

        for _ in rows:
            new_row[self.column] += 1

        yield new_row


class Sum(Reducer):
    """
    Sum values aggregated by key
    Example for key=('a',) and column='b'
        {'a': 1, 'b': 2, 'c': 4}
        {'a': 1, 'b': 3, 'c': 5}
        =>
        {'a': 1, 'b': 5}
    """

    def __init__(self, column: str) -> None:
        """
        :param column: name for sum columns
        """
        self.column = column

    def __call__(self, group_key: tuple[str, ...], rows: TRowsIterable) -> TRowsGenerator:
        row = next(iter(rows))

        new_row: dict[str, tp.Any] = {key: row[key] for key in group_key}

        new_row[self.column] = row[self.column]
        for row in rows:
            new_row[self.column] += row[self.column]

        yield new_row


class Mean(Reducer):
    """
    Mean values aggregated by key
    Example for key=('a',) and column='b'
        {'a': 1, 'b': 2, 'c': 4}
        {'a': 1, 'b': 3, 'c': 5}
        =>
        {'a': 1, 'b': 5}
    """

    def __init__(self, column: str) -> None:
        """
        :param column: name for mean column
        """
        self.column = column

    def __call__(self, group_key: tuple[str, ...], rows: TRowsIterable) -> TRowsGenerator:
        row = next(iter(rows))

        new_row: dict[str, tp.Any] = {key: row[key] for key in group_key}

        new_row[self.column] = row[self.column]
        stream_size = 1

        for row in rows:
            new_row[self.column] += row[self.column]
            stream_size += 1

        new_row[self.column] /= stream_size
        yield new_row


# Joiners


class InnerJoiner(Joiner):
    """Join with inner strategy"""

    def __call__(self, keys: Sequence[str], rows_a: TRowsIterable, rows_b: TRowsIterable) -> TRowsGenerator:
        yield from self.common_join_part(keys, rows_a, list(rows_b))


class OuterJoiner(Joiner):
    """Join with outer strategy"""

    def __call__(self, keys: Sequence[str], rows_a: TRowsIterable, rows_b: TRowsIterable) -> TRowsGenerator:
        materialized_a = list(rows_a)
        materialized_b = list(rows_b)

        if materialized_a and materialized_b:
            yield from self.common_join_part(keys, materialized_a, materialized_b)
        elif materialized_a:
            yield from materialized_a
        elif materialized_b:
            yield from materialized_b


class LeftJoiner(Joiner):
    """Join with left strategy"""

    def __call__(self, keys: Sequence[str], rows_a: TRowsIterable, rows_b: TRowsIterable) -> TRowsGenerator:
        materialized_b = list(rows_b)

        if materialized_b:
            yield from self.common_join_part(keys, rows_a, materialized_b)
        else:
            yield from rows_a


class RightJoiner(Joiner):
    """Join with right strategy"""

    def __call__(self, keys: Sequence[str], rows_a: TRowsIterable, rows_b: TRowsIterable) -> TRowsGenerator:
        materialized_a = list(rows_a)

        if materialized_a:
            yield from self.common_join_part(keys, rows_b, materialized_a, join_type='right')
        else:
            yield from rows_b
