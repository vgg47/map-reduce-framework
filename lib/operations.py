from abc import abstractmethod, ABC
from operator import itemgetter
from collections import defaultdict

import typing as tp
import itertools
import re
from math import radians, cos, sin, asin, sqrt
import math
import heapq
from dateutil import parser

TRow = tp.Dict[str, tp.Any]
TRowsIterable = tp.Iterable[TRow]
TRowsGenerator = tp.Generator[TRow, None, None]


class Operation(ABC):
    @abstractmethod
    def __call__(self, rows: TRowsIterable, *args: tp.Any, **kwargs: tp.Any) -> TRowsGenerator:
        pass


# Operations


class Mapper(ABC):
    """Base class for mappers"""

    @abstractmethod
    def __call__(self, row: tp.Any) -> TRowsGenerator:
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


class Reducer(ABC):
    """Base class for reducers"""

    @abstractmethod
    def __call__(self, rows: TRowsIterable) -> TRowsGenerator:
        """
        :param rows: table rows
        """
        pass


class Reduce(Operation):
    """Reduce object factory"""
    def __init__(self, reducer: Reducer, keys: tp.Sequence[str]) -> None:
        """Initialize reduce factory

        @param reducer: reducer which object will generate
        @param keys: keys that will be used for grouping
        """
        self.reducer = reducer
        self.keys = keys

    def __call__(self, rows: TRowsIterable, *args: tp.Any, **kwargs: tp.Any) -> TRowsGenerator:
        """Reduce data by self.keys, applying self.reducer to each group

        @param rows: input data generator
        @return: result generator
        """
        for group_values, group in itertools.groupby(rows, key=lambda x: [x[key] for key in self.keys]):
            for row in self.reducer(group):
                row.update(dict(zip(self.keys, group_values)))
                yield row


class Joiner(ABC):
    """Base class for joiners"""

    def __init__(self, suffix_a: str = '', suffix_b: str = '') -> None:
        """ Initialize joiner object

        @param suffix_a: суффикс для имени столбца со значениями из первого графа, если столбец с данным названием есть
        в обоих графах
        @param suffix_b: суффикс для имени столбца со значениями из второго графа, если столбец с данным названием есть
        в обоих графах
        """
        self._a_suffix = suffix_a
        self._b_suffix = suffix_b

    def merge_rows(self, row_a: TRow, row_b: TRow, keys: tp.Sequence[str]) -> TRow:
        """Метод обрабатывает строку в графе, который является результатом джоина, добавляя суффиксы к ключам, которые
        есть в обоих исходных графах, но по которым не осуществляется джоин.

        @param a: строка из первого графа
        @param b: строка из второго графа
        @param keys: ключи, по которым выполняется джоин
        @return: обработанная строка
        """
        common_keys = (row_a.keys() & row_b.keys()) - set(keys)

        merged: TRow = {}
        for key, value in itertools.chain(row_a.items(), row_b.items()):
            if key not in common_keys:
                suffix = ''
            elif key in row_a.keys() and key + self._a_suffix not in merged.keys():
                suffix = self._a_suffix
            else:
                suffix = self._b_suffix
            merged[key + suffix] = value
        return merged

    def common_join(self, rows_a: TRowsIterable, rows_b: TRowsIterable, keys: tp.Sequence[str]) -> TRowsGenerator:
        """ Auxiliary method for any type of join. Generally implements InnerJoin.

        @param rows_a: the first data generator
        @param rows_b: the second data generator
        @param keys: keys for join operation
        @return: result generator
        """
        for b in rows_b:
            for a in rows_a:
                yield self.merge_rows(a, b, keys)

    @abstractmethod
    def __call__(self, keys: tp.Sequence[str], rows_a: TRowsIterable, rows_b: TRowsIterable) -> TRowsGenerator:
        """
        :param keys: join keys
        :param rows_a: left table rows
        :param rows_b: right table rows
        """
        pass


class Join(Operation):
    def __init__(self, joiner: Joiner, keys: tp.Sequence[str]):
        self.keys = keys
        self.joiner = joiner

    def __call__(self, rows: tp.Any, *args: tp.Any, **kwargs: tp.Any) -> TRowsGenerator:
        """Groups data for joining them

        @param rows: left table
        @param args: there lies right table
        @return: generator result of join
        """
        left_grouper = itertools.groupby(rows, key=lambda x: {key: x[key] for key in self.keys})  # упражнение читателю
        right_grouper = itertools.groupby(args[0], key=lambda x: {key: x[key] for key in self.keys})
        left_key, left_group = next(left_grouper, (None, None))
        right_key, right_group = next(right_grouper, (None, None))

        while left_key is not None and right_key is not None:
            left_values, right_values = list(left_key.values()), list(right_key.values())
            if left_values < right_values:
                yield from self.joiner(self.keys, left_group or [], [])
                left_key, left_group = next(left_grouper, (None, None))
                continue
            if left_values == right_values:
                yield from self.joiner(self.keys, left_group or [], right_group or [])
                left_key, left_group = next(left_grouper, (None, None))
                right_key, right_group = next(right_grouper, (None, None))
                continue
            if left_values > right_values:
                yield from self.joiner(self.keys, [], right_group or [])
                right_key, right_group = next(right_grouper, (None, None))

        while left_key is not None:
            yield from self.joiner(self.keys, left_group or [], [])
            left_key, left_group = next(left_grouper, (None, None))

        while right_key is not None:
            yield from self.joiner(self.keys, [], right_group or [])
            right_key, right_group = next(right_grouper, (None, None))

# Dummy operators


class DummyMapper(Mapper):
    """Yield exactly the row passed"""

    def __call__(self, row: TRow) -> TRowsGenerator:
        yield row


class AddField(Mapper):
    """add useless field column"""

    def __init__(self, column: str, def_value: tp.Any = None):
        """
        :param column: name of column to process
        """
        self.column = column
        self.def_value = def_value

    def __call__(self, row: TRow) -> TRowsGenerator:
        row[self.column] = self.def_value
        yield row


class RemoveField(Mapper):
    """add useless field column"""

    def __init__(self, column: str):
        """
        :param column: name of column to process
        """
        self.column = column

    def __call__(self, row: TRow) -> TRowsGenerator:
        row.pop(self.column)
        yield row


class FirstReducer(Reducer):
    """Yield only first row from passed ones"""

    def __call__(self, rows: TRowsIterable) -> TRowsGenerator:
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
        self.regexp = re.compile(r'([^\w\s]|_)+')

    def __call__(self, row: TRow) -> TRowsGenerator:
        row[self.column] = re.sub(self.regexp, '',  row[self.column])
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
        row[self.column] = LowerCase._lower_case(row[self.column])
        yield row


class Split(Mapper):
    """Split row on multiple rows by separator"""

    def __init__(self, column: str, separator: tp.Optional[str] = None) -> None:
        """
        :param column: name of column to split
        :param separator: string to separate by
        """
        self.column = column
        self.separator = separator

    def __call__(self, row: TRow) -> TRowsGenerator:
        for value in row[self.column].split(sep=self.separator):
            tmp_row: TRow = row.copy()
            tmp_row[self.column] = value
            yield tmp_row


class Product(Mapper):
    """Calculates product of multiple columns"""

    def __init__(self, columns: tp.Sequence[str], result_column: str = 'product') -> None:
        """
        :param columns: column names to product
        :param result_column: column name to save product in
        """
        self.columns = columns
        self.result_column = result_column

    def __call__(self, row: TRow) -> TRowsGenerator:
        prod = 1
        for column in self.columns:
            prod *= row[column]
        row[self.result_column] = prod
        yield row


class Filter(Mapper):
    """Remove records that don't satisfy some condition"""

    def __init__(self, condition: tp.Callable[[TRow], bool]) -> None:
        """
        :param condition: if condition is not true - remove record
        """
        self.condition = condition

    def __call__(self, row: TRow) -> TRowsGenerator:
        if self.condition(row):
            yield row


class InverseDocumentFrequency(Mapper):
    """Calculate IDF"""
    def __init__(self, row_count_colunm: str, docs_per_word_column: str, idf_column: str = "idf") -> None:
        """
        @param row_count_colunm: имя столбца содержащего общее количество слов
        @param docs_per_word_column: имя столбца содержащего количество документов, в которых встречается слово
        @param idf_column: имя столбца, в который записывается результат
        """
        self.row_count_column = row_count_colunm
        self.docs_per_word_column = docs_per_word_column
        self.idf_column = idf_column

    def __call__(self, row: TRow) -> TRowsGenerator:
        row[self.idf_column] = math.log(row[self.row_count_column] / row[self.docs_per_word_column])
        yield row


class TFIDF(Mapper):
    """Calculate IDF"""

    def __init__(self, tf_column: str, idf_column: str, tfidf_column: str = "tfidf") -> None:
        """TFIDF(word_i, doc_i) = (frequency of word_i in doc_i) *
        log((total number of docs) / (docs where word_i is present))

        @param tf_column: имя столбца содержащего tf
        @param idf_column: имя столбца содержащего idf
        @param tfidf_column: имя столбца, в который записывается результат
        """
        self.tf_column = tf_column
        self.idf_column = idf_column
        self.tfidf_column = tfidf_column

    def __call__(self, row: TRow) -> TRowsGenerator:
        row[self.tfidf_column] = row[self.tf_column] * row[self.idf_column]
        yield row


class PMI(Mapper):
    """Calculate PMI"""
    def __init__(self, tf_column: str, cf_column: str, pmi_column: str = "pmi") -> None:
        """pmi(word_i, doc_i) = log((frequency of word_i in doc_i) / (frequency of word_i in all documents combined))

        @param tf_column: frequency of word_i in doc_i
        @param cf_column: frequency of word_i in all documents combined
        @param pmi_column: result column
        """
        self.tf_column = tf_column
        self.cf_column = cf_column
        self.pmi_column = pmi_column

    def __call__(self, row: TRow) -> TRowsGenerator:
        row[self.pmi_column] = math.log(row[self.tf_column] / row[self.cf_column])
        yield row


class Project(Mapper):
    """Leave only mentioned columns"""

    def __init__(self, columns: tp.Sequence[str]) -> None:
        """
        :param columns: names of columns
        """
        self.columns = columns

    def __call__(self, row: TRow) -> TRowsGenerator:
        yield {column: row[column] for column in self.columns}


class StreetLength(Mapper):
    """Compute streets lenght by coordinates"""

    def __init__(self, start_column: str, end_column: str, result_column: str = "length") -> None:
        """
        @param start_column: название столбца с координатами начала улицы
        @param end_column: название столбца с координатами конца улицы
        @param result_column: название слолбца в который записывается результат
        """
        self.start_column = start_column
        self.end_column = end_column
        self.result_column = result_column

    def haversine(self, *args: tp.SupportsFloat) -> float:
        """
        Calculate the great circle distance between two points
        on the earth (specified in decimal degrees)
        """
        # convert decimal degrees to radians
        lon1, lat1, lon2, lat2 = map(radians, args)

        # haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        c = 2 * asin(sqrt(a))
        r = 6371  # Radius of earth in kilometers. Use 3956 for miles
        return c * r

    def __call__(self, row: TRow) -> TRowsGenerator:
        row[self.result_column] = self.haversine(*row[self.start_column], *row[self.end_column])
        yield row


class ProcessDate(Mapper):
    """Excract from 2 string in datetime format duration of interval, weekday and hour of start."""

    def __init__(self, enter_time_column: str, leave_time_column: str,
                 week_day_column: str = "week_day", hour_column: str = "hour",
                 duration_column: str = "duration") -> None:
        """

        @param enter_time_column: название столбца со временем начала
        @param leave_time_column: название столбца со временем конца
        @param week_day_column: название столбца, в который запишется день недели
        @param hour_column: название столбца, в который запишется час
        @param duration_column: название столбца, в который запишется длительность интервала
        """
        self.enter_time_column = enter_time_column
        self.leave_time_column = leave_time_column
        self.week_day_column = week_day_column
        self.hour_column = hour_column
        self.duration_column = duration_column

    def __call__(self, row: TRow) -> TRowsGenerator:

        start_date = parser.parse(row[self.enter_time_column])
        end_date = parser.parse(row[self.leave_time_column])
        row[self.duration_column] = (end_date - start_date).total_seconds() / 3600
        row[self.week_day_column] = start_date.strftime('%a')
        row[self.hour_column] = start_date.hour
        yield row


class ReadFromFile(Mapper):
    """Read from filename line-by-line and process every string using parser"""

    def __init__(self, parser: tp.Callable[[str], TRow]) -> None:
        """
        @param parser: функция преобразующая строку в TRow
        """
        self.parser = parser

    def __call__(self, row: str) -> TRowsGenerator:
        yield self.parser(row)


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

    def __call__(self, rows: TRowsIterable) -> TRowsGenerator:
        yield from heapq.nlargest(self.n, rows, key=itemgetter(self.column_max))


class TermFrequency(Reducer):
    """Calculate frequency of values in column"""

    def __init__(self, words_column: str, result_column: str = 'tf', by_field: tp.Union[str, None] = None) -> None:
        """
        :param words_column: name for column with words
        :param result_column: name for result column
        """
        self.words_column = words_column
        self.result_column = result_column
        self.by_field = by_field

    def __call__(self, rows: TRowsIterable) -> TRowsGenerator:
        dictionary: tp.DefaultDict[str, int] = defaultdict(int)
        cumsum = 0
        if self.by_field is None:
            for row in rows:
                dictionary[row[self.words_column]] += 1
                cumsum += 1
        else:
            for row in rows:
                dictionary[row[self.words_column]] += row[self.by_field]
                cumsum += row[self.by_field]

        for key, value in dictionary.items():
            yield {self.words_column: key, self.result_column: value / cumsum}


class Count(Reducer):
    """Count rows passed and yield single row as a result"""

    def __init__(self, column: str) -> None:
        """
        :param column: name of column to count
        """
        self.column = column

    def __call__(self, rows: TRowsIterable) -> TRowsGenerator:
        cumsum = 0
        for _ in rows:
            cumsum += 1
        yield {self.column: cumsum}


class MeanSpeed(Reducer):
    """Compute mean speed at concrete data"""

    def __init__(self, duration_column: str, length_column: str, result_column: str, count_column: str) -> None:
        """
        @param duration_column: название столбца c длительностью интервала
        @param length_column: название столбца с длинной улицы
        @param result_column: название столбца, в который запишется результат
        @param count_column: название столбца c количеством таких поездок
        """
        self.duration_column = duration_column
        self.length_column = length_column
        self.result_column = result_column
        self.count_column = count_column

    def __call__(self, rows: TRowsIterable) -> TRowsGenerator:
        total_time = 0
        total_length = 0
        for row in rows:
            total_time += row[self.duration_column] * row[self.count_column]
            total_length += row[self.length_column] * row[self.count_column]
        yield {self.result_column: total_length / total_time}


class Sum(Reducer):
    """Sum values in column passed and yield single row as a result"""

    def __init__(self, column: str, delete_others: bool = True) -> None:
        """
        :param column: name of column to sum
        """
        self.column = column
        self.delete_others = delete_others

    def __call__(self, rows: TRowsIterable) -> TRowsGenerator:
        cumsum = 0
        for row in rows:
            cumsum += row[self.column]
        yield {self.column: cumsum}

# Joiners


class InnerJoiner(Joiner):
    """Join with inner strategy"""

    def __call__(self, keys: tp.Sequence[str], rows_a: TRowsIterable, rows_b: TRowsIterable) -> TRowsGenerator:
        yield from self.common_join(rows_a, rows_b, keys)


class OuterJoiner(Joiner):
    """Join with outer strategy"""

    def __call__(self, keys: tp.Sequence[str], rows_a: TRowsIterable, rows_b: TRowsIterable) -> TRowsGenerator:
        if not rows_a:
            yield from rows_b
        elif not rows_b:
            yield from rows_a
        else:
            yield from self.common_join(rows_a, rows_b, keys)


class LeftJoiner(Joiner):
    """Join with left strategy"""

    def __call__(self, keys: tp.Sequence[str], rows_a: TRowsIterable, rows_b: TRowsIterable) -> TRowsGenerator:
        if not rows_b:
            yield from rows_a
        else:
            yield from self.common_join(rows_a, rows_b, keys)


class RightJoiner(Joiner):
    """Join with right strategy"""

    def __call__(self, keys: tp.Sequence[str], rows_a: TRowsIterable, rows_b: TRowsIterable) -> TRowsGenerator:
        if not rows_a:
            yield from rows_b
        else:
            yield from self.common_join(rows_a, rows_b, keys)
