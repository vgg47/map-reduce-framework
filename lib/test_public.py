import typing as tp

from compgraph import graphs
from compgraph.lib import memory_watchdog
from operator import itemgetter
from pytest import approx
from . import operations as ops


def test_dummy_map() -> None:
    tests: ops.TRowsIterable = [
        {'test_id': 1, 'text': 'one two three'},
        {'test_id': 2, 'text': 'testing out stuff'}
    ]

    result = ops.Map(ops.DummyMapper())

    assert tests == list(result(tests))


def test_add_field_map() -> None:
    tests: ops.TRowsIterable = [
        {'test_id': 1, 'text': 'one two three'},
        {'test_id': 2, 'text': 'testing out stuff'}
    ]

    etalon: ops.TRowsIterable = [
        {'test_id': 1, 'text': 'one two three', "fake": 1},
        {'test_id': 2, 'text': 'testing out stuff', "fake": 1}
    ]

    result = ops.Map(ops.AddField("fake", 1))

    assert etalon == list(result(tests))


def test_remove_field_map() -> None:
    tests: ops.TRowsIterable = [
        {'test_id': 1, 'text': 'one two three', "fake": 1},
        {'test_id': 2, 'text': 'testing out stuff', "fake": 1}
    ]

    etalon: ops.TRowsIterable = [
        {'test_id': 1, 'text': 'one two three'},
        {'test_id': 2, 'text': 'testing out stuff'}
    ]

    result = ops.Map(ops.RemoveField("fake"))

    assert etalon == list(result(tests))

def test_inverse_document_frequency_map() -> None:
    rows: ops.TRowsIterable = [
        {'text': 'hello', 'doc_count': 4, 'row_count': 6},
        {'text': 'little', 'doc_count': 4, 'row_count': 6},
        {'text': 'world', 'doc_count': 4, 'row_count': 6},

    ]

    etalon: ops.TRowsIterable = [
        {'text': 'hello', 'doc_count': 4, 'row_count': 6, 'idf': approx(0.4054, abs=0.001)},
        {'text': 'little', 'doc_count': 4, 'row_count': 6, 'idf': approx(0.4054, abs=0.001)},
        {'text': 'world', 'doc_count': 4, 'row_count': 6, 'idf': approx(0.4054, abs=0.001)},
    ]

    result = ops.Map(ops.InverseDocumentFrequency("row_count", "doc_count", "idf"))

    assert etalon == list(result(rows))

def test_tfidf_map() -> None:
    rows: ops.TRowsIterable = [
        {'doc_id': 1, 'text': 'hello', 'tf': 0.3333333333333333, 'idf': 0.4054651081081644},
        {'doc_id': 4, 'text': 'hello', 'tf': 0.25, 'idf': 0.4054651081081644},
        {'doc_id': 5, 'text': 'hello', 'tf': 0.6666666666666666, 'idf': 0.4054651081081644},
        {'doc_id': 6, 'text': 'hello', 'tf': 0.2, 'idf': 0.4054651081081644},
        {'doc_id': 1, 'text': 'little', 'tf': 0.3333333333333333, 'idf': 0.4054651081081644},
        {'doc_id': 2, 'text': 'little', 'tf': 1.0, 'idf': 0.4054651081081644},
        {'doc_id': 3, 'text': 'little', 'tf': 1.0, 'idf': 0.4054651081081644},
        {'doc_id': 4, 'text': 'little', 'tf': 0.5, 'idf': 0.4054651081081644},
        {'doc_id': 1, 'text': 'world', 'tf': 0.3333333333333333, 'idf': 0.4054651081081644},
        {'doc_id': 4, 'text': 'world', 'tf': 0.25, 'idf': 0.4054651081081644},
        {'doc_id': 5, 'text': 'world', 'tf': 0.3333333333333333, 'idf': 0.4054651081081644},
        {'doc_id': 6, 'text': 'world', 'tf': 0.8, 'idf': 0.4054651081081644}
    ]

    etalon: ops.TRowsIterable = [
        {'doc_id': 1, 'text': 'hello', 'tf': 0.3333333333333333, 'idf': 0.4054651081081644,
         'tf_idf': approx(0.1351, abs=0.001)},
        {'doc_id': 4, 'text': 'hello', 'tf': 0.25, 'idf': 0.4054651081081644, 'tf_idf': approx(0.1013, abs=0.001)},
        {'doc_id': 5, 'text': 'hello', 'tf': 0.6666666666666666, 'idf': 0.4054651081081644,
         'tf_idf': approx(0.2703, abs=0.001)},
        {'doc_id': 6, 'text': 'hello', 'tf': 0.2, 'idf': 0.4054651081081644, 'tf_idf': approx(0.0810, abs=0.001)},
        {'doc_id': 1, 'text': 'little', 'tf': 0.3333333333333333, 'idf': 0.4054651081081644,
         'tf_idf': approx(0.1351, abs=0.001)},
        {'doc_id': 2, 'text': 'little', 'tf': 1.0, 'idf': 0.4054651081081644, 'tf_idf': approx(0.4054, abs=0.001)},
        {'doc_id': 3, 'text': 'little', 'tf': 1.0, 'idf': 0.4054651081081644, 'tf_idf': approx(0.4054, abs=0.001)},
        {'doc_id': 4, 'text': 'little', 'tf': 0.5, 'idf': 0.4054651081081644, 'tf_idf': approx(0.2027, abs=0.001)},
        {'doc_id': 1, 'text': 'world', 'tf': 0.3333333333333333, 'idf': 0.4054651081081644,
         'tf_idf': approx(0.1351, abs=0.001)},
        {'doc_id': 4, 'text': 'world', 'tf': 0.25, 'idf': 0.4054651081081644, 'tf_idf': approx(0.1013, abs=0.001)},
        {'doc_id': 5, 'text': 'world', 'tf': 0.3333333333333333, 'idf': 0.4054651081081644,
         'tf_idf': approx(0.1351, abs=0.001)},
        {'doc_id': 6, 'text': 'world', 'tf': 0.8, 'idf': 0.4054651081081644, 'tf_idf': approx(0.3243, abs=0.001)}
    ]

    result = ops.Map(ops.TFIDF("tf", "idf","tf_idf"))

    assert etalon == list(result(rows))


def test_pmi_map() -> None:
    rows: ops.TRowsIterable = [
        {'doc_id': 5, 'text': 'hello', 'tf': 1.0, 'atf': 0.3076923076923077},
        {'doc_id': 6, 'text': 'hello', 'tf': 0.3333333333333333, 'atf': 0.3076923076923077},
        {'doc_id': 3, 'text': 'little', 'tf': 1.0, 'atf': 0.38461538461538464},
        {'doc_id': 4, 'text': 'little', 'tf': 1.0, 'atf': 0.38461538461538464},
        {'doc_id': 6, 'text': 'world', 'tf': 0.6666666666666666, 'atf': 0.3076923076923077}
    ]

    etalon: ops.TRowsIterable = [
        {'doc_id': 5, 'text': 'hello', 'tf': 1.0, 'atf': 0.3076923076923077, 'pmi': approx(1.1786, abs=0.0001)},
        {'doc_id': 6, 'text': 'hello', 'tf': 0.3333333333333333, 'atf': 0.3076923076923077,
         'pmi': approx(0.0800, abs=0.0001)},
        {'doc_id': 3, 'text': 'little', 'tf': 1.0, 'atf': 0.38461538461538464, 'pmi': approx(0.9555, abs=0.0001)},
        {'doc_id': 4, 'text': 'little', 'tf': 1.0, 'atf': 0.38461538461538464, 'pmi': approx(0.9555, abs=0.0001)},
        {'doc_id': 6, 'text': 'world', 'tf': 0.6666666666666666, 'atf': 0.3076923076923077,
         'pmi': approx(0.7731, abs=0.0001)}
    ]

    result = ops.Map(ops.PMI("tf", "atf", "pmi"))

    assert etalon == list(result(rows))


def test_street_length_map() -> None:
    lengths: ops.TRowsIterable = [
        {'start': [37.84870228730142, 55.73853974696249], 'end': [37.8490418381989, 55.73832445777953],
         'edge_id': 8414926848168493057},
        {'start': [37.524768467992544, 55.88785375468433], 'end': [37.52415172755718, 55.88807155843824],
         'edge_id': 5342768494149337085},
        {'start': [37.56963176652789, 55.846845586784184], 'end': [37.57018438540399, 55.8469259692356],
         'edge_id': 5123042926973124604},
        {'start': [37.41463478654623, 55.654487907886505], 'end': [37.41442892700434, 55.654839486815035],
         'edge_id': 5726148664276615162},
        {'start': [37.584684155881405, 55.78285809606314], 'end': [37.58415022864938, 55.78177368734032],
         'edge_id': 451916977441439743},
        {'start': [37.736429711803794, 55.62696328852326], 'end': [37.736344216391444, 55.626937723718584],
         'edge_id': 7639557040160407543},
        {'start': [37.83196756616235, 55.76662947423756], 'end': [37.83191015012562, 55.766647034324706],
         'edge_id': 1293255682152955894},
    ]

    etalon: ops.TRowsIterable = [
        {'start': [37.84870228730142, 55.73853974696249], 'end': [37.8490418381989, 55.73832445777953],
         'edge_id': 8414926848168493057, 'length': approx(0.0320, abs=0.0001)},
        {'start': [37.524768467992544, 55.88785375468433], 'end': [37.52415172755718, 55.88807155843824],
         'edge_id': 5342768494149337085, 'length': approx(0.0454, abs=0.0001)},
        {'start': [37.56963176652789, 55.846845586784184], 'end': [37.57018438540399, 55.8469259692356],
         'edge_id': 5123042926973124604, 'length': approx(0.0356, abs=0.0001)},
        {'start': [37.41463478654623, 55.654487907886505], 'end': [37.41442892700434, 55.654839486815035],
         'edge_id': 5726148664276615162, 'length': approx(0.0411, abs=0.0001)},
        {'start': [37.584684155881405, 55.78285809606314], 'end': [37.58415022864938, 55.78177368734032],
         'edge_id': 451916977441439743, 'length': approx(0.1251, abs=0.0001)},
        {'start': [37.736429711803794, 55.62696328852326], 'end': [37.736344216391444, 55.626937723718584],
         'edge_id': 7639557040160407543, 'length': approx(0.0060, abs=0.0001)},
        {'start': [37.83196756616235, 55.76662947423756], 'end': [37.83191015012562, 55.766647034324706],
         'edge_id': 1293255682152955894, 'length': approx(0.0040, abs=0.0001)}
    ]

    result = ops.Map(ops.StreetLength("start", "end", "length"))

    assert etalon == list(result(lengths))


def test_process_date_map() -> None:
    rows: ops.TRowsIterable = [
        {'leave_time': '20171020T112238.723000', 'enter_time': '20171020T112237.427000',
         'edge_id': 8414926848168493057},
        {'leave_time': '20171011T145553.040000', 'enter_time': '20171011T145551.957000',
         'edge_id': 8414926848168493057},
        {'leave_time': '20171020T090548.939000', 'enter_time': '20171020T090547.463000',
         'edge_id': 8414926848168493057},
        {'leave_time': '20171024T144101.879000', 'enter_time': '20171024T144059.102000',
         'edge_id': 8414926848168493057},
        {'leave_time': '20171022T131828.330000', 'enter_time': '20171022T131820.842000',
         'edge_id': 5342768494149337085},
        {'leave_time': '20171014T134826.836000', 'enter_time': '20171014T134825.215000',
         'edge_id': 5342768494149337085},
        {'leave_time': '20171010T060609.897000', 'enter_time': '20171010T060608.344000',
         'edge_id': 5342768494149337085},
        {'leave_time': '20171027T082600.201000', 'enter_time': '20171027T082557.571000',
         'edge_id': 5342768494149337085}
    ]

    etalon: ops.TRowsIterable = [
        {'leave_time': '20171020T112238.723000', 'enter_time': '20171020T112237.427000', 'edge_id': 8414926848168493057,
         'duration': 0.00036, 'weekday': 'Fri', 'hour': 11},
        {'leave_time': '20171011T145553.040000', 'enter_time': '20171011T145551.957000', 'edge_id': 8414926848168493057,
         'duration': 0.00030083333333333335, 'weekday': 'Wed', 'hour': 14},
        {'leave_time': '20171020T090548.939000', 'enter_time': '20171020T090547.463000', 'edge_id': 8414926848168493057,
         'duration': 0.00041, 'weekday': 'Fri', 'hour': 9},
        {'leave_time': '20171024T144101.879000', 'enter_time': '20171024T144059.102000', 'edge_id': 8414926848168493057,
         'duration': 0.000771388888888889, 'weekday': 'Tue', 'hour': 14},
        {'leave_time': '20171022T131828.330000', 'enter_time': '20171022T131820.842000', 'edge_id': 5342768494149337085,
         'duration': 0.0020800000000000003, 'weekday': 'Sun', 'hour': 13},
        {'leave_time': '20171014T134826.836000', 'enter_time': '20171014T134825.215000', 'edge_id': 5342768494149337085,
         'duration': 0.0004502777777777778, 'weekday': 'Sat', 'hour': 13},
        {'leave_time': '20171010T060609.897000', 'enter_time': '20171010T060608.344000', 'edge_id': 5342768494149337085,
         'duration': 0.00043138888888888887, 'weekday': 'Tue', 'hour': 6},
        {'leave_time': '20171027T082600.201000', 'enter_time': '20171027T082557.571000', 'edge_id': 5342768494149337085,
         'duration': 0.0007305555555555555, 'weekday': 'Fri', 'hour': 8}
    ]

    result = ops.Map(ops.ProcessDate("enter_time", "leave_time", "weekday", "hour", "duration"))

    assert etalon == list(result(rows))

def test_lower_case() -> None:
    tests: ops.TRowsIterable = [
        {'test_id': 1, 'text': 'camelCaseTest'},
        {'test_id': 2, 'text': 'UPPER_CASE_TEST'},
        {'test_id': 3, 'text': 'wEiRdTeSt'}
    ]

    etalon: ops.TRowsIterable = [
        {'test_id': 1, 'text': 'camelcasetest'},
        {'test_id': 2, 'text': 'upper_case_test'},
        {'test_id': 3, 'text': 'weirdtest'}
    ]

    result = ops.Map(ops.LowerCase(column='text'))(tests)

    assert etalon == list(result)


def test_filtering_punctuation() -> None:
    tests: ops.TRowsIterable = [
        {'test_id': 1, 'text': 'Hello, world!'},
        {'test_id': 2, 'text': 'Test. with. a. lot. of. dots.'},
        {'test_id': 3, 'text': r'!"#$%&\'()*+,-./:;<=>?@[\]^_`{|}~'}
    ]

    etalon: ops.TRowsIterable = [
        {'test_id': 1, 'text': 'Hello world'},
        {'test_id': 2, 'text': 'Test with a lot of dots'},
        {'test_id': 3, 'text': ''}
    ]

    result = ops.Map(ops.FilterPunctuation(column='text'))(tests)

    assert etalon == list(result)


def test_splitting() -> None:
    tests: ops.TRowsIterable = [
        {'test_id': 1, 'text': 'one two three'},
        {'test_id': 2, 'text': 'tab\tsplitting\ttest'},
        {'test_id': 3, 'text': 'more\nlines\ntest'},
        {'test_id': 4, 'text': 'tricky\u00A0test'}
    ]

    etalon: ops.TRowsIterable = [
        {'test_id': 1, 'text': 'one'},
        {'test_id': 1, 'text': 'three'},
        {'test_id': 1, 'text': 'two'},

        {'test_id': 2, 'text': 'splitting'},
        {'test_id': 2, 'text': 'tab'},
        {'test_id': 2, 'text': 'test'},

        {'test_id': 3, 'text': 'lines'},
        {'test_id': 3, 'text': 'more'},
        {'test_id': 3, 'text': 'test'},

        {'test_id': 4, 'text': 'test'},
        {'test_id': 4, 'text': 'tricky'}
    ]

    result = ops.Map(ops.Split(column='text'))(tests)

    assert etalon == sorted(result, key=itemgetter('test_id', 'text'))


def test_product() -> None:
    tests: ops.TRowsIterable = [
        {'test_id': 1, 'speed': 5, 'distance': 10},
        {'test_id': 2, 'speed': 60, 'distance': 2},
        {'test_id': 3, 'speed': 3, 'distance': 15},
        {'test_id': 4, 'speed': 100, 'distance': 0.5},
        {'test_id': 5, 'speed': 48, 'distance': 15},
    ]

    etalon: ops.TRowsIterable = [
        {'test_id': 1, 'speed': 5, 'distance': 10, 'time': 50},
        {'test_id': 2, 'speed': 60, 'distance': 2, 'time': 120},
        {'test_id': 3, 'speed': 3, 'distance': 15, 'time': 45},
        {'test_id': 4, 'speed': 100, 'distance': 0.5, 'time': 50},
        {'test_id': 5, 'speed': 48, 'distance': 15, 'time': 720},
    ]

    result = ops.Map(ops.Product(columns=['speed', 'distance'], result_column='time'))(tests)

    assert etalon == list(result)


def test_filter() -> None:
    tests: ops.TRowsIterable = [
        {'test_id': 1, 'f': 0, 'g': 0},
        {'test_id': 2, 'f': 0, 'g': 1},
        {'test_id': 3, 'f': 1, 'g': 0},
        {'test_id': 4, 'f': 1, 'g': 1}
    ]

    etalon: ops.TRowsIterable = [
        {'test_id': 2, 'f': 0, 'g': 1},
        {'test_id': 3, 'f': 1, 'g': 0}
    ]

    def xor(row: ops.TRow) -> bool:
        return row['f'] ^ row['g']

    result = ops.Map(ops.Filter(condition=xor))(tests)

    assert etalon == list(result)


def test_projection() -> None:
    tests: ops.TRowsIterable = [
        {'test_id': 1, 'junk': 'x', 'value': 42},
        {'test_id': 2, 'junk': 'y', 'value': 1},
        {'test_id': 3, 'junk': 'z', 'value': 144}
    ]

    etalon: ops.TRowsIterable = [
        {'value': 42},
        {'value': 1},
        {'value': 144}
    ]

    result = ops.Map(ops.Project(columns=['value']))(tests)

    assert etalon == list(result)


def test_dummy_reduce() -> None:
    tests: ops.TRowsIterable = [
        {'test_id': 1, 'text': 'hello, world'},
        {'test_id': 2, 'text': 'bye!'}
    ]

    result = ops.Reduce(ops.FirstReducer(), keys=['test_id'])(tests)
    assert tests == list(result)


def test_top_n() -> None:
    matches: ops.TRowsIterable = [
        {'match_id': 1, 'player_id': 1, 'rank': 42},
        {'match_id': 1, 'player_id': 2, 'rank': 7},
        {'match_id': 1, 'player_id': 3, 'rank': 0},
        {'match_id': 1, 'player_id': 4, 'rank': 39},

        {'match_id': 2, 'player_id': 5, 'rank': 15},
        {'match_id': 2, 'player_id': 6, 'rank': 39},
        {'match_id': 2, 'player_id': 7, 'rank': 27},
        {'match_id': 2, 'player_id': 8, 'rank': 7}
    ]

    etalon: ops.TRowsIterable = [
        {'match_id': 1, 'player_id': 1, 'rank': 42},
        {'match_id': 1, 'player_id': 2, 'rank': 7},
        {'match_id': 1, 'player_id': 4, 'rank': 39},

        {'match_id': 2, 'player_id': 5, 'rank': 15},
        {'match_id': 2, 'player_id': 6, 'rank': 39},
        {'match_id': 2, 'player_id': 7, 'rank': 27}
    ]

    presorted_matches = sorted(matches, key=itemgetter('match_id'))  # !!!
    result = ops.Reduce(ops.TopN(column='rank', n=3), keys=['match_id'])(presorted_matches)

    assert etalon == sorted(result, key=itemgetter('match_id', 'player_id'))


def test_term_frequency() -> None:
    docs: ops.TRowsIterable = [
        {'doc_id': 1, 'text': 'hello', 'count': 1},
        {'doc_id': 1, 'text': 'little', 'count': 1},
        {'doc_id': 1, 'text': 'world', 'count': 1},

        {'doc_id': 2, 'text': 'little', 'count': 1},

        {'doc_id': 3, 'text': 'little', 'count': 3},
        {'doc_id': 3, 'text': 'little', 'count': 3},
        {'doc_id': 3, 'text': 'little', 'count': 3},

        {'doc_id': 4, 'text': 'little', 'count': 2},
        {'doc_id': 4, 'text': 'hello', 'count': 1},
        {'doc_id': 4, 'text': 'little', 'count': 2},
        {'doc_id': 4, 'text': 'world', 'count': 1},

        {'doc_id': 5, 'text': 'hello', 'count': 2},
        {'doc_id': 5, 'text': 'hello', 'count': 2},
        {'doc_id': 5, 'text': 'world', 'count': 1},

        {'doc_id': 6, 'text': 'world', 'count': 4},
        {'doc_id': 6, 'text': 'world', 'count': 4},
        {'doc_id': 6, 'text': 'world', 'count': 4},
        {'doc_id': 6, 'text': 'world', 'count': 4},
        {'doc_id': 6, 'text': 'hello', 'count': 1}
    ]

    etalon: ops.TRowsIterable = [
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
    ]

    presorted_docs = sorted(docs, key=itemgetter('doc_id'))  # !!!
    result = ops.Reduce(ops.TermFrequency(words_column='text'), keys=['doc_id'])(presorted_docs)
    # print(list(result))
    assert etalon == sorted(result, key=itemgetter('doc_id', 'text'))


def test_counting() -> None:
    sentences: ops.TRowsIterable = [
        {'sentence_id': 1, 'word': 'hello'},
        {'sentence_id': 1, 'word': 'my'},
        {'sentence_id': 1, 'word': 'little'},
        {'sentence_id': 1, 'word': 'world'},

        {'sentence_id': 2, 'word': 'hello'},
        {'sentence_id': 2, 'word': 'my'},
        {'sentence_id': 2, 'word': 'little'},
        {'sentence_id': 2, 'word': 'little'},
        {'sentence_id': 2, 'word': 'hell'}
    ]

    etalon: ops.TRowsIterable = [
        {'count': 1, 'word': 'hell'},
        {'count': 1, 'word': 'world'},
        {'count': 2, 'word': 'hello'},
        {'count': 2, 'word': 'my'},
        {'count': 3, 'word': 'little'}
    ]

    presorted_words = sorted(sentences, key=itemgetter('word'))  # !!!
    result = ops.Reduce(ops.Count(column='count'), keys=['word'])(presorted_words)

    assert etalon == sorted(result, key=itemgetter('count', 'word'))


def test_sum() -> None:
    matches: ops.TRowsIterable = [
        {'match_id': 1, 'player_id': 1, 'score': 42},
        {'match_id': 1, 'player_id': 2, 'score': 7},
        {'match_id': 1, 'player_id': 3, 'score': 0},
        {'match_id': 1, 'player_id': 4, 'score': 39},

        {'match_id': 2, 'player_id': 5, 'score': 15},
        {'match_id': 2, 'player_id': 6, 'score': 39},
        {'match_id': 2, 'player_id': 7, 'score': 27},
        {'match_id': 2, 'player_id': 8, 'score': 7}
    ]

    etalon: ops.TRowsIterable = [
        {'match_id': 1, 'score': 88},
        {'match_id': 2, 'score': 88}
    ]

    presorted_matches = sorted(matches, key=itemgetter('match_id'))  # !!!
    result = ops.Reduce(ops.Sum(column='score'), keys=['match_id'])(presorted_matches)

    assert etalon == sorted(result, key=itemgetter('match_id'))


def test_simple_join() -> None:
    players: ops.TRowsIterable = [
        {'player_id': 1, 'username': 'XeroX'},
        {'player_id': 2, 'username': 'jay'},
        {'player_id': 3, 'username': 'Destroyer'},
    ]

    games: ops.TRowsIterable = [
        {'game_id': 1, 'player_id': 3, 'score': 99},
        {'game_id': 2, 'player_id': 1, 'score': 17},
        {'game_id': 3, 'player_id': 1, 'score': 22}
    ]

    etalon: ops.TRowsIterable = [
        {'game_id': 1, 'player_id': 3, 'score': 99, 'username': 'Destroyer'},
        {'game_id': 2, 'player_id': 1, 'score': 17, 'username': 'XeroX'},
        {'game_id': 3, 'player_id': 1, 'score': 22, 'username': 'XeroX'}
    ]

    presorted_games = iter(sorted(games, key=itemgetter('player_id')))    # !!!
    presorted_players = iter(sorted(players, key=itemgetter('player_id')))  # !!!
    result = ops.Join(ops.InnerJoiner(), keys=['player_id'])(presorted_games, presorted_players)

    assert etalon == sorted(result, key=itemgetter('game_id'))

# test_simple_join()
def test_inner_join() -> None:
    players: ops.TRowsIterable = [
        {'player_id': 0, 'username': 'root'},
        {'player_id': 1, 'username': 'XeroX'},
        {'player_id': 2, 'username': 'jay'}
    ]

    games: ops.TRowsIterable = [
        {'game_id': 1, 'player_id': 3, 'score': 9999999},
        {'game_id': 2, 'player_id': 1, 'score': 17},
        {'game_id': 3, 'player_id': 2, 'score': 22}
    ]

    etalon: ops.TRowsIterable = [
        # player 3 is unknown
        # no games for player 0
        {'game_id': 2, 'player_id': 1, 'score': 17, 'username': 'XeroX'},
        {'game_id': 3, 'player_id': 2, 'score': 22, 'username': 'jay'}
    ]

    presorted_games = iter(sorted(games, key=itemgetter('player_id')))    # !!!
    presorted_players = iter(sorted(players, key=itemgetter('player_id')))  # !!!
    result = ops.Join(ops.InnerJoiner(), keys=['player_id'])(presorted_games, presorted_players)

    assert etalon == sorted(result, key=itemgetter('game_id'))


def test_outer_join() -> None:
    players: ops.TRowsIterable = [
        {'player_id': 0, 'username': 'root'},
        {'player_id': 1, 'username': 'XeroX'},
        {'player_id': 2, 'username': 'jay'}
    ]

    games: ops.TRowsIterable = [
        {'game_id': 1, 'player_id': 3, 'score': 9999999},
        {'game_id': 2, 'player_id': 1, 'score': 17},
        {'game_id': 3, 'player_id': 2, 'score': 22}
    ]

    etalon: ops.TRowsIterable = [
        {'player_id': 0, 'username': 'root'},              # no such game
        {'game_id': 1, 'player_id': 3, 'score': 9999999},  # no such player
        {'game_id': 2, 'player_id': 1, 'score': 17, 'username': 'XeroX'},
        {'game_id': 3, 'player_id': 2, 'score': 22, 'username': 'jay'}
    ]

    presorted_games = iter(sorted(games, key=itemgetter('player_id')))    # !!!
    presorted_players = iter(sorted(players, key=itemgetter('player_id')))  # !!!
    result = ops.Join(ops.OuterJoiner(), keys=['player_id'])(presorted_games, presorted_players)

    assert etalon == sorted(result, key=lambda x: x.get('game_id', -1))


def test_left_join() -> None:
    players: ops.TRowsIterable = [
        {'player_id': 0, 'username': 'root'},
        {'player_id': 1, 'username': 'XeroX'},
        {'player_id': 2, 'username': 'jay'}
    ]

    games: ops.TRowsIterable = [
        {'game_id': 1, 'player_id': 3, 'score': 0},
        {'game_id': 2, 'player_id': 1, 'score': 17},
        {'game_id': 3, 'player_id': 2, 'score': 22},
        {'game_id': 4, 'player_id': 2, 'score': 41}
    ]

    etalon: ops.TRowsIterable = [
        # ignore player 0 with 0 games
        {'game_id': 1, 'player_id': 3, 'score': 0},  # unknown player 3
        {'game_id': 2, 'player_id': 1, 'score': 17, 'username': 'XeroX'},
        {'game_id': 3, 'player_id': 2, 'score': 22, 'username': 'jay'},
        {'game_id': 4, 'player_id': 2, 'score': 41, 'username': 'jay'}
    ]

    presorted_games = iter(sorted(games, key=itemgetter('player_id')))    # !!!
    presorted_players = iter(sorted(players, key=itemgetter('player_id'))) # !!!
    result = ops.Join(ops.LeftJoiner(), keys=['player_id'])(presorted_games, presorted_players)
    # for line in result:
    #     print(line)

    assert etalon == sorted(result, key=itemgetter('game_id'))

# test_left_join()

def test_right_join() -> None:
    players: ops.TRowsIterable = [
        {'player_id': 0, 'username': 'root'},
        {'player_id': 1, 'username': 'XeroX'},
        {'player_id': 2, 'username': 'jay'}
    ]

    games: ops.TRowsIterable = [
        {'game_id': 1, 'player_id': 3, 'score': 0},
        {'game_id': 2, 'player_id': 1, 'score': 17},
        {'game_id': 3, 'player_id': 2, 'score': 22},
        {'game_id': 4, 'player_id': 2, 'score': 41},
        {'game_id': 5, 'player_id': 1, 'score': 34}
    ]

    etalon: ops.TRowsIterable = [
        # ignore game with unknown player 3
        {'player_id': 0, 'username': 'root'},  # no games for root
        {'game_id': 2, 'player_id': 1, 'score': 17, 'username': 'XeroX'},
        {'game_id': 3, 'player_id': 2, 'score': 22, 'username': 'jay'},
        {'game_id': 4, 'player_id': 2, 'score': 41, 'username': 'jay'},
        {'game_id': 5, 'player_id': 1, 'score': 34, 'username': 'XeroX'}
    ]

    presorted_games = iter(sorted(games, key=itemgetter('player_id')))    # !!!
    presorted_players = iter(sorted(players, key=itemgetter('player_id')))  # !!!
    result = ops.Join(ops.RightJoiner(), keys=['player_id'])(presorted_games, presorted_players)

    assert etalon == sorted(result, key=lambda x: x.get('game_id', -1))


def test_simple_join_with_collision() -> None:
    players: ops.TRowsIterable = [
        {'player_id': 1, 'username': 'XeroX', 'score': 400},
        {'player_id': 2, 'username': 'jay', 'score': 451},
        {'player_id': 3, 'username': 'Destroyer', 'score': 999},
    ]

    games: ops.TRowsIterable = [
        {'game_id': 1, 'player_id': 3, 'score': 99},
        {'game_id': 2, 'player_id': 1, 'score': 17},
        {'game_id': 3, 'player_id': 1, 'score': 22}
    ]

    etalon: ops.TRowsIterable = [
        {'game_id': 1, 'player_id': 3, 'score_game': 99, 'score_max': 999, 'username': 'Destroyer'},
        {'game_id': 2, 'player_id': 1, 'score_game': 17, 'score_max': 400, 'username': 'XeroX'},
        {'game_id': 3, 'player_id': 1, 'score_game': 22, 'score_max': 400, 'username': 'XeroX'}
    ]

    presorted_games = iter(sorted(games, key=itemgetter('player_id')))    # !!!
    presorted_players = iter(sorted(players, key=itemgetter('player_id')))  # !!!
    result = ops.Join(ops.InnerJoiner(suffix_a='_game', suffix_b='_max'),
                      keys=['player_id'])(presorted_games, presorted_players)

    assert etalon == sorted(result, key=itemgetter('game_id'))
