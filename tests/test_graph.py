from itertools import islice, cycle
from operator import itemgetter

from pytest import approx

from compgraph import algorithms
from compgraph.graph import Graph
from compgraph import operations as ops


def test_graph_map() -> None:
    graph = Graph.graph_from_iter('texts').map(ops.LowerCase('text'))

    rows = [
        {'doc_id': 1, 'text': 'HELLO HELLO! WORLD...'},
        {'doc_id': 2, 'text': 'world? world... world!!! WORLD!!! HELLO!!!'}
    ]

    expected = [
        {'doc_id': 1, 'text': 'hello hello! world...'},
        {'doc_id': 2, 'text': 'world? world... world!!! world!!! hello!!!'}
    ]

    result = graph.run(texts=lambda: iter(rows))

    assert sorted(result, key=itemgetter('doc_id', 'text')) == expected


def test_graph_reduce() -> None:
    graph = Graph.graph_from_iter('texts').reduce(ops.Sum('count'), ['doc_id'])

    rows = [
        {'doc_id': 1, 'count': 22},
        {'doc_id': 1, 'count': 20},

        {'doc_id': 2, 'count': 39},
        {'doc_id': 2, 'count': 1},
    ]

    expected = [
        {'doc_id': 1, 'count': 42},
        {'doc_id': 2, 'count': 40}
    ]

    result = graph.run(texts=lambda: iter(rows))

    assert sorted(result, key=itemgetter('doc_id', 'count')) == expected


def test_graph_sort() -> None:
    graph = Graph.graph_from_iter('texts').sort(['doc_id'])

    rows = [
        {'doc_id': 7, 'count': 1},
        {'doc_id': 3, 'count': 2},
        {'doc_id': 1, 'count': 3},
        {'doc_id': 5, 'count': 4},
        {'doc_id': 45, 'count': 4},
        {'doc_id': 11, 'count': 4}
    ]

    expected = [
        {'doc_id': 1, 'count': 3},
        {'doc_id': 3, 'count': 2},
        {'doc_id': 5, 'count': 4},
        {'doc_id': 7, 'count': 1},
        {'doc_id': 11, 'count': 4},
        {'doc_id': 45, 'count': 4}
    ]

    result = graph.run(texts=lambda: iter(rows))

    assert list(result) == expected


def test_graph_join() -> None:
    graph_to_join = Graph.graph_from_iter('texts2')
    graph = Graph.graph_from_iter('texts1').join(ops.InnerJoiner(), graph_to_join.copy(), ['doc_id'])

    rows1 = [
        {'doc_id': 1, 'text1': 'hello'},
        {'doc_id': 2, 'text1': 'world'}
    ]
    rows2 = [
        {'doc_id': 1, 'text2': 'bye'},
        {'doc_id': 2, 'text2': 'world'}
    ]

    expected = [
        {'doc_id': 1, 'text1': 'hello', 'text2': 'bye'},
        {'doc_id': 2, 'text1': 'world', 'text2': 'world'}

    ]

    result = graph.run(texts1=lambda: iter(rows1), texts2=lambda: iter(rows2))

    assert sorted(result, key=itemgetter('doc_id', 'text1', 'text2')) == expected


def test_tf_idf_multiple_call() -> None:
    graph = algorithms.inverted_index_graph('texts', doc_column='doc_id', text_column='text', result_column='tf_idf')

    rows1 = [
        {'doc_id': 1, 'text': 'hello, little world'},
        {'doc_id': 2, 'text': 'little'},
        {'doc_id': 3, 'text': 'little little little'},
        {'doc_id': 4, 'text': 'little? hello little world'},
        {'doc_id': 5, 'text': 'HELLO HELLO! WORLD...'},
        {'doc_id': 6, 'text': 'world? world... world!!! WORLD!!! HELLO!!!'}
    ]

    expected1 = [
        {'doc_id': 1, 'text': 'hello', 'tf_idf': approx(0.1351, 0.001)},
        {'doc_id': 1, 'text': 'world', 'tf_idf': approx(0.1351, 0.001)},

        {'doc_id': 2, 'text': 'little', 'tf_idf': approx(0.4054, 0.001)},

        {'doc_id': 3, 'text': 'little', 'tf_idf': approx(0.4054, 0.001)},

        {'doc_id': 4, 'text': 'hello', 'tf_idf': approx(0.1013, 0.001)},
        {'doc_id': 4, 'text': 'little', 'tf_idf': approx(0.2027, 0.001)},

        {'doc_id': 5, 'text': 'hello', 'tf_idf': approx(0.2703, 0.001)},
        {'doc_id': 5, 'text': 'world', 'tf_idf': approx(0.1351, 0.001)},

        {'doc_id': 6, 'text': 'world', 'tf_idf': approx(0.3243, 0.001)}
    ]

    result1 = graph.run(texts=lambda: iter(rows1))

    assert sorted(result1, key=itemgetter('doc_id', 'text')) == expected1

    rows2 = [
        {'doc_id': 1, 'text': 'hello, little world'},
        {'doc_id': 2, 'text': 'hello'},
        {'doc_id': 3, 'text': 'little little little'},
        {'doc_id': 4, 'text': 'little? hello little world'},
        {'doc_id': 5, 'text': 'HELLO HELLO! little...'},
        {'doc_id': 6, 'text': 'world? world... world!!! WORLD!!! HELLO!!!'}
    ]

    expected2 = [
        {'doc_id': 1, 'text': 'hello', 'tf_idf': approx(0.0607, abs=0.001)},
        {'doc_id': 1, 'text': 'little', 'tf_idf': approx(0.1351, abs=0.001)},
        {'doc_id': 1, 'text': 'world', 'tf_idf': approx(0.2310, abs=0.001)},

        {'doc_id': 2, 'text': 'hello', 'tf_idf': approx(0.1823, abs=0.001)},

        {'doc_id': 3, 'text': 'little', 'tf_idf': approx(0.4054, abs=0.001)},

        {'doc_id': 4, 'text': 'little', 'tf_idf': approx(0.2027, abs=0.001)},
        {'doc_id': 4, 'text': 'world', 'tf_idf': approx(0.1732, abs=0.001)},

        {'doc_id': 5, 'text': 'hello', 'tf_idf': approx(0.1215, abs=0.001)},

        {'doc_id': 6, 'text': 'world', 'tf_idf': approx(0.5545, abs=0.001)}
    ]

    result2 = graph.run(texts=lambda: iter(rows2))

    assert sorted(result2, key=itemgetter('doc_id', 'text')) == expected2


def test_pmi_multiple_calls() -> None:
    graph = algorithms.pmi_graph('texts', doc_column='doc_id', text_column='text', result_column='pmi')

    rows1 = [
        {'doc_id': 1, 'text': 'hello, little world'},
        {'doc_id': 2, 'text': 'little'},
        {'doc_id': 3, 'text': 'little little little'},
        {'doc_id': 4, 'text': 'little? hello little world'},
        {'doc_id': 5, 'text': 'HELLO HELLO! WORLD...'},
        {'doc_id': 6, 'text': 'world? world... world!!! WORLD!!! HELLO!!! HELLO!!!!!!!'}
    ]

    expected1 = [
        {'doc_id': 3, 'text': 'little', 'pmi': approx(0.9555, 0.001)},
        {'doc_id': 4, 'text': 'little', 'pmi': approx(0.9555, 0.001)},
        {'doc_id': 5, 'text': 'hello', 'pmi': approx(1.1786, 0.001)},
        {'doc_id': 6, 'text': 'world', 'pmi': approx(0.7731, 0.001)},
        {'doc_id': 6, 'text': 'hello', 'pmi': approx(0.0800, 0.001)},
    ]

    result1 = graph.run(texts=lambda: iter(rows1))

    assert list(result1) == expected1

    rows2 = [
        {'doc_id': 1, 'text': 'hello, little world'},
        {'doc_id': 2, 'text': 'hello'},
        {'doc_id': 3, 'text': 'little little little'},
        {'doc_id': 4, 'text': 'little? hello little world'},
        {'doc_id': 5, 'text': 'HELLO HELLO! little...'},
        {'doc_id': 6, 'text': 'world? world... world!!! WORLD!!! HELLO!!!'}
    ]

    expected2 = [
        {'doc_id': 3, 'text': 'little', 'pmi': approx(0.7884, abs=0.001)},
        {'doc_id': 4, 'text': 'little', 'pmi': approx(0.7884, abs=0.001)},
        {'doc_id': 5, 'text': 'hello', 'pmi': approx(1.7047, abs=0.001)},
        {'doc_id': 6, 'text': 'world', 'pmi': approx(1.0116, abs=0.001)}
    ]

    result2 = graph.run(texts=lambda: iter(rows2))

    assert list(result2) == expected2


def test_yandex_maps() -> None:
    graph = algorithms.yandex_maps_graph(
        'travel_time', 'edge_length',
        enter_time_column='enter_time', leave_time_column='leave_time', edge_id_column='edge_id',
        start_coord_column='start', end_coord_column='end',
        weekday_result_column='weekday', hour_result_column='hour', speed_result_column='speed'
    )

    lengths = [
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

    times = [
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

    expected = [
        {'weekday': 'Fri', 'hour': 8, 'speed': approx(62.2322, 0.001)},
        {'weekday': 'Fri', 'hour': 9, 'speed': approx(78.1070, 0.001)},
        {'weekday': 'Fri', 'hour': 11, 'speed': approx(88.9552, 0.001)},
        {'weekday': 'Sat', 'hour': 13, 'speed': approx(100.9690, 0.001)},
        {'weekday': 'Sun', 'hour': 13, 'speed': approx(21.8577, 0.001)},
        {'weekday': 'Tue', 'hour': 6, 'speed': approx(105.3901, 0.001)},
        {'weekday': 'Tue', 'hour': 14, 'speed': approx(41.5145, 0.001)},
        {'weekday': 'Wed', 'hour': 14, 'speed': approx(106.4505, 0.001)}
    ]

    result = graph.run(travel_time=lambda: islice(cycle(iter(times)), len(times)), edge_length=lambda: iter(lengths))

    assert sorted(result, key=itemgetter('weekday', 'hour')) == expected

    result = graph.run(travel_time=lambda: islice(cycle(iter(times)), len(times)), edge_length=lambda: iter(lengths))

    assert sorted(result, key=itemgetter('weekday', 'hour')) == expected
