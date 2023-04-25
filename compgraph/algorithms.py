from collections.abc import Callable
import json
import typing as tp

from . import operations
from .graph import Graph


def reader(input_stream_name: str, filename: str | None = None,
           parser: Callable[[str], dict[str, tp.Any]] = json.loads) -> Graph:
    if filename is not None:
        return Graph.graph_from_file(filename, lambda x: parser(x))
    return Graph.graph_from_iter(input_stream_name)


def word_count_graph(input_stream_name: str, text_column: str = 'text', count_column: str = 'count',
                     filename: str | None = None,
                     parser: Callable[[str], dict[str, tp.Any]] = json.loads) -> Graph:
    """Constructs graph which counts words in text_column of all rows passed"""
    reader_graph = reader(input_stream_name, filename, parser)

    return reader_graph.copy() \
        .map(operations.FilterPunctuation(text_column)) \
        .map(operations.LowerCase(text_column)) \
        .map(operations.Split(text_column)) \
        .sort([text_column]) \
        .reduce(operations.Count(count_column), [text_column]) \
        .sort([count_column, text_column])


def inverted_index_graph(input_stream_name: str, doc_column: str = 'doc_id', text_column: str = 'text',
                         result_column: str = 'tf_idf', filename: str | None = None,
                         parser: Callable[[str], dict[str, tp.Any]] = json.loads) -> Graph:
    """Constructs graph which calculates td-idf for every word/document pair"""
    reader_graph = reader(input_stream_name, filename, parser)

    split_graph = reader_graph.copy() \
        .map(operations.FilterPunctuation(text_column)) \
        .map(operations.LowerCase(text_column)) \
        .map(operations.Split(text_column))

    count_graph = reader_graph.copy() \
        .reduce(operations.CountRows('doc_ctr'), [])

    idf_graph = split_graph.copy() \
        .sort([doc_column, text_column]) \
        .reduce(operations.FirstReducer(), [doc_column, text_column]) \
        .sort([text_column]) \
        .reduce(operations.Count('doc_text_ctr'), [text_column]) \
        .join(operations.InnerJoiner(), count_graph, []) \
        .map(operations.LogTransform('doc_ctr', 'doc_text_ctr', 'idf'))

    tf_graph = split_graph.copy() \
        .sort([doc_column]) \
        .reduce(operations.TermFrequency(text_column, 'tf'), [doc_column]) \
        .sort([text_column])

    return idf_graph.copy() \
        .join(operations.InnerJoiner(), tf_graph.copy(), [text_column]) \
        .map(operations.Product(['idf', 'tf'], result_column)) \
        .map(operations.Project([doc_column, text_column, result_column])) \
        .sort([text_column, doc_column]) \
        .reduce(operations.TopN(result_column, 3), [text_column])


def pmi_graph(input_stream_name: str, doc_column: str = 'doc_id', text_column: str = 'text',
              result_column: str = 'pmi', filename: str | None = None,
              parser: Callable[[str], dict[str, tp.Any]] = json.loads) -> Graph:
    """Constructs graph which gives for every document the top 10 words ranked by pointwise mutual information"""
    reader_graph = reader(input_stream_name, filename, parser)

    split_graph = reader_graph \
        .map(operations.FilterPunctuation(text_column)) \
        .map(operations.LowerCase(text_column)) \
        .map(operations.Split(text_column)) \
        .map(operations.LongerThanN(text_column, 4)) \
        .sort([doc_column, text_column]) \
        .reduce(operations.Count('ctr'), [doc_column, text_column]) \
        .map(operations.AtLeastNTimes('ctr', 2))

    tf_graph = split_graph.copy() \
        .sort([doc_column]) \
        .reduce(operations.TermFrequencyFromCounts(text_column, 'ctr', 'tf'), [doc_column]) \
        .sort([text_column])

    tf_combined_graph = split_graph.copy() \
        .sort([text_column]) \
        .reduce(operations.Sum('ctr'), [text_column]) \
        .reduce(operations.TermFrequencyFromCounts(text_column, 'ctr', 'tf_combined'), [])

    return tf_graph.copy() \
        .sort([text_column]) \
        .join(operations.InnerJoiner(), tf_combined_graph.copy(), [text_column]) \
        .map(operations.LogTransform('tf', 'tf_combined', result_column)) \
        .map(operations.Project([doc_column, text_column, result_column])) \
        .sort([doc_column]) \
        .reduce(operations.TopN(result_column, 10), [doc_column])


def yandex_maps_graph(input_stream_name_time: str, input_stream_name_length: str,
                      enter_time_column: str = 'enter_time', leave_time_column: str = 'leave_time',
                      edge_id_column: str = 'edge_id', start_coord_column: str = 'start', end_coord_column: str = 'end',
                      weekday_result_column: str = 'weekday', hour_result_column: str = 'hour',
                      speed_result_column: str = 'speed', filename_time: str | None = None,
                      filename_length: str | None = None,
                      parser: Callable[[str], dict[str, tp.Any]] = json.loads) -> Graph:
    """Constructs graph which measures average speed in km/h depending on the weekday and hour"""
    time_reader_graph = reader(input_stream_name_time, filename_time, parser)
    length_reader_graph = reader(input_stream_name_length, filename_length, parser)

    time_graph = time_reader_graph \
        .map(operations.HourWeekday(enter_time_column, weekday_result_column, hour_result_column)) \
        .map(operations.TimeDiff('diff', enter_time_column, leave_time_column)) \
        .map(operations.Project([edge_id_column, weekday_result_column, hour_result_column, 'diff'])) \
        .sort([edge_id_column])

    length_graph = length_reader_graph \
        .map(operations.Haversine('length', start_coord_column, end_coord_column)) \
        .map(operations.Project([edge_id_column, 'length'])) \
        .sort([edge_id_column]) \

    total_length_graph = time_graph.copy().join(operations.InnerJoiner(), length_graph.copy(), [edge_id_column]) \
        .sort([weekday_result_column, hour_result_column]) \
        .reduce(operations.Sum('length'), [weekday_result_column, hour_result_column])

    total_time_graph = time_graph.copy().join(operations.InnerJoiner(), length_graph.copy(), [edge_id_column]) \
        .sort([weekday_result_column, hour_result_column]) \
        .reduce(operations.Sum('diff'), [weekday_result_column, hour_result_column])

    return total_time_graph \
        .join(operations.InnerJoiner(), total_length_graph,  [weekday_result_column, hour_result_column]) \
        .map(operations.Divide('length', 'diff', speed_result_column)) \
        .map(operations.Project([hour_result_column, speed_result_column, weekday_result_column]))
