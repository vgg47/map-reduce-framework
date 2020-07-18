# from compgraph.lib.graph import Graph
# from compgraph.lib import operations
from .lib.graph import Graph
from .lib import operations


def word_count_graph(input_stream_name: str, text_column: str = 'text', count_column: str = 'count') -> Graph:
    """Constructs graph which counts words in text_column of all rows passed"""
    return Graph.graph_from_iter(input_stream_name) \
        .map(operations.FilterPunctuation(text_column)) \
        .map(operations.LowerCase(text_column)) \
        .map(operations.Split(text_column)) \
        .sort([text_column]) \
        .reduce(operations.Count(count_column), [text_column]) \
        .sort([count_column, text_column])


def inverted_index_graph(input_stream_name: str, doc_column: str = 'doc_id', text_column: str = 'text',
                         result_column: str = 'tf_idf') -> Graph:
    """Constructs graph which calculates td-idf for every word/document pair"""
    graph1 = Graph.graph_from_iter(input_stream_name) \
        .map(operations.FilterPunctuation(text_column)) \
        .map(operations.LowerCase(text_column)) \
        .map(operations.Split(text_column))

    graph2 = Graph.graph_from_iter(input_stream_name) \
        .map(operations.AddField("tmp", 1)) \
        .reduce(operations.Count("row_count"), ["tmp"])

    graph3 = Graph.graph_from_graph(graph1) \
        .sort([doc_column, text_column]) \
        .reduce(operations.FirstReducer(), [doc_column, text_column]) \
        .sort([text_column]) \
        .reduce(operations.Count("doc_count"), [text_column]) \
        .map(operations.AddField("tmp", 1)) \
        .join(operations.InnerJoiner(), graph2, ["tmp"]) \
        .map(operations.RemoveField("tmp")) \
        .map(operations.InverseDocumentFrequency("row_count", "doc_count")) \
        .map(operations.Project([text_column, "idf"])) \
        .sort([text_column])

    graph4 = Graph.graph_from_graph(graph1) \
        .sort([doc_column]) \
        .reduce(operations.TermFrequency(text_column), [doc_column]) \
        .sort([text_column]) \
        .join(operations.LeftJoiner(), graph3, [text_column]) \
        .map(operations.TFIDF("tf", "idf", result_column)) \
        .map(operations.Project([doc_column, text_column, result_column])) \
        .sort([text_column]) \
        .reduce(operations.TopN(result_column, 3), [text_column])

    return graph4


def pmi_graph(input_stream_name: str, doc_column: str = 'doc_id', text_column: str = 'text',
              result_column: str = 'pmi') -> Graph:
    """Constructs graph which gives for every document the top 10 words ranked by pointwise mutual information"""

    graph1 = Graph.graph_from_iter(input_stream_name) \
        .map(operations.FilterPunctuation(text_column)) \
        .map(operations.LowerCase(text_column)) \
        .map(operations.Split(text_column)) \
        .sort([doc_column, text_column]) \
        .reduce(operations.Count("count_in_doc"), [text_column, doc_column]) \
        .map(operations.Filter(lambda x: len(x[text_column]) > 4 and x["count_in_doc"] >= 2))

    graph2 = Graph.graph_from_graph(graph1) \
        .map(operations.AddField("tmp", 1)) \
        .sort([text_column]) \
        .reduce(operations.TermFrequency(text_column, "atf", by_field="count_in_doc"), ["tmp"]) \
        .map(operations.RemoveField("tmp")) \
        .sort([text_column])

    graph3 = Graph.graph_from_graph(graph1) \
        .sort([doc_column]) \
        .reduce(operations.TermFrequency(text_column, by_field="count_in_doc"), [doc_column]) \
        .sort([text_column]) \
        .join(operations.InnerJoiner(), graph2, [text_column]) \
        .map(operations.PMI("tf", "atf", result_column)) \
        .map(operations.Project([doc_column, text_column, result_column])) \
        .sort([doc_column]) \
        .reduce(operations.TopN(result_column, 10), [doc_column])

    return graph3


def yandex_maps_graph(input_stream_name_time: str, input_stream_name_length: str,
                      enter_time_column: str = 'enter_time', leave_time_column: str = 'leave_time',
                      edge_id_column: str = 'edge_id', start_coord_column: str = 'start', end_coord_column: str = 'end',
                      weekday_result_column: str = 'weekday', hour_result_column: str = 'hour',
                      speed_result_column: str = 'speed') -> Graph:
    """Constructs graph which measures average speed in km/h depending on the weekday and hour"""

    graph1 = Graph.graph_from_iter(input_stream_name_length) \
        .map(operations.StreetLength(start_coord_column, end_coord_column, "length")) \
        .map(operations.Project([edge_id_column, "length"])) \
        .sort([edge_id_column])

    graph2 = Graph.graph_from_iter(input_stream_name_time) \
        .map(operations.ProcessDate(enter_time_column, leave_time_column, weekday_result_column, hour_result_column,
                                    "duration")) \
        .map(operations.Project([edge_id_column, weekday_result_column, hour_result_column, "duration"])) \
        .sort([edge_id_column, weekday_result_column, hour_result_column, "duration"]) \
        .reduce(operations.Count("count"), [edge_id_column, weekday_result_column, hour_result_column, "duration"]) \
        .join(operations.InnerJoiner(), graph1, [edge_id_column]) \
        .map(operations.RemoveField(edge_id_column)) \
        .sort([weekday_result_column, hour_result_column]) \
        .reduce(operations.MeanSpeed("duration", "length", speed_result_column, "count"),
                [weekday_result_column, hour_result_column])

    return graph2


def yandex_maps_graph_from_file(input_filename_name_time: str, input_filename_name_length: str,
                                enter_time_column: str = 'enter_time', leave_time_column: str = 'leave_time',
                                edge_id_column: str = 'edge_id', start_coord_column: str = 'start',
                                end_coord_column: str = 'end',
                                weekday_result_column: str = 'weekday', hour_result_column: str = 'hour',
                                speed_result_column: str = 'speed') -> Graph:
    """Constructs graph which measures average speed in km/h depending on the weekday and hour.
    Reads data from file"""

    graph1 = Graph.graph_from_file(input_filename_name_length) \
        .map(operations.StreetLength(start_coord_column, end_coord_column, "length")) \
        .map(operations.Project([edge_id_column, "length"])) \
        .sort([edge_id_column])

    graph2 = Graph.graph_from_file(input_filename_name_time) \
        .map(operations.ProcessDate(enter_time_column, leave_time_column, weekday_result_column, hour_result_column,
                                    "duration")) \
        .map(operations.Project([edge_id_column, weekday_result_column, hour_result_column, "duration"])) \
        .sort([edge_id_column, weekday_result_column, hour_result_column, "duration"]) \
        .reduce(operations.Count("count"), [edge_id_column, weekday_result_column, hour_result_column, "duration"]) \
        .join(operations.InnerJoiner(), graph1, [edge_id_column]) \
        .map(operations.RemoveField(edge_id_column)) \
        .sort([weekday_result_column, hour_result_column]) \
        .reduce(operations.MeanSpeed("duration", "length", speed_result_column, "count"),
                [weekday_result_column, hour_result_column])

    return graph2


def word_count_graph_from_file(input_file_name: str, text_column: str = 'text', count_column: str = 'count') -> Graph:
    """Constructs graph which counts words in text_column of all rows passed.
     Reads data from file"""

    return Graph.graph_from_file(input_file_name) \
        .map(operations.FilterPunctuation(text_column)) \
        .map(operations.LowerCase(text_column)) \
        .map(operations.Split(text_column)) \
        .sort([text_column]) \
        .reduce(operations.Count(count_column), [text_column]) \
        .sort([count_column, text_column])
