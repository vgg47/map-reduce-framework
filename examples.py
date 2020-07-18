from compgraph import graphs


def yandex_maps_from_file_heavy() -> None:
    """ По двум файлам из ресурсов считает среднюю скорость движения в каждый час каждого дня недели
        и печатает результат в stdout.
    """
    graph = graphs.yandex_maps_graph_from_file(
        './resource/travel_times.txt', './resource/road_graph_data.txt',
        enter_time_column='enter_time', leave_time_column='leave_time', edge_id_column='edge_id',
        start_coord_column='start', end_coord_column='end',
        weekday_result_column='weekday', hour_result_column='hour', speed_result_column='speed'
    )

    result = graph.run()

    for row in result:
        print(row)


def word_count_from_file_heavy() -> None:
    """ Cчитает количество вхождений для каждого слова в файле из ресурсов
        и печатает результат в stdout.
    """

    graph = graphs.word_count_graph_from_file('./resource/text_corpus.txt')

    result = graph.run()

    for row in result:
        print(row)
