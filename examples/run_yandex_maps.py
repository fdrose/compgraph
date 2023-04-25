import click
import json

from compgraph.algorithms import yandex_maps_graph


@click.group()
def cli() -> None:  # pragma: no cover
    pass


@cli.command()
@click.argument('filename_time')
@click.argument('filename_length')
@click.argument('filename_out')
def run_maps(filename_time: str, filename_length: str, filename_out: str) -> None:
    graph = yandex_maps_graph(input_stream_name_time='input_time', input_stream_name_length='input_length',
                              enter_time_column='enter_time', leave_time_column='leave_time',
                              edge_id_column='edge_id', start_coord_column='start', end_coord_column='end',
                              weekday_result_column='weekday', hour_result_column='hour',
                              speed_result_column='speed', filename_time=filename_time,
                              filename_length=filename_length)

    result = graph.run(input_time=lambda: filename_time, input_length=filename_length)
    with open(filename_out, 'w') as out:
        for row in result:
            json.dump(row, out)
            out.write('\n')


if __name__ == '__main__':
    run_maps()
