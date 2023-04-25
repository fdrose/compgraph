import click
import json

from compgraph.algorithms import word_count_graph


@click.group()
def cli() -> None:  # pragma: no cover
    pass


@cli.command()
@click.argument('filename_in')
@click.argument('filename_out')
def run_count(filename_in: str, filename_out: str) -> None:
    graph = word_count_graph(input_stream_name='input', text_column='text', count_column='count', filename=filename_in)

    result = graph.run(input=lambda: filename_in)
    with open(filename_out, 'w') as out:
        for row in result:
            json.dump(row, out)
            out.write('\n')


if __name__ == '__main__':
    run_count()
