import click
import json

from compgraph.algorithms import pmi_graph


@click.group()
def cli() -> None:  # pragma: no cover
    pass


@cli.command()
@click.argument('filename_in')
@click.argument('filename_out')
def run_pmi(filename_in: str, filename_out: str) -> None:
    graph = pmi_graph(input_stream_name='input', doc_column='doc_id', text_column='text',
                      result_column='pmi', filename=filename_in)

    result = graph.run(input=lambda: filename_in)
    with open(filename_out, 'w') as out:
        for row in result:
            json.dump(row, out)
            out.write('\n')


if __name__ == '__main__':
    run_pmi()
