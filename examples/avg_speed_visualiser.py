import click
import json

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


@click.group()
def cli() -> None:  # pragma: no cover
    pass


@cli.command()
@click.argument('filename_in')
def visualiser(filename_in: str) -> None:  # pragma: no cover
    df = []
    with open(filename_in) as f:
        for line in f:
            df.append(json.loads(line))

    df = pd.DataFrame(df)
    hue_order = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

    fig = sns.relplot(data=df, x='hour', y='speed', kind='line', hue='weekday', hue_order=hue_order)
    plt.xlabel('hour')
    plt.ylabel('average speed')
    plt.title('Average speed depending on weekday and hour', fontdict={'fontsize': 12})
    fig.figure.set_size_inches(10, 5)
    plt.savefig('avg_speed.png', bbox_inches='tight')


if __name__ == '__main__':
    visualiser()
