import click

from sdm.cli.compute_score_distributions import compute_score_distributions
from sdm.cli.visualize.line_chart import line_chart


@click.group()
def main():
    """
    Score Distribution Model (SDM)

    Predicting Recall in large-scale retrieval using score distributions modeling.
    """
    pass


@click.group()
def visualize():
    """
    Visualize the results of the experiments.
    """
    pass


main.add_command(compute_score_distributions)

main.add_command(visualize)
visualize.add_command(line_chart)


if __name__ == "__main__":
    main()
