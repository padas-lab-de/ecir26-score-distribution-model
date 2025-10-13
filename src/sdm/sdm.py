import click
from sdm.cli.evaluate import evaluate


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


main.add_command(evaluate)


if __name__ == "__main__":
    main()
