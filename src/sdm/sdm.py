import click

from sdm.cli.compute_empirical_results import compute_empirical_results
from sdm.cli.compute_score_distributions import compute_score_distributions
from sdm.cli.visualize.distribution import distribution
from sdm.cli.visualize.prediction import prediction


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
main.add_command(compute_empirical_results)

main.add_command(visualize)
visualize.add_command(distribution)
visualize.add_command(prediction)


if __name__ == "__main__":
    main()
