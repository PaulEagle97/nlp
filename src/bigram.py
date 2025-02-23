import logging
from rich.logging import RichHandler
from rich.console import Console

from utils import (
    read_dataset_with_newline_separation,
    get_alphabet_mappings,
    get_alphabet,
    create_lookup_table,
    plot_lookup_table,
    create_probability_table,
    generate_with_bigrams,
    compute_sequence_score
)

logging.basicConfig(
    level=logging.INFO,  # Set the desired level (DEBUG, INFO, etc.)
    format="%(message)s",  # Customize the log format if needed
    datefmt="[%X]",
    handlers=[RichHandler()]
)


def main():
    words = read_dataset_with_newline_separation('data/names.txt')
    chars = get_alphabet(words)
    stoi, itos = get_alphabet_mappings(chars)
    lookup_table = create_lookup_table(words, stoi)
    plot_lookup_table(lookup_table, itos)
    prob_table = create_probability_table(lookup_table)

    score = compute_sequence_score(words, stoi, prob_table)
    logging.info(f'Score of the dataset: {score}')

    bigrams = generate_with_bigrams(len(words), prob_table, itos)
    for bg in bigrams[:10]:
        Console().print(bg)

    score = compute_sequence_score(bigrams, stoi, prob_table)
    logging.info(f'Score of the generated sequence: {score}')

    score = compute_sequence_score(['lena'], stoi, prob_table)
    logging.info(f'Score of the custom sequence: {score}')


if __name__ == "__main__":
    main()