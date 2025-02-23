import logging
import torch
from rich.logging import RichHandler
from rich.console import Console

from utils import (
    read_dataset_with_newline_separation,
    get_alphabet,
    split_dataset,
    setup_mlp,
    generate_range_of_learning_rates,
    train_mlp,
    evaluate_loss,
    generate_with_mlp,
    get_alphabet_mappings
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
    alphabet_size = len(chars) + 1 # +1 for the '.'

    config = {'BLOCK_SIZE': 3,
              'SPLIT_RATIOS': [0.8, 0.1, 0.1],
              'INNER_LAYER_SIZE': 200,
              'EMBEDDING_DIM': 10,
              'NUM_EPOCHS': 200000,
              'BATCH_SIZE': 512,
              'LOAD_SAVED_WEIGHTS': True}

    train_slice, dev_slice, test_slice = split_dataset(
        words,
        config['BLOCK_SIZE'],
        config['SPLIT_RATIOS'][0],
        config['SPLIT_RATIOS'][1],
        config['SPLIT_RATIOS'][2])


    saved_weights_path = 'data/weights/mlp_params.pth'

    if config['LOAD_SAVED_WEIGHTS']:
        mlp_params = torch.load(saved_weights_path)

    else:
        mlp_params = setup_mlp(alphabet_size, config)
        
        learning_rates = generate_range_of_learning_rates(config['NUM_EPOCHS'], -1.0, -2.0)

        train_loss = evaluate_loss(train_slice, mlp_params, config)
        dev_loss = evaluate_loss(dev_slice, mlp_params, config)
        logging.info(f"Train Loss before training: {train_loss}")
        logging.info(f"Dev Loss before training: {dev_loss}")

        learning_rate_per_epoch, loss_per_epoch = train_mlp(config, mlp_params, train_slice, learning_rates)
        torch.save(mlp_params, saved_weights_path)

        train_loss = evaluate_loss(train_slice, mlp_params, config)
        dev_loss = evaluate_loss(dev_slice, mlp_params, config)
        logging.info(f"Train Loss after training: {train_loss}")
        logging.info(f"Dev Loss after training: {dev_loss}")

    _, itos = get_alphabet_mappings(chars)
    generated_names = generate_with_mlp(20, mlp_params, config, itos)
    for name in generated_names:
        Console().print(name)


if __name__ == "__main__":
    main()
