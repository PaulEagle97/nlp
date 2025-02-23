import torch
import random
from torch import Tensor
from torch.nn import functional
import logging
from typing import List, Dict, Tuple
from tqdm import tqdm
import matplotlib.pyplot as plt

logger = logging.getLogger("utils")


def read_dataset_with_newline_separation(path: str) -> List[str]:
    with open(path, 'r', encoding='utf-8') as f:
        entries = f.read().splitlines()
    logger.info(f'Dataset with {len(entries)} lines was read from {path}')
    return entries


def count_bigrams(words: List[str], verbose: bool = False) -> Dict[Tuple[str, str], int]:
    bigram_counter = {}
    for word in words:
        chars = ['<S>'] + list(word) + ['<E>']  # 'word' -> ['<S>', 'w', 'o', 'r', 'd', '<E>']
        for char_1, char_2 in zip(chars, chars[1:]):
            bigram = (char_1, char_2)
            bigram_counter[bigram] = bigram_counter.get(bigram, 0) + 1
    if verbose:
        logger.debug(f'Total number of unique bigrams: {len(bigram_counter)}')
        top_10_bigrams = sorted(bigram_counter.items(), key = lambda kv: kv[1], reverse = True)[:10]
        logger.debug(f'Top-10 most frequent bigrams\n{top_10_bigrams}')
    return bigram_counter


def get_alphabet(words: List[str]) -> List[str]:
    alphabet = sorted(list(set(''.join(words))))
    logger.info(f"Alphabet of length {len(alphabet)} was created.")
    return alphabet


def get_alphabet_mappings(chars: List[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
    stoi = {s:i+1 for i, s in enumerate(chars)}
    stoi['.'] = 0
    logger.debug(f'str -> int mapping: {stoi}')
    itos = {i:s for s, i in stoi.items()}
    logger.debug(f'int -> str mapping: {itos}')
    return stoi, itos


def create_lookup_table(words: List[str], stoi_map: Dict[str, int]) -> Tensor:
    table = torch.zeros((27, 27), dtype = torch.int32)
    for word in tqdm(words):
        chars = ['.'] + list(word) + ['.']
        for ch1, ch2 in zip(chars, chars[1:]):
            ix1 = stoi_map[ch1]
            ix2 = stoi_map[ch2]
            table[ix1, ix2] += 1
    return table


def plot_lookup_table(table: Tensor, itos_map: Dict[int, str]):
    plt.figure(figsize=(16,16))
    plt.imshow(table, cmap="Blues")
    for i in range(27):
        for j in range(27):
            chstr = itos_map[i] + itos_map[j]
            plt.text(j, i, chstr, ha="center", va="bottom", color="gray")
            plt.text(j, i, table[i, j].item(), ha="center", va="top", color="gray")
    plt.axis("off")
    plt.show()


def create_probability_table(lookup_table: Tensor) -> Tensor:
    prob_table = (lookup_table+1).float()
    prob_table /= prob_table.sum(1, keepdim=True)
    return prob_table


def generate_with_bigrams(num_words: int, prob_table: Tensor, itos_map: Dict[int, str]) -> List[str]:
    g = torch.Generator()
    bigrams = []
    for _ in range(num_words):
        out = []
        row_idx = 0
        while True:
            row = prob_table[row_idx]
            row_idx = torch.multinomial(row, num_samples=1, replacement=True, generator=g).item()
            out.append(itos_map[row_idx])
            if row_idx == 0:
                break
        bigrams.append(''.join(out))
    return bigrams


def compute_sequence_score(sequence: List[str], stoi_map: Dict[str, int], prob_table: Tensor) -> float:
    negative_log_likelyhood = 0.0
    n = 0
    for word in tqdm(sequence):
        chs = ['.'] + list(word) + ['.']
        for ch1, ch2 in zip(chs, chs[1:]):
            ix1 = stoi_map[ch1]
            ix2 = stoi_map[ch2]
            prob = prob_table[ix1, ix2]
            logprob = torch.log(prob)
            negative_log_likelyhood -= logprob
            n += 1

    return negative_log_likelyhood / n


def create_context_to_char_mappings(sequence: List[str], block_size: int, stoi_map: Dict[str, int], itos_map: Dict[int, str]) -> Tuple[Tensor, Tensor]:
    X, Y = [], []
    for word in tqdm(sequence):
        context = [0] * block_size
        for ch in word + '.':
            ix = stoi_map[ch]
            X.append(context)
            Y.append(ix)
            context_str = ''.join(itos_map[i] for i in context)
            logging.debug(f"{context_str} ---> {ch}")
            context = context[1:] + [ix] # crop and append

    # symmetric arrays between chunks of context and the next character
    X = torch.tensor(X)
    Y = torch.tensor(Y)
    return X, Y


def split_dataset(dataset: List[str], block_size: int, train_ratio: float, dev_ratio: float, test_ratio: float):
    assert train_ratio + dev_ratio + test_ratio == 1.0
    random.shuffle(dataset)

    n1 = int(len(dataset) * train_ratio)
    n2 = int(len(dataset) * (train_ratio + dev_ratio))

    chars = get_alphabet(dataset)
    stoi, itos = get_alphabet_mappings(chars)    

    train = create_context_to_char_mappings(dataset[:n1], block_size, stoi, itos)
    dev = create_context_to_char_mappings(dataset[n1:n2], block_size, stoi, itos)
    test = create_context_to_char_mappings(dataset[n2:], block_size, stoi, itos)
    return train, dev, test


def setup_mlp(alphabet_size: int, config: Dict):
    g = torch.Generator()

    c = torch.randn((alphabet_size, config['EMBEDDING_DIM']), generator=g)
    w1 = torch.randn((config['EMBEDDING_DIM'] * config['BLOCK_SIZE'], config['INNER_LAYER_SIZE']), generator=g)
    b1 = torch.randn(config['INNER_LAYER_SIZE'], generator=g)
    w2 = torch.randn((config['INNER_LAYER_SIZE'], alphabet_size), generator=g)
    b2 = torch.randn(alphabet_size, generator=g)

    params = [c, w1, w2, b1, b2]
    for p in params:
        p.requires_grad = True

    logging.info(f"MLP with {sum(p.nelement() for p in params)} parameters was created")
    return params


def generate_range_of_learning_rates(num_epochs: int, start_exp: float, end_exp: float) -> Tensor:
    lre = torch.linspace(start_exp, end_exp, num_epochs) # learning rate exponents
    return 10 ** lre # learning rate steps


def generate_fixed_learning_rates(num_epochs: int, learning_rate: float) -> Tensor:
    return torch.full((num_epochs,), learning_rate)


def train_mlp(config: Dict, params: List, train_data: Tuple[Tensor, Tensor], learning_rates: Tensor):
    Xtr, Ytr = train_data
    c, w1, w2, b1, b2 = params
    learning_rate_per_epoch = []
    loss_per_epoch = []
    logging.info(f"Starting training on {len(Xtr)} samples")

    for epoch in tqdm(range(config['NUM_EPOCHS'])):
        # minibatch construct
        ixs = torch.randint(0, Xtr.shape[0], (config['BATCH_SIZE'],))

        # forward pass
        emb = c[Xtr[ixs]] # (BATCH_SIZE, BLOCK_SIZE, EMBEDDING_DIM)
        h = torch.tanh(emb.view(-1, config['EMBEDDING_DIM'] * config['BLOCK_SIZE']) @ w1 + b1) # (BATCH_SIZE, INNER_LAYER_SIZE)
        logits = h @ w2 + b2 # (BATCH_SIZE, alphabet_size)
        # this is equivalent (but less efficient) to F.cross_entropy()
        # counts = logits.exp()
        # prob = counts / counts.sum(1, keepdims=True)
        # loss = -prob[torch.arange(32), Y].log().mean()
        loss = functional.cross_entropy(logits, Ytr[ixs])
        logging.debug(f"Loss = {loss.item()}")

        # backward pass
        for p in params:
            p.grad = None
        loss.backward()

        # update
        lr = learning_rates[epoch] # using dynamic learning rate
        for p in params:
            p.data += -lr * p.grad

        learning_rate_per_epoch.append(lr) # for finding the best learning rate
        loss_per_epoch.append(loss.log10().item())

    plt.plot(range(config['NUM_EPOCHS']), loss_per_epoch)
    plt.show()

    return learning_rate_per_epoch, loss_per_epoch


def evaluate_loss(data_for_evaluation: Tuple[Tensor, Tensor], params: List, config: Dict):
    X, Y = data_for_evaluation
    c, w1, w2, b1, b2 = params

    emb = c[X]
    h = torch.tanh(emb.view(-1, config['EMBEDDING_DIM'] * config['BLOCK_SIZE']) @ w1 + b1)
    logits = h @ w2 + b2
    loss = functional.cross_entropy(logits, Y)
    return loss.item()


def generate_with_mlp(num_words: int, params: List, config: Dict, itos_map: Dict[int, str]) -> List[str]:
    c, w1, w2, b1, b2 = params
    sequences = []
    for _ in range(num_words):
        out = []
        context = [0] * config['BLOCK_SIZE']
        while True:
            emb = c[torch.tensor([context])]
            h = torch.tanh(emb.view(1, -1) @ w1 + b1)
            logits = h @ w2 + b2
            probs = functional.softmax(logits, dim=1)
            ix = torch.multinomial(probs, num_samples=1).item()
            context = context[1:] + [ix]
            out.append(itos_map[ix])
            if ix == 0:
                break
        sequences.append(''.join(out))
    return sequences
