import random
import pandas as pd


def generate_sequence(target_length=200, min_repeats=3, max_repeats=20):
    sequence = []
    while len(sequence) < target_length:
        # random float with variable precision
        value = round(random.uniform(0, 1), random.randint(3, 9))
        repeats = random.randint(min_repeats, max_repeats)
        sequence.extend([value] * repeats)
    return sequence[:target_length]  # cut exactly to 200


# Example usage
random_sequence_float = generate_sequence()
random_sequence_float = pd.DataFrame(random_sequence_float)


def generate_sequence(target_length=200, min_repeats=3, max_repeats=20):
    sequence = []
    while len(sequence) < target_length:
        # random integer between 1 and 250
        value = random.randint(1, 29)
        repeats = random.randint(min_repeats, max_repeats)
        sequence.extend([value] * repeats)
    return sequence[:target_length]  # cut exactly to 200


# Example usage
random_sequence_INT = generate_sequence()
for num in random_sequence_INT:
    print(num)
random_sequence_INT = pd.DataFrame(random_sequence_INT)
