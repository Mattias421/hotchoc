import kenlm
import random
import argparse


def top_k_sample(model, vocabulary, k=10, max_len=10000, temperature=1.0):
    """
    Top-K autoregressive sampling with KenLM.

    Args:
        model: kenlm.Model
        vocabulary: list of tokens (strings)
        k: top-k cutoff
        max_len: maximum generation length
        temperature: softmax temperature (>0)

    Returns:
        generated string
    """

    vocabulary.append("</s>")
    state = kenlm.State()
    model.BeginSentenceWrite(state)

    generated = []

    for step in range(max_len):
        candidates = []

        # Score all candidate words
        for word in vocabulary:
            next_state = kenlm.State()
            score = model.BaseScore(state, word, next_state)  # log10 prob

            candidates.append((word, score, next_state))

        # Sort by score descending
        candidates.sort(key=lambda x: x[1], reverse=True)

        # Keep only top-k
        top_candidates = candidates[:k]

        # Convert log10 scores to probabilities
        # Apply temperature scaling
        scaled_probs = []
        for word, score, next_state in top_candidates:
            scaled = score / temperature
            prob = 10**scaled
            scaled_probs.append(prob)

        # Normalize
        total = sum(scaled_probs)
        probs = [p / total for p in scaled_probs]

        # Sample
        chosen_index = random.choices(range(len(top_candidates)), weights=probs)[0]
        chosen_word, chosen_score, chosen_state = top_candidates[chosen_index]

        if chosen_word != "</s>":
            generated.append(chosen_word)

        # Advance state
        state = chosen_state

        if chosen_word == "</s>":
            break

    return " ".join(generated)


def main():
    parser = argparse.ArgumentParser(description="Generate sentences using KenLM")
    parser.add_argument(
        "--vocab_file",
        type=str,
        required=True,
        help="Path to vocabulary file (one token per line)",
    )
    parser.add_argument(
        "--lm_file",
        type=str,
        required=True,
        help="Path to KenLM binary file (.bin)",
    )
    parser.add_argument(
        "--num_sentences", type=int, default=10, help="Number of sentences to generate"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Top-K sampling parameter (default: 10)",
    )
    parser.add_argument(
        "--max_len",
        type=int,
        default=10000,
        help="Maximum generation length (default: 10000)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (default: 1.0)",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Output file path (default: print to stdout)",
    )

    args = parser.parse_args()

    # Load vocabulary
    with open(args.vocab_file, encoding="utf-8") as f:
        vocabulary = [line.split()[0] for line in f if line.strip()]

    # Load language model
    lm = kenlm.Model(args.lm_file)

    # Generate sentences
    sentences = []
    for i in range(args.num_sentences):
        sample = top_k_sample(
            lm,
            vocabulary=vocabulary,
            k=args.top_k,
            max_len=args.max_len,
            temperature=args.temperature,
        )
        sentences.append(sample)

    # Output sentences
    if args.output_file:
        with open(args.output_file, "w", encoding="utf-8") as f:
            for sentence in sentences:
                f.write(sentence + "\n")
        print(
            f"Generated {args.num_sentences} sentences and saved to {args.output_file}"
        )
    else:
        for sentence in sentences:
            print(sentence)


if __name__ == "__main__":
    main()
