"""List of all string messages emitted by the utils."""

ERRORS = {
    0: "Not enough data. Make seq_length and batch_size small.",
    1: "Input file not found at path {path}",
    2: "{init} must be a a path",
    3: "config.pkl file does not exist in path {init}",
    4: "chars_vocab.pkl file does not exist in path {}",
    5: "No checkpoint found",
    6: "No model path found in checkpoint",
    7: "Command line argument and saved model disagree on '{arg}'",
    8: "Data and loaded model disagree on dictionary mappings!",
    9: "Training model not found for evaluation",
    10: "WARNING - Reusing old vocabulary",
    11: "WARNING - Reusing old IPA vocabulary",
    12: "Error in frequency file type",
    13: "Invalid loss mode",
    14: "Context not found in frequency map"
}

LOGS = {
    0: "Reading text file..",
    1: "Reading pre-processed vocabulary..",
    2: "{}/{} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}",
    3: "model saved to {}",
    4: "Evaluation data loaded.",
    5: "Batch {} of {}",
    6: "Loaded weights from weights/ folder",
    7: "Using {} as a separator",
    8: "Preparing eval data",
    9: "The current perplexity is {}",
    10: "Calculating evaluation perplexity for epoch {}",
    11: "Preprocessing the evaluation data",
    12: "Choosing top {} words",
    13: "{} / {} inverted",
    14: "Finished caching frequency data"
}

FILES = {
    0: "vocab.pkl",
    1: "data.npy",
    2: "config.pkl",
    3: "chars_vocab.pkl",
    4: "args_summary.txt",
    5: "plot_data.csv",
    6: "utils/final_encoding.txt",
    7: "ipa_vocab.pkl",
    8: "ipa_data.npy",
    9: "vocab_splits.pkl",
    10: "plot_data_eval.csv",
    11: "prob_data.csv",
    12: "final_data.csv",
    13: "prob_data.txt"
}

WORD_CHAR_DEFAULTS = [
    ["filename", "input.txt", "char_input.txt"],
    ["vocab_file", "vocab.pkl", "char_vocab.pkl"],
    ["constants_file", "const.npy", "char_const.npy"],
    ["frequencies_file", "freq.json", "char_freq.pkl"],
    ["map_file", "map.pkl", "char_map.pkl"],
    ["freqfile_type", "json", "pkl"]
]
