#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import sys
import numpy as np

from collections import defaultdict
from pathlib import Path
from tabulate import tabulate


def parse_config_string(config: str) -> str:
    """
    Parse strings like 'bs32-e30-lr5e-05' and return '32,5e-05'.
    """
    match = re.search(r'bs(\d+).*?lr([0-9.e-]+)', config)
    if not match:
        raise ValueError(f"Could not parse '{config}'")
    batch_size = match.group(1)
    learning_rate = match.group(2)
    return f"{learning_rate},{batch_size}"

# pattern = "bert-tiny-historic-multilingual-cased-*"  # sys.argv[1]
pattern = sys.argv[1]
total_training_time = 0.0
log_dirs = Path("./").rglob(f"{pattern}")

dev_results = defaultdict(list)
test_results_micro = defaultdict(list)
test_results_macro = defaultdict(list)

log_dirs = [log_dir for log_dir in log_dirs if ".cache" not in str(log_dir)]

epoch_last_iter_time = dict()

for log_dir in log_dirs:
    training_log = log_dir / "training.log"

    if not training_log.exists():
        print(f"No training.log found in {log_dir}")
        continue

    matches = re.match(r".*(bs.*?)-(e.*?)-(lr.*?)-(\d+)$", str(log_dir))

    batch_size = matches.group(1)
    epochs = matches.group(2)
    lr = matches.group(3)
    seed = matches.group(4)

    result_identifier = f"{batch_size}-{epochs}-{lr}"

    with open(training_log, "rt") as f_p:
        all_dev_results = []
        for line in f_p:
            line = line.rstrip()

            # Extract time from last iter of each epoch
            # Match lines like: epoch 13 - iter 690/699 - loss ... - time (sec): ...
            iter_match = re.match(
                r".*epoch (\d+) - iter (\d+)/(\d+) - loss .* - time \(sec\): ([\d.]+) .*", line
            )

            
            if iter_match:
                epoch = int(iter_match.group(1))
                time = float(iter_match.group(4))
                # Always overwrite: last iter line for each epoch will be the last one seen
                epoch_last_iter_time[epoch] = time

            if "f1-score (micro avg)" in line or "f1-score (macro avg)" in line:
                dev_result = line.split(" ")[-1]
                all_dev_results.append(dev_result)
                # dev_results[result_identifier].append(dev_result)

            if "F-score (micro" in line:
                test_result = line.split(" ")[-1]
                test_results_micro[result_identifier].append(float(test_result))

            if "F-score (macro" in line:
                test_result = line.split(" ")[-1]
                test_results_macro[result_identifier].append(float(test_result))

        best_dev_result = max([float(value) for value in all_dev_results])
        dev_results[result_identifier].append(best_dev_result)
        total_training_time += sum(epoch_last_iter_time.values())

mean_dev_results = {}

print("Debug:", dev_results)

for dev_result in dev_results.items():
    result_identifier, results = dev_result

    mean_result = np.mean([float(value) for value in results])

    mean_dev_results[result_identifier] = mean_result

print("Averaged Development Results:")

sorted_mean_dev_results = dict(sorted(mean_dev_results.items(), key=lambda item: item[1], reverse=True))

for mean_dev_config, score in sorted_mean_dev_results.items():
    print(f"{mean_dev_config} : {round(score * 100, 2)}")

best_dev_configuration = max(mean_dev_results, key=mean_dev_results.get)

print("Markdown table:")

print("")

print("Best configuration:", best_dev_configuration)

print("\n")

print("Best Development Score:",
      round(mean_dev_results[best_dev_configuration] * 100, 2))

print("\n")

header = ["Configuration"] + [f"Run {i + 1}" for i in range(len(dev_results[best_dev_configuration]))] + ["Avg."]

table = []

for mean_dev_config, score in sorted_mean_dev_results.items():
    current_std = np.std(dev_results[mean_dev_config])
    current_row = [f"`{mean_dev_config}`", *[round(res * 100, 2) for res in dev_results[mean_dev_config]],
                   f"{round(score * 100, 2)} ± {round(current_std * 100, 2)}"]
    table.append(current_row)

print(tabulate(table, headers=header, tablefmt="github") + "\n")

print("")

print(f"Test Score (Micro F1) for best configuration ({best_dev_configuration}):\n")

test_table_micro = [f"{parse_config_string(best_dev_configuration)}", *[round(res * 100, 2) for res in test_results_micro[best_dev_configuration]],
                    f"{round(np.mean(test_results_micro[best_dev_configuration]) * 100, 2)} ± {round(np.std(test_results_micro[best_dev_configuration]) * 100, 2)}"]

print(tabulate([test_table_micro], headers=header, tablefmt="github") + "\n")

print("")

print(f"Test Score (Macro F1) for best configuration ({best_dev_configuration}):\n")

test_table_macro = [f"{parse_config_string(best_dev_configuration)}", *[round(res * 100, 2) for res in test_results_macro[best_dev_configuration]],
                    f"{round(np.mean(test_results_macro[best_dev_configuration]) * 100, 2)} ± {round(np.std(test_results_macro[best_dev_configuration]) * 100, 2)}"]

print(tabulate([test_table_macro], headers=header, tablefmt="github") + "\n")


print(f"\nTotal training time (sum of last logged iter times per epoch): {round(total_training_time, 2)} seconds\n")