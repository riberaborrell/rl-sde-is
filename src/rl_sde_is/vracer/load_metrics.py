import os
import re

import numpy as np

from rl_sde_is.utils.path import save_data
from rl_sde_is.utils.config import OUTPUT_ROOT_DIR

def read_first_line(file_path: str):
    with open(file_path, "rb") as f:
        return f.readline().decode("utf-8")

def find_output_files_with_dirpath(dir_path: str) -> list[str]:
    """
    Find output files in the directory that contain the given directory path.

    Parameters:
    - dir_path: directory path of the vracer simulation.

    Returns:
    - A list of file paths that contain the search string and match the given file extension.
    """

    matching_files = []

    try:
        for root, _, files in os.walk(OUTPUT_ROOT_DIR):
            for file in files:

                # skip files that don't match the .out extension
                if not file.endswith('.out'):
                    continue

                file_path = os.path.join(root, file)
                try:
                    # read the first line of the file
                    first_line = read_first_line(file_path)

                    # skip if empty file or not vracer output file
                    if not first_line or 'Korali' not in first_line:
                        continue

                    # get dir path and check if it matches
                    match = re.search(r"Directory Path", first_line)
                    if not match:
                        continue
                    found_dir_path = first_line.strip().split('Directory Path: ')[1]
                    if dir_path == found_dir_path:
                        matching_files.append(file_path)

                except Exception as e:
                    print(f"Error reading file '{file_path}': {e}")

    except Exception as e:
        print(f"Error scanning directory '{directory}': {e}")

    return matching_files

def extract_korali_metrics_from_file(file_path: str):
    """
    Extract generation time, policy update time, and running time from a Korali `.out` file.

    Parameters:
    - file_path: Path to the Korali output file.

    Returns:
    - Tuple of lists containing the different computational time metrics for all generations
    """
    generations, policy_updates, policy_eval_times, policy_update_times, running_times, generation_times \
        = [], [], [], [], [], []

    try:
        with open(file_path, "r") as file:
            for line in file:
                line = line.strip()

                # extract current generation number
                if "Current Generation:" in line:
                    try:
                        gen = int(line.split("Current Generation: #")[1])
                        generations.append(gen)
                    except (ValueError, IndexError):
                        generations.append(None)

                # extract number of policy updates 
                if "Policy Update Count:" in line:
                    try:
                        k = int(line.split("Policy Update Count: ")[1])
                        policy_updates.append(k)
                    except (ValueError, IndexError):
                        policy_updates.append(None)

                 # extract policy evaluation time 
                if "Avg Policy Evaluation Time:" in line:
                    try:
                        match = re.search(r"\[([0-9\.]+)s\]", line)
                        time = float(match.group(1)) if match else None
                        policy_eval_times.append(time)
                    except (ValueError, IndexError):
                        policy_eval_times.append(None)

                # extract policy update time
                if "Policy Update Time:" in line:
                    try:
                        match = re.search(r"\[([0-9\.]+)s\]", line)
                        time = float(match.group(1)) if match else None
                        policy_update_times.append(time)
                    except (ValueError, IndexError):
                        policy_update_times.append(None)

                # extract running time
                if "Running Time:" in line:
                    try:
                        match = re.search(r"\[([0-9\.]+)s\]", line)
                        time = float(match.group(1)) if match else None
                        running_times.append(time)
                    except (ValueError, IndexError):
                        running_times.append(None)

                # extract generation time
                if "Generation Time:" in line:
                    try:
                        time_str = line.split("Generation Time:")[1].strip()
                        time = float(time_str.replace("s", "").strip())
                        generation_times.append(time)
                    except (ValueError, IndexError):
                        generation_times.append(None)

    except Exception as e:
        print(f"Error processing file '{file_path}': {e}")

    return generations, policy_updates, policy_eval_times, policy_update_times, running_times, generation_times


def load_metrics(data: dict):
    """
    Find output file corresponding to the vracer simulation and extract computational time metrics.

    Parameters:
    - dictionary: A dictionary containing the results from the VRacer simulation.
    """

    # find output files for the vracer simulation
    matching_files = find_output_files_with_dirpath(data['dir_path'])
    if not matching_files:
        print(f"No vracer output files found for '{data['dir_path']}'.")
        return

    # extract ct metrics from the last output file
    generations, policy_updates, policy_eval_times, policy_update_times, running_times, generation_times \
        = extract_korali_metrics_from_file(matching_files[-1])

    data['generations'] = np.array(generations, dtype=np.int32)
    data['policy_updates'] = np.array(policy_updates, dtype=np.int32)
    data['policy_eval_times'] = np.array(policy_eval_times, dtype=np.float32)
    data['policy_update_times'] = np.array(policy_update_times, dtype=np.float32)
    data['cts'] = np.array(running_times, dtype=np.float32)
    data['generation_times'] = np.array(generation_times, dtype=np.float32)

    save_data(data, data['dir_path'])
