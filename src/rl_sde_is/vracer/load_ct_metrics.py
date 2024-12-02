import os
import re

import numpy as np

from rl_sde_is.utils.path import save_data
from rl_sde_is.utils.config import PROJECT_ROOT_DIR

def find_output_files_with_dirpath(dir_path: str) -> list[str]:
    """
    Find output files in the directory that contain the given directory path.

    Parameters:
    - dir_path: directory path of the vracer simulation.

    Returns:
    - A list of file paths that contain the search string and match the given file extension.
    """

    directory = os.path.join(PROJECT_ROOT_DIR, 'output')
    matching_files = []

    try:
        for root, _, files in os.walk(directory):
            for file in files:

                # skip files that don't match the .out extension
                if not file.endswith('.out'):
                    continue

                file_path = os.path.join(root, file)
                try:
                    with open(file_path, "r") as f:

                        # read the last line of the file
                        lines = f.readlines()
                        last_line = lines[-1] if lines else None

                        # skip if empty file or not vracer output file
                        if not last_line or 'korali' not in last_line:
                            continue

                        # get dir path and check if it matches
                        found_dir_path = last_line.strip().split('Directory Path: ')[1]
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
    current_generation = None
    policy_eval_times, policy_update_times, running_times, generation_times = [], [], [], []

    try:
        with open(file_path, "r") as file:
            for line in file:
                line = line.strip()

                # update the current generation number
                if "Current Generation:" in line:
                    try:
                        current_generation = int(line.split("Current Generation: #")[1])
                    except (ValueError, IndexError):
                       current_generation = None

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

    return policy_eval_times, policy_update_times, running_times, generation_times


def load_ct_metrics(data: dict):
    """
    Find output file corresponding to the vracer simulation and extract computational time metrics.

    Parameters:
    - dictionary: A dictionary containing the results from the VRacer simulation.
    """

    # find output files for the vracer simulation
    matching_files = find_output_files_with_dirpath(data['dir_path'])
    if not matching_files:
        return

    # extract ct metrics from the last output file
    policy_eval_times, policy_update_times, running_times, generation_times \
        = extract_korali_metrics_from_file(matching_files[-1])

    data['policy_eval_times'] = np.array(policy_eval_times, dtype=np.float32)
    data['policy_update_times'] = np.array(policy_update_times, dtype=np.float32)
    data['cts'] = np.array(running_times, dtype=np.float32)
    data['generation_times'] = np.array(generation_times, dtype=np.float32)

    save_data(data, data['dir_path'])
