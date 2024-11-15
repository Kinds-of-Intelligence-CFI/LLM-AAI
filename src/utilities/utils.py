import errno
import os
import csv
import yaml
from typing import List, Optional, Dict, Any

from mlagents_envs.base_env import DecisionSteps, TerminalSteps


def get_change_in_total_reward(dec: DecisionSteps, term: TerminalSteps) -> float:
    # Note: Whenever the final timestep is reached but we don't get the reward, dec is 0.0 and term has the decrement
    change_reward = 0
    if len(dec.reward) > 0:
        change_reward += dec.reward.item()
    if len(term.reward) > 0:
        change_reward += term.reward.item()
    if len(dec.reward) == 0 and len(term.reward) == 0:
        raise ValueError("Either dec.reward or term.reward should be non 0.")
    return change_reward


def try_mkdir(path: str) -> None:
    try:
        os.mkdir(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def populate_csv(results_csv_path: str,
                 column_labels: List[str],
                 column_data: List[str]) -> None:
    file_exists = os.path.isfile(results_csv_path)
    with open(results_csv_path, 'a' if file_exists else 'w', newline='') as csv_file:
        csv_write = csv.writer(csv_file)
        if not file_exists:
            csv_write.writerow(column_labels)
        csv_write.writerow(column_data)

def check_episode_pass(total_reward: float, config_path: str, arena_index: int) -> bool:
    class CustomLoader(yaml.SafeLoader):
        def ignore_unknown(self, node: yaml.Node) -> None:
            return None

    def construct_arena_config(loader: CustomLoader, node: yaml.MappingNode) -> Dict[str, Any]:
        return loader.construct_mapping(node)

    def construct_arena(loader: CustomLoader, node: yaml.MappingNode) -> Dict[str, Any]:
        return loader.construct_mapping(node)

    def construct_item(loader: CustomLoader, node: yaml.MappingNode) -> Dict[str, Any]:
        return loader.construct_mapping(node)

    def construct_vector3(loader: CustomLoader, node: yaml.MappingNode) -> Dict[str, float]:
        return loader.construct_mapping(node)

    def construct_rgb(loader: CustomLoader, node: yaml.MappingNode) -> Dict[str, int]:
        return loader.construct_mapping(node)

    CustomLoader.add_constructor('!ArenaConfig', construct_arena_config)
    CustomLoader.add_constructor('!Arena', construct_arena)
    CustomLoader.add_constructor('!Item', construct_item)
    CustomLoader.add_constructor('!Vector3', construct_vector3)
    CustomLoader.add_constructor('!RGB', construct_rgb)

    CustomLoader.add_multi_constructor('', CustomLoader.ignore_unknown)
    with open(config_path, 'r') as file:
        data = yaml.load(file, Loader=CustomLoader)

    pass_mark: Optional[float] = data.get('arenas', [{}])[arena_index].get('passMark', None)

    if pass_mark is None:
        return True

    return pass_mark <= total_reward
