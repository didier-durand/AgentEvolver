import random
import re
from loguru import logger
from dataclasses import dataclass, field
from omegaconf import DictConfig
from typing import List, Dict, Any, Optional, Literal, Tuple
from beyondagent.schema.task import Task
from beyondagent.schema.trajectory import Trajectory
from beyondagent.client.em_client import EMClient


@dataclass
class TaskExpConfig:
    add_exp: List[bool]
    train_mode: str = "discard"     # "keep" | "discard"

@dataclass
class TrajExpConfig:
    add_exp: bool = True
    train_mode: str = "discard"
    task_id: str = ""
    data_id: str = ""
    rollout_id: str = ""
    query: str = ""
    mode: str = "sample"            # "sample" | "validate"
    experience_list: List[str] = field(default_factory=list)



class ExperienceManager(object):

    def __init__(self, config: DictConfig):
        """
        Initializes the ExperienceManager with the provided configuration.

        Args:
            config (DictConfig): The configuration dictionary containing settings for the experience manager, rollout, and other components.
        """
        self.config: DictConfig = config
        self.rollout_config = config.actor_rollout_ref.rollout
        self.exp_manager_config = config.exp_manager
        self.reme_config = config.exp_manager.reme

        self.val_rollout_mode = self.exp_manager_config.val_rollout_mode
        self.train_rollout_mode = self.exp_manager_config.train_rollout_mode
        self.rollout_ratio = self.exp_manager_config.rollout_ratio
        self.train_sample_mode = self.exp_manager_config.train_sample_mode
        self.train_sample_keepratio = self.exp_manager_config.train_sample_keepratio


    def get_complete_exp_configs(self, tasks: List[Task], mode: Literal["sample", "validate"]) -> List[TaskExpConfig]:
        """
        Generates complete experience configurations for the given tasks.

        Args:
            tasks (List[Task]): A list of Task objects for which to generate configurations.
            mode (Literal["sample", "validate"]): The mode of operation, either "sample" or "validate".

        Returns:
            List[TaskExpConfig]: A list of TaskExpConfig objects with allocated training modes and experience addition settings.
        """
        exp_manager_configs = self.allocate_train_mode(tasks)
        exp_manager_configs = self.allocate_add_exp(exp_manager_configs, mode)
        return exp_manager_configs

    def allocate_train_mode(self, tasks: List[Task]) -> List[TaskExpConfig]:
        """
        Allocates training modes for the given tasks based on the configured training sample experience mode.

        Args:
            tasks (List[Task]): A list of Task objects for which to allocate training modes.

        Returns:
            List[TaskExpConfig]: A list of TaskExpConfig objects with allocated training modes.
        """
        mode_to_ratio = {
            "allkeep": 1.0,
            "alldiscard": 0.0,
            "hybrid": self.train_sample_keepratio
        }
        keep_ratio = mode_to_ratio.get(
            self.train_sample_mode, self.train_sample_keepratio
        )
        keep_count = int(len(tasks) * keep_ratio)
        exp_modes = ['keep'] * keep_count + ['discard'] * (len(tasks) - keep_count)
        random.shuffle(exp_modes)
        return [TaskExpConfig(add_exp=[], train_mode=exp_mode) for exp_mode in exp_modes]
    
    def allocate_add_exp(self, exp_configs: List[TaskExpConfig], mode: Literal["sample", "validate"]) -> List[TaskExpConfig]:
        """
        Allocates experience addition settings for the given tasks based on the mode and configured experience modes.

        Args:
            exp_configs (List[TaskExpConfig]): A list of TaskExpConfig objects to be updated.
            mode (Literal["sample", "validate"]): The mode of operation, either "sample" or "validate".

        Returns:
            List[TaskExpConfig]: An updated list of TaskExpConfig objects with allocated experience addition settings.
        """
        is_validate = mode == "validate"
        rollout_n = self.rollout_config.val_kwargs.n if is_validate else self.rollout_config.n
        exp_mode = self.val_rollout_mode if is_validate else self.train_rollout_mode
        for task_exp_config in exp_configs:
            add_exp_choices = {
                "woexp": [False] * rollout_n,
                "mixed": sorted([i < round(rollout_n*self.rollout_ratio) for i in range(rollout_n)], key=lambda _: random.random()),
                "all": [True] * rollout_n
            }[exp_mode]
            task_exp_config.add_exp = add_exp_choices
        
        return exp_configs
        

    def schedule_exp_update(self, current_step: int, trajectories: List[Trajectory]) -> Dict[str, float]:
        """
        Schedules and executes experience updates based on the current step.

        Args:
            current_step (int): The current step in the training process.
            trajectories (List[Trajectory]): A list of trajectories to be potentially updated.

        Returns:
            Dict[str, float]: Metrics from the update process, if an update was performed.
        """
        metrics = {}
        if self.need_update(current_step):
            metrics = self.update_experience_pool(trajectories)
        return metrics

    def need_update(self, current_step: int) -> bool:
        """
        Determines if an experience update is needed based on the current step and configuration.

        Args:
            current_step (int): The current step in the training process.

        Returns:
            bool: True if an update is needed, False otherwise.
        """
        em_config = self.em_config
        return (em_config.enable_summarizer and 
                em_config.updated_freq and 
                current_step % em_config.updated_freq == 0)
        

    def update_experience_pool(self, trajectories: List[Trajectory]) -> Dict[str, float]:
        # ⚠️Not validated!
        """
        Updates the experience pool by summarizing and processing new trajectories.

        Args:
            trajectories (List[Trajectory]): A list of trajectories to be summarized and added to the experience pool.

        Returns:
            Dict[str, float]: Metrics from the update process, including summarization time.
        """
        summary_task = None
        summary_task = self.thread_pool.submit(
            self.em_client.call_summarizer,
            trajectories=trajectories,
            workspace_id=self.em_config.workspace_id)  # ⭐ Submit a summary task asynchronously
        print("async submit summary_task~")

        metrics = {}
        if summary_task is not None:
            experience_list, time_cost = summary_task.result()
            metrics["experience_maker/summary"] = time_cost
            # metrics.update({
            #     "experience_maker/summary": time_cost,
            # })
            print("Summary task completed~")
            if experience_list:
                for i, experience in enumerate(experience_list):
                    print(f"index={i} experience={experience}")
        return metrics




class ExperienceWorker(object):
    def __init__(self, config: DictConfig):
        """
        Initializes the ExperienceWorker with the provided configuration.

        Args:
            config (DictConfig): Configuration settings for the experience worker.
        """
        self.config: DictConfig = config
        self.experience_template = self.config.exp_manager.experience_template
    
    def manage_rollout_context(self, init_messages: List[dict], traj_exp_config: TrajExpConfig) -> Tuple[List[dict], TrajExpConfig]:
        """
        Manages the context for the rollout phase, potentially adding historical experience.

        Args:
            init_messages (List[dict]): Initial messages for the rollout.
            traj_exp_config (TrajExpConfig): Configuration for the trajectory experience.

        Returns:
            Tuple[List[dict], TrajExpConfig]: Updated messages and modified trajectory experience config.
        """
        # check experience conditions
        if not self._should_process_experience(traj_exp_config):
            return init_messages, traj_exp_config
        
        # initialize em client
        self._ensure_em_client()
        
        # construct trajectory
        trajectory = Trajectory(
            data_id=traj_exp_config.data_id,
            rollout_id=traj_exp_config.rollout_id,
            steps=init_messages,
            query=traj_exp_config.query
        )

        # retrieve experience
        reme_config = self.config.exp_manager.reme
        history_experience = self.em_client.call_context_generator(
            trajectory=trajectory,
            retrieve_top_k=reme_config.retrieve_top_k,
            workspace_id=reme_config.workspace_id
        )

        # check empty condition
        if not history_experience:
            logger.info("Experience is empty!")
            return init_messages, traj_exp_config

        # apply experience to trajectory
        logger.info(f"Retrieved history experience: {history_experience}")
        formatted_experience = self.experience_template.format(history_experience)
        new_content = formatted_experience + trajectory.steps[-1]["content"]
        trajectory.steps[-1]["content"] = new_content
        traj_exp_config.experience_list = traj_exp_config.experience_list + [formatted_experience]

        return trajectory.steps, traj_exp_config
    
    def _should_process_experience(self, traj_exp_config: TrajExpConfig) -> bool:
        """
        Checks if experience processing should be performed.

        Args:
            traj_exp_config (TrajExpConfig): Configuration for the trajectory experience.

        Returns:
            bool: True if experience should be processed, False otherwise.
        """
        return (traj_exp_config.add_exp and
                self.config.exp_manager.reme.enable_context_generator)
    
    def _ensure_em_client(self) -> None:
        """
        Initializes the EM client if it doesn't exist.
        """
        if not hasattr(self, 'em_client'):
            self.em_client = EMClient(
                base_url=self.config.exp_manager.reme.base_url
            )



    def manage_training_context(self, message: str, metadata_config: Dict) -> Tuple[str, str]:
        """
        Extracts and removes experience information from the given message.

        Args:
            message (str): Input message potentially containing experience information.
            metadata_config (Dict): Configuration for the trajectory experience.

        Returns:
            Tuple[str, str]: Extracted experience and the message with experience information removed.
        """
        experience = ""
        cleaned_message = message

        if metadata_config.get("task_train_mode", "discard") == "discard": 
            pattern = re.escape(self.experience_template).replace(r'\{\}', '(.*?)')
            match = re.search(pattern, message, re.DOTALL)
            if match:
                experience = match.group(1)
                cleaned_message = re.sub(pattern, '', message, flags=re.DOTALL)

        
        return experience, cleaned_message

