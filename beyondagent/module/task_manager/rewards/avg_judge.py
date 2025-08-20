import re
import threading
from typing import Any, Optional, Type, cast
from concurrent.futures import ThreadPoolExecutor, as_completed
from loguru import logger
from beyondagent.client.env_client import EnvClient
from beyondagent.client.llm_client import DashScopeClient
from beyondagent.module.agent_flow.reward_calculator import GraderResult, RewardCalculator
from beyondagent.module.task_manager.rewards.binary_judge_gt import LlmAsJudgeBinaryRewardCalculatorWithGT
from beyondagent.module.task_manager.rewards.reward import LlmAsJudgeRewardCalculator
from beyondagent.schema.task import Task
from beyondagent.schema.trajectory import Trajectory
from . import grader_manager


class AvgJudge(RewardCalculator):
    def __init__(self, task: Task):
        super().__init__(task)
        self._judges: list[RewardCalculator] = []

    def add_judge(self, x: RewardCalculator):
        self._judges.append(x)

    def calculate_reward(
        self, trajectory: Trajectory, env: EnvClient, instance_id: str, max_workers: int = 4
    ) -> GraderResult:
        """并行计算多个 judge 的分数，控制最大线程数"""

        rewards: list[float] = []

        def worker(judge: RewardCalculator):
            try:
                result = judge.calculate_reward(trajectory, env, instance_id)
                return result["score"]
            except Exception as e:
                logger.error(f"Judge failed: {e}")
                return 0.0

        # 并发执行
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(worker, j) for j in self._judges]
            for f in as_completed(futures):
                rewards.append(f.result())

        if not rewards:
            return {"score": 0.0, "reason": "No valid rewards"}

        return {
            "score": sum(rewards) / len(rewards),
            "reason": "AvgJudge (threaded)"
        }


@grader_manager.reg("avg-llm-binary-gt")
class AvgBinaryGTJudge(AvgJudge):
    def __init__(self, task: Task, n: int = 3):
        super().__init__(task)
        for i in range(n):
            self.add_judge(
                LlmAsJudgeBinaryRewardCalculatorWithGT(
                    task, model_name="qwq-plus", use_mean_constraint=True
                )
            )


@grader_manager.reg("avg-llm")
class AvgLlmJudge(AvgJudge):
    def __init__(self, task: Task, n: int = 3):
        super().__init__(task)
        for i in range(n):
            self.add_judge(
                LlmAsJudgeRewardCalculator(
                    task, model_name="qwq-plus"
                )
            )