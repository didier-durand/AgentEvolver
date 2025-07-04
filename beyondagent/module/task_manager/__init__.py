

from concurrent.futures import ThreadPoolExecutor
import copy
import json
import os
import time
from typing import Callable, Optional, Sequence, TypedDict, Unpack

import hydra
from loguru import logger
from omegaconf import DictConfig
from beyondagent.client.llm_client import DashScopeClient
from beyondagent.module.agent_flow.agent_flow import AgentFlow
from beyondagent.module.agent_flow.base_agent_flow import BaseAgentFlow
from beyondagent.module.task_manager.env_worker import EnvWorker
from beyondagent.module.task_manager.prompt_explore import AGENT_INTERACTION_SYSTEM_PROMPT
from beyondagent.module.task_manager.prompt_summarize import AGENT_SUMMARIZE_SYSTEM_PROMPT, get_task_summarize_prompt, parse_tasks_from_response
from beyondagent.module.task_manager.protocols import LlmClient
from beyondagent.module.task_manager.schema import TaskObjective
from beyondagent.schema.task import Task
from beyondagent.schema.trajectory import Trajectory


class TaskManagerProps(TypedDict):
    max_llm_retries:int
    max_explore_step:int
    num_explore_threads:int
    n:int

class TaskManager(object):

    def __init__(self, config:DictConfig, llm_client: LlmClient,tokenizer, env_service_url:str,**kwargs:Unpack[TaskManagerProps]):
        self._config=config
        self._llm_client = llm_client
        self._env_service_url = env_service_url
        self._max_llm_retries = kwargs["max_llm_retries"] or 3
        self._max_explore_step = kwargs["max_explore_step"] or 20
        self._num_exploration_threads = kwargs["num_explore_threads"] or 10
        self._n=kwargs["n"]
        
        self._tokenizer = tokenizer # TODO: 这玩意似乎不该在这
    
    def generate_task(self,tasks:Sequence[Task])->list[TaskObjective]:
        # TODO: 应当按照不同的任务划分队列，使得后一次对同任务的探索能够避开已经探索的任务
        task_q=list(copy.copy(tasks))*self._n
        res=[]
        # 每次最多探索所有不同任务，或者最大线程个任务
        parallel_num=min(self._num_exploration_threads,len(tasks))
        for i in range(0,len(task_q),parallel_num):
            trajectories=self._step_explore_batch(task_q[i:i+parallel_num])
            
            # FIXME: for debug, save all trajectories in readable format
            with open("debug-trajectories.json","a") as f:
                json.dump([x.dict() for x in trajectories],f,indent=2,ensure_ascii=False)
            
            task_objectives=self._step_summarize_batch(task_q[i:i+parallel_num],trajectories)
            res.extend(task_objectives)
            # TODO: 把已经有的 task 加入 experience，阻止再次探索重复任务
        
        return res
    
    def _step_explore_batch(self,tasks:Sequence[Task]):
        with ThreadPoolExecutor(max_workers=self._num_exploration_threads) as executor:
            # TODO: I have no idea what data_id and rollout_id are.
            futures = [executor.submit(self._step_explore, task, "data_id", "rollout_id") for task in tasks]
            results = [future.result() for future in futures]
            return results
    
    def _step_explore(self,task:Task, data_id: str, rollout_id: str):
        """
        Step 1: explore the environment to find out possible actions and their results.
        """
        # reset env every time
        env_worker=EnvWorker(env_type=task.env_type, task_id=task.task_id, instance_id=None, env_service_url=self._env_service_url)
        llm_chat_fn = self._get_llm_chat_fn() # TODO: better sampling_params for exploring
        agent_flow: BaseAgentFlow = AgentFlow(enable_context_generator=False,
                                            llm_chat_fn=llm_chat_fn, 
                                            tokenizer=self._tokenizer, 
                                            config=self._config)
        agent_flow.max_steps=self._max_explore_step # TODO(cc): this is ugly
        
        assert isinstance(task.query,str)
        traj=env_worker.execute(data_id=data_id, rollout_id=rollout_id,system_prompt=AGENT_INTERACTION_SYSTEM_PROMPT, agent_flow=agent_flow)
        
        return traj
    
    def _step_summarize_batch(self,tasks:Sequence[Task],trajectories:Sequence[Trajectory])->list[TaskObjective]:
        with ThreadPoolExecutor(max_workers=self._num_exploration_threads) as executor:
            futures = [executor.submit(self._step_summarize, task, traj) for task, traj in zip(tasks, trajectories)]
            results = [future.result() for future in futures]
            return sum(results,[])
    
    
    def _step_summarize(self,task:Task,trajectory:Trajectory)->list[TaskObjective]:
        """
        Step 2: summarize the results of the exploration to generate the TASK (query and gt).
        
        Args:
            task: Task
            trajectories: Trajectory.
        """
        # 这个方法从现在看基本上是固定的
        llm_fn=self._get_llm_chat_fn()
        system_prompt,user_prompt=get_task_summarize_prompt([trajectory])
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        llm_output=llm_fn(messages=messages)['content']
        tasks=parse_tasks_from_response(task,llm_output)
        return tasks
    
    
    def _get_llm_chat_fn(self, sampling_params: Optional[dict] = None) -> Callable:
        def llm_chat(messages: list[dict[str, str]],
                     custom_sampling_params: Optional[dict] = None,
                     request_id: Optional[str] = None) -> dict:
            """
            input messages: [{"role": "system", "value": "..."}, {"role": "user", "value": "..."}]
            output messages: [{"role": "assistant", "value": "..."}]
            """
            # TODO: sending sampling_params to rollout server
            updated_sampling_params = {}
            if sampling_params:
                updated_sampling_params.update(sampling_params)
            if custom_sampling_params:
                updated_sampling_params.update(custom_sampling_params)

            # output_messages = []
            input_messages = copy.deepcopy(messages)
            res=None
            for i in range(self._max_llm_retries):
                try:
                    res=self._llm_client.chat(messages=input_messages,sampling_params=updated_sampling_params)
                    break

                except Exception as e:
                    logger.exception(f"rollout_server.{i} error: {e.args}")
                    time.sleep(i + 1)
            
            assert res is not None, f"LLM client failed to chat"
            return {
                "role": "assistant",
                "content": res,
            }
        return llm_chat



@hydra.main(config_path="/Users/cc/projects/BeyondAgent/config", config_name="beyond_agent_dataflow", version_base=None)
def test(config):
    import transformers
    import json
    tokenizer=transformers.AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", trust_remote_code=True)
    manager=TaskManager(config,DashScopeClient(),tokenizer=tokenizer,env_service_url="http://localhost:8000",max_explore_step=10,max_llm_retries=3,num_explore_threads=2,n=3)
    task=Task(task_id="0a9d82a_1",env_type="appworld")
    res=manager.generate_task([task])
    with open("debug-taskobjectives.json","w") as f:
        json.dump([x.dict() for x in res],f,indent=2,ensure_ascii=False)
    import pdb;pdb.set_trace()

if __name__=="__main__":
    test()