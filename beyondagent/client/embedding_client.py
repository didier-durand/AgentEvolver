import os
import time
import threading
from loguru import logger
import requests
import json
from typing import List, Sequence, Union, Optional, Dict, Any


class RateLimiter:
    """
    线程安全的限流器，使用令牌桶算法
    """
    
    def __init__(self, max_calls: int, time_window: int = 60):
        """
        初始化限流器
        
        Args:
            max_calls (int): 时间窗口内最大调用次数
            time_window (int): 时间窗口，单位秒，默认60秒
        """
        self.max_calls = max_calls
        self.time_window = time_window
        self.interval = time_window / max_calls  # 每次调用的最小间隔
        
        self._lock = threading.Lock()
        self._last_call_time = 0
        
        logger.info(f"初始化限流器: {max_calls}次/{time_window}秒, 最小间隔: {self.interval:.2f}秒")
    
    def acquire(self):
        """
        获取执行权限，如果超过限制则阻塞等待
        """
        with self._lock:
            current_time = time.time()
            time_since_last_call = current_time - self._last_call_time
            
            if time_since_last_call < self.interval:
                wait_time = self.interval - time_since_last_call
                # logger.debug(f"触发限流，等待 {wait_time:.2f} 秒")
                # 在锁内等待，确保线程安全
                time.sleep(wait_time)
                current_time = time.time()
            
            self._last_call_time = current_time
            # logger.debug(f"获得执行权限，时间: {current_time}")


class OpenAIEmbeddingClient:
    """
    OpenAI Embedding API客户端类
    支持调用符合OpenAI格式的embedding接口，带限流功能
    """
    
    def __init__(self, api_key: str, base_url: str = "https://api.openai.com/v1", 
                 model_name: str = "text-embedding-ada-002",
                 rate_limit_calls: int = 60, rate_limit_window: int = 60):
        """
        初始化客户端
        
        Args:
            api_key (str): API密钥
            base_url (str): API基础URL，默认为OpenAI官方地址
            model_name (str): 模型名称，默认为text-embedding-ada-002
            rate_limit_calls (int): 限流次数，默认60次
            rate_limit_window (int): 限流时间窗口，默认60秒
        """
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.model_name = model_name
        
        # 初始化限流器
        self.rate_limiter = RateLimiter(rate_limit_calls, rate_limit_window)
        
        # 设置请求头
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        logger.info(f"初始化OpenAI Embedding客户端，限流: {rate_limit_calls}次/{rate_limit_window}秒")
    
    def get_embeddings(self, texts: Union[str, Sequence[str]], 
                      model: Optional[str] = None,
                      encoding_format: str = "float",
                      dimensions: Optional[int] = None,
                      user: Optional[str] = None) -> Dict[str, Any]:
        """
        获取文本的嵌入向量（带限流）
        
        Args:
            texts (Union[str, Sequence[str]]): 要获取嵌入向量的文本，可以是单个字符串或字符串列表
            model (Optional[str]): 模型名称，如果不指定则使用初始化时的模型
            encoding_format (str): 编码格式，默认为"float"
            dimensions (Optional[int]): 输出维度（某些模型支持）
            user (Optional[str]): 用户标识符
            
        Returns:
            Dict[str, Any]: API响应结果
            
        Raises:
            requests.RequestException: 请求异常
            ValueError: 参数错误
        """
        # 限流控制
        self.rate_limiter.acquire()
        
        # 参数验证
        if not texts:
            raise ValueError("texts不能为空")
        
        # 构建请求数据
        payload = {
            "input": texts,
            "model": model or self.model_name,
            "encoding_format": encoding_format
        }
        
        # 添加可选参数
        if dimensions is not None:
            payload["dimensions"] = dimensions
        if user is not None:
            payload["user"] = user
        
        # 发送请求
        url = f"{self.base_url}/embeddings"
        
        try:
            response = requests.post(
                url, 
                headers=self.headers, 
                json=payload,
                timeout=30
            )
            if not response.ok:
                logger.error(f"请求失败: {response.status_code} {response.reason}")
                logger.error(f"失败json: {response.json()}")
                response.raise_for_status()
            
            return response.json()
            
        except requests.RequestException as e:
            raise requests.RequestException(f"请求失败: {e}")
    
    def get_single_embedding(self, text: str, **kwargs) -> List[float]:
        """
        获取单个文本的嵌入向量（简化方法）
        
        Args:
            text (str): 要获取嵌入向量的文本
            **kwargs: 其他参数传递给get_embeddings方法
            
        Returns:
            List[float]: 嵌入向量
        """
        result = self.get_embeddings(text, **kwargs)
        return result['data'][0]['embedding']
    
    def get_multiple_embeddings(self, texts: Sequence[str], **kwargs) -> List[List[float]]:
        """
        获取多个文本的嵌入向量（简化方法）
        
        Args:
            texts (List[str]): 要获取嵌入向量的文本列表
            **kwargs: 其他参数传递给get_embeddings方法
            
        Returns:
            List[List[float]]: 嵌入向量列表
        """
        result = self.get_embeddings(texts, **kwargs)
        return [item['embedding'] for item in result['data']]
    
    def set_model(self, model_name: str):
        """设置默认模型名称"""
        self.model_name = model_name
    
    def set_base_url(self, base_url: str):
        """设置base URL"""
        self.base_url = base_url.rstrip('/')
    
    def set_api_key(self, api_key: str):
        """设置API密钥"""
        self.api_key = api_key
        self.headers["Authorization"] = f"Bearer {self.api_key}"
    
    def set_rate_limit(self, max_calls: int, time_window: int = 60):
        """
        设置限流参数
        
        Args:
            max_calls (int): 时间窗口内最大调用次数
            time_window (int): 时间窗口，单位秒，默认60秒
        """
        self.rate_limiter = RateLimiter(max_calls, time_window)
        logger.info(f"更新限流设置: {max_calls}次/{time_window}秒")


# 使用示例
if __name__ == "__main__":
    import threading
    import concurrent.futures
    
    # 初始化客户端，设置每分钟最多10次请求
    client = OpenAIEmbeddingClient(
        api_key=os.environ.get('OPENAI_API_KEY', 'test-key'),
        base_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
        model_name="text-embedding-v4",
        rate_limit_calls=10,  # 每分钟10次
        rate_limit_window=60
    )
    
    def test_embedding(thread_id: int, text: str):
        """测试函数，用于多线程测试"""
        try:
            start_time = time.time()
            embedding = client.get_single_embedding(f"{text} - Thread {thread_id}")
            end_time = time.time()
            print(f"线程 {thread_id}: 成功获取embedding，耗时 {end_time - start_time:.2f}秒，维度: {len(embedding)}")
            return True
        except Exception as e:
            print(f"线程 {thread_id}: 错误 - {e}")
            return False
    
    try:
        print("=== 单线程测试 ===")
        # 单线程测试
        for i in range(3):
            start_time = time.time()
            embedding = client.get_single_embedding(f"Test text {i}")
            end_time = time.time()
            print(f"请求 {i+1}: 完成，耗时 {end_time - start_time:.2f}秒")
        
        print("\n=== 多线程测试 ===")
        # 多线程测试
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = []
            for i in range(8):  # 提交8个任务，超过限流数量
                future = executor.submit(test_embedding, i+1, "Hello world")
                futures.append(future)
            
            # 等待所有任务完成
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
            successful_count = sum(results)
            print(f"多线程测试完成，成功: {successful_count}/{len(results)}")
        
    except Exception as e:
        print(f"测试过程中出现错误: {e}")