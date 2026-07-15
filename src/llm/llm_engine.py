"""
端侧大模型推理引擎 — LLM 本地推理 + RAG 知识库
Windows/LoongArch: llama.cpp (GGUF 量化模型)
"""
import asyncio
from typing import Optional, AsyncIterator, List, Dict
from pathlib import Path

from loguru import logger
from src.utils.event_bus import EventBus
from configs.settings import LLMConfig


class LLMEngine:
    """
    大模型推理引擎

    推理方案:
    - llama-cpp-python: GGUF 模型加载 + CPU 高效推理
    - embedding 模型: sentence-transformers (RAG 向量化)
    - LoongArch: 需要从源码编译 llama.cpp + OpenBLAS

    内存规划:
    - 7B INT4 量化: ~4.5GB
    - 1.5B INT8 量化: ~1.8GB
    - Embedding 模型: ~1.2GB
    - 总计预留: ~7GB (龙芯 3A6000 通常 16-32GB RAM)
    """

    def __init__(self, event_bus: EventBus, config: LLMConfig):
        self.event_bus = event_bus
        self.config = config
        self.llm = None          # llama-cpp-python Llama 实例
        self.embedding_model = None
        self.knowledge_base: List[Dict] = []
        self._stop_generation = False

    async def initialize(self):
        """加载 LLM 模型和 Embedding 模型"""
        logger.info("加载端侧大模型...")

        # ----- LLM 推理引擎 -----
        model_path = self.config.model_path
        if model_path and Path(model_path).exists():
            try:
                from llama_cpp import Llama
                self.llm = Llama(
                    model_path=model_path,
                    n_ctx=self.config.n_ctx,
                    n_threads=self.config.n_threads,
                    n_batch=self.config.n_batch,
                    verbose=False,
                )
                logger.info(f"  LLM 模型加载成功: {model_path}")
                logger.info(f"    上下文长度: {self.config.n_ctx}")
                logger.info(f"    CPU 线程数: {self.config.n_threads}")
            except ImportError:
                logger.warning("  llama-cpp-python 未安装，LLM 对话功能降级")
                logger.info("  安装方法: pip install llama-cpp-python")
            except Exception as e:
                logger.error(f"  LLM 模型加载失败: {e}")
        else:
            logger.warning(f"  GGUF 模型文件未找到: {model_path}")
            logger.info("  将使用预设回复模板（演示模式）")
            logger.info("  下载模型: huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF")

        # ----- RAG Embedding 模型 -----
        if self.config.rag_enabled:
            try:
                from sentence_transformers import SentenceTransformer
                self.embedding_model = SentenceTransformer(self.config.embedding_model)
                logger.info(f"  Embedding 模型加载成功: {self.config.embedding_model}")
            except ImportError:
                logger.warning("  sentence-transformers 未安装，RAG 功能禁用")
            except Exception as e:
                logger.warning(f"  Embedding 模型加载失败: {e}")

        # ----- 加载知识库 -----
        await self._load_knowledge_base()

        logger.info("大模型推理引擎初始化完成")

    async def generate_response(
        self,
        user_message: str,
        system_prompt: str = "",
        temperature: Optional[float] = None,
    ) -> str:
        """
        生成对话回复

        Args:
            user_message: 用户输入
            system_prompt: 系统提示词
            temperature: 温度参数
        """
        # RAG 检索增强
        rag_context = ""
        if self.embedding_model and self.knowledge_base:
            rag_context = self._retrieve_knowledge(user_message, top_k=3)

        # 构建完整 Prompt
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if rag_context:
            messages.append({
                "role": "system",
                "content": f"参考以下知识:\n{rag_context}"
            })
        messages.append({"role": "user", "content": user_message})

        temp = temperature or self.config.temperature

        if self.llm:
            return await self._generate_with_llama(messages, temp)
        else:
            return self._generate_with_template(user_message)

    async def generate_stream(
        self,
        user_message: str,
        system_prompt: str = "",
    ) -> AsyncIterator[str]:
        """流式生成对话回复"""
        self._stop_generation = False

        if self.llm:
            # 使用 llama.cpp 流式输出
            messages = [{"role": "user", "content": user_message}]
            if system_prompt:
                messages.insert(0, {"role": "system", "content": system_prompt})

            # llama.cpp 流式生成
            prompt = self._format_chat_prompt(messages)
            loop = asyncio.get_event_loop()
            for token in self.llm(
                prompt,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                stream=True,
            ):
                if self._stop_generation:
                    break
                text = token["choices"][0].get("text", "")
                if text:
                    await loop.run_in_executor(None, lambda: None)  # 让出控制权
                    yield text
        else:
            # 模板回复（模拟流式）
            response = self._generate_with_template(user_message)
            for char in response:
                yield char
                await asyncio.sleep(0.03)

    async def _generate_with_llama(self, messages: List[Dict], temperature: float) -> str:
        """使用 llama.cpp 生成回复"""
        try:
            prompt = self._format_chat_prompt(messages)
            loop = asyncio.get_event_loop()
            output = await loop.run_in_executor(
                None,
                lambda: self.llm(
                    prompt,
                    max_tokens=self.config.max_tokens,
                    temperature=temperature,
                    top_p=self.config.top_p,
                    stop=["<|im_end|>", "<|endoftext|>", "用户:", "User:"],
                )
            )
            response = output["choices"][0]["text"].strip()
            return response
        except Exception as e:
            logger.error(f"LLM 生成失败: {e}")
            return self._generate_with_template("")

    def _format_chat_prompt(self, messages: List[Dict]) -> str:
        """将消息列表格式化为模型输入格式 (ChatML)"""
        formatted = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                formatted += f"<|im_start|>system\n{content}<|im_end|>\n"
            elif role == "user":
                formatted += f"<|im_start|>user\n{content}<|im_end|>\n"
            elif role == "assistant":
                formatted += f"<|im_start|>assistant\n{content}<|im_end|>\n"
        formatted += "<|im_start|>assistant\n"
        return formatted

    def _generate_with_template(self, user_message: str) -> str:
        """模板回复 — LLM 不可用时的降级方案"""
        msg = user_message.lower()

        # 问候
        if any(w in msg for w in ["你好", "早", "嗨"]):
            return "你好！我是小航，你的智能座舱伙伴。有什么可以帮你的吗？"

        # 导航
        if "导航" in msg or "去" in msg:
            # 提取目的地
            for kw in ["去", "到", "前往", "导航到"]:
                if kw in msg:
                    dest = msg.split(kw)[-1].strip()
                    return f"好的，正在为您规划到{dest}的路线。请确认是否开始导航？"
            return "请问您想去哪里？"

        # 音乐
        if "播放" in msg or "听" in msg or "音乐" in msg:
            song = msg.replace("播放", "").replace("听", "").replace("音乐", "").strip()
            if song:
                return f"好的，为您播放{song}"
            return "好的，为您播放您喜欢的音乐"

        # 空调
        if "空调" in msg or "温度" in msg or "冷" in msg or "热" in msg:
            return "好的，已为您调节空调温度。当前车内温度22°C。"

        # 窗户
        if "窗" in msg:
            return "好的，已为您操作车窗。"

        # 情绪
        if any(w in msg for w in ["心情", "难过", "不开心", "郁闷", "烦"]):
            return "我理解你的感受。深呼吸，放松一下。要不要听首轻松的音乐，或者和我聊聊？"

        if any(w in msg for w in ["累", "困", "疲劳"]):
            return "你看起来需要休息了。安全第一，前面可以靠边停车休息一下，或者让我提醒您最近的服务区？"

        # 默认回复
        return "好的，我收到了。作为你的智能座舱助手，我会全力保障你的驾驶安全和舒适体验。"

    # ============== RAG 知识库 ==============

    async def _load_knowledge_base(self):
        """加载车载知识库"""
        kb_dir = Path(__file__).parent.parent.parent / "data" / "knowledge"
        kb_dir.mkdir(parents=True, exist_ok=True)

        # 预置知识条目
        self.knowledge_base = [
            {
                "id": "kb_001",
                "title": "疲劳驾驶危害",
                "content": ("连续驾驶超过2小时即为疲劳驾驶。疲劳驾驶时反应速度下降60%，"
                          "判断能力下降50%。建议每2小时休息15分钟。"),
                "category": "安全知识",
            },
            {
                "id": "kb_002",
                "title": "胎压安全标准",
                "content": "乘用车标准胎压一般为2.3-2.5 bar。胎压过低会导致油耗增加和爆胎风险，"
                          "胎压过高会降低抓地力和舒适性。建议每月检查一次胎压。",
                "category": "车辆知识",
            },
            {
                "id": "kb_003",
                "title": "夜间驾驶注意事项",
                "content": ("夜间驾驶应降低车速，增加跟车距离。正确使用远近光灯，"
                          "会车时提前切换近光。特别注意行人和非机动车。"),
                "category": "安全知识",
            },
            {
                "id": "kb_004",
                "title": "雨天驾驶技巧",
                "content": ("雨天路滑，制动距离增加2-3倍。应开启雾灯和近光灯，"
                          "降低车速，避免急刹和急转。积水路段应低速匀速通过。"),
                "category": "安全知识",
            },
            {
                "id": "kb_005",
                "title": "儿童安全座椅",
                "content": ("12岁以下儿童必须使用安全座椅。4岁以下应使用反向安装座椅。"
                          "安全座椅应安装在后排，切勿在前排使用后向安全座椅。"),
                "category": "安全知识",
            },
            {
                "id": "kb_006",
                "title": "路怒症应对",
                "content": ("路怒症表现为驾驶时易怒、攻击性驾驶。应对方法：深呼吸、"
                          "听舒缓音乐、换位思考。牢记安全比面子更重要。"),
                "category": "心理健康",
            },
        ]

        # 加载外部知识文件
        for txt_file in kb_dir.glob("*.txt"):
            try:
                content = txt_file.read_text(encoding="utf-8")
                self.knowledge_base.append({
                    "id": f"file_{txt_file.stem}",
                    "title": txt_file.stem,
                    "content": content[:2000],  # 限制长度
                    "category": "用户知识",
                })
            except Exception:
                pass

        if self.embedding_model and self.knowledge_base:
            # 预计算所有知识条目的向量
            texts = [kb["content"] for kb in self.knowledge_base]
            self._kb_embeddings = self.embedding_model.encode(texts)
            logger.info(f"  知识库已加载: {len(self.knowledge_base)} 条知识")
        else:
            self._kb_embeddings = None

    def _retrieve_knowledge(self, query: str, top_k: int = 3) -> str:
        """检索相关知识"""
        if not self.embedding_model or not self._kb_embeddings:
            return ""

        try:
            query_embedding = self.embedding_model.encode([query])
            from scipy.spatial.distance import cosine
            similarities = [
                1 - cosine(query_embedding[0], kb_emb)
                for kb_emb in self._kb_embeddings
            ]
            # 排序取 Top-K
            top_indices = sorted(
                range(len(similarities)),
                key=lambda i: similarities[i],
                reverse=True
            )[:top_k]

            results = []
            for idx in top_indices:
                if similarities[idx] > 0.3:  # 相似度阈值
                    kb = self.knowledge_base[idx]
                    results.append(f"【{kb['title']}】{kb['content']}")

            return "\n".join(results)
        except Exception as e:
            logger.debug(f"RAG 检索失败: {e}")
            return ""

    def stop_generation(self):
        """停止当前生成"""
        self._stop_generation = True

    async def shutdown(self):
        """关闭 LLM 引擎"""
        self.llm = None
        self.embedding_model = None
        logger.info("LLM 引擎已关闭")
