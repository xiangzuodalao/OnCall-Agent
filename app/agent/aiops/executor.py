"""
Executor 节点：执行单个步骤
基于 LangGraph 官方教程实现
"""

import json
from typing import Dict, Any
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_qwq import ChatQwen
from loguru import logger

from app.config import config
from app.tools import get_current_time, retrieve_knowledge
from app.agent.mcp_client import get_mcp_client_with_retry
from .state import PlanExecuteState


async def executor(state: PlanExecuteState) -> Dict[str, Any]:
    """
    执行节点：执行计划中的下一个步骤
    
    使用 LangGraph 的 ToolNode 自动处理工具调用
    """
    logger.info("=== Executor：执行步骤 ===")

    plan = state.get("plan", [])

    # 如果计划为空，不执行
    if not plan:
        logger.info("计划为空，跳过执行")
        return {}

    # 取出第一个步骤
    task = plan[0]
    logger.info(f"当前任务: {task}")

    try:
        # 获取本地工具
        local_tools = [
            get_current_time,
            retrieve_knowledge
        ]

        # 获取 MCP 工具
        mcp_client = await get_mcp_client_with_retry()
        mcp_tools = await mcp_client.get_tools()
        logger.info(f"可用工具数量: 本地 {len(local_tools)} + MCP {len(mcp_tools)}")

        # 合并所有工具
        all_tools = local_tools + mcp_tools

        # 创建 LLM（绑定工具）
        llm = ChatQwen(
            model=config.rag_model,
            api_key=config.dashscope_api_key,
            temperature=0
        )
        llm_with_tools = llm.bind_tools(all_tools)

        # 构建消息（只包含当前步骤，避免原始任务干扰）
        messages = [
            SystemMessage(content="""你是一个能力强大的助手，负责执行具体的任务步骤。

你可以使用各种工具来完成任务。对于每个步骤：
1. 理解步骤的目标
2. 选择合适的工具，如果已经指定了工具，则使用指定的工具
3. 调用工具获取信息
4. 返回执行结果

注意：
- 如果工具调用失败，请说明失败原因
- 不要编造数据，只返回实际获取的信息
- 执行结果要清晰、准确
- 专注于当前步骤，不要考虑其他任务"""),
            HumanMessage(content=f"请执行以下任务: {task}")
        ]

        # 第一步：LLM 决定是否调用工具
        llm_response = await llm_with_tools.ainvoke(messages)
        logger.info(f"LLM 响应类型: {type(llm_response)}")

        # 第二步：如果有工具调用，执行工具
        if hasattr(llm_response, "tool_calls") and llm_response.tool_calls:
            logger.info(f"检测到 {len(llm_response.tool_calls)} 个工具调用")
            
            tool_results = await _run_tool_calls(llm_response.tool_calls, all_tools)
            tool_context = "\n\n".join(tool_results)
            final_response = await llm.ainvoke([
                *messages,
                HumanMessage(content=f"Tool execution results:\n{tool_context}\n\nPlease summarize the current step result based only on the tool results above.")
            ])
            result = final_response.content if hasattr(final_response, 'content') else str(final_response)
        else:
            # 没有工具调用，直接使用 LLM 的输出
            logger.info("LLM 未调用工具，直接返回结果")
            result = llm_response.content if hasattr(llm_response, 'content') else str(llm_response)

        logger.info(f"步骤执行完成，结果长度: {len(result)}")

        # 返回更新：移除已执行的步骤，添加执行历史
        return {
            "plan": plan[1:],  # 移除第一个步骤
            "past_steps": [(task, result)],  # 使用 operator.add 追加
        }

    except Exception as e:
        logger.error(f"执行步骤失败: {e}", exc_info=True)
        return {
            "plan": plan[1:],
            "past_steps": [(task, f"执行失败: {str(e)}")],
        }


async def _run_tool_calls(tool_calls: list, tools: list) -> list[str]:
    tool_by_name = {getattr(tool, "name", str(tool)): tool for tool in tools}
    results: list[str] = []

    for call in tool_calls:
        name = call.get("name") if isinstance(call, dict) else getattr(call, "name", None)
        args = call.get("args", {}) if isinstance(call, dict) else getattr(call, "args", {})

        if not name:
            results.append(f"Unknown tool call: {_stringify_tool_result(call)}")
            continue

        tool = tool_by_name.get(name)
        if tool is None:
            results.append(f"Tool {name} is not available")
            continue

        try:
            logger.info(f"Executing tool manually: {name}, args={args}")
            if hasattr(tool, "ainvoke"):
                value = await tool.ainvoke(args or {})
            else:
                value = tool.invoke(args or {})
            results.append(f"Tool {name} result:\n{_stringify_tool_result(value)}")
        except Exception as e:
            logger.exception(f"Tool {name} execution failed: {e}")
            results.append(f"Tool {name} failed: {e}")

    return results


def _stringify_tool_result(value: Any) -> str:
    if hasattr(value, "content"):
        value = getattr(value, "content")
    if isinstance(value, (dict, list, tuple)):
        try:
            return json.dumps(value, ensure_ascii=False, default=str)
        except TypeError:
            return str(value)
    return str(value)
