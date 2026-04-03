#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
微信聊天记录分析程序
使用ModelScope API分析聊天记录，评估客户业务意向程度
支持PyInstaller打包版本
"""

import os
import sys
import json
import csv
import time
import logging
from datetime import datetime
from openai import OpenAI
from configparser import ConfigParser

def create_default_config(config_path):
    """创建默认配置文件模板"""
    default_config = """[DEFAULT]
# ModelScope API配置
base_url = https://api-inference.modelscope.cn/v1/
api_key = your_api_key
model_name = Qwen/Qwen3-VL-8B-Instruct

# 使用说明：
# 1. 请将your_api_key替换为您的ModelScope API密钥
# 2. 可以从 https://modelscope.cn/ 获取API密钥
# 3. 确保API密钥有足够的调用额度
"""
    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write(default_config)
        print(f"已创建默认配置文件: {config_path}")
        print("请修改配置文件中的API密钥后重新运行程序")
    except Exception as e:
        print(f"创建默认配置文件失败: {e}")

def load_config():
    """加载配置文件"""
    config = ConfigParser()
    
    # 支持PyInstaller打包后的路径处理
    if getattr(sys, 'frozen', False):
        # 打包后的可执行文件路径
        base_path = os.path.dirname(sys.executable)
    else:
        # 开发环境下的脚本路径
        base_path = os.path.dirname(os.path.abspath(__file__))
    
    config_path = os.path.join(base_path, 'config2.ini')
    
    if not os.path.exists(config_path):
        print(f"配置文件不存在: {config_path}")
        print("请确保config2.ini文件与程序在同一目录下")
        # 创建默认配置文件模板
        create_default_config(config_path)
        return {
            'base_url': 'https://api-inference.modelscope.cn/v1/',
            'api_key': 'your_api_key',
            'model_name': 'Qwen/Qwen3-VL-8B-Instruct'
        }
    
    config.read(config_path, encoding='utf-8')

    # 从配置文件获取API配置
    url = config.get('DEFAULT', 'base_url', fallback='https://api-inference.modelscope.cn/v1/')
    key = config.get('DEFAULT', 'api_key', fallback='your_api_key')

    # 从配置文件获取模型名称
    model = config.get('DEFAULT', 'model_name', fallback='Qwen/Qwen3-VL-8B-Instruct')

    return {
        'base_url': url,
        'api_key': key,
        'model_name': model
    }

def read_chat_file(file_path):
    """读取聊天文件内容"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return content
    except Exception as e:
        print(f"读取文件 {file_path} 失败: {e}")
        return ""

def truncate_chat_content(chat_content, max_length=2000):
    """截断聊天记录内容，如果超过指定长度则只保留前1000字和后1000字

    Args:
        chat_content: 聊天记录内容
        max_length: 最大长度限制，默认2000字

    Returns:
        处理后的聊天记录内容
    """
    if len(chat_content) <= max_length:
        return chat_content

    # 截取前1000字和后1000字
    first_part = chat_content[:1000]
    last_part = chat_content[-1000:]

    # 添加分隔符，表示中间内容被省略
    truncated_content = f"{first_part}\n\n[...中间内容已省略...]\n\n{last_part}"

    return truncated_content


def get_chat_send_stats(original_text: str, truncated_text: str):
    """计算本次发送给模型的字符数与截断信息（用于终端简洁展示）"""
    original_len = len(original_text or "")
    send_len = len(truncated_text or "")
    truncated = send_len < original_len
    removed = max(0, original_len - send_len)
    return {
        "original_len": original_len,
        "send_len": send_len,
        "truncated": truncated,
        "removed": removed,
    }


def extract_contact_name(file_name):
    """从文件名提取联系人昵称"""
    # 去掉扩展名和前缀
    if file_name.startswith('私聊_'):
        name = file_name[3:-4]  # 去掉'私聊_'和'.txt'
    else:
        name = file_name[:-4]  # 去掉'.txt'
    return name

def setup_logging():
    """设置日志功能"""
    # 支持PyInstaller打包后的路径处理
    if getattr(sys, 'frozen', False):
        base_path = os.path.dirname(sys.executable)
    else:
        base_path = os.path.dirname(os.path.abspath(__file__))
    
    # 创建logs目录
    logs_dir = os.path.join(base_path, 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    
    # 生成日志文件名（包含时间戳）
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(logs_dir, f'analysis_{timestamp}.log')
    
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )

    # 关闭第三方库的HTTP请求噪声日志（如：HTTP Request: POST ... 200 OK）
    for noisy_logger in (
        'httpx',
        'httpcore',
        'openai',
        'urllib3',
        'requests',
    ):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)

    # 防止某些库重复打印
    logging.getLogger().propagate = False
    
    return log_file

def call_modelscope_api_with_retry(prompt_text, api_config, max_retries=3, retry_delay=2):
    """调用ModelScope API进行文本分析，支持重试机制"""
    for attempt in range(max_retries):
        try:
            client = OpenAI(
                api_key=api_config['api_key'],
                base_url=api_config['base_url']
            )

            response = client.chat.completions.create(
                model=api_config['model_name'],  # 使用配置文件中指定的模型
                messages=[
                    {
                        'role': 'system',
                        'content': '你是业务意向分析助手。请严格遵循以下要求：\n1. 分析微信聊天记录，其中"我"是业务员角色\n2. 如果聊天记录没有客户名字，文件名中"私聊_"后面的字段就是客户微信昵称\n3. 必须返回严格符合JSON格式的结果，不要包含任何额外文本或说明\n4. 使用双引号包裹所有字段名和字符串值\n5. 确保JSON格式完全正确，可以直接被json.loads()解析\n6. 字段名必须与要求完全一致，不要添加空格或修改格式'
                    },
                    {
                        'role': 'user',
                        'content': prompt_text
                    }
                ],
                temperature=0.01,
                max_tokens=1500
            )

            # 检查响应是否有效
            if not response or not hasattr(response, 'choices') or not response.choices:
                error_msg = f"API响应无效或为空: {response}"
                logging.warning(f"第{attempt+1}次尝试失败: {error_msg}")
                if attempt < max_retries - 1:
                    logging.info(f"等待{retry_delay}秒后重试...")
                    time.sleep(retry_delay)
                    continue
                else:
                    logging.error(f"API调用失败，已达到最大重试次数: {error_msg}")
                    return None

            if not response.choices[0] or not hasattr(response.choices[0], 'message'):
                error_msg = f"API响应格式错误: {response.choices}"
                logging.warning(f"第{attempt+1}次尝试失败: {error_msg}")
                if attempt < max_retries - 1:
                    logging.info(f"等待{retry_delay}秒后重试...")
                    time.sleep(retry_delay)
                    continue
                else:
                    logging.error(f"API调用失败，已达到最大重试次数: {error_msg}")
                    return None

            result = response.choices[0].message.content.strip()
            logging.info(f"API调用成功 (第{attempt+1}次尝试)")
            return result

        except Exception as e:
            error_msg = f"调用ModelScope API失败: {e}"
            import traceback
            error_details = traceback.format_exc()
            logging.warning(f"第{attempt+1}次尝试失败: {error_msg}")
            
            if attempt < max_retries - 1:
                logging.info(f"等待{retry_delay}秒后重试...")
                time.sleep(retry_delay)
            else:
                logging.error(f"API调用失败，已达到最大重试次数: {error_msg}")
                logging.error(f"详细错误信息: {error_details}")
                return None
    
    return None

def parse_analysis_result(analysis_text):
    """解析分析结果，提取结构化数据"""
    try:
        # 清理分析文本 - 移除可能的markdown代码块标记
        cleaned_text = analysis_text.strip()
        
        # 移除各种可能的markdown代码块标记
        if cleaned_text.startswith('```json'):
            cleaned_text = cleaned_text[7:].strip()
        elif cleaned_text.startswith('```'):
            cleaned_text = cleaned_text[3:].strip()
        if cleaned_text.endswith('```'):
            cleaned_text = cleaned_text[:-3].strip()
        
        # 移除可能的JSON前缀说明文字
        if '{"' in cleaned_text:
            json_start = cleaned_text.find('{"')
            cleaned_text = cleaned_text[json_start:]
        
        # 移除可能的JSON后缀说明文字
        if '"}' in cleaned_text:
            json_end = cleaned_text.rfind('"}') + 2
            cleaned_text = cleaned_text[:json_end]
        
        cleaned_text = cleaned_text.strip()

        # 尝试解析JSON格式的响应
        if cleaned_text.startswith('{') and cleaned_text.endswith('}'):
            try:
                result = json.loads(cleaned_text)
                print("✅ JSON解析成功")
                return result
            except json.JSONDecodeError as e:
                print(f"❌ JSON解析失败，尝试修复格式: {e}")
                
                # 多层级修复尝试
                for attempt in range(3):
                    try:
                        if attempt == 0:
                            # 第一层修复：引号问题
                            fixed_text = cleaned_text.replace("'", '"')
                            # 修复可能的逗号问题
                            fixed_text = fixed_text.replace(',}', '}').replace(',]', ']')
                            # 修复可能的换行问题
                            fixed_text = fixed_text.replace('\n', ' ').replace('\r', ' ')
                        elif attempt == 1:
                            # 第二层修复：处理可能的字段名不一致
                            fixed_text = cleaned_text
                            # 标准化字段名
                            field_mappings = [
                                ('跟踪话术 1', '跟踪话术1'),
                                ('跟踪话术 2', '跟踪话术2'), 
                                ('跟踪话术 3', '跟踪话术3'),
                                ('是否咨询业务：', '"是否咨询业务":'),
                                ('客户昵称：', '"客户昵称":'),
                                ('订单状态：', '"订单状态":'),
                                ('咨询业务：', '"咨询业务":'),
                                ('关心问题：', '"关心问题":'),
                                ('当前态度：', '"当前态度":'),
                                ('聊天总结：', '"聊天总结":')
                            ]
                            for old, new in field_mappings:
                                fixed_text = fixed_text.replace(old, new)
                        else:
                            # 第三层修复：使用正则表达式提取JSON
                            import re
                            json_pattern = r'\{[^{}]*\{[^{}]*\}[^{}]*\}|\{[^{}]*\}'
                            matches = re.findall(json_pattern, cleaned_text)
                            if matches:
                                fixed_text = matches[0]
                            else:
                                continue
                        
                        result = json.loads(fixed_text)
                        print(f"✅ JSON修复成功 (第{attempt+1}层修复)")
                        return result
                        
                    except json.JSONDecodeError:
                        if attempt == 2:
                            print("❌ 所有JSON修复尝试失败，回退到手动解析")
                            break
                        continue
        else:
            print("❌ 响应不是JSON格式，使用手动解析")

        # 如果没有JSON格式，尝试手动解析
        lines = analysis_text.split('\n')
        result = {}

        for line in lines:
            # 提取是否咨询业务
            if '是否咨询业务' in line or '咨询业务：' in line:
                if '是' in line:
                    result['是否咨询业务'] = '是'
                elif '否' in line:
                    result['是否咨询业务'] = '否'

            # 提取订单状态
            if '订单状态' in line or '订单状态：' in line:
                if '售前' in line:
                    result['订单状态'] = '售前'
                elif '售后' in line:
                    result['订单状态'] = '售后'

            # 提取咨询业务
            if '咨询业务' in line or '咨询了什么业务' in line:
                summary_start = line.find('：') if '：' in line else line.find(':')
                if summary_start != -1:
                    business = line[summary_start+1:].strip()
                    if business:
                        result['咨询业务'] = business

            # 提取关心问题
            if '关心问题' in line or '关心的问题' in line:
                summary_start = line.find('：') if '：' in line else line.find(':')
                if summary_start != -1:
                    concern = line[summary_start+1:].strip()
                    if concern:
                        result['关心问题'] = concern

            # 提取当前态度
            if '当前态度' in line or '态度' in line:
                summary_start = line.find('：') if '：' in line else line.find(':')
                if summary_start != -1:
                    attitude = line[summary_start+1:].strip()
                    if attitude:
                        result['当前态度'] = attitude

            # 提取聊天总结
            if '聊天总结' in line or '当前聊天总结' in line:
                summary_start = line.find('：') if '：' in line else line.find(':')
                if summary_start != -1:
                    summary = line[summary_start+1:].strip()
                    if summary:
                        result['聊天总结'] = summary

            # 提取跟踪话术
            if '跟踪话术1' in line or '跟踪话术 1' in line:
                summary_start = line.find('：') if '：' in line else line.find(':')
                if summary_start != -1:
                    follow_up1 = line[summary_start+1:].strip()
                    if follow_up1:
                        result['跟踪话术1'] = follow_up1

            if '跟踪话术2' in line or '跟踪话术 2' in line:
                summary_start = line.find('：') if '：' in line else line.find(':')
                if summary_start != -1:
                    follow_up2 = line[summary_start+1:].strip()
                    if follow_up2:
                        result['跟踪话术2'] = follow_up2

            if '跟踪话术3' in line or '跟踪话术 3' in line:
                summary_start = line.find('：') if '：' in line else line.find(':')
                if summary_start != -1:
                    follow_up3 = line[summary_start+1:].strip()
                    if follow_up3:
                        result['跟踪话术3'] = follow_up3

        # 设置默认值
        if '是否咨询业务' not in result:
            result['是否咨询业务'] = '否'
        if '订单状态' not in result:
            result['订单状态'] = '售前'
        if '咨询业务' not in result:
            result['咨询业务'] = '无明确业务咨询'
        if '关心问题' not in result:
            result['关心问题'] = '无明确关注问题'
        if '当前态度' not in result:
            result['当前态度'] = '无明确态度'
        if '聊天总结' not in result:
            result['聊天总结'] = analysis_text[:200]
        if '跟踪话术1' not in result:
            result['跟踪话术1'] = '暂无跟踪话术'
        if '跟踪话术2' not in result:
            result['跟踪话术2'] = '暂无跟踪话术'
        if '跟踪话术3' not in result:
            result['跟踪话术3'] = '暂无跟踪话术'

        return result

    except Exception as e:
        print(f"解析分析结果失败: {e}")
        return {
            '是否咨询业务': '否',
            '订单状态': '售前',
            '咨询业务': '无明确业务咨询',
            '关心问题': '无明确关注问题',
            '当前态度': '无明确态度',
            '聊天总结': analysis_text[:200],
            '跟踪话术1': '暂无跟踪话术',
            '跟踪话术2': '暂无跟踪话术',
            '跟踪话术3': '暂无跟踪话术'
        }

def create_analysis_prompt(contact_name, chat_content):
    """创建用于分析的提示词"""
    prompt = f"""
请分析以下微信聊天记录：

联系人：{contact_name}

聊天记录内容：
{chat_content}

分析要求：
1. 判断客户是否是咨询业务（是 / 否)
2. 客户昵称
3. 订单状态（售前/售后）
4. 咨询了什么业务
5. 关心的问题（问了哪些问题和我如何销冠回答）
6. 当前态度
7. 当前聊天总结
8. 跟踪话术1：如果你是销冠（下一步要下一步跟踪思路是什么，你要发什么给他口语（分三段发有逻辑性的避免一大段 + 亲切的话 给他～请叫他，姓 - 工。比如李工（只有第一句发某工，后面不发）
9. 跟踪话术2：如果客户这次不回，第二次跟踪思路话术（分三段发 + 亲切的话）
10. 跟踪话术3：如果客户不回、第三次跟踪话术（分三段发 + 亲切的话）

请以JSON格式输出，包含以下字段：
- 是否咨询业务：是 / 否
- 客户昵称：{contact_name}
- 订单状态：售前 / 售后
- 咨询业务：客户咨询的具体业务内容
- 关心问题：客户关心的问题和销冠回答策略
- 当前态度：客户的当前态度
- 聊天总结：当前聊天的总结
- 跟踪话术1：第一次跟踪的话术
- 跟踪话术2：第二次跟踪的话术
- 跟踪话术3：第三次跟踪的话术
"""
    return prompt

def process_all_chat_files(api_config):
    """处理所有聊天文件"""
    # 支持PyInstaller打包后的路径处理
    if getattr(sys, 'frozen', False):
        # 打包后的可执行文件路径
        base_path = os.path.dirname(sys.executable)
    else:
        # 开发环境下的脚本路径
        base_path = os.path.dirname(os.path.abspath(__file__))
    
    chat_folder = os.path.join(base_path, "聊天记录")

    if not os.path.exists(chat_folder):
        print(f"❌ 聊天文件夹不存在: {chat_folder}")
        print("   请确保'聊天记录'文件夹与程序在同一目录下")
        return 0

    # 获取所有聊天文件
    chat_files = [f for f in os.listdir(chat_folder) if f.endswith('.txt')]

    if not chat_files:
        print("❌ 未找到聊天文件")
        print("   请在'聊天记录'文件夹中放置.txt格式的聊天文件")
        return 0

    print(f"📁 找到 {len(chat_files)} 个聊天文件，开始分析...\n")
    
    processed_files = 0

    for i, file_name in enumerate(chat_files, 1):
        file_path = os.path.join(chat_folder, file_name)
        contact_name = extract_contact_name(file_name)

        print(f"[{i}/{len(chat_files)}] {file_name}（客户：{contact_name}）")

        chat_content_original = read_chat_file(file_path)

        if not chat_content_original.strip():
            print("   ⚠️ 文件内容为空，跳过")
            # 删除空文件
            try:
                os.remove(file_path)
                print(f"   🗑️ 已删除空文件: {file_name}")
                logging.info(f"已删除空文件: {file_name}")
            except Exception as e:
                print(f"   ❌ 删除空文件失败: {e}")
                logging.error(f"删除空文件 {file_name} 失败: {e}")
            continue

        # 截断聊天记录长度（仅用于发送给模型）
        chat_content_send = truncate_chat_content(chat_content_original)
        stats = get_chat_send_stats(chat_content_original, chat_content_send)

        if stats['truncated']:
            print(f"   ✂️ 聊天记录：原始 {stats['original_len']} 字；发送 {stats['send_len']} 字；已截断 {stats['removed']} 字")
        else:
            print(f"   📏 聊天记录：发送 {stats['send_len']} 字（未截断）")

        # 创建分析提示词
        prompt = create_analysis_prompt(contact_name, chat_content_send)

        # 调用API进行分析（带重试机制）
        print("   📡 分析中...", end="", flush=True)
        analysis_result = call_modelscope_api_with_retry(prompt, api_config, max_retries=3, retry_delay=2)

        if analysis_result:
            # 解析分析结果
            parsed_result = parse_analysis_result(analysis_result)

            # 添加客户昵称
            parsed_result['客户昵称'] = contact_name

            # 获取要写入的CSV文件路径
            output_csv_file = get_output_csv_file(base_path)
            # 立即保存到CSV
            append_result_to_csv(parsed_result, output_csv_file)

            print(" ✅")
            logging.info(f"文件 {file_name} 分析成功")

            # 删除聊天记录文件
            try:
                os.remove(file_path)
                print(f"   🗑️ 已删除聊天记录文件: {file_name}")
                logging.info(f"已删除聊天记录文件: {file_name}")
            except Exception as e:
                print(f"   ❌ 删除聊天记录文件失败: {e}")
                logging.error(f"删除聊天记录文件 {file_name} 失败: {e}")

            processed_files += 1
        else:
            print(" ❌")
            logging.error(f"文件 {file_name} 分析失败，已达到最大重试次数")

            # 显示详细错误信息并等待用户手动处理
            print("\n" + "="*60)
            print("❌ API调用失败，已达到最大重试次数")
            print("   请检查以下可能的问题：")
            print("   1. API密钥是否正确配置")
            print("   2. 网络连接是否正常")
            print("   3. API服务是否可用")
            print("   4. 账户余额是否充足")
            print("="*60)
            print("\n💡 建议：")
            print("   1. 检查config2.ini文件中的API配置")
            print("   2. 确认网络连接正常")
            print("   3. 查看logs目录下的日志文件获取详细信息")
            print("\n🛑 程序已暂停，等待用户处理...")
            
            # 等待用户输入
            user_input = input("输入任意值继续处理下一个文件，或输入'quit'退出程序: ").strip().lower()
            
            if user_input == 'quit':
                logging.info("用户选择退出程序")
                print("\n👋 程序已退出")
                return processed_files

            # 继续处理，但记录失败结果
            default_result = {
                '客户昵称': contact_name,
                '是否咨询业务': '否',
                '订单状态': '售前',
                '咨询业务': 'API调用失败',
                '关心问题': 'API调用失败',
                '当前态度': 'API调用失败',
                '聊天总结': 'API调用失败，已达到最大重试次数',
                '跟踪话术1': 'API调用失败',
                '跟踪话术2': 'API调用失败',
                '跟踪话术3': 'API调用失败'
            }
            # 即使失败也保存记录
            output_csv_file = get_output_csv_file(base_path)
            append_result_to_csv(default_result, output_csv_file)
            logging.warning(f"文件 {file_name} 已标记为失败状态，并记录到 {os.path.basename(output_csv_file)}")

            # 失败后不删除文件，以便后续检查
            print(f"   ⚠️ 文件 {file_name} 分析失败，将保留文件以便检查。")

    return processed_files

def get_output_csv_file(base_path, max_rows=200):
    """获取当前应写入的CSV文件路径，如果超过行数限制则创建新文件"""
    file_index = 0
    while True:
        file_name = f"汇总结果_{file_index}.csv" if file_index > 0 else "汇总结果.csv"
        output_file_path = os.path.join(base_path, file_name)

        if not os.path.exists(output_file_path):
            initialize_csv_file(output_file_path)
            return output_file_path
        else:
            try:
                with open(output_file_path, 'r', encoding='utf-8-sig') as csvfile:
                    # -1 for header
                    row_count = sum(1 for row in csv.reader(csvfile)) - 1
                if row_count < max_rows:
                    return output_file_path
                else:
                    file_index += 1
            except Exception as e:
                logging.error(f"检查CSV文件 {output_file_path} 行数时出错: {e}")
                # 如果出错，则尝试下一个文件
                file_index += 1

def initialize_csv_file(output_file_path):
    """初始化CSV文件，如果不存在则创建并写入表头"""
    fieldnames = ['客户昵称', '是否咨询业务', '订单状态', '咨询业务', '关心问题', '当前态度', '聊天总结', '跟踪话术1', '跟踪话术2', '跟踪话术3']
    try:
        if not os.path.exists(output_file_path):
            with open(output_file_path, 'w', newline='', encoding='utf-8-sig') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
            print(f"📄 创建新的CSV文件: {output_file_path}")
            logging.info(f"创建新的CSV文件: {output_file_path}")
    except Exception as e:
        print(f"❌ 初始化CSV文件失败: {e}")
        logging.error(f"初始化CSV文件失败: {e}")

def append_result_to_csv(result, output_file_path):
    """将单个分析结果追加到CSV文件，并检查重复"""
    fieldnames = ['客户昵称', '是否咨询业务', '订单状态', '咨询业务', '关心问题', '当前态度', '聊天总结', '跟踪话术1', '跟踪话术2', '跟踪话术3']
    
    try:
        # 检查文件是否存在，如果不存在则创建并写入表头
        file_exists = os.path.exists(output_file_path)
        if not file_exists:
            initialize_csv_file(output_file_path)

        # 读取已存在的结果，避免重复添加相同的联系人
        existing_contacts = set()
        try:
            with open(output_file_path, 'r', newline='', encoding='utf-8-sig') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    if '客户昵称' in row and row['客户昵称']:
                        existing_contacts.add(row['客户昵称'])
        except FileNotFoundError:
            pass # 文件不存在是正常情况，上面已经处理
        except Exception as e:
            print(f"读取现有CSV文件失败: {e}")
            logging.error(f"读取现有CSV文件 {output_file_path} 失败: {e}")

        contact_name = result.get('客户昵称', '')
        if contact_name in existing_contacts:
            print(f"   ⏭️ 跳过已存在的联系人: {contact_name}")
            logging.info(f"跳过已存在的联系人: {contact_name}")
            return

        with open(output_file_path, 'a', newline='', encoding='utf-8-sig') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow(result)
        
        print(f"   💾 结果已追加到: {os.path.basename(output_file_path)}")
        logging.info(f"联系人 {contact_name} 的结果已追加到 {os.path.basename(output_file_path)}")

    except Exception as e:
        print(f"❌ 追加CSV文件失败: {e}")
        logging.error(f"追加结果到CSV文件 {output_file_path} 失败: {e}")


def save_results_to_csv(results, output_file="汇总结果.csv"):
    """将分析结果保存到CSV文件"""
    # 支持PyInstaller打包后的路径处理
    if getattr(sys, 'frozen', False):
        # 打包后的可执行文件路径
        base_path = os.path.dirname(sys.executable)
    else:
        # 开发环境下的脚本路径
        base_path = os.path.dirname(os.path.abspath(__file__))
    
    output_file_path = os.path.join(base_path, output_file)
    fieldnames = ['客户昵称', '是否咨询业务', '订单状态', '咨询业务', '关心问题', '当前态度', '聊天总结', '跟踪话术1', '跟踪话术2', '跟踪话术3']

    try:
        # 检查文件是否已存在
        file_exists = os.path.exists(output_file_path)

        # 读取已存在的结果，避免重复添加相同的联系人
        existing_contacts = set()
        if file_exists:
            try:
                with open(output_file_path, 'r', newline='', encoding='utf-8-sig') as csvfile:
                    reader = csv.DictReader(csvfile)
                    for row in reader:
                        if '客户昵称' in row:
                            existing_contacts.add(row['客户昵称'])
            except Exception as e:
                print(f"读取现有CSV文件失败: {e}")

        with open(output_file_path, 'a' if file_exists else 'w', newline='', encoding='utf-8-sig') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            # 如果文件不存在，写入表头
            if not file_exists:
                writer.writeheader()
                print(f"📄 创建新的CSV文件: {output_file_path}")

            # 只写入新的联系人数据（避免重复）
            new_results_count = 0
            for result in results:
                contact_name = result.get('客户昵称', '')
                if contact_name not in existing_contacts:
                    writer.writerow(result)
                    existing_contacts.add(contact_name)
                    new_results_count += 1
                else:
                    print(f"   ⏭️ 跳过已存在的联系人: {contact_name}")

        print(f"\n💾 分析结果已保存到: {output_file_path}")
        if file_exists:
            print(f"   ➕ 新增 {new_results_count} 个联系人，总计 {len(existing_contacts)} 个联系人")
        else:
            print(f"   📊 共分析了 {len(results)} 个联系人")

    except Exception as e:
        print(f"❌ 保存CSV文件失败: {e}")

def wait_for_keypress():
    """等待用户按任意键退出"""
    print("\n" + "="*50)
    print("程序执行完成，按任意键退出...")
    try:
        # Windows系统
        if os.name == 'nt':
            import msvcrt
            msvcrt.getch()
        else:
            # Linux/Mac系统
            import sys
            import tty
            import termios
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            try:
                tty.setraw(sys.stdin.fileno())
                sys.stdin.read(1)
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    except Exception:
        # 如果上述方法失败，使用简单的input方法
        input("按回车键退出...")

def main():
    """主函数"""
    # 设置日志
    log_file = setup_logging()
    
    print("🚀 微信聊天记录分析程序")
    print("📊 使用大模型聊天记录，评估客户业务意向程度")
    print("📦 v0.2.2整合包版本，作者:@苏青淮")
    print("="*50)
    print(f"📝 日志文件: {log_file}")
    
    # 显示程序运行路径信息
    if getattr(sys, 'frozen', False):
        print(f"📁 程序运行路径: {os.path.dirname(sys.executable)}")
        logging.info(f"程序运行路径: {os.path.dirname(sys.executable)}")
    else:
        print(f"📁 程序运行路径: {os.path.dirname(os.path.abspath(__file__))}")
        logging.info(f"程序运行路径: {os.path.dirname(os.path.abspath(__file__))}")

    # 加载配置
    print("\n🔧 正在加载配置...")
    logging.info("开始加载配置")
    api_config = load_config()

    if api_config['api_key'] == 'your_api_key':
        error_msg = "配置错误: 请先在config2.ini文件中配置您的API密钥"
        print(f"\n❌ {error_msg}")
        config_path = os.path.join(os.path.dirname(sys.executable) if getattr(sys, 'frozen', False) else os.path.dirname(os.path.abspath(__file__)), "config2.ini")
        print(f"   配置文件路径: {config_path}")
        logging.error(error_msg)
        wait_for_keypress()
        return

    logging.info("配置加载成功")

    # 处理所有聊天文件
    print("\n📋 开始分析聊天记录...")
    logging.info("开始分析聊天记录")
    processed_count = process_all_chat_files(api_config)

    if processed_count > 0:
        print(f"\n✅ 分析完成！共处理了 {processed_count} 个新文件。")
        logging.info(f"分析完成，共处理 {processed_count} 个新文件")
    else:
        print("\n⚠️ 没有找到新的或可分析的聊天记录")
        logging.warning("没有找到新的或可分析的聊天记录")

    # 等待用户按任意键退出
    logging.info("程序执行完成")
    wait_for_keypress()

if __name__ == "__main__":
    main()