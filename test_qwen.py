import os
from dotenv import load_dotenv
from openai import OpenAI

# 1. 加载 .env 文件中的环境变量
load_dotenv()

# 从环境变量中获取 API Key 和 Base URL
api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_BASE_URL")

if not api_key:
    raise ValueError("未找到 OPENAI_API_KEY，请检查 .env 文件。")

# 2. 初始化 OpenAI 客户端
client = OpenAI(
    api_key=api_key,
    base_url=base_url
)

def generate_article_with_qwen(prompt: str) -> str:
    """使用 qwen3-max 模型生成文章"""
    print(f"正在使用 qwen3-max 生成文章，基于提示词：\n{prompt}\n")
    print("-" * 40)
    
    try:
        response = client.chat.completions.create(
            model="qwen3-max",
            messages=[
                {"role": "system", "content": "你是一个专业的AI写作助手，擅长撰写结构清晰、内容详实的文章。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=2048
        )
        
        # 提取生成的文本内容
        article = response.choices[0].message.content
        return article

    except Exception as e:
        print(f"API 请求出错: {e}")
        return ""

if __name__ == "__main__":
    # 设定要生成文章的提示词
    user_prompt = "请写一篇关于人工智能在未来医疗领域应用的短文（约500字），重点探讨诊断和新药研发。"
    
    # 3. 调用模型生成文章
    result = generate_article_with_qwen(user_prompt)
    
    if result:
        print("【生成文章结果】：\n")
        print(result)
        
        # 可以选择把生成的文章保存到本地文件
        with open("qwen_article_output.txt", "w", encoding="utf-8") as f:
            f.write(result)
        print("\n" + "-" * 40)
        print("文章已成功保存至 qwen_article_output.txt")
