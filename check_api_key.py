#!/usr/bin/env python3
"""检查 OpenAI API Key 是否可用"""

import os
from dotenv import load_dotenv

def check_api_key():
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    if not api_key:
        print("❌ OPENAI_API_KEY 未设置")
        return False

    print(f"API Key: {api_key[:20]}...{api_key[-4:]}")
    print(f"Model: {model}")

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)

        print("\n正在测试 API 连接...")
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Say 'OK' if you can hear me."}],
            max_tokens=10,
        )

        result = response.choices[0].message.content
        print(f"✅ API 响应: {result}")
        print(f"✅ API Key 有效！")
        return True

    except Exception as e:
        print(f"❌ API 调用失败: {e}")
        return False

if __name__ == "__main__":
    check_api_key()
