#!/usr/bin/env python3
"""
把 LangGraph 的 checkpoints.db 导出为人类可读的 Markdown
用法:
    python export_md.py
    python export_md.py -t thread-typhoon-1
"""

from pathlib import Path
from typing import Any, Dict

from langgraph.checkpoint.sqlite import SqliteSaver


def export_md(db_path: str, thread_id: str, out_file: str) -> None:
    """
    读取指定 thread_id 的对话并写入 Markdown
    """
    with SqliteSaver.from_conn_string(db_path) as saver:  # <-- 必须加 with
        config = {"configurable": {"thread_id": thread_id}}
        state = saver.get(config)

        if state is None:
            print(f"❌ thread_id={thread_id} 无记录")
            return

        messages = state.get("messages", [])
        lines = [f"# LangGraph 对话记录 (thread_id={thread_id})\n"]

        for idx, msg in enumerate(messages, 1):
            name = getattr(msg, "name", msg.type) or "unknown"
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            lines.append(f"## Step {idx} - {name}\n")
            lines.append(f"{content}\n\n")

        Path(out_file).write_text("".join(lines), encoding="utf-8")
        print(f"✅ 已导出：{out_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export LangGraph checkpoints → Markdown")
    parser.add_argument("-d", "--db", default="checkpoints.db", help="SQLite file")
    parser.add_argument("-t", "--thread", default="thread-typhoon-1", help="thread_id")
    args = parser.parse_args()

    out_file = Path(args.db).with_suffix(f".{args.thread}.md")
    export_md(args.db, args.thread, str(out_file))