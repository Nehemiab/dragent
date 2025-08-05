from langgraph.checkpoint.sqlite import SqliteSaver

with SqliteSaver.from_conn_string("checkpoints.db") as saver:
    for c in saver.list(None):
        tid = c[0]["configurable"]["thread_id"]
        print(tid)