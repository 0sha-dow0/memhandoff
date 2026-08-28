# Agent A to Agent B

This offline example turns a small Agent A transcript into a portable `.ctx`,
checks it against its local archive, and compiles provider-neutral context for
Agent B. It makes no model call and needs no API key.

From the repository root, with `open-context` installed:

```bash
./examples/cross-agent-handoff/run.sh
```

The script refuses to overwrite an existing `memhandoff-demo` directory. Pass a
different destination when you want to run it again:

```bash
./examples/cross-agent-handoff/run.sh /tmp/my-memhandoff-demo
```

Open `memhandoff-demo/agent-b-context.json` and paste its `text` value into a
different agent. Ask: “What should you do next, and what must you not change?”
The receiving agent should recover all of these details:

- write UTF-16LE with a byte-order mark
- use batches of exactly 500 rows
- expect a final batch of 418 from 12,418 rows
- never use the shared NFS mount
- do not change deployment configuration
- Kafka was considered and rejected
- port 9443 belongs to the metrics sidecar

If it loses or changes one of them, that is useful evidence. Open a
[Context Loss Report](https://github.com/0sha-dow0/memhandoff/issues/new?template=context-loss.yml)
with the smallest sanitized reproduction you can share.

This example deliberately runs without `--extract`. Its six spoken turns fit in
the recent-context window, so they are carried verbatim. On a long session,
older facts need extraction or retrieval and may be lost; the project does not
claim otherwise.
