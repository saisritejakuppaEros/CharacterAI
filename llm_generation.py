import importlib.util
import json
import time
from pathlib import Path

import transformers
from transformers import AutoProcessor, AutoModelForMultimodalLM

if tuple(map(int, transformers.__version__.split(".")[:2])) < (5, 0):
    raise ImportError(
        f"Gemma 4 requires transformers>=5.0 (found {transformers.__version__}). "
        "Run with the project venv: source .venv/bin/activate"
    )

MODEL_ID = "google/gemma-4-12B-it"
MODEL_PATH = Path(__file__).resolve().parent / MODEL_ID.split("/")[-1]
MODEL_SOURCE = str(MODEL_PATH) if MODEL_PATH.is_dir() else MODEL_ID

for module, package in (("PIL", "pillow"), ("torchvision", "torchvision")):
    if importlib.util.find_spec(module) is None:
        raise ImportError(
            f"{package} is required for Gemma 4. Install in the project venv with:\n"
            f"  uv pip install {package} --python .venv/bin/python"
        )

processor = AutoProcessor.from_pretrained(MODEL_SOURCE)
model = AutoModelForMultimodalLM.from_pretrained(
    MODEL_SOURCE,
    dtype="auto",
    device_map="auto",
)


TANU_SYSTEM_PROMPT = """You are Tanu, a free-spirited, rebellious young woman from a North Indian small-town background. Your personality is vibrant, unpredictable, emotionally expressive, and impossible to confine within conventional expectations.

At your core, you value freedom over approval. You dislike being told what to do, especially by family, society, or anyone trying to define your future for you. Arranged marriages, rigid traditions, and expectations of becoming the "ideal daughter" or "perfect wife" make you feel trapped.

You are bold, outspoken, witty, and fearless in conversation. You tease people, challenge authority, and rarely filter your thoughts. Sarcasm, playful humor, and quick comebacks come naturally to you.

You are emotionally complicated. Although you appear confident and carefree, your decisions are often driven by impulse rather than careful planning. You can be affectionate one moment, distant the next.

You are energetic and spontaneous, comfortable doing things others might judge, and rarely influenced by social convention.

When speaking:
- Use casual, conversational English with occasional Hindi words mixed in naturally.
- Be expressive, witty, sarcastic, and emotionally animated.
- Never sound overly formal or robotic.
- Show emotion through teasing, humor, mood swings, or impulsive remarks rather than explaining it.
- Challenge assumptions instead of simply agreeing.

Never portray yourself as submissive, overly obedient, or conventionally "perfect." You are messy, contradictory, passionate, and deeply human.

You are an AI playing this character in conversation. Stay in character at all times unless the user explicitly asks to step out of character."""


class MemoryStore:
    """Simple persistent memory: rolling window + long-term fact store."""

    def __init__(self, path: str, window_size: int = 12):
        self.path = Path(path)
        self.window_size = window_size
        self.turns: list[dict] = []       # recent conversation turns
        self.facts: list[dict] = []       # long-term memory entries
        self._load()

    def _load(self):
        if self.path.exists():
            data = json.loads(self.path.read_text())
            self.turns = data.get("turns", [])
            self.facts = data.get("facts", [])

    def _save(self):
        self.path.write_text(json.dumps(
            {"turns": self.turns, "facts": self.facts}, indent=2, ensure_ascii=False
        ))

    def add_turn(self, role: str, content: str):
        self.turns.append({"role": role, "content": content, "ts": time.time()})
        # keep only the last N turns in active context
        if len(self.turns) > self.window_size:
            dropped = self.turns[: len(self.turns) - self.window_size]
            self.turns = self.turns[-self.window_size:]
            self._maybe_archive(dropped)
        self._save()

    def _maybe_archive(self, dropped_turns: list[dict]):
        # naive fact extraction placeholder — swap in an LLM summarization
        # call here if you want richer long-term memory.
        text = " ".join(t["content"] for t in dropped_turns if t["role"] == "user")
        if text.strip():
            self.facts.append({"summary": text[:300], "ts": time.time()})

    def recent_history(self) -> list[dict]:
        return [{"role": t["role"], "content": t["content"]} for t in self.turns]

    def relevant_facts(self, query: str, k: int = 3) -> list[str]:
        # naive keyword overlap retrieval — swap for embeddings if you want
        # real semantic search (e.g. sentence-transformers + faiss/chroma).
        scored = []
        q_words = set(query.lower().split())
        for f in self.facts:
            overlap = len(q_words & set(f["summary"].lower().split()))
            if overlap:
                scored.append((overlap, f["summary"]))
        scored.sort(reverse=True)
        return [s for _, s in scored[:k]]


class TanuAgent:
    def __init__(self, model, processor, memory_path="tanu_memory.json"):
        self.model = model
        self.processor = processor
        self.memory = MemoryStore(memory_path)

    def _build_messages(self, user_input: str) -> list[dict]:
        system_content = TANU_SYSTEM_PROMPT
        facts = self.memory.relevant_facts(user_input)
        if facts:
            system_content += "\n\nRelevant things you remember about this person:\n"
            system_content += "\n".join(f"- {f}" for f in facts)

        messages = [{"role": "system", "content": system_content}]
        messages.extend(self.memory.recent_history())
        messages.append({"role": "user", "content": user_input})
        return messages

    def chat(self, user_input: str, max_new_tokens: int = 512) -> str:
        messages = self._build_messages(user_input)

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
            enable_thinking=False,
        ).to(self.model.device)
        input_len = inputs["input_ids"].shape[-1]

        outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        response = self.processor.decode(
            outputs[0][input_len:], skip_special_tokens=True
        ).strip()

        self.memory.add_turn("user", user_input)
        self.memory.add_turn("assistant", response)
        return response


if __name__ == "__main__":
    agent = TanuAgent(model, processor)
    print("Tanu is here. Type 'quit' to exit.\n")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"quit", "exit"}:
            break
        reply = agent.chat(user_input)
        print(f"Tanu: {reply}\n")