"""
AI Agents / Tool Use — Demo
================================
Implements a ReAct (Reason + Act) agent loop from scratch.
The agent receives a question, decides which tool to call,
observes the result, reasons about it, and repeats until it
has the answer.

No external LLM API needed — uses a rule-based "brain" to
demonstrate the architecture. The pattern is identical to what
LangChain / OpenAI function-calling do under the hood.
"""

import os
import re
import math
import json
import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "output.txt")

lines = []
def log(msg=""):
    print(msg)
    lines.append(str(msg))


# ═══════════════════════════════════════════════════════════
# TOOLS — functions the agent can call
# ═══════════════════════════════════════════════════════════

TOOL_REGISTRY = {}

def tool(name, description):
    """Decorator to register a tool."""
    def decorator(func):
        TOOL_REGISTRY[name] = {
            "function": func,
            "description": description,
            "name": name,
        }
        return func
    return decorator


@tool("calculator", "Evaluate a math expression. Input: a math expression string like '2 + 3 * 4'")
def calculator(expression: str) -> str:
    try:
        allowed = set("0123456789+-*/.() ")
        if not all(c in allowed for c in expression):
            return f"Error: invalid characters in expression"
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error: {e}"


@tool("lookup", "Look up a fact from a knowledge base. Input: a topic string")
def lookup(topic: str) -> str:
    kb = {
        "python": "Python is a programming language created by Guido van Rossum in 1991.",
        "earth radius": "Earth's mean radius is 6,371 km.",
        "earth circumference": "Earth's circumference is approximately 40,075 km.",
        "speed of light": "The speed of light is 299,792,458 m/s.",
        "population of france": "France has a population of approximately 68 million.",
        "boiling point of water": "Water boils at 100°C (212°F) at standard atmospheric pressure.",
        "pi": "Pi (π) is approximately 3.14159265358979.",
        "avogadro": "Avogadro's number is 6.022 × 10²³ mol⁻¹.",
    }
    topic_lower = topic.lower().strip()
    for key, value in kb.items():
        if key in topic_lower or topic_lower in key:
            return value
    return f"No information found for '{topic}'."


@tool("date", "Get the current date and time. Input: ignored")
def get_date(_: str = "") -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@tool("unit_convert", "Convert between units. Input: 'value from_unit to_unit' e.g. '100 km miles'")
def unit_convert(query: str) -> str:
    conversions = {
        ("km", "miles"): 0.621371,
        ("miles", "km"): 1.60934,
        ("kg", "lbs"): 2.20462,
        ("lbs", "kg"): 0.453592,
        ("celsius", "fahrenheit"): lambda x: x * 9/5 + 32,
        ("fahrenheit", "celsius"): lambda x: (x - 32) * 5/9,
        ("m", "ft"): 3.28084,
        ("ft", "m"): 0.3048,
    }
    parts = query.strip().split()
    if len(parts) != 3:
        return "Error: format should be 'value from_unit to_unit'"
    try:
        value = float(parts[0])
    except ValueError:
        return f"Error: '{parts[0]}' is not a number"
    from_u, to_u = parts[1].lower(), parts[2].lower()
    key = (from_u, to_u)
    if key not in conversions:
        return f"Error: unknown conversion {from_u} → {to_u}"
    conv = conversions[key]
    if callable(conv):
        result = conv(value)
    else:
        result = value * conv
    return f"{value} {from_u} = {result:.4f} {to_u}"


# ═══════════════════════════════════════════════════════════
# AGENT — ReAct loop
# ═══════════════════════════════════════════════════════════

class ReActAgent:
    """
    ReAct agent: Reason → Act → Observe → Repeat.

    In production, the 'think' step would be an LLM call.
    Here we use pattern matching to demonstrate the architecture.
    """

    def __init__(self, tools: dict, max_steps: int = 5):
        self.tools = tools
        self.max_steps = max_steps
        self.trace = []  # full reasoning trace

    def think(self, question: str, observations: list) -> tuple:
        """
        Decide what to do next. Returns (action, action_input) or ("finish", answer).
        In a real agent, this would be an LLM prompt like:
            "Given the question and observations so far, what tool should I use next?"
        """
        q = question.lower()

        if not observations:
            if any(w in q for w in ["calculate", "compute", "what is", "how much"]):
                numbers = re.findall(r'[\d.]+', question)
                ops = re.findall(r'[+\-*/]', question)
                if numbers and ops:
                    expr = question
                    for word in ["calculate", "compute", "what is", "how much is"]:
                        expr = expr.lower().replace(word, "")
                    expr = expr.strip().rstrip("?").strip()
                    return ("calculator", expr)

            if "convert" in q:
                match = re.search(r'([\d.]+)\s*(\w+)\s+(?:to|in)\s+(\w+)', q)
                if match:
                    return ("unit_convert", f"{match.group(1)} {match.group(2)} {match.group(3)}")

            if any(w in q for w in ["date", "time", "today", "now"]):
                return ("date", "")

            if "circumference" in q and "earth" in q:
                return ("lookup", "earth circumference")

            if "radius" in q and "earth" in q:
                return ("lookup", "earth radius")

            for topic in ["python", "speed of light", "population", "boiling point", "pi", "avogadro"]:
                if topic in q:
                    return ("lookup", topic)

            if "radius" in q or "circumference" in q:
                return ("lookup", "earth radius")

            return ("lookup", question.strip("?").strip())

        last_obs = observations[-1]

        if "radius" in q and "circumference" in q and len(observations) == 1:
            radius_match = re.search(r'([\d,]+)\s*km', last_obs)
            if radius_match:
                radius = float(radius_match.group(1).replace(",", ""))
                return ("calculator", f"2 * 3.14159 * {radius}")

        if "convert" in q and len(observations) == 1:
            numbers = re.findall(r'[\d,.]+', last_obs)
            if numbers:
                val = numbers[0].replace(",", "")
                if "miles" in q:
                    return ("unit_convert", f"{val} km miles")
                if "feet" in q or "ft" in q:
                    return ("unit_convert", f"{val} m ft")

        return ("finish", last_obs)

    def run(self, question: str) -> str:
        """Execute the full ReAct loop."""
        self.trace = []
        observations = []

        log(f"\n  Question: {question}")
        self.trace.append({"step": "question", "content": question})

        for step in range(1, self.max_steps + 1):
            action, action_input = self.think(question, observations)
            log(f"  Step {step} — Thought: I should use '{action}' with input '{action_input}'")
            self.trace.append({"step": step, "action": action, "input": action_input})

            if action == "finish":
                log(f"  Step {step} — Final Answer: {action_input}")
                self.trace.append({"step": step, "type": "finish", "answer": action_input})
                return action_input

            if action not in self.tools:
                obs = f"Error: unknown tool '{action}'"
            else:
                obs = self.tools[action]["function"](action_input)

            log(f"  Step {step} — Observation: {obs}")
            self.trace.append({"step": step, "type": "observation", "content": obs})
            observations.append(obs)

        final = observations[-1] if observations else "Could not determine answer."
        log(f"  (Max steps reached) Final Answer: {final}")
        return final


# ═══════════════════════════════════════════════════════════
# DEMO
# ═══════════════════════════════════════════════════════════

def run_demo():
    log("AI AGENTS / TOOL USE — DEMO")
    log("=" * 60)

    # Show available tools
    log("\n  Available tools:")
    for name, info in TOOL_REGISTRY.items():
        log(f"    • {name}: {info['description']}")

    agent = ReActAgent(TOOL_REGISTRY, max_steps=5)

    questions = [
        "What is the speed of light?",
        "Calculate 2.5 * 3.14159 * 6371",
        "What date is it today?",
        "Convert 100 km to miles",
        "What is the radius of Earth, and what is its circumference?",
        "Convert 212 fahrenheit to celsius",
    ]

    log(f"\n{'=' * 60}")
    log("RUNNING AGENT ON QUESTIONS")
    log("=" * 60)

    results = []
    for q in questions:
        log(f"\n{'─' * 50}")
        answer = agent.run(q)
        results.append((q, answer, len(agent.trace)))

    log(f"\n{'=' * 60}")
    log("SUMMARY")
    log("=" * 60)
    log(f"\n  {'Question':<50} {'Steps':>6}")
    log(f"  {'-' * 58}")
    for q, ans, steps in results:
        log(f"  {q:<50} {steps:>6}")

    log(f"\n{'=' * 60}")
    log("AGENT ARCHITECTURE — ReAct PATTERN")
    log("=" * 60)
    log("""
  ┌─────────┐
  │ Question│
  └────┬────┘
       ▼
  ┌─────────┐     ┌──────────┐     ┌──────────┐
  │ THINK   │────▶│   ACT    │────▶│ OBSERVE  │
  │ (LLM    │     │ (call    │     │ (get     │
  │  decides │◀───│  tool)   │     │  result) │
  │  next   │     └──────────┘     └────┬─────┘
  │  action)│◀──────────────────────────┘
  └────┬────┘
       │ action = "finish"
       ▼
  ┌─────────┐
  │ ANSWER  │
  └─────────┘

  Real implementations:
    • OpenAI function calling — tools defined as JSON schemas
    • LangChain agents      — tool decorator + LLM router
    • AutoGPT / CrewAI      — multi-agent collaboration
    • MCP (Model Context Protocol) — standardized tool interface

  Key insight: the agent is just a LOOP — think, act, observe, repeat.
  The "intelligence" comes from the LLM's ability to reason about which
  tool to call and how to interpret results.
""")

    with open(OUTPUT_PATH, "w") as f:
        f.write("\n".join(lines))
    log(f"\n→ Output saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    run_demo()
