system_prompt = "Generate a complete Python code to solve math problem."

few_shot_prompt = ""

question_format = """Problem: {problem}

Note:
- The final answer must be reported **modulo 1000**.
- If the computed solution is **greater than 1000**, return only the last three digits (ignoring any leading zeros).
  - Example: 65521 → **521**, 1009 → **9**
- If the solution is **negative**, report its positive equivalent modulo 1000.
  - Example: -900 → **100**
- If the problem explicitly requires computing the result **modulo m**, first compute the result **modulo m** (ensuring 0 ≤ a' < m), and then take the final result **modulo 1000**.
  - Example:
    - If computing **2025 mod 999**, the result is **27**.
    - If computing **2025 mod 1013**, the result is **12**.
- Some problems may involve basic mathematical operations such as rounding, floor functions, or square roots.
  - Example: ⌊1002–√⌋ = **141**
"""