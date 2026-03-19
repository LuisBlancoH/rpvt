"""Training tasks that require inference over memory, not just retrieval.

These tasks cannot be solved by retrieving a stored fact — the answer
must be derived by combining, comparing, or reasoning over multiple
pieces of stored information.

Task types:
  1. Multi-hop: A→B, B→C, ask A→C
  2. Comparison: A has X, B has Y, who has more?
  3. Constraint satisfaction: A can't do X, B can't do Y, assign tasks
  4. Temporal reasoning: A happened before B, what order?
  5. Negation: everything except X
"""

import random


def generate_multihop_task(rng):
    """A→B, B→C chain. Answer requires following the chain.

    Example:
        Memory 1: "Alice works at Nexus Corp."
        Memory 2: "Nexus Corp is located in Berlin."
        Question: "What city does Alice work in?"
        Answer: "Berlin"
    """
    names = ["Alice", "Bob", "Carol", "David", "Elena", "Frank", "Grace", "Hugo"]
    companies = ["Nexus Corp", "Helios Labs", "Quantum AI", "Atlas Systems",
                 "Omega Research", "Prism Technologies", "Vertex Dynamics"]
    cities = ["Berlin", "Tokyo", "Sydney", "Toronto", "Lagos", "Seoul", "Oslo"]

    name = rng.choice(names)
    company = rng.choice(companies)
    city = rng.choice(cities)

    passages = [
        f"{name} works at {company}. They joined as a senior researcher.",
        f"{company} is headquartered in {city}. The company was founded in 2015.",
    ]

    qa = {
        "question": f"What city does {name} work in?",
        "answer": city,
        "type": "multihop",
        "reasoning": f"{name} → {company} → {city}",
    }

    return passages, [qa]


def generate_comparison_task(rng):
    """Two entities with numeric attributes. Answer requires comparison.

    Example:
        Memory 1: "Project Alpha has a budget of 45000 dollars."
        Memory 2: "Project Beta has a budget of 72000 dollars."
        Question: "Which project has a larger budget?"
        Answer: "Project Beta"
    """
    project_names = ["Alpha", "Beta", "Gamma", "Delta", "Epsilon",
                     "Zeta", "Theta", "Omega"]
    p1, p2 = rng.sample(project_names, 2)

    budget1 = rng.randint(10, 90) * 1000
    budget2 = rng.randint(10, 90) * 1000
    while budget1 == budget2:
        budget2 = rng.randint(10, 90) * 1000

    passages = [
        f"Project {p1} has a budget of {budget1} dollars and a team of {rng.randint(3, 20)} people.",
        f"Project {p2} has a budget of {budget2} dollars and a team of {rng.randint(3, 20)} people.",
    ]

    larger = p1 if budget1 > budget2 else p2
    qa = {
        "question": f"Which project has a larger budget, {p1} or {p2}?",
        "answer": f"Project {larger}",
        "type": "comparison",
        "reasoning": f"{p1}={budget1} vs {p2}={budget2} → {larger}",
    }

    return passages, [qa]


def generate_constraint_task(rng):
    """Constraint satisfaction — who can do what given restrictions.

    Example:
        Memory 1: "Alice is allergic to peanuts."
        Memory 2: "The Thai restaurant serves pad thai with peanut sauce."
        Question: "Can Alice eat at the Thai restaurant?"
        Answer: "No"
    """
    names = ["Alice", "Bob", "Carol", "David"]
    allergies = [
        ("peanuts", "pad thai with peanut sauce", "Thai"),
        ("gluten", "fresh pasta with wheat flour", "Italian"),
        ("shellfish", "lobster bisque", "Seafood"),
        ("dairy", "cream-based risotto", "French"),
    ]

    name = rng.choice(names)
    allergy, dish, restaurant_type = rng.choice(allergies)

    passages = [
        f"{name} has a severe allergy to {allergy}. They must avoid all foods containing {allergy}.",
        f"The {restaurant_type} restaurant's signature dish is {dish}.",
    ]

    qa = {
        "question": f"Can {name} safely eat at the {restaurant_type} restaurant?",
        "answer": "No",
        "type": "constraint",
        "reasoning": f"{name} allergic to {allergy}, {dish} contains {allergy}",
    }

    return passages, [qa]


def generate_temporal_task(rng):
    """Temporal ordering from separate facts.

    Example:
        Memory 1: "Alice joined the company in 2018."
        Memory 2: "Bob joined the company in 2015."
        Question: "Who joined first?"
        Answer: "Bob"
    """
    names = rng.sample(["Alice", "Bob", "Carol", "David", "Elena", "Frank"], 2)
    year1 = rng.randint(2010, 2023)
    year2 = rng.randint(2010, 2023)
    while year1 == year2:
        year2 = rng.randint(2010, 2023)

    passages = [
        f"{names[0]} joined the company in {year1}. They started in the engineering department.",
        f"{names[1]} joined the company in {year2}. They started in the research department.",
    ]

    first = names[0] if year1 < year2 else names[1]
    qa = {
        "question": f"Who joined the company first, {names[0]} or {names[1]}?",
        "answer": first,
        "type": "temporal",
        "reasoning": f"{names[0]}={year1}, {names[1]}={year2} → {first}",
    }

    return passages, [qa]


def generate_aggregation_task(rng):
    """Aggregate information across memories.

    Example:
        Memory 1: "Team A has 5 members."
        Memory 2: "Team B has 8 members."
        Memory 3: "Team C has 3 members."
        Question: "How many total members across all teams?"
        Answer: "16"
    """
    team_names = rng.sample(["Alpha", "Beta", "Gamma", "Delta", "Epsilon"], 3)
    sizes = [rng.randint(3, 15) for _ in range(3)]

    passages = [
        f"Team {team_names[i]} has {sizes[i]} members and is led by a senior manager."
        for i in range(3)
    ]

    total = sum(sizes)
    qa = {
        "question": f"How many total members are there across teams {team_names[0]}, {team_names[1]}, and {team_names[2]}?",
        "answer": str(total),
        "type": "aggregation",
        "reasoning": f"{'+'.join(str(s) for s in sizes)} = {total}",
    }

    return passages, [qa]


TASK_GENERATORS = {
    "multihop": generate_multihop_task,
    "comparison": generate_comparison_task,
    "constraint": generate_constraint_task,
    "temporal": generate_temporal_task,
    "aggregation": generate_aggregation_task,
}


def generate_inference_tasks(rng, n_tasks, task_types=None):
    """Generate a mix of inference tasks.

    Args:
        rng: random.Random instance
        n_tasks: number of tasks to generate
        task_types: list of task type names, or None for all types

    Returns:
        list of (passages, qa_pairs) tuples
    """
    if task_types is None:
        task_types = list(TASK_GENERATORS.keys())

    generators = [TASK_GENERATORS[t] for t in task_types]
    tasks = []

    for _ in range(n_tasks):
        gen = rng.choice(generators)
        passages, qas = gen(rng)
        tasks.append((passages, qas))

    return tasks
