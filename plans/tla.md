NEVER WRITE CODE WITHOUT TLA+ FIRST FULL ANALYSIS, VALIDATION BY TLC AND HUMAN - YOU ARE ONLY ALLOWED TO INTRODUCE CODE ONCE YOU HAVE TLA+ PROVING THE CODE IS CORRECT, THEN PASSING TLC MODEL CHECK, THEN A YES FROM THE HUMAN AFTER TELLING IT WITH NATURAL LANGUAGE WHAT IT IS AND FINALLY HAVING FULL COVERAGE TESTS WRITTEN TO ENSURE THE GOALS WILL BE REACHED.

## Project Organization Requirements

0. ALWAYS CREATE BRANCH-SPECIFIC FOLDERS

For every feature/branch, create a dedicated folder structure in the `plans` directory:

```
plans/
├── feature-[INTEGER_ID]-[FEATURE_NAME]/
│   ├── [feature-name].md                    # Main implementation plan
│   ├── tla-validation-summary.md           # TLA+ validation results
│   ├── tla-validation-final.md            # Final validation summary
│   └── tla/                               # TLA+ specification files
│       ├── [FeatureName].tla              # Main TLA+ specification
│       ├── [FeatureName].cfg              # TLA+ configuration
│       ├── [FeatureName]_TTrace_*.tla     # Trace files (if any)
│       ├── states/                        # TLC state exploration files
│       └── tla2tools.jar                  # TLA+ tools JAR
└── tla.md                                 # This guide (root level)
```

FOLDER NAMING CONVENTION

**Format**: `feature-[INTEGER_ID]-[FEATURE_NAME]`

**Example**: `feature-2027331687-auto-generate-chat-title`

Where:

- `INTEGER_ID`: The numeric identifier from the git branch
- `FEATURE_NAME`: Descriptive name using kebab-case


1. **Create Feature Branch**: `git checkout -b feature/[INTEGER_ID]/[feature-name]`

2. **Create Plans Folder**:

   ```bash
   cd plans
   mkdir feature-[INTEGER_ID]-[feature-name]
   cd feature-[INTEGER_ID]-[feature-name]
   mkdir tla
   ```

3. Create a "logic" folder where you will create a full predicate logic circuit intended to be used as a TLA+ specification. Use Horn clauses to represent the logic that will be written in the `tla` folder thereafter following the full logic circuit that fully describes the feature.

4. **ALWAYS WRITE THE TLA+ PLAN** in the feature folder BEFORE beginning coding, THEN FOLLOW IT PRECISELY.

5. **VALIDATE TLA+ BEFORE BEGINNING**. Example:

   ```bash
   cd /Users/capanema/Work/title/Monorepo/plans/feature-[INTEGER_ID]-[feature-name]/tla
   java -jar ../tla/tla2tools.jar -workers auto -config [FeatureName].cfg -coverage 1 [FeatureName].tla
   ```

6. Once TLC model checker is validated, clarify with user by translating the validated model into natural language if it is correct.

7. Proceed with code implementation.

7.1. Very important is that the code we produce is flake8 and MyPy compliant.
7.2. All the tests we write must reflect the TLA+ validated features so that the code follows what has been validated and not only what has been implemented.

8. Write full coverage tests for the code to be executed so that we can assure the solution produced follows the TLA+ validations. DO NOT WRITE TESTS THAT VALIDATE THE IMPLMENTED CODE, WRITE TESTS THAT ENSURE THE IMPLEMENTED CODE FOLLOWS THE TLA+ VALIDATIONS.

   - Use `pytest` for writing tests.
   - Ensure all tests are in the `tests/` directory.
   - Use `pydantic_ai` to create agents that can reason about the data and make decisions based on the TLA+ specifications.
   - Test the agents with the same TLA+ specifications to ensure they can reason correctly about the system.
   - Ensure compliance with the TLA+ specifications by running the agents against the TLA+ model checker.

9. After that write unit tests that ensure implemented code works correctly without errors.

   - Use `pytest` for writing tests.
   - Ensure all tests are in the `tests/` directory.
   - Use `pydantic_ai` to create agents that can reason about the data and make decisions based on the TLA+ specifications.
   - Test the agents with the same TLA+ specifications to ensure they can reason correctly about the system.
   - Ensure compliance with the TLA+ specifications by running the agents against the TLA+ model checker.

10. Once the unit tests pass with higher than at least 80% coverage, write and test integration tests to ensure the system works as expected.

- Use `pytest` for writing tests.
- Ensure all tests are in the `tests/` directory.
- Use `pydantic_ai` to create agents that can reason about the data and make decisions based on the TLA+ specifications.
- Test the agents with the same TLA+ specifications to ensure they can reason correctly about the system.
- Ensure compliance with the TLA+ specifications by running the agents against the TLA+ model checker.
- Ensure the integration tests pass with higher than at least 80% coverage.

11. DEFINITION OF DONE: Once all is done and pass tests we can say it is a success. Do a full report

- Write a `tla-validation-summary.md` file in the feature folder summarizing the TLA+ validation results.
- Write a `tla-validation-final.md` file in the feature folder summarizing the final validation results.
- Write a `README.md` file in the feature folder explaining how to run the TLA+ model checker and how to run the tests.
- Write a `CHANGELOG.md` file in the feature folder summarizing the changes made in this feature.

## APPENDIX 1: HOW TO WRITE PLANS IN PREDICATE LOGIC

# 📘 Predicate Logic — Full Explanation

## 1. Overview

**Predicate Logic**, also known as **First-Order Logic (FOL)**, is a formal system used to express facts, rules, and relationships in a structured and interpretable way. It is built on the foundation of **predicates**, **terms**, **variables**, **constants**, and **logical connectives**. Predicate logic is more expressive than propositional logic, enabling reasoning about the internal structure of statements and the objects they refer to.

It serves as the backbone for rule-based reasoning systems, expert systems, and symbolic inference engines.

---

## 2. Core Components

### 🔹 Predicates

A **predicate** represents a property of an object or a relationship between multiple objects.

- **Syntax**: `Predicate(arg1, arg2, ..., argN)`
- **Examples**:
  - `is_mammal(dog)`
  - `parent(alice, bob)`
  - `greater_than(7, 3)`

Predicates begin with lowercase letters and express declarative statements about the world.

### 🔹 Terms

A **term** is a building block of logical statements and can be:

- A **constant**: a specific object (e.g., `alice`, `3`)
- A **variable**: a placeholder (e.g., `X`, `Y`)
- A **compound term**: another predicate or function result

### 🔹 Constants

Constants refer to specific, named objects or entities. They are fixed and do not vary.

- Examples: `earth`, `sun`, `42`, `john_doe`

### 🔹 Variables

Variables are generic placeholders for any object in the domain.

- Examples: `X`, `Y`, `Person`, `Animal`
- Variables are **universally** or **existentially** quantified (see Quantifiers below).

### 🔹 Functions

Functions are terms that return other terms and may be nested.

- Example: `mother_of(father_of(bob))`

---

## 3. Logical Structures

### 🔹 Atoms

An **atom** is a predicate applied to the right number of terms.

- Example: `likes(alice, chocolate)`

### 🔹 Literals

A **literal** is an atom or its negation.

- Examples:
  - Positive literal: `hungry(dog)`
  - Negative literal: `¬hungry(dog)` or `not hungry(dog)`

---

## 4. Logical Connectives

Used to build complex expressions from atomic predicates:

| Connective | Symbol     | Meaning                 |
|------------|------------|-------------------------|
| Negation   | ¬ or `not` | Logical NOT             |
| Conjunction | ∧ or `and` | Logical AND             |
| Disjunction | ∨ or `or`  | Logical OR              |
| Implication | → or `:-`  | IF ... THEN (rule form) |
| Biconditional | ↔ or `<->` | Logical equivalence     |

---

## 5. Quantifiers

Quantifiers specify the scope of variables:

- **Universal Quantifier** (∀): means the statement applies to **all** instances.
  - Example: `∀X. human(X) → mortal(X)` → "All humans are mortal"

- **Existential Quantifier** (∃): means the statement applies to **at least one** instance.
  - Example: `∃X. loves(john, X)` → "There exists someone John loves"

In practical systems, rules use **implicitly universal quantification** over free variables.

---

## 6. Rules and Inference

A **rule** expresses a conditional relationship: *if the body holds, then the head is true*.

- **Syntax**:  
  `Head :- Body1, Body2, ..., BodyN.`  
  Meaning: "Head is true if all Body conditions are true"

- **Example**:  
  `grandparent(X, Z) :- parent(X, Y), parent(Y, Z).`  
  → "X is a grandparent of Z if X is a parent of Y and Y is a parent of Z."

The left-hand side is called the **head**, the right-hand side is the **body** (conjunction of conditions).

---

## 7. Facts

Facts are unconditional atomic statements declared as true.

- Syntax: `predicate(constant1, constant2, ..., constantN).`
- Example:
  - `parent(mary, john).`
  - `animal(dog).`

They form the **knowledge base** upon which inference is performed.

---

## 8. Queries

Queries ask whether a certain statement can be logically inferred from the facts and rules.

- Syntax: `?- predicate(arguments).`
- Example:
  - `?- parent(mary, john).` → True
  - `?- grandparent(mary, Z).` → Returns all Z such that `grandparent(mary, Z)` holds

---

## 9. Unification

**Unification** is the process of matching two predicates by finding variable substitutions that make them identical.

- Example:
  - Query: `?- parent(mary, X).`
  - Fact: `parent(mary, john).`
  - Result: `X = john`

---

## 10. Resolution and Inference Engine

A logic engine deduces new facts by applying **resolution**: chaining rules and facts to infer conclusions.

- Uses **backward chaining** (goal-driven): starts from a query and looks for supporting facts/rules
- Or **forward chaining** (data-driven): starts from facts and applies rules to derive new facts

---

## 11. Negation-as-Failure

If a predicate cannot be proven to be true from the facts and rules, it is assumed false. This is known as **negation as failure**, a form of **non-monotonic reasoning**.

- Example:

  ```prolog
  not_believed(X) :- not believes(X).
  ```

This is useful in practical systems with incomplete knowledge.

---

## 12. Example Knowledge Base

```prolog
% Facts
parent(john, mary).
parent(mary, alice).
parent(mary, bob).

% Rules
grandparent(X, Z) :- parent(X, Y), parent(Y, Z).
sibling(X, Y) :- parent(Z, X), parent(Z, Y), X \= Y.

% Query
?- grandparent(john, alice).  % True
?- sibling(alice, bob).       % True
```

---

## 13. Best Practices

- Use **short, meaningful predicate names** that reflect relationships
- Prefer **lowercase for constants**, **uppercase for variables**
- Avoid deep nesting unless necessary
- Break complex logic into **multiple simple rules**

---

## 14. Use Case in Instruction Interpretation

When interpreting natural language instructions, sentences are transformed into:

- **Facts**: Declarative knowledge
- **Rules**: Generalizations or conditionals
- **Queries**: Intentions or goals

For example:

- Instruction: “John is Alice’s father.”  
  → `parent(john, alice).`

- Instruction: “If someone is a parent of someone who is a parent, then they are a grandparent.”  
  → `grandparent(X, Z) :- parent(X, Y), parent(Y, Z).`

---

This system forms the foundation for any reasoning agent that must interpret, act upon, or respond to human logic-like instructions.

## APPENDIX 2: TLA+ Complete Guide

### TLA+ Complete Guide

*A hands-on, end-to-end manual for engineers who want to adopt **TLA+** as a standard pre-coding step. Written for day-to-day use, it moves from first principles to advanced patterns and integrates seamlessly with modern software-delivery pipelines.*

---

## Table of Contents - TLA+ Guide

1. @Why TLA+?  
2. @Core Concepts  
   2.1. Mathematical Foundations  
   2.2. Specs, Modules & Expressions  
3. @Tools & Installation  
   3.1. TLA+ Toolbox  
   3.2. TLC Model Checker  
   3.3. PlusCal Translator  
   3.4. Apalache & TLAPS  
4. @Hello, World – Your First Spec  
5. @The Authoring Workflow  
   5.1. Step-by-Step Checklist  
   5.2. Coding Standards for Specs  
6. @Specifying Real-World Systems  
   6.1. State Machines & Protocols  
   6.2. Data-Centric Services  
   6.3. Distributed Algorithms  
7. @Common Patterns & Idioms  
8. @Property-Driven Design  
9. @Integrating TLA+ into SDLC  
10. @Cheat Sheet  
11. @Troubleshooting & FAQ  
12. @Further Resources

---

<a name="why"></a>

## 1  Why TLA+?

- **Eliminate design bugs early.** Formal specs surface edge-cases far cheaper than tests after coding.  
- **Executable math.** Combines set theory & temporal logic—powerful yet precise.  
- **Scale-invariant.** Same language expresses single-thread invariants or complex distributed protocols.  
- **Tooling.** *TLC* model checker explores state spaces; *Apalache* handles large systems symbolically; *TLAPS* provides machine-checked proofs.

> **Mandate:** Every critical module **must ship with**  
>
> 1) a top-level spec, 2) at least one safety/property check, and 3) a TLC run in CI.

---

<a name="core"></a>

## 2  Core Concepts

### 2.1  Mathematical Foundations

| Concept | Meaning |
|---------|---------|
| **Set** | Collection of values, `{1,2,3}` or `{x ∈ Nat : x < 10}` |
| **Function** | Map `f == [x ∈ S |-> e]` |
| **Record** | `[field1 ↦ val1, field2 ↦ val2]` |
| **Tuple / Sequence** | `<<a,b,c>>` / `<<>>` |
| **Predicate** | Boolean expression over variables |
| **Temporal Operator** | `□` (always), `◇` (eventually), `→` (leads-to) |
| **Next-state relation** | `Next` describes legal transitions |

### 2.2  Specs, Modules & Expressions

```tla
---- MODULE Example ----
EXTENDS Naturals, Sequences
VARIABLES x, y

Init == /\\ x = 0 /\\ y = 0
Next == \\/ x' = x + 1 /\\ y' = y
        \\/ x' = x     /\\ y' = y + 1

Spec == Init /\\ □[Next]_<<x,y>>
Inv  == x ≥ 0 /\\ y ≥ 0
====
```

- `Init` – initial state predicate.  
- `Next` – how variables change (`x'` = next state).  
- `Spec` – full behaviour (always apply `Next`).  
- `Inv` – invariant checked by TLC.

---

<a name="tools"></a>

## 3  Tools & Installation

### 3.1  TLA+ Toolbox (IDE)

- Java-based GUI bundling editor, TLC launchers & trace viewer.  
- **Install:** download ZIP, unzip, run `toolbox`.  
- **Headless:** `tlc2.TLC <spec>` for CI.

### 3.2  TLC Model Checker

- **Explicit-state explorer.** Generates all states reachable from `Init` using `Next`.  
- **Limits:** state explosion → mitigate with symmetry sets, stuttering steps, parameter bounds.

### 3.3  PlusCal Translator

- *Algorithmic* pseudo-code → TLA+. Easier for algorithmic thinking.

```tla
--algorithm EuclidGCD {
  variables a \in Nat, b \in Nat;
  begin
    while (b /= 0) {
      a, b := b, a % b;
    }
  end while;
  return a;
}
```

### 3.4  Apalache & TLAPS

- **Apalache** – SMT-based symbolic model checker; handles unbounded ints, large params.  
- **TLAPS** – Proof assistant; write hierarchical proofs for full generality.

---

<a name="hello"></a>

## 4  Hello, World – Your First Spec

1. **Create module** `Counter.tla`.  
2. `VARIABLE counter`.  
3. `Init == counter = 0`.  
4. `Next == counter' = counter + 1`.  
5. `TypeInv == counter ∈ Nat`.  
6. Run TLC with bound `counter < 10` (`CONSTANTS Max ← 9`).  
7. Examine trace → refine.  
8. Commit `Counter.tla`, `Counter.cfg`, and TLC log.

---

<a name="workflow"></a>

## 5  The Authoring Workflow

### 5.1  Step-by-Step Checklist

1. **Scope:** define boundaries & abstractions.  
2. **Identify state variables** (minimal but sufficient).  
3. **Write `Init`** – valid starting conditions.  
4. **Write `Next`** – disjunctive actions.  
5. **State constraints** (`TypeInv`).  
6. **Specify properties** – invariants & liveness.  
7. **Model check** small yet representative parameters.  
8. **Debug traces**, iterate until no violations.  
9. **(Optional) Prove** with TLAPS.  
10. **Link to ticket / PR** – spec is acceptance criterion.

### 5.2  Coding Standards for Specs

- UpperCamelCase module names, lowerCamel actions.  
- One **module per component**; reuse via `EXTENDS`.  
- Keep formulas short; factor with definitions.  
- Comment every operator; include references.

---

<a name="real"></a>

## 6  Specifying Real-World Systems

### 6.1  State Machines & Protocols

- Example: **Leader Election**  
  - Variables: `state`, `term`, `votedFor`.  
  - *Safety*: at most one leader per term.  
  - *Liveness*: eventually elect a leader.

### 6.2  Data-Centric Services

- Key-value store with linearizability property.  
- Use *history variables* to capture API calls & responses.

### 6.3  Distributed Algorithms

- Two-phase commit, Paxos, Raft.  
- Patterns: quorum intersection, majority sets, conflict-free increments.

---

<a name="patterns"></a>

## 7  Common Patterns & Idioms

| Pattern | Purpose | Sketch |
|---------|---------|--------|
| **Stuttering** | Allow no-op steps | `Next == ... ∨ UNCHANGED vars` |
| **History Variable** | Record operation log | `history' = history ⧺ <<op>>` |
| **Symmetry** | Reduce state space | Declare symmetry sets in TLC cfg |
| **Fairness** | Ensure progress | `Spec == Init ∧ □[Next]_vars ∧ WF_vars(Next)` |

---

<a name="properties"></a>

## 8  Property-Driven Design

- **Safety** – nothing bad happens (invariants).  
- **Liveness** – something good eventually happens (`□◇ condition`).  
- **Refinement** – show implementation spec refines abstract spec (`SpecImpl ⇒ SpecAbs`).

---

<a name="integration"></a>

## 9  Integrating TLA+ into SDLC

| Stage | Action |
|-------|--------|
| **Design Review** | Spec & TLC run attached to design doc. |
| **CI Pipeline** | `java -jar tla2tools.jar tlc2.TLC -deadlock -coverage 100 <spec>`; fail build on violation. |
| **Code Review** | PR must reference spec commit; reviewer cross-checks invariants vs implementation. |
| **Regression** | Store traces; add spec tests when fixing a bug. |

---

<a name="cheatsheet"></a>

## 10  Cheat Sheet

```text
∀ x ∈ S : P       \\A
∃ x ∈ S : P       \\E
UNCHANGED vars     ≜ vars' = vars
[Next]_vars        ≜ Next ∨ UNCHANGED vars
WF_vars(A)         Weak fairness
SF_vars(A)         Strong fairness
\\* comment         Block comment
```

Concept | Syntax | Example
---|---|---
Set comprehension | `{e : p}` | `{x + 1 : x ∈ 1..10}`
Function update | `[f EXCEPT ![k] = v]` | `f' = [f EXCEPT ![id] = newVal]`
Sequence append | `seq ⧺ <<e>>` | `history' = history ⧺ <<op>>`

---

<a name="troubleshooting"></a>

## 11  Troubleshooting & FAQ

**Q: TLC reports “Deadlock found”.**  
A: Ensure `Next` always enables a transition; include stuttering step.

**Q: State explosion!**  
A: Use smaller constants, symmetry, abstract data types, or Apalache.

**Q: How do I model timeouts?**  
A: Represent logical timers as variables; use fairness to guarantee expiry.

---

<a name="resources"></a>

## 12  Further Resources

- *Specifying Systems* – Leslie Lamport (free book)  
- TLA+ Video Course – lamport.azurewebsites.net/video  
- **Apalache** docs – <https://apalache.informal.systems>  
- Hillel Wayne’s *Learn TLA+*  
- Community chat – `#tla-plus` on Fosstodon & chat.stackexchange.com

---

### License & Authors

Feel free to copy, adapt, and embed this guide within your engineering handbook.  
Written by **ChatGPT** for **Eduardo Capanema** (2025).
