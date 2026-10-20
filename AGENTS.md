# AGENTS.md — LLM_from_scratch

<!-- architect:rules start — shared rules, edited in The_Architect/rules/AGENTS.rules.md -->

## Contribution rules

1. **This file is not an agentic dump.** Its rules are entered only by a
   human: in writing, or by dictation assisted by an agent.
2. **Follow a proper software development life cycle.** Every feature and
   every bug has proper requirements, a proper specification and a proper
   implementation, and everything lands by merge request. A merge request
   may be reviewed by an agent, depending on its severity and the size of
   the change, but not necessarily.
3. **Requirements and specifications may be written by the planning agent
   or by a human.**
4. **Humans or agents write the code, after proper adherence to the
   software development life cycle.** Keep the build and the tests honest;
   write the docs.
5. **Requirements and specification are different things, and everyone,
   human or agent, keeps them apart.** A requirement says what the system
   must do, in general terms, and names no technology. A specification is
   the specific instance of the requirements that led to the current
   program; it names the technology. A technology is named in a
   requirement only when the system is defined by it: if the system must
   run on Linux, Linux is in the requirements; if it happens to be written
   in Python, Python is in the specification.

## Communication

**If something can be explained with a picture, a graph, a workflow or an
animated GIF, do it before writing.**

**Write in ASD-STE100, Simplified Technical English, the aerospace
standard.** At least 80% of the text follows it; connective words may be
added if needed.

<!-- architect:rules end -->

## Objective

LLM_from_scratch is part of EVA. Its objective is to learn the internals of a
large language model by building one from scratch, from the first principle
to a model that runs.

The language is C++, standard 23 and above. The project is defined by this
choice.

## Source references

- Stanford's course on language modeling from scratch (CS336).
- Andrej Karpathy's work on building language models from scratch.
