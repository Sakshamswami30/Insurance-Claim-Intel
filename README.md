# Insurance Claim Document Intelligence

A system that reads insurance claim documents, extracts structured fields using LLMs, validates them, and retrieves relevant policy clauses — reducing manual review time.

## What it does

- Ingests insurance claim documents (PDFs / images)
- Extracts structured fields using an LLM with schema validation
- Flags high-risk claims using a classical ML classifier
- Retrieves relevant policy clauses via hybrid retrieval
- Generates a grounded reviewer summary with citations

## Status

Work in progress — being built step by step.

## Setup

```bash
py -3.11 -m venv venv
venv\Scripts\activate
pip install -r requirements.txt