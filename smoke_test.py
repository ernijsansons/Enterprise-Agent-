#!/usr/bin/env python3
"""Orchestrator smoke test."""
import os
os.environ['OPENAI_API_KEY'] = 'stub-key'

from src.agent_orchestrator import AgentOrchestrator

def main():
    try:
        result = AgentOrchestrator().run_mode('coding', 'build_from_docs:: Hello World Check')
        confidence = result.get('confidence', 0.0)
        print(f'confidence {confidence}')
        return confidence >= 0.8
    except Exception as e:
        print(f'Error: {e}')
        return False

if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)




