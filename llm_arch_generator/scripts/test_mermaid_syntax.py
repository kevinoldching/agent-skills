#!/usr/bin/env python3
"""Verify mermaid syntax in SKILL.md"""

import re

def check_mermaid_syntax():
    with open('/home/omni/code_repos/agent-skills/llm_arch_generator/SKILL.md', 'r') as f:
        content = f.read()

    lines = content.split('\n')
    in_block = False
    block_num = 0
    issues = []

    for i, line in enumerate(lines, 1):
        if line.strip().startswith('```mermaid'):
            in_block = True
            block_num += 1
            block_start = i
            block_lines = []
        elif in_block and line.strip() == '```':
            block_content = '\n'.join(block_lines)

            if block_num == 1:
                if 'graph LR' not in block_content:
                    issues.append(f'Block 1 (line {block_start}): Missing graph LR')
            elif block_num == 2:
                if 'graph TD' not in block_content:
                    issues.append(f'Block 2 (line {block_start}): Missing graph TD')

            in_block = False
            block_lines = []
        elif in_block:
            block_lines.append(line)

    return issues

if __name__ == '__main__':
    issues = check_mermaid_syntax()
    if issues:
        print('DIAGRAM ISSUES:')
        for issue in issues:
            print(f'  - {issue}')
    else:
        print('All mermaid diagrams have proper graph direction')