#!/usr/bin/env python3
"""
Clean Schema and Add Umbrella Concepts
=======================================
"""

import json
import sys
from typing import Any, Iterable

def add_umbrella_concepts(schema):
    """Add umbrella_concepts to schema."""
    
    # Update lists to hold umbrella concepts
    for cat in schema.get('entities_schema', []):
        cat_name = cat['category']
        
        if cat_name not in ["Primary_measures", "Comparison", "Recommendation", "Clinical_indication"]:
            for subcat_name, subcat_data in cat.get('subcategories', {}).items():
                
                if subcat_name in ["overall_intepretation", "other_impressions", "Other_findings"]:
                    continue
                
                subcat_name = subcat_name.replace('_', ' ').capitalize()  # Normalize subcategory name (to match concept format)
                
                if subcat_name.lower() not in subcat_data.get("concepts", []).lower():
                    subcat_data["concepts"].append(subcat_name)  # Include subcategory name itself
    
    return schema

def dumps_inline_arrays(obj: Any, indent: int = 2, ensure_ascii: bool = False,
                        wrap_threshold: int = 6, wrap_width: int = 5) -> str:
    """
    Pretty-print JSON with:
      - Objects/lists indented by `indent`
      - Leaf lists (all scalars) inline
      - BUT if a leaf list has more than `wrap_threshold` items,
        wrap it across lines with up to `wrap_width` items per line.
    """
    def is_scalar(x):
        return isinstance(x, (str, int, float, bool)) or x is None

    def chunk(it: Iterable[Any], n: int):
        it = list(it)
        for i in range(0, len(it), n):
            yield it[i:i+n]

    def _dump(x, level=0):
        pad = " " * (indent * level)
        pad_in = " " * (indent * (level + 1))

        if isinstance(x, dict):
            if not x:
                return "{}"
            items = []
            for k, v in x.items():  # preserves insertion order (Py3.7+)
                items.append(f'{pad_in}{json.dumps(k, ensure_ascii=ensure_ascii)}: {_dump(v, level+1)}')
            return "{\n" + ",\n".join(items) + f"\n{pad}" + "}"

        if isinstance(x, list):
            if not x:
                return "[]"
            # Leaf list of scalars
            if all(is_scalar(e) for e in x):
                if len(x) > wrap_threshold:
                    # wrap lines with up to wrap_width items each
                    lines = []
                    for block in chunk(x, wrap_width):
                        inner = ", ".join(json.dumps(e, ensure_ascii=ensure_ascii) for e in block)
                        lines.append(f"{pad_in}{inner}")
                    return "[\n" + ",\n".join(lines) + f"\n{pad}]"
                # inline if not exceeding threshold
                inner = ", ".join(json.dumps(e, ensure_ascii=ensure_ascii) for e in x)
                return "[" + inner + "]"
            # Nested/non-leaf lists: pretty-print elements vertically
            parts = [f"{pad_in}{_dump(e, level+1)}" for e in x]
            return "[\n" + ",\n".join(parts) + f"\n{pad}]"

        # Scalars
        return json.dumps(x, ensure_ascii=ensure_ascii)

    return _dump(obj, 0)

def main():
    if len(sys.argv) < 3:
        print("Usage: python schema_add_umbrella_concepts.py input.json output.json")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    
    # Load schema
    with open(input_path, 'r') as f:
        schema = json.load(f)
    
    print(f"Loaded schema from {input_path}")
    
    # Add umbrella concepts
    schema = add_umbrella_concepts(schema)
    
    # Save
    with open(output_path, 'w') as f:
        f.write(dumps_inline_arrays(schema))  # For pretty inline arrays
        # json.dump(schema, f, indent=2)
    
    print(f"✓ Saved to {output_path}")
    
if __name__ == '__main__':
    main()