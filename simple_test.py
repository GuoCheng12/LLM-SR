#!/usr/bin/env python3
"""
Simple test for JSON formatting changes
"""
import json
import tempfile
import os

def test_json_formatting():
    """Test the improved JSON formatting"""
    
    # Sample data structure similar to what profile.py generates
    sample_data = {
        'sample_order': 0,
        'function': 'def equation(x, v, params):\n    return params[0] * x + params[1] * v + 0.5',
        'score': 8.95,
        'params': [1.2, -0.5, 0.1],
        'top_3_scores': [
            {
                'score': 10.0,
                'sample_order': 0,
                'function': 'def equation(x, v, params):\n    return params[0] * x + params[1] * v',
                'params': [1.0, -0.5, 0.0]
            },
            {
                'score': 9.5,
                'sample_order': 1,
                'function': 'def equation(x, v, params):\n    return params[0] * x + params[1] * v + params[2]',
                'params': [1.1, -0.4, 0.1]
            },
            {
                'score': 8.95,
                'sample_order': 2,
                'function': 'def equation(x, v, params):\n    return params[0] * x + params[1] * v + 0.5',
                'params': [1.2, -0.5, 0.1]
            }
        ]
    }
    
    # Test old format (compact)
    print("--- OLD FORMAT (compact) ---")
    old_json = json.dumps(sample_data)
    print(old_json[:200] + "...")
    
    # Test new format (with indentation)
    print("\n--- NEW FORMAT (indented) ---")
    new_json = json.dumps(sample_data, indent=4, ensure_ascii=False)
    print(new_json[:400] + "...")
    
    # Save to file to test actual file writing
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
        json.dump(sample_data, f, indent=4, ensure_ascii=False)
        temp_file = f.name
    
    # Read back and verify
    with open(temp_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    print("\n--- FILE CONTENT ---")
    print(content[:500] + "...")
    
    # Clean up
    os.unlink(temp_file)
    
    # Verify improvements
    print("\n--- VERIFICATION ---")
    if '"sample_order": 0' in content and '    "function":' in content:
        print("✓ JSON has proper indentation")
    else:
        print("✗ JSON indentation may have issues")
    
    if 'top_3_scores' in content and '"function":' in content:
        print("✓ Top_3_scores contains function code")
    else:
        print("✗ Top_3_scores structure may have issues")
    
    # Count lines to show readability improvement
    old_lines = old_json.count('\n')
    new_lines = new_json.count('\n')
    print(f"✓ Old format: {old_lines} lines, New format: {new_lines} lines")

if __name__ == "__main__":
    test_json_formatting()