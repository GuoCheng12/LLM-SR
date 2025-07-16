#!/usr/bin/env python3
"""
Test script for profile.py changes
"""
import os
import sys
import tempfile
import json
from llmsr import profile
from llmsr import code_manipulation

def test_profile_changes():
    """Test the improved JSON formatting and top_3_scores functionality"""
    
    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        # Initialize profiler
        profiler = profile.Profiler(log_dir=temp_dir, max_log_nums=5)
        
        # Create sample functions to test
        sample_functions = []
        for i in range(5):
            # Create a mock function
            func = code_manipulation.Function(
                name="equation",
                args=["x", "v", "params"],
                body=f"    return params[0] * x + params[1] * v + {i}\n",
                docstring=f"Test function {i}"
            )
            
            # Add required attributes
            func.global_sample_nums = i
            func.score = float(10 - i)  # Decreasing scores: 10, 9, 8, 7, 6
            func.params = [1.0 + i * 0.1, -0.5 + i * 0.1, 0.1 + i * 0.1]
            func.sample_time = 0.1 + i * 0.01
            func.evaluate_time = 0.2 + i * 0.02
            
            sample_functions.append(func)
        
        # Register functions with profiler
        print("Registering sample functions...")
        for func in sample_functions:
            profiler.register_function(func)
        
        # Check JSON files
        samples_dir = os.path.join(temp_dir, 'samples')
        json_files = [f for f in os.listdir(samples_dir) if f.endswith('.json')]
        
        print(f"\nGenerated {len(json_files)} JSON files:")
        for json_file in sorted(json_files):
            file_path = os.path.join(samples_dir, json_file)
            with open(file_path, 'r', encoding='utf-8') as f:
                content = json.load(f)
            
            print(f"\n--- {json_file} ---")
            print(f"Sample Order: {content['sample_order']}")
            print(f"Score: {content['score']}")
            print(f"Function (first 100 chars): {content['function'][:100]}...")
            
            # Check top_3_scores format
            if content['top_3_scores']:
                print(f"Top 3 scores count: {len(content['top_3_scores'])}")
                for i, entry in enumerate(content['top_3_scores']):
                    print(f"  Rank {i+1}: Score={entry['score']:.2f}, Sample={entry['sample_order']}")
                    print(f"    Function preview: {entry['function'][:50]}...")
        
        # Test JSON formatting
        sample_file = os.path.join(samples_dir, 'samples_0.json')
        with open(sample_file, 'r', encoding='utf-8') as f:
            raw_content = f.read()
        
        print(f"\n--- JSON Formatting Test ---")
        print("First 300 characters of samples_0.json:")
        print(raw_content[:300])
        print("...")
        
        # Verify indentation
        if '"sample_order": 0' in raw_content and '    "function":' in raw_content:
            print("✓ JSON formatting with proper indentation detected")
        else:
            print("✗ JSON formatting may have issues")
        
        # Verify top_3_scores structure
        with open(sample_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if data['top_3_scores'] and 'function' in data['top_3_scores'][0]:
            print("✓ Top_3_scores contains function code")
        else:
            print("✗ Top_3_scores may not contain function code")
        
        print("\n--- Test completed ---")

if __name__ == "__main__":
    test_profile_changes()