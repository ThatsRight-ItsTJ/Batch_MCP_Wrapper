#!/usr/bin/env python3
import csv
import re

# Read the CSV data for ImageRouter
with open('g4f_providers_example.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row['Name'] == 'ImageRouter':
            print("ImageRouter CSV data:")
            for key, value in row.items():
                print(f"  '{key}': '{value}'")
            
            # Test template replacement
            template_vars = row.copy()
            
            # Test simple variable replacement
            test_expr = "{{ APIKey }}"
            pattern = r'\{\{\s*([^}]+)\s*\}\}'
            match = re.search(pattern, test_expr)
            if match:
                var_name = match.group(1).strip()
                print(f"\nTemplate expression: {test_expr}")
                print(f"Variable name extracted: '{var_name}'")
                print(f"Value from CSV: '{template_vars.get(var_name, 'NOT FOUND')}'")
            
            # Test complex expression
            complex_expr = "{{ 'Bearer ' + APIKey if AuthMode == 'bearer' and APIKey else '' }}"
            match = re.search(pattern, complex_expr)
            if match:
                expr = match.group(1).strip()
                print(f"\nComplex expression: {complex_expr}")
                print(f"Expression content: '{expr}'")
                
                # Manual evaluation
                api_key = template_vars.get('APIKey', '')
                auth_mode = template_vars.get('AuthMode', '')
                print(f"APIKey value: '{api_key}'")
                print(f"AuthMode value: '{auth_mode}'")
                
                if auth_mode == 'bearer' and api_key:
                    result = f'Bearer {api_key}'
                else:
                    result = ''
                print(f"Expected result: '{result}'")
            
            break