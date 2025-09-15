#!/usr/bin/env python3
"""
Script to generate YAML configuration files for each provider in g4f_providers_example.csv
using the yaml_template.txt template with logic to pull models from endpoints or use delimited lists
"""

import csv
import yaml
import os
import requests
import json
from pathlib import Path
from datetime import datetime
import re

def clean_filename(name):
    """Clean provider name for safe filename usage"""
    # Replace spaces and special characters with underscores
    return name.replace(' ', '_').replace('-', '_').replace('.', '_')

def fetch_models_from_endpoint(endpoint_url, api_key=None, auth_mode=None):
    """Fetch models from a models endpoint URL"""
    try:
        headers = {'Content-Type': 'application/json'}
        
        # Add authentication headers if needed
        if auth_mode and auth_mode.lower() == 'bearer' and api_key:
            headers['Authorization'] = f'Bearer {api_key}'
        
        response = requests.get(endpoint_url, headers=headers, timeout=30)
        response.raise_for_status()
        
        # Try to parse JSON response
        data = response.json()
        
        # Extract models from different possible response formats
        models = []
        if isinstance(data, dict):
            # Common patterns in API responses
            if 'data' in data and isinstance(data['data'], list):
                # OpenAI format: {"data": [{"id": "model1"}, {"id": "model2"}]}
                models = [item.get('id', '') for item in data['data'] if isinstance(item, dict)]
            elif 'models' in data and isinstance(data['models'], list):
                # OpenRouter format: {"models": [{"id": "model1"}, {"id": "model2"}]}
                models = [item.get('id', '') for item in data['models'] if isinstance(item, dict)]
            elif 'items' in data and isinstance(data['items'], list):
                # Alternative format: {"items": [{"id": "model1"}, {"id": "model2"}]}
                models = [item.get('id', '') for item in data['items'] if isinstance(item, dict)]
            elif 'model' in data and isinstance(data['model'], list):
                # Another format: {"model": ["model1", "model2"]}
                models = [str(item) for item in data['model'] if item]
            else:
                # Try to find any string that looks like a model ID
                models = [str(key) for key in data.keys() if re.match(r'^[a-zA-Z0-9\-_\.]+$', str(key))]
        elif isinstance(data, list):
            # Direct list of models
            models = [str(item) for item in data if item]
        
        # Filter out empty strings and return unique models
        return list(set([model.strip() for model in models if model.strip()]))
        
    except Exception as e:
        print(f"Warning: Failed to fetch models from {endpoint_url}: {e}")
        return []

def process_models_field(models_field, base_url, api_key=None, auth_mode=None):
    """Process models field - fetch from endpoint if URL, otherwise parse delimited list"""
    if not models_field:
        return []
    
    # Check if it's a URL (starts with http)
    if models_field.startswith('http'):
        # It's an endpoint URL, fetch models from it
        return fetch_models_from_endpoint(models_field, api_key, auth_mode)
    else:
        # It's a delimited list, parse it
        models = []
        if '|' in models_field:
            # Pipe-separated list
            models = [model.strip() for model in models_field.split('|') if model.strip()]
        elif models_field:
            # Single model
            models = [models_field.strip()]
        
        return models

def render_template(template_content, provider_data):
    """Render the template with provider data using direct string replacement"""
    import re
    
    # Clean the provider data - handle empty strings and None values
    clean_data = {}
    for key, value in provider_data.items():
        if value is None or value == '':
            clean_data[key] = ''
        else:
            clean_data[key] = str(value).strip()
    
    # Add timestamp
    clean_data['timestamp'] = datetime.now().isoformat()
    
    # Process models field to get actual model list
    models_list = process_models_field(
        clean_data.get('Model(s)', ''),
        clean_data.get('Base_URL', ''),
        clean_data.get('APIKey', ''),
        clean_data.get('AuthMode', '')
    )
    
    # Add processed model data
    clean_data['models_list'] = models_list
    clean_data['first_model'] = models_list[0] if models_list else 'gpt-3.5-turbo'
    clean_data['models_endpoint'] = clean_data.get('Model(s)', '') if clean_data.get('Model(s)', '').startswith('http') else ''
    
    def evaluate_complex_expression(expr, data):
        """Evaluate complex template expressions"""
        expr = expr.strip()
        
        # Handle if/else expressions
        if ' if ' in expr and ' else ' in expr:
            # Parse: "true_value if condition else false_value"
            parts = expr.split(' if ')
            true_part = parts[0].strip()
            condition_else = parts[1].split(' else ')
            condition = condition_else[0].strip()
            false_part = condition_else[1].strip()
            
            # Remove quotes from true/false values
            if true_part.startswith("'") and true_part.endswith("'"):
                true_part = true_part[1:-1]
            if false_part.startswith("'") and false_part.endswith("'"):
                false_part = false_part[1:-1]
            
            # Evaluate the condition
            condition_result = False
            
            if ' and ' in condition:
                # Handle AND conditions
                and_parts = [p.strip() for p in condition.split(' and ')]
                condition_result = True
                for part in and_parts:
                    if '==' in part:
                        var_name, expected = [p.strip().strip("'\"") for p in part.split('==')]
                        actual = data.get(var_name, '')
                        if actual.lower() != expected.lower():
                            condition_result = False
                            break
                    elif '!=' in part:
                        var_name, expected = [p.strip().strip("'\"") for p in part.split('!=')]
                        actual = data.get(var_name, '')
                        if actual.lower() == expected.lower():
                            condition_result = False
                            break
                    elif part in data:
                        # Simple variable check
                        if not data[part]:
                            condition_result = False
                            break
            elif '==' in condition:
                # Simple equality check
                var_name, expected = [p.strip().strip("'\"") for p in condition.split('==')]
                actual = data.get(var_name, '')
                condition_result = actual.lower() == expected.lower()
            elif '!=' in condition:
                # Simple inequality check
                var_name, expected = [p.strip().strip("'\"") for p in condition.split('!=')]
                actual = data.get(var_name, '')
                condition_result = actual.lower() != expected.lower()
            elif ".startswith('http')" in condition:
                # URL check
                var_name = condition.split('.')[0]
                var_value = data.get(var_name, '')
                condition_result = var_value.startswith('http')
            elif ' in ' in condition:
                # Contains check
                parts = condition.split(' in ')
                search_term = parts[0].strip().strip("'\"")
                var_name = parts[1].strip()
                var_value = data.get(var_name, '')
                condition_result = search_term in var_value
            elif condition in data:
                # Simple variable existence check
                condition_result = bool(data[condition])
            
            # Handle string concatenation in true_part
            if '+' in true_part:
                concat_parts = [p.strip() for p in true_part.split('+')]
                result = ''
                for part in concat_parts:
                    if part.startswith("'") and part.endswith("'"):
                        result += part[1:-1]
                    else:
                        result += data.get(part, '')
                return result if condition_result else false_part
            else:
                return true_part if condition_result else false_part
        
        # Handle split operations - THIS IS THE KEY FIX
        elif '.split(' in expr:
            if 'Model(s).split' in expr:
                models_field = data.get('Model(s)', '')
                if '|' in models_field and not models_field.startswith('http'):
                    # Parse pipe-separated models and return as proper array format
                    model_list = [model.strip() for model in models_field.split('|') if model.strip()]
                    # Return as YAML-compatible array format
                    return str(model_list).replace("'", '"')
                elif models_field and not models_field.startswith('http'):
                    return f'["{models_field}"]'
                else:
                    return '[]'
        
        # Handle array access - THIS IS ANOTHER KEY FIX
        elif '[0]' in expr and 'Model(s)' in expr:
            models_field = data.get('Model(s)', '')
            if '|' in models_field and not models_field.startswith('http'):
                first_model = models_field.split('|')[0].strip()
                return first_model if first_model else 'gpt-3.5-turbo'
            elif models_field and not models_field.startswith('http'):
                return models_field
            else:
                return 'gpt-3.5-turbo'
        
        # Handle string concatenation
        elif '+' in expr:
            parts = [p.strip() for p in expr.split('+')]
            result = ''
            for part in parts:
                if part.startswith("'") and part.endswith("'"):
                    result += part[1:-1]
                else:
                    result += data.get(part, '')
            return result
        
        # Handle URL construction - THIS FIXES THE Base_URL ISSUE
        elif 'Base_URL' in expr and ('chat/completions' in expr or '/health' in expr):
            base_url = data.get('Base_URL', '')
            if 'chat/completions' in expr and 'not in' in expr:
                return base_url + '/chat/completions' if 'chat/completions' not in base_url else base_url
            elif '/health' in expr and 'not in' in expr:
                return base_url + '/health' if '/health' not in base_url else base_url
            return base_url
        
        # Simple variable lookup
        elif expr in data:
            return data[expr]
        
        # Return empty string for unhandled expressions
        return ''
    
    # Replace all template expressions
    def replace_template_vars(match):
        expr = match.group(1).strip()
        
        # First try simple variable replacement
        if expr in clean_data:
            return clean_data[expr]
        
        # Then try complex expression evaluation
        result = evaluate_complex_expression(expr, clean_data)
        return str(result)
    
    # Apply template replacements
    template_pattern = r'\{\{\s*([^}]+)\s*\}\}'
    rendered_content = re.sub(template_pattern, replace_template_vars, template_content)
    
    return rendered_content

def main():
    """Main function to generate YAML files using template"""
    csv_file = 'g4f_providers_example.csv'
    template_file = 'yaml_template.txt'
    output_dir = 'request_templates'
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(exist_ok=True)
    
    # Read template file
    try:
        with open(template_file, 'r', encoding='utf-8') as file:
            template_content = file.read()
    except FileNotFoundError:
        print(f"Error: Template file '{template_file}' not found")
        return
    
    # Read CSV file
    with open(csv_file, 'r', newline='', encoding='utf-8') as file:
        csv_reader = csv.DictReader(file)
        providers = list(csv_reader)
    
    # Generate YAML file for each provider
    for provider in providers:
        # Generate filename
        clean_name = clean_filename(provider['Name'])
        yaml_filename = f"{clean_name}.yaml"
        yaml_path = os.path.join(output_dir, yaml_filename)
        
        # Render template with provider data
        rendered_content = render_template(template_content, provider)
        
        # Write rendered content to YAML file
        with open(yaml_path, 'w', encoding='utf-8') as yaml_file:
            yaml_file.write(rendered_content)
        
        print(f"Generated: {yaml_path}")
    
    print(f"\nSuccessfully generated {len(providers)} YAML configuration files in '{output_dir}' directory")

if __name__ == "__main__":
    main()