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
    
    # Create a copy of the provider data for template rendering
    template_vars = provider_data.copy()
    
    # Add timestamp
    template_vars['timestamp'] = datetime.now().isoformat()
    
    # Rename some fields to match template expectations
    template_vars['Rate_Limit_Cost_Info'] = template_vars.get('Rate Limit/Cost Info', '')
    template_vars['Template'] = template_vars.get('Template', '')
    
    # Process models field
    models_list = process_models_field(
        template_vars['Model(s)'],
        template_vars['Base_URL'],
        template_vars['APIKey'],
        template_vars['AuthMode']
    )
    
    # Create replacement dictionary with all variables
    replacements = {}
    
    # Add basic variables
    for key, value in template_vars.items():
        replacements[key] = str(value) if value is not None else ''
    
    # Add processed variables
    replacements['models_endpoint'] = template_vars['Model(s)'] if template_vars['Model(s)'].startswith('http') else ''
    replacements['models_list'] = models_list
    replacements['first_model'] = models_list[0] if models_list else ''
    
    # Helper function to evaluate expressions
    def evaluate_expression(expr):
        """Evaluate a template expression and return the result"""
        expr = expr.strip()
        
        # Handle if/else expressions
        if ' if ' in expr and ' else ' in expr:
            parts = expr.split(' if ')
            true_value = parts[0].strip()
            condition_else = parts[1].split(' else ')
            condition = condition_else[0].strip()
            false_value = condition_else[1].strip()
            
            # Remove quotes
            if true_value.startswith("'") and true_value.endswith("'"):
                true_value = true_value[1:-1]
            if false_value.startswith("'") and false_value.endswith("'"):
                false_value = false_value[1:-1]
            
            # Evaluate condition
            if condition in replacements:
                condition_value = replacements[condition]
                if condition == 'APIKey':
                    return true_value if condition_value else false_value
                elif condition == 'AuthMode':
                    return true_value if condition_value.lower() == 'bearer' else false_value
                elif condition == 'Name':
                    if 'Image' in condition_value:
                        return 'image'
                    elif 'Text' in condition_value:
                        return 'text'
                    else:
                        return 'mixed'
                elif condition == 'Rate_Limit_Cost_Info':
                    return true_value if condition_value else false_value
                elif condition == 'Template':
                    return true_value if condition_value else 'default'
                elif condition_value:
                    return true_value
                else:
                    return false_value
            else:
                return false_value
        
        # Handle split expressions
        elif '.split(' in expr:
            parts = expr.split('.')
            var_name = parts[0].strip()
            if var_name in replacements and replacements[var_name]:
                if '|' in replacements[var_name]:
                    return str([item.strip() for item in replacements[var_name].split('|') if item.strip()])
                else:
                    return f'["{replacements[var_name]}"]'
            return '[]'
        
        # Handle string concatenation
        elif '+' in expr:
            parts = [p.strip() for p in expr.split('+')]
            result = ''
            for part in parts:
                if part.startswith("'") and part.endswith("'"):
                    part = part[1:-1]
                result += replacements.get(part, '')
            return result
        
        # Handle contains expressions
        elif ' in ' in expr:
            parts = expr.split(' in ')
            search_term = parts[0].strip()
            if search_term.startswith("'") and search_term.endswith("'"):
                search_term = search_term[1:-1]
            var_name = parts[1].strip()
            if var_name in replacements and search_term in replacements[var_name]:
                return 'true'
            else:
                return 'false'
        
        # Handle simple variable
        else:
            return replacements.get(expr, '')
    
    # First, handle all template expressions
    def replace_template(match):
        expr = match.group(1).strip()
        return evaluate_expression(expr)
    
    # Handle all template expressions
    template_pattern = r'\{\{\s*([^}]+)\s*\}\}'
    rendered_content = re.sub(template_pattern, replace_template, template_content)
    
    return rendered_content

def main():
    """Main function to generate YAML files using template"""
    csv_file = 'g4f_providers_example.csv'
    template_file = 'yaml_template.txt'
    output_dir = 'request templates'
    
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
        # Skip header row
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