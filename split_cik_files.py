#!/usr/bin/env python3
"""
Split dual_class_output.json into two files:
1. dual_class_output.json - only companies with CIK
2. dual_class_output_nocik.json - only companies without CIK
"""

import json

def split_cik_files():
    # Load the original data
    with open('dual_class_output.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    companies_with_cik = []
    companies_without_cik = []
    
    for company in data['companies']:
        cik = company.get('cik', '')
        if cik and cik.strip():
            companies_with_cik.append(company)
        else:
            companies_without_cik.append(company)
    
    # Create file for companies with CIK
    data_with_cik = {
        'as_of': data['as_of'],
        'total_companies': len(companies_with_cik),
        'companies_with_cik': len(companies_with_cik),
        'companies': companies_with_cik
    }
    
    # Create file for companies without CIK
    data_without_cik = {
        'as_of': data['as_of'],
        'total_companies': len(companies_without_cik),
        'companies_without_cik': len(companies_without_cik),
        'companies': companies_without_cik
    }
    
    # Write the files
    with open('dual_class_output.json', 'w', encoding='utf-8') as f:
        json.dump(data_with_cik, f, indent=2, ensure_ascii=False)
    
    with open('dual_class_output_nocik.json', 'w', encoding='utf-8') as f:
        json.dump(data_without_cik, f, indent=2, ensure_ascii=False)
    
    print(f"Split complete:")
    print(f"  Companies with CIK: {len(companies_with_cik)} -> dual_class_output.json")
    print(f"  Companies without CIK: {len(companies_without_cik)} -> dual_class_output_nocik.json")

if __name__ == '__main__':
    split_cik_files()
