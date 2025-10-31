#!/usr/bin/env python3
"""
ECG Schema Updater
==================
Automatically updates ECG reports when schema category/subcategory/concept names change.

Usage:
    python schema_report_updater.py --old-schema schema_v2.json --new-schema schema_v3.json --reports merged_v2.json --output updated_reports.json
"""

import json
import argparse
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from pathlib import Path


class SchemaMapper:
    """Maps old schema names to new schema names."""
    
    def __init__(self, old_schema: dict, new_schema: dict):
        self.old_schema = old_schema
        self.new_schema = new_schema
        self.mapping = self._build_mapping()
        self.unmapped_items = []
        
    def _build_mapping(self) -> Dict[Tuple[str, str, str], Tuple[str, str, str]]:
        """
        Build a mapping from (old_category, old_subcategory, old_concept) 
        to (new_category, new_subcategory, new_concept).
        """
        mapping = {}
        
        old_flat = self._flatten_schema(self.old_schema)
        new_flat = self._flatten_schema(self.new_schema)
        
        old_set = set(old_flat)
        new_set = set(new_flat)
        
        # First pass: exact matches (when names haven't changed)
        for (old_cat, old_sub, old_con) in old_flat:
            if (old_cat, old_sub, old_con) in new_flat:
                mapping[(old_cat, old_sub, old_con)] = (old_cat, old_sub, old_con)
        
        # Second pass: match concepts that moved to different subcategories
        # The pattern is: subcategory name changed but concept name stayed the same
        only_old = old_set - new_set
        only_new = new_set - old_set
        
        # Build index of concepts in new schema
        new_concepts_index = defaultdict(list)
        for new_cat, new_sub, new_con in only_new:
            new_concepts_index[new_con].append((new_cat, new_sub, new_con))
        
        for old_cat, old_sub, old_con in only_old:
            # Look for same concept in different subcategory
            if old_con in new_concepts_index:
                candidates = new_concepts_index[old_con]
                # If same category, different subcategory, it's a match
                for new_cat, new_sub, new_con in candidates:
                    if old_cat == new_cat and old_sub != new_sub:
                        mapping[(old_cat, old_sub, old_con)] = (new_cat, new_sub, new_con)
                        # Remove from candidates to avoid duplicate mapping
                        new_concepts_index[old_con].remove((new_cat, new_sub, new_con))
                        break
        
        # Track unmapped items for reporting
        mapped_old = set(mapping.keys())
        self.unmapped_items = sorted(old_set - mapped_old)
        
        return mapping
    
    def _flatten_schema(self, schema: dict) -> List[Tuple[str, str, str]]:
        """
        Flatten schema to list of (category, subcategory, concept) tuples.
        """
        flat = []
        for category_item in schema.get('entities_schema', []):
            category = category_item['category']
            for subcategory, subcat_data in category_item.get('subcategories', {}).items():
                for concept in subcat_data.get('concepts', []):
                    flat.append((category, subcategory, concept))
                    
                # Also add concept aliases
                if 'concept_aliases' in subcat_data:
                    for alias in subcat_data['concept_aliases'].keys():
                        flat.append((category, subcategory, alias))
        
        return flat
    
    def add_manual_mapping(self, 
                          old_category: str, old_subcategory: str, old_concept: str,
                          new_category: str, new_subcategory: str, new_concept: str):
        """Manually add a mapping for renamed items."""
        self.mapping[(old_category, old_subcategory, old_concept)] = \
            (new_category, new_subcategory, new_concept)
        # Remove from unmapped if it was there
        key = (old_category, old_subcategory, old_concept)
        if key in self.unmapped_items:
            self.unmapped_items.remove(key)
    
    def add_manual_mappings_from_file(self, mapping_file: str):
        """Load manual mappings from a JSON file.
        
        Format:
        {
            "mappings": [
                {
                    "old": {"category": "X", "subcategory": "Y", "concept": "Z"},
                    "new": {"category": "A", "subcategory": "B", "concept": "C"}
                }
            ]
        }
        """
        with open(mapping_file, 'r') as f:
            data = json.load(f)
        
        for mapping_entry in data.get('mappings', []):
            old = mapping_entry['old']
            new = mapping_entry['new']
            self.add_manual_mapping(
                old['category'], old['subcategory'], old['concept'],
                new['category'], new['subcategory'], new['concept']
            )
    
    def get_mapping(self, category: str, subcategory: str, concept: str) -> Optional[Tuple[str, str, str]]:
        """Get the new names for given old names."""
        return self.mapping.get((category, subcategory, concept))
    
    def get_unmapped_items(self) -> List[Tuple[str, str, str]]:
        """Get list of items that couldn't be automatically mapped."""
        return self.unmapped_items


class ECGReportUpdater:
    """Updates ECG reports based on schema changes."""
    
    def __init__(self, mapper: SchemaMapper, verbose: bool = False):
        self.mapper = mapper
        self.verbose = verbose
        self.stats = {
            'reports_processed': 0,
            'entities_updated': 0,
            'entities_unchanged': 0,
            'entities_not_found': 0
        }
        self.not_found_details = defaultdict(list)
    
    def update_report(self, report: dict) -> dict:
        """Update a single report."""
        updated_report = report.copy()
        
        if 'entities' in updated_report:
            updated_entities = []
            for entity in updated_report['entities']:
                updated_entity = self._update_entity(entity, report.get('report_id', 'unknown'))
                updated_entities.append(updated_entity)
            updated_report['entities'] = updated_entities
        
        self.stats['reports_processed'] += 1
        return updated_report
    
    def _update_entity(self, entity: dict, report_id) -> dict:
        """Update a single entity."""
        updated_entity = entity.copy()
        
        if 'entity' in updated_entity:
            old_cat = updated_entity['entity'].get('category')
            old_sub = updated_entity['entity'].get('subcategory')
            old_con = updated_entity['entity'].get('concept')
            
            new_mapping = self.mapper.get_mapping(old_cat, old_sub, old_con)
            
            if new_mapping:
                new_cat, new_sub, new_con = new_mapping
                
                # Check if anything changed
                if (old_cat, old_sub, old_con) != (new_cat, new_sub, new_con):
                    if self.verbose:
                        print(f"Report {report_id}, Entity {entity.get('id', 'unknown')}:")
                        print(f"  {old_cat}/{old_sub}/{old_con} -> {new_cat}/{new_sub}/{new_con}")
                    
                    updated_entity['entity'] = {
                        'category': new_cat,
                        'subcategory': new_sub,
                        'concept': new_con
                    }
                    self.stats['entities_updated'] += 1
                else:
                    self.stats['entities_unchanged'] += 1
            else:
                if self.verbose:
                    print(f"WARNING: No mapping found for {old_cat}/{old_sub}/{old_con} in report {report_id}")
                self.stats['entities_not_found'] += 1
                self.not_found_details[(old_cat, old_sub, old_con)].append(report_id)
        
        return updated_entity
    
    def update_reports(self, reports: List[dict]) -> List[dict]:
        """Update multiple reports."""
        return [self.update_report(report) for report in reports]
    
    def print_stats(self):
        """Print update statistics."""
        print("\n" + "="*60)
        print("UPDATE STATISTICS")
        print("="*60)
        print(f"Reports processed: {self.stats['reports_processed']}")
        print(f"Entities updated: {self.stats['entities_updated']}")
        print(f"Entities unchanged: {self.stats['entities_unchanged']}")
        print(f"Entities not found in mapping: {self.stats['entities_not_found']}")
        print("="*60)
    
    def print_unmapped_entities(self):
        """Print details of entities that couldn't be mapped."""
        if not self.not_found_details:
            return
        
        print("\n" + "="*60)
        print("UNMAPPED ENTITIES (require manual review)")
        print("="*60)
        for (cat, sub, con), report_ids in sorted(self.not_found_details.items()):
            print(f"\n{cat} / {sub} / {con}")
            report_ids_str = [str(rid) for rid in report_ids]
            print(f"  Found in {len(report_ids)} report(s): {', '.join(report_ids_str[:5])}")
            if len(report_ids) > 5:
                print(f"  ... and {len(report_ids) - 5} more")
        print("="*60)


def generate_mapping_differences(old_schema: dict, new_schema: dict, output_file: str):
    """
    Generate a template mapping file showing differences between schemas.
    """
    mapper = SchemaMapper(old_schema, new_schema)
    old_flat = set(mapper._flatten_schema(old_schema))
    new_flat = set(mapper._flatten_schema(new_schema))
    
    only_in_old = old_flat - new_flat
    only_in_new = new_flat - old_flat
    
    # Remove items that were automatically mapped
    auto_mapped_old = set(mapper.mapping.keys())
    unmapped_old = only_in_old - auto_mapped_old
    
    template = {
        "description": "Manual mapping template for schema changes",
        "auto_mapped_count": len(mapper.mapping),
        "unmapped_in_old_schema": [
            {"category": cat, "subcategory": sub, "concept": con}
            for cat, sub, con in sorted(unmapped_old)
        ],
        "only_in_new_schema": [
            {"category": cat, "subcategory": sub, "concept": con}
            for cat, sub, con in sorted(only_in_new)
        ],
        "example_mappings": [
            {
                "old": {"category": "OLD_CAT", "subcategory": "OLD_SUB", "concept": "OLD_CON"},
                "new": {"category": "NEW_CAT", "subcategory": "NEW_SUB", "concept": "NEW_CON"}
            }
        ]
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(template, f, indent=2)
    
    print(f"\nMapping analysis saved to: {output_file}")
    print(f"Automatically mapped: {len(mapper.mapping)}")
    print(f"Items requiring manual mapping: {len(unmapped_old)}")
    print(f"New items in v3 schema: {len(only_in_new)}")


def validate_reports_against_schema(reports: List[dict], schema: dict, verbose: bool = False) -> Dict:
    """
    Validate that all entities in reports exist in the schema.
    Returns a dictionary with validation results.
    """
    # Build set of valid (category, subcategory, concept) tuples from schema
    valid_entities = set()
    for category_item in schema.get('entities_schema', []):
        category = category_item['category']
        for subcategory, subcat_data in category_item.get('subcategories', {}).items():
            for concept in subcat_data.get('concepts', []):
                valid_entities.add((category, subcategory, concept))
            
            # Also add concept aliases
            if 'concept_aliases' in subcat_data:
                for alias in subcat_data['concept_aliases'].keys():
                    valid_entities.add((category, subcategory, alias))
    
    # Check all entities in reports
    invalid_entities = defaultdict(list)  # Maps (cat, sub, con) -> list of report_ids
    total_entities = 0
    
    for report in reports:
        report_id = report.get('report_id', 'unknown')
        for entity in report.get('entities', []):
            total_entities += 1
            if 'entity' in entity:
                cat = entity['entity'].get('category')
                sub = entity['entity'].get('subcategory')
                con = entity['entity'].get('concept')
                
                entity_key = (cat, sub, con)
                if entity_key not in valid_entities:
                    invalid_entities[entity_key].append(report_id)
                    if verbose:
                        print(f"Invalid entity in report {report_id}: {cat}/{sub}/{con}")
    
    results = {
        'total_reports': len(reports),
        'total_entities': total_entities,
        'valid_entities': total_entities - sum(len(v) for v in invalid_entities.values()),
        'invalid_entities_count': sum(len(v) for v in invalid_entities.values()),
        'invalid_entity_types': len(invalid_entities),
        'invalid_entities': dict(invalid_entities)
    }
    
    return results


def print_validation_results(results: Dict):
    """Print validation results in a readable format."""
    print("\n" + "="*60)
    print("SCHEMA VALIDATION RESULTS")
    print("="*60)
    print(f"Total reports checked: {results['total_reports']}")
    print(f"Total entities: {results['total_entities']}")
    print(f"Valid entities: {results['valid_entities']}")
    print(f"Invalid entities: {results['invalid_entities_count']}")
    print(f"Invalid entity types: {results['invalid_entity_types']}")
    
    if results['invalid_entities']:
        print("\n" + "-"*60)
        print("ENTITIES NOT FOUND IN SCHEMA:")
        print("-"*60)
        for (cat, sub, con), report_ids in sorted(results['invalid_entities'].items()):
            print(f"\n{cat} / {sub} / {con}")
            report_ids_str = [str(rid) for rid in report_ids]
            print(f"  Found in {len(report_ids)} report(s): {', '.join(report_ids_str[:10])}")
            if len(report_ids) > 10:
                print(f"  ... and {len(report_ids) - 10} more")
    else:
        print("\n✓ All entities are valid!")
    
    print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description='Update ECG reports when schema names change',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate reports against schema
  python schema_report_updater.py --validate schema_v3.json --reports merged_v3.json
  
  # Update all reports
  python schema_report_updater.py --old-schema schema_v2.json --new-schema schema_v3.json \\
      --reports merged_v2.json --output updated_reports.json
  
  # Update single report
  python schema_report_updater.py --old-schema schema_v2.json --new-schema schema_v3.json \\
      --single-report report.json --output updated_report.json
  
  # Generate mapping template to identify differences
  python schema_report_updater.py --old-schema schema_v2.json --new-schema schema_v3.json \\
      --generate-mapping mapping_analysis.json
  
  # Update with manual mappings
  python schema_report_updater.py --old-schema schema_v2.json --new-schema schema_v3.json \\
      --reports merged_v2.json --output updated_reports.json \\
      --manual-mapping custom_mappings.json
        """
    )
    
    parser.add_argument('--old-schema', help='Path to old schema JSON file')
    parser.add_argument('--new-schema', help='Path to new schema JSON file')
    parser.add_argument('--reports', help='Path to merged reports JSON file')
    parser.add_argument('--single-report', help='Path to single report JSON file')
    parser.add_argument('--output', help='Path to output file')
    parser.add_argument('--manual-mapping', help='Path to manual mapping JSON file')
    parser.add_argument('--generate-mapping', help='Generate mapping template and exit')
    parser.add_argument('--validate', help='Validate reports against schema (provide schema path)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Validation mode
    if args.validate:
        if not args.reports and not args.single_report:
            print("ERROR: Please provide --reports or --single-report for validation")
            return
        
        print(f"Loading schema from {args.validate}...")
        with open(args.validate, 'r') as f:
            schema = json.load(f)
        
        if args.reports:
            print(f"Loading reports from {args.reports}...")
            with open(args.reports, 'r', encoding='utf-8') as f:
                reports = json.load(f)
        else:
            print(f"Loading report from {args.single_report}...")
            with open(args.single_report, 'r', encoding='utf-8') as f:
                report = json.load(f)
                reports = [report]
        
        print(f"Validating {len(reports)} report(s) against schema...")
        results = validate_reports_against_schema(reports, schema, verbose=args.verbose)
        print_validation_results(results)
        return
    
    # Require schemas for update mode
    if not args.old_schema or not args.new_schema:
        print("ERROR: --old-schema and --new-schema are required for update mode")
        print("Use --validate for validation mode, or provide both schemas for update mode")
        parser.print_help()
        return
    
    # Load schemas
    print(f"Loading schemas...")
    with open(args.old_schema, 'r') as f:
        old_schema = json.load(f)
    with open(args.new_schema, 'r') as f:
        new_schema = json.load(f)
    
    # Generate mapping template if requested
    if args.generate_mapping:
        generate_mapping_differences(old_schema, new_schema, args.generate_mapping)
        return
    
    # Create mapper
    mapper = SchemaMapper(old_schema, new_schema)
    
    # Load manual mappings if provided
    if args.manual_mapping:
        print(f"Loading manual mappings from {args.manual_mapping}")
        mapper.add_manual_mappings_from_file(args.manual_mapping)
    
    # Create updater
    updater = ECGReportUpdater(mapper, verbose=args.verbose)
    
    # Update reports
    if args.reports:
        print(f"Loading reports from {args.reports}...")
        with open(args.reports, 'r', encoding='utf-8') as f:
            reports = json.load(f)
        
        print(f"Updating {len(reports)} reports...")
        updated_reports = updater.update_reports(reports)
        
        if args.output:
            print(f"Saving updated reports to {args.output}...")
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(updated_reports, f, indent=2, ensure_ascii=False)
            print(f"✓ Updated reports saved to {args.output}")
    
    elif args.single_report:
        print(f"Loading single report from {args.single_report}...")
        with open(args.single_report, 'r', encoding='utf-8') as f:
            report = json.load(f)
        
        print(f"Updating report...")
        updated_report = updater.update_report(report)
        
        if args.output:
            print(f"Saving updated report to {args.output}...")
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(updated_report, f, indent=2, ensure_ascii=False)
            print(f"✓ Updated report saved to {args.output}")
    
    else:
        print("ERROR: Please provide either --reports or --single-report")
        parser.print_help()
        return
    
    # Print statistics
    updater.print_stats()
    updater.print_unmapped_entities()
    
    # Show unmapped items from mapper
    unmapped = mapper.get_unmapped_items()
    if unmapped:
        print(f"\n{len(unmapped)} concept(s) in old schema have no mapping:")
        print("These may be deleted concepts or require manual mapping.")


if __name__ == '__main__':
    main()