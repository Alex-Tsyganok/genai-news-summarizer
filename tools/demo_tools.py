#!/usr/bin/env python3
"""
Quick demo of the ChromaDB database tools
Run this from the tools/ directory to test all inspection utilities
"""

import subprocess
import sys
import os

def run_command_safe(cmd, description):
    """Run a command safely, handling encoding issues."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print('='*60)
    
    try:
        # For Windows, use a simpler approach - just show first few lines
        if sys.platform == "win32":
            print(f"Executing: python {cmd.split()[-1]}")
            print("Note: Run this command directly to see full output with emojis")
            
            # Run with error handling
            result = subprocess.run(
                cmd, 
                shell=True, 
                capture_output=True, 
                text=True, 
                cwd='.', 
                encoding='utf-8',
                errors='replace'
            )
            
            if result.stdout:
                # Show just first 20 lines to avoid encoding issues
                lines = result.stdout.split('\n')[:20]
                for line in lines:
                    # Replace problematic Unicode with placeholders
                    clean_line = line.encode('ascii', 'replace').decode('ascii')
                    print(clean_line)
                if len(result.stdout.split('\n')) > 20:
                    print("... (output truncated)")
            
            if result.stderr:
                print(f"Errors occurred - run command directly for details")
                
        else:
            # On non-Windows, run normally
            result = subprocess.run(cmd, shell=True, cwd='.')
            
    except Exception as e:
        print(f"Command execution info: {e}")
        print(f"Try running: python {cmd.split()[-1]} directly")

def main():
    """Demonstrate all database tools."""
    print("ChromaDB Database Tools Demo")
    print("This script demonstrates the database inspection tools")
    
    # Check if we're in the tools directory
    if not os.path.exists('inspect_db.py'):
        print("Please run this script from the tools/ directory")
        sys.exit(1)
    
    # Test 1: Quick overview
    run_command_safe("python quick_db_view.py", "Quick Database Overview")
    
    # Test 2: Direct ChromaDB access
    run_command_safe("python explore_chromadb.py", "Direct ChromaDB Exploration")
    
    # Test 3: Interactive inspector demo (just show help)
    print(f"\n{'='*60}")
    print("Interactive Inspector Commands")
    print('='*60)
    print("The interactive inspector (inspect_db.py) supports these commands:")
    print("  stats          - Show collection statistics")
    print("  list [N]       - List all records")
    print("  search <query> - Search records")
    print("  show <id>      - Show specific record")
    print("  topics         - Analyze topics")
    print("  export [file]  - Export to JSON")
    print("\nTo use: python inspect_db.py")
    
    print(f"\n{'='*60}")
    print("Demo Complete!")
    print("Choose the tool that best fits your needs:")
    print("• quick_db_view.py - Fast overview")
    print("• explore_chromadb.py - Raw database access") 
    print("• inspect_db.py - Interactive exploration")
    print('='*60)

if __name__ == "__main__":
    main()
