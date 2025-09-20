#!/usr/bin/env python3
"""
Secure .env file setup script for Enterprise Coding Agent v3.4
This script helps you create a .env file with your API keys securely.
"""

import os
import sys
from pathlib import Path

def create_env_file():
    """Create .env file from .env.example template."""
    project_root = Path(__file__).parent
    env_example = project_root / '.env.example'
    env_file = project_root / '.env'
    
    if not env_example.exists():
        print("❌ Error: .env.example file not found!")
        print("Please make sure .env.example exists in the project root.")
        return False
    
    if env_file.exists():
        response = input("⚠️  .env file already exists. Overwrite? (y/N): ").strip().lower()
        if response != 'y':
            print("✅ Keeping existing .env file.")
            return True
    
    try:
        # Copy .env.example to .env
        with open(env_example, 'r') as f:
            content = f.read()
        
        with open(env_file, 'w') as f:
            f.write(content)
        
        print("✅ Created .env file from template.")
        print("📝 Please edit .env file and add your actual API keys.")
        print("🔒 Remember: Never commit .env to version control!")
        return True
        
    except Exception as e:
        print(f"❌ Error creating .env file: {e}")
        return False

def main():
    """Main setup function."""
    print("🔐 Enterprise Coding Agent v3.4 - Environment Setup")
    print("=" * 50)
    
    if create_env_file():
        print("\n📋 Next steps:")
        print("1. Edit the .env file with your actual API keys")
        print("2. Install dependencies: pip install -r requirements.txt")
        print("3. Run the agent: python src/agent_orchestrator.py")
        print("\n🔒 Security reminder:")
        print("- .env is already in .gitignore")
        print("- Never share your .env file")
        print("- Use .env.example for team templates")
    else:
        print("\n❌ Setup failed. Please check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
