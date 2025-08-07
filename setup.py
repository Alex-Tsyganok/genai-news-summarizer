"""
Setup and installation script for the AI News Summarizer.
"""
import os
import sys
import subprocess
import shutil
from pathlib import Path

def run_command(command, description=""):
    """Run a shell command and handle errors."""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ is required")
        return False
    
    version_str = sys.version.split()[0]
    print(f"✅ Python {version_str} detected")
    
    # Warning for Python 3.13+
    if sys.version_info >= (3, 13):
        print("⚠️  WARNING: Python 3.13+ may require additional setup for some packages")
        print("   Recommended: Use Python 3.12.x for better compatibility with scientific libraries")
        print("   If installation fails, consider downgrading or installing C++ Build Tools")
    
    return True

def create_directories():
    """Create necessary directories."""
    directories = [
        "data",
        "data/chromadb",
        "logs",
        "test_data"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"📁 Created directory: {directory}")

def setup_environment():
    """Set up the project environment."""
    print("🚀 Setting up AI News Summarizer")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Create directories
    create_directories()
    
    # Install dependencies
    if not run_command("pip install -r requirements.txt", "Installing dependencies"):
        print("💡 Try: pip install --upgrade pip")
        return False
    
    # Create .env file if it doesn't exist
    if not os.path.exists(".env"):
        shutil.copy(".env.example", ".env")
        print("📝 Created .env file from template")
        print("⚠️  Please edit .env and add your OpenAI API key")
    else:
        print("✅ .env file already exists")
    
    # Download NLTK data
    try:
        import nltk
        print("📚 Downloading NLTK data...")
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        print("✅ NLTK data downloaded")
    except ImportError:
        print("⚠️  NLTK not available, skipping data download")
    
    print("\n🎉 Setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Edit .env file and add your OpenAI API key")
    print("2. Run: python examples/basic_usage.py")
    print("3. Start web app: streamlit run src/ui/app.py")
    print("4. Or use CLI: python cli.py --help")
    
    return True

def run_tests():
    """Run the test suite."""
    print("\n🧪 Running tests...")
    if run_command("python tests/test_pipeline.py", "Running test suite"):
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed")

def setup_virtual_environment():
    """Guide user through virtual environment setup if needed."""
    venv_path = Path("venv")
    
    if not venv_path.exists():
        print("🔧 Creating virtual environment...")
        result = subprocess.run([sys.executable, "-m", "venv", "venv"], 
                              capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"❌ Failed to create virtual environment: {result.stderr}")
            return False
            
        print("✅ Virtual environment created successfully!")
        print("\n📝 To activate the virtual environment:")
        print("   Windows: venv\\Scripts\\activate")
        print("   macOS/Linux: source venv/bin/activate")
        print("\nThen run this setup script again.")
        return False
    
    return True

def main():
    """Main setup function."""
    print("🚀 AI News Summarizer Setup")
    
    # Check virtual environment
    if not setup_virtual_environment():
        return
        
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        run_tests()
    else:
        if setup_environment():
            if input("\nRun tests? (y/n): ").lower().strip() == 'y':
                run_tests()

if __name__ == "__main__":
    main()
