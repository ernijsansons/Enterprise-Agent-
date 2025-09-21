#!/bin/bash
# Enterprise Agent Installation Script
# One-line install: curl -sSL https://yoursite.com/install.sh | bash

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
REPO_URL="https://github.com/yourorg/enterprise-agent.git"
INSTALL_DIR="$HOME/.enterprise-agent"
BIN_DIR="$HOME/.local/bin"

# Functions
print_header() {
    echo -e "\n${BLUE}============================================${NC}"
    echo -e "${BLUE}     Enterprise Agent Installer${NC}"
    echo -e "${BLUE}============================================${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

# Check dependencies
check_dependencies() {
    local missing_deps=()
    local warnings=()

    # Check Git
    if ! command -v git &> /dev/null; then
        missing_deps+=("git")
    else
        # Verify git version (need >= 2.0)
        git_version=$(git --version | grep -oE '[0-9]+\.[0-9]+' | head -1)
        if [[ $(echo "$git_version 2.0" | tr " " "\n" | sort -V | head -1) != "2.0" ]] && [[ "$git_version" != "2.0" ]]; then
            warnings+=("Git version $git_version is old, recommend >= 2.0")
        fi
    fi

    # Check Python and version
    if ! command -v python3 &> /dev/null; then
        missing_deps+=("python3")
    else
        # Verify Python version (need >= 3.9)
        python_version=$(python3 --version | grep -oE '[0-9]+\.[0-9]+' | head -1)
        if python3 -c "import sys; exit(0 if sys.version_info >= (3, 9) else 1)" 2>/dev/null; then
            print_success "Python $python_version detected"
        else
            print_error "Python version $python_version is too old. Need Python >= 3.9"
            exit 1
        fi
    fi

    # Check pip
    if ! command -v pip3 &> /dev/null; then
        missing_deps+=("pip3")
    else
        # Check if pip is working
        if ! pip3 --version &> /dev/null; then
            print_error "pip3 is installed but not working properly"
            exit 1
        fi
    fi

    # Check Node.js (optional but recommended)
    if ! command -v node &> /dev/null; then
        warnings+=("Node.js not found - Claude Code CLI won't be available")
    else
        # Verify Node version (need >= 16)
        node_version=$(node --version | grep -oE '[0-9]+' | head -1)
        if [[ $node_version -lt 16 ]]; then
            warnings+=("Node.js version $node_version is old, recommend >= 16 for Claude Code CLI")
        fi
    fi

    # Check npm (optional but recommended)
    if ! command -v npm &> /dev/null; then
        warnings+=("npm not found - Claude Code CLI won't be available")
    fi

    # Check for virtual environment support
    if ! python3 -m venv --help &> /dev/null; then
        print_error "Python venv module not available. Install python3-venv package"
        exit 1
    fi

    # Display warnings
    for warning in "${warnings[@]}"; do
        print_warning "$warning"
    done

    if [ ${#missing_deps[@]} -ne 0 ]; then
        print_error "Missing critical dependencies: ${missing_deps[*]}"
        echo ""
        echo "Please install missing dependencies:"
        echo "  Ubuntu/Debian: sudo apt-get install ${missing_deps[*]} python3-venv"
        echo "  macOS: brew install ${missing_deps[*]}"
        echo "  Fedora: sudo dnf install ${missing_deps[*]} python3-venv"
        echo "  CentOS/RHEL: sudo yum install ${missing_deps[*]} python3-venv"
        exit 1
    fi

    print_success "All critical dependencies satisfied"
}

# Clone or update repository
install_agent() {
    if [ -d "$INSTALL_DIR" ]; then
        print_info "Updating existing installation..."
        cd "$INSTALL_DIR" || {
            print_error "Failed to change to $INSTALL_DIR"
            exit 1
        }
        
        if git pull origin main; then
            print_success "Updated from git"
        else
            print_warning "Could not update from git, continuing with existing code"
        fi
    else
        print_info "Cloning Enterprise Agent..."
        if git clone "$REPO_URL" "$INSTALL_DIR"; then
            print_success "Cloned repository"
        else
            print_error "Failed to clone repository from $REPO_URL"
            exit 1
        fi
        
        cd "$INSTALL_DIR" || {
            print_error "Failed to change to $INSTALL_DIR"
            exit 1
        }
    fi

    print_success "Agent code installed to $INSTALL_DIR"
}

# Install Python dependencies
install_python_deps() {
    print_info "Installing Python dependencies..."

    cd "$INSTALL_DIR" || {
        print_error "Failed to change to $INSTALL_DIR"
        exit 1
    }

    # Create virtual environment
    if [ ! -d "venv" ]; then
        print_info "Creating virtual environment..."
        if python3 -m venv venv; then
            print_success "Created virtual environment"
        else
            print_error "Failed to create virtual environment"
            # Try alternative location in case of permission issues
            print_info "Trying alternative location..."
            TEMP_VENV="/tmp/enterprise-agent-venv-$$"
            if python3 -m venv "$TEMP_VENV"; then
                mv "$TEMP_VENV" venv
                print_success "Created virtual environment in alternative location"
            else
                print_error "Failed to create virtual environment. Check Python installation."
                exit 1
            fi
        fi
    else
        print_info "Virtual environment already exists"
    fi

    # Activate virtual environment with error handling
    if [ -f "venv/bin/activate" ]; then
        # shellcheck source=/dev/null
        if source venv/bin/activate; then
            print_success "Activated virtual environment"
        else
            print_error "Failed to activate virtual environment"
            exit 1
        fi
    else
        print_error "Virtual environment activation script not found"
        exit 1
    fi

    # Verify Python in virtual environment
    if ! python --version &> /dev/null; then
        print_error "Python not available in virtual environment"
        exit 1
    fi

    # Upgrade pip with retry
    print_info "Upgrading pip..."
    local pip_retry=3
    while [ $pip_retry -gt 0 ]; do
        if pip install --upgrade pip --no-warn-script-location; then
            print_success "Upgraded pip"
            break
        else
            pip_retry=$((pip_retry - 1))
            if [ $pip_retry -eq 0 ]; then
                print_warning "Failed to upgrade pip after 3 attempts, continuing..."
            else
                print_warning "Pip upgrade failed, retrying... ($pip_retry attempts left)"
                sleep 2
            fi
        fi
    done

    # Choose installation method
    local install_method=""
    if [ -f "pyproject.toml" ] && command -v poetry &> /dev/null; then
        install_method="poetry"
        print_info "Using Poetry for dependency management"
    elif [ -f "requirements.txt" ]; then
        install_method="requirements"
        print_info "Using requirements.txt for dependency management"
    else
        install_method="minimal"
        print_info "Creating minimal requirements for basic functionality"
    fi

    # Install dependencies based on method
    case $install_method in
        "poetry")
            if poetry install --no-dev; then
                print_success "Dependencies installed via Poetry"
            else
                print_warning "Poetry installation failed, falling back to pip"
                install_method="requirements"
            fi
            ;;
    esac

    if [ "$install_method" = "requirements" ]; then
        if [ -f "requirements.txt" ]; then
            print_info "Installing from requirements.txt..."
            if pip install -r requirements.txt --no-warn-script-location; then
                print_success "Python dependencies installed from requirements.txt"
            else
                print_error "Failed to install dependencies from requirements.txt"
                print_info "Checking for specific error details..."
                pip install -r requirements.txt --no-warn-script-location --verbose || {
                    print_error "Detailed installation failed. Check network connection and PyPI access."
                    exit 1
                }
            fi
        fi
    elif [ "$install_method" = "minimal" ]; then
        # Create minimal requirements with fallback versions
        print_info "Creating minimal requirements.txt..."
        cat > requirements.txt <<'EOF'
# Minimal requirements for Enterprise Agent
pyyaml>=6.0.1,<7.0
python-dotenv>=1.0.1,<2.0
requests>=2.31.0,<3.0
# AI providers (optional)
anthropic>=0.20.0,<1.0
openai>=1.2.0,<2.0
# Utilities
click>=8.0.0,<9.0
rich>=13.0.0,<14.0
EOF

        print_info "Installing minimal dependencies..."
        if pip install -r requirements.txt --no-warn-script-location; then
            print_success "Minimal Python dependencies installed"
        else
            print_error "Failed to install minimal dependencies"
            print_info "Trying to install core packages individually..."

            # Try installing core packages one by one
            local core_packages=("pyyaml" "python-dotenv" "requests" "click")
            local failed_packages=()

            for package in "${core_packages[@]}"; do
                if pip install "$package" --no-warn-script-location; then
                    print_success "Installed $package"
                else
                    failed_packages+=("$package")
                    print_warning "Failed to install $package"
                fi
            done

            if [ ${#failed_packages[@]} -ne 0 ]; then
                print_error "Failed to install core packages: ${failed_packages[*]}"
                print_error "Please check your internet connection and try again"
                exit 1
            fi
        fi
    fi

    # Verify installation
    print_info "Verifying installation..."
    if python -c "import yaml, dotenv, requests" 2>/dev/null; then
        print_success "Core dependencies verified"
    else
        print_error "Core dependency verification failed"
        exit 1
    fi

    # Test CLI entry point
    if [ -f "enterprise_agent_cli.py" ]; then
        if python enterprise_agent_cli.py --version &>/dev/null; then
            print_success "CLI entry point verified"
        else
            print_warning "CLI entry point test failed, but installation may still work"
        fi
    fi

    deactivate
}

# Create executable wrapper
create_wrapper() {
    print_info "Creating executable wrapper..."

    # Create bin directory if it doesn't exist
    mkdir -p "$BIN_DIR"

    # Create wrapper script
    cat > "$BIN_DIR/enterprise-agent" <<'EOF'
#!/bin/bash
# Enterprise Agent wrapper script

AGENT_DIR="$HOME/.enterprise-agent"

# Activate virtual environment and run CLI
source "$AGENT_DIR/venv/bin/activate"
python "$AGENT_DIR/enterprise_agent_cli.py" "$@"
deactivate
EOF

    chmod +x "$BIN_DIR/enterprise-agent"

    # Create short alias
    ln -sf "$BIN_DIR/enterprise-agent" "$BIN_DIR/ea"

    print_success "Created executable: $BIN_DIR/enterprise-agent"
}

# Setup Claude Code CLI
setup_claude_code() {
    print_info "Checking Claude Code CLI..."

    if command -v claude &> /dev/null; then
        print_success "Claude Code CLI is installed"

        # Check login status
        if claude --version 2>&1 | grep -q "logged in"; then
            print_success "Already logged in to Claude Code"
        else
            print_warning "Not logged in to Claude Code"
            echo "  Run: claude login"
        fi
    else
        print_warning "Claude Code CLI not installed"
        echo ""
        echo "To use Claude Code (zero API costs with Max subscription):"
        echo "  1. Install: npm install -g @anthropic-ai/claude-code"
        echo "  2. Login: claude login"
        echo ""
        read -p "Install Claude Code CLI now? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            npm install -g @anthropic-ai/claude-code
            print_success "Claude Code CLI installed"
            echo "Now run: claude login"
        fi
    fi
}

# Configure environment
configure_environment() {
    print_info "Configuring environment..."

    # Create global config directory
    mkdir -p "$HOME/.enterprise-agent"

    # Create default config if it doesn't exist
    if [ ! -f "$HOME/.enterprise-agent/config.yml" ]; then
        cat > "$HOME/.enterprise-agent/config.yml" <<EOF
# Enterprise Agent Global Configuration
use_claude_code: false  # Set to true after 'claude login'
models:
  primary: claude-3-5-sonnet-20241022
  fallback: gpt-4o-mini
default_domain: coding
EOF
        print_success "Created global configuration"
    fi

    # Add to PATH
    if [[ ":$PATH:" != *":$BIN_DIR:"* ]]; then
        echo ""
        echo "Add to your PATH by adding this to ~/.bashrc or ~/.zshrc:"
        echo "  export PATH=\"\$PATH:$BIN_DIR\""
        echo ""

        # Detect shell and update config
        if [ -n "$ZSH_VERSION" ]; then
            echo "export PATH=\"\$PATH:$BIN_DIR\"" >> ~/.zshrc
            print_success "Added to ~/.zshrc"
        elif [ -n "$BASH_VERSION" ]; then
            echo "export PATH=\"\$PATH:$BIN_DIR\"" >> ~/.bashrc
            print_success "Added to ~/.bashrc"
        fi
    fi
}

# Main installation flow
main() {
    print_header

    echo "This will install Enterprise Agent to: $INSTALL_DIR"
    echo ""

    # Check dependencies
    print_info "Checking dependencies..."
    check_dependencies

    # Install agent
    install_agent

    # Install Python dependencies
    install_python_deps

    # Create wrapper
    create_wrapper

    # Setup Claude Code
    setup_claude_code

    # Configure environment
    configure_environment

    # Success message
    echo ""
    print_success "Installation complete!"
    echo ""
    echo "Quick Start:"
    echo "  1. Reload your shell or run: source ~/.bashrc"
    echo "  2. Initialize in a project: enterprise-agent init"
    echo "  3. Run agent: enterprise-agent run --input \"Your prompt\""
    echo "  4. Interactive mode: enterprise-agent interactive"
    echo ""
    echo "For Claude Code (zero API costs):"
    echo "  1. Install: npm install -g @anthropic-ai/claude-code"
    echo "  2. Login: claude login"
    echo "  3. Update config: ~/.enterprise-agent/config.yml"
    echo "     Set: use_claude_code: true"
    echo ""
    print_success "Happy coding with Enterprise Agent! 🚀"
}

# Run main
main "$@"