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

    # Check Git
    if ! command -v git &> /dev/null; then
        missing_deps+=("git")
    fi

    # Check Python
    if ! command -v python3 &> /dev/null; then
        missing_deps+=("python3")
    fi

    # Check pip
    if ! command -v pip3 &> /dev/null; then
        missing_deps+=("pip3")
    fi

    # Check Node.js
    if ! command -v node &> /dev/null; then
        missing_deps+=("nodejs")
    fi

    # Check npm
    if ! command -v npm &> /dev/null; then
        missing_deps+=("npm")
    fi

    if [ ${#missing_deps[@]} -ne 0 ]; then
        print_error "Missing dependencies: ${missing_deps[*]}"
        echo ""
        echo "Please install missing dependencies:"
        echo "  Ubuntu/Debian: sudo apt-get install ${missing_deps[*]}"
        echo "  macOS: brew install ${missing_deps[*]}"
        echo "  Fedora: sudo dnf install ${missing_deps[*]}"
        exit 1
    fi

    print_success "All dependencies satisfied"
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
        if python3 -m venv venv; then
            print_success "Created virtual environment"
        else
            print_error "Failed to create virtual environment"
            exit 1
        fi
    fi

    # Activate virtual environment
    if source venv/bin/activate; then
        print_success "Activated virtual environment"
    else
        print_error "Failed to activate virtual environment"
        exit 1
    fi

    # Upgrade pip
    if pip install --upgrade pip; then
        print_success "Upgraded pip"
    else
        print_warning "Failed to upgrade pip, continuing..."
    fi

    # Install requirements
    if [ -f "requirements.txt" ]; then
        if pip install -r requirements.txt; then
            print_success "Python dependencies installed from requirements.txt"
        else
            print_error "Failed to install dependencies from requirements.txt"
            exit 1
        fi
    else
        # Create minimal requirements
        print_info "Creating minimal requirements.txt..."
        cat > requirements.txt <<EOF
pyyaml>=6.0.1
python-dotenv>=1.0.1
requests>=2.31.0
anthropic>=0.20.0
openai>=1.2.0
EOF
        if pip install -r requirements.txt; then
            print_success "Python dependencies installed from minimal requirements"
        else
            print_error "Failed to install minimal dependencies"
            exit 1
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