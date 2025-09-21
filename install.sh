#!/bin/bash
# Enterprise Agent Installation Script v3.4
# Production-ready installer with comprehensive error handling
# One-line install: curl -sSL https://raw.githubusercontent.com/ernijsansons/Enterprise-Agent-/main/install.sh | bash

set -euo pipefail

# Enable error tracing
trap 'error_handler $? $LINENO' ERR

# Global error handler
error_handler() {
    local exit_code=$1
    local line_num=$2
    echo -e "\n${RED}Installation failed at line $line_num with exit code $exit_code${NC}"
    echo -e "${YELLOW}Troubleshooting:${NC}"
    echo "  - Check the installation log at /tmp/enterprise-agent-install.log"
    echo "  - Ensure you have proper permissions"
    echo "  - Verify internet connectivity"
    echo "  - Report issues at: https://github.com/ernijsansons/Enterprise-Agent-/issues"
    cleanup_on_error
    exit $exit_code
}

# Cleanup on error
cleanup_on_error() {
    if [ -n "${TEMP_VENV:-}" ] && [ -d "$TEMP_VENV" ]; then
        rm -rf "$TEMP_VENV"
    fi
    if [ -n "${INSTALL_DIR:-}" ] && [ -d "$INSTALL_DIR/.install_backup" ]; then
        echo -e "${YELLOW}Restoring previous installation...${NC}"
        rm -rf "$INSTALL_DIR"
        mv "$INSTALL_DIR.install_backup" "$INSTALL_DIR"
    fi
}

# Logging
LOG_FILE="/tmp/enterprise-agent-install.log"
exec 2> >(tee -a "$LOG_FILE" >&2)
date > "$LOG_FILE"
echo "Enterprise Agent Installation Log" >> "$LOG_FILE"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
REPO_URL="${REPO_URL:-https://github.com/ernijsansons/Enterprise-Agent-.git}"
INSTALL_DIR="${INSTALL_DIR:-$HOME/.enterprise-agent}"
BIN_DIR="${BIN_DIR:-$HOME/.local/bin}"
MAX_RETRIES=3
RETRY_DELAY=2
MIN_PYTHON_VERSION="3.9"
MIN_NODE_VERSION="16"
MIN_GIT_VERSION="2.0"

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
    local python_cmd=""
    for cmd in python3 python python3.12 python3.11 python3.10 python3.9; do
        if command -v "$cmd" &> /dev/null; then
            if "$cmd" -c "import sys; exit(0 if sys.version_info >= (3, 9) else 1)" 2>/dev/null; then
                python_cmd="$cmd"
                python_version=$("$cmd" --version | grep -oE '[0-9]+\.[0-9]+' | head -1)
                print_success "Python $python_version detected (using $cmd)"
                export PYTHON_CMD="$cmd"
                break
            fi
        fi
    done

    if [ -z "$python_cmd" ]; then
        print_error "Python >= $MIN_PYTHON_VERSION not found"
        missing_deps+=("python3")
    fi

    # Check pip
    local pip_cmd=""
    for cmd in pip3 pip "${PYTHON_CMD:-python3}" -m pip; do
        if $cmd --version &> /dev/null 2>&1; then
            pip_cmd="$cmd"
            print_success "pip detected (using $cmd)"
            export PIP_CMD="$cmd"
            break
        fi
    done

    if [ -z "$pip_cmd" ]; then
        print_warning "pip not found, will attempt to install"
        if "${PYTHON_CMD:-python3}" -m ensurepip &> /dev/null 2>&1; then
            print_success "pip installed via ensurepip"
            export PIP_CMD="${PYTHON_CMD:-python3} -m pip"
        else
            missing_deps+=("pip3")
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
    if ! "${PYTHON_CMD:-python3}" -m venv --help &> /dev/null 2>&1; then
        print_warning "Python venv module not available, attempting to install..."
        if [ -f /etc/debian_version ]; then
            print_info "Debian/Ubuntu detected, install python3-venv with: sudo apt-get install python3-venv"
        elif [ -f /etc/redhat-release ]; then
            print_info "RHEL/CentOS detected, install with: sudo yum install python3-venv"
        elif [ "$(uname)" = "Darwin" ]; then
            print_info "macOS detected, venv should be included with Python"
        fi
        # Try using virtualenv as fallback
        if command -v virtualenv &> /dev/null; then
            print_info "Using virtualenv as fallback"
            export USE_VIRTUALENV=true
        else
            missing_deps+=("python3-venv or virtualenv")
        fi
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
        print_info "Backing up existing installation..."
        if [ -d "$INSTALL_DIR.install_backup" ]; then
            rm -rf "$INSTALL_DIR.install_backup"
        fi
        cp -r "$INSTALL_DIR" "$INSTALL_DIR.install_backup" 2>/dev/null || true

        print_info "Updating existing installation..."
        cd "$INSTALL_DIR" || {
            print_error "Failed to change to $INSTALL_DIR"
            exit 1
        }

        # Stash local changes if any
        if git diff --quiet 2>/dev/null; then
            if git pull origin main; then
                print_success "Updated from git"
            else
                print_warning "Could not update from git, trying reset..."
                git fetch origin main
                git reset --hard origin/main || print_warning "Could not reset to origin/main"
            fi
        else
            print_warning "Local changes detected, stashing..."
            git stash push -m "Auto-stash before update $(date)"
            if git pull origin main; then
                print_success "Updated from git"
                print_info "Local changes were stashed. Run 'git stash pop' to restore."
            else
                print_warning "Could not update from git, continuing with existing code"
            fi
        fi
    else
        print_info "Cloning Enterprise Agent..."
        local clone_retry=$MAX_RETRIES
        while [ $clone_retry -gt 0 ]; do
            if git clone --depth 1 "$REPO_URL" "$INSTALL_DIR" 2>/dev/null; then
                print_success "Cloned repository"
                break
            else
                clone_retry=$((clone_retry - 1))
                if [ $clone_retry -eq 0 ]; then
                    print_error "Failed to clone repository after $MAX_RETRIES attempts"
                    print_info "Trying alternative download method..."

                    # Try downloading as archive
                    local archive_url="${REPO_URL%.git}/archive/refs/heads/main.zip"
                    if command -v wget &> /dev/null; then
                        if wget -O /tmp/enterprise-agent.zip "$archive_url" 2>/dev/null; then
                            unzip -q /tmp/enterprise-agent.zip -d /tmp/
                            mv /tmp/Enterprise-Agent--main "$INSTALL_DIR"
                            rm /tmp/enterprise-agent.zip
                            print_success "Downloaded as archive"
                            break
                        fi
                    elif command -v curl &> /dev/null; then
                        if curl -L -o /tmp/enterprise-agent.zip "$archive_url" 2>/dev/null; then
                            unzip -q /tmp/enterprise-agent.zip -d /tmp/
                            mv /tmp/Enterprise-Agent--main "$INSTALL_DIR"
                            rm /tmp/enterprise-agent.zip
                            print_success "Downloaded as archive"
                            break
                        fi
                    fi

                    print_error "All download methods failed. Check internet connection."
                    exit 1
                else
                    print_warning "Clone failed, retrying... ($clone_retry attempts left)"
                    sleep $RETRY_DELAY
                fi
            fi
        done
        
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
        local venv_created=false

        if [ "${USE_VIRTUALENV:-false}" = "true" ]; then
            # Use virtualenv command
            if virtualenv -p "${PYTHON_CMD:-python3}" venv 2>/dev/null; then
                print_success "Created virtual environment with virtualenv"
                venv_created=true
            fi
        else
            # Use venv module
            if "${PYTHON_CMD:-python3}" -m venv venv 2>/dev/null; then
                print_success "Created virtual environment"
                venv_created=true
            fi
        fi

        if [ "$venv_created" = "false" ]; then
            print_warning "Failed to create virtual environment in default location"
            # Try alternative locations
            for alt_dir in "/tmp" "$HOME/tmp" "$HOME/.cache"; do
                if [ -w "$alt_dir" ]; then
                    TEMP_VENV="$alt_dir/enterprise-agent-venv-$$"
                    print_info "Trying alternative location: $alt_dir"
                    if "${PYTHON_CMD:-python3}" -m venv "$TEMP_VENV" 2>/dev/null; then
                        mv "$TEMP_VENV" venv
                        print_success "Created virtual environment in $alt_dir"
                        venv_created=true
                        break
                    fi
                fi
            done
        fi

        if [ "$venv_created" = "false" ]; then
            print_error "Failed to create virtual environment. Check Python installation."
            exit 1
        fi
    else
        print_info "Virtual environment already exists"
        # Verify it's working
        if [ ! -f "venv/bin/activate" ] && [ ! -f "venv/Scripts/activate" ]; then
            print_warning "Virtual environment seems corrupted, recreating..."
            rm -rf venv
            install_python_deps  # Recursive call
            return
        fi
    fi

    # Activate virtual environment with error handling
    local activate_script=""
    if [ -f "venv/bin/activate" ]; then
        activate_script="venv/bin/activate"
    elif [ -f "venv/Scripts/activate" ]; then
        activate_script="venv/Scripts/activate"  # Windows
    else
        print_error "Virtual environment activation script not found"
        exit 1
    fi

    # shellcheck source=/dev/null
    if source "$activate_script" 2>/dev/null; then
        print_success "Activated virtual environment"
    else
        print_warning "Failed to activate virtual environment, using direct paths"
        export PATH="$INSTALL_DIR/venv/bin:$PATH"
        export VIRTUAL_ENV="$INSTALL_DIR/venv"
    fi

    # Verify Python in virtual environment
    if ! python --version &> /dev/null; then
        print_error "Python not available in virtual environment"
        exit 1
    fi

    # Upgrade pip with retry
    print_info "Upgrading pip..."
    local pip_retry=$MAX_RETRIES
    while [ $pip_retry -gt 0 ]; do
        if pip install --upgrade pip --no-warn-script-location 2>/dev/null || \
           python -m pip install --upgrade pip --no-warn-script-location 2>/dev/null; then
            print_success "Upgraded pip"
            break
        else
            pip_retry=$((pip_retry - 1))
            if [ $pip_retry -eq 0 ]; then
                print_warning "Failed to upgrade pip after $MAX_RETRIES attempts, continuing..."
            else
                print_warning "Pip upgrade failed, retrying... ($pip_retry attempts left)"
                sleep $RETRY_DELAY
            fi
        fi
    done

    # Install wheel for faster installations
    pip install wheel 2>/dev/null || true

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
    mkdir -p "$BIN_DIR" || {
        print_warning "Could not create $BIN_DIR, trying alternative location"
        BIN_DIR="$HOME/bin"
        mkdir -p "$BIN_DIR"
    }

    # Create wrapper script
    cat > "$BIN_DIR/enterprise-agent" <<EOF
#!/bin/bash
# Enterprise Agent wrapper script v3.4

AGENT_DIR="\${ENTERPRISE_AGENT_HOME:-$HOME/.enterprise-agent}"

# Check if installation exists
if [ ! -d "\$AGENT_DIR" ]; then
    echo "Enterprise Agent not found at \$AGENT_DIR"
    echo "Run installation script: curl -sSL https://raw.githubusercontent.com/ernijsansons/Enterprise-Agent-/main/install.sh | bash"
    exit 1
fi

# Activate virtual environment and run CLI
if [ -f "\$AGENT_DIR/venv/bin/activate" ]; then
    source "\$AGENT_DIR/venv/bin/activate"
elif [ -f "\$AGENT_DIR/venv/Scripts/activate" ]; then
    source "\$AGENT_DIR/venv/Scripts/activate"
else
    export PATH="\$AGENT_DIR/venv/bin:\$PATH"
fi

if [ -f "\$AGENT_DIR/enterprise_agent_cli.py" ]; then
    python "\$AGENT_DIR/enterprise_agent_cli.py" "\$@"
elif [ -f "\$AGENT_DIR/src/agent_orchestrator.py" ]; then
    python -m src.agent_orchestrator "\$@"
else
    echo "Error: CLI entry point not found"
    exit 1
fi

deactivate 2>/dev/null || true
EOF

    chmod +x "$BIN_DIR/enterprise-agent" || {
        print_error "Failed to make wrapper executable"
        exit 1
    }

    # Create short alias
    ln -sf "$BIN_DIR/enterprise-agent" "$BIN_DIR/ea" 2>/dev/null || \
        print_warning "Could not create 'ea' alias"

    print_success "Created executable: $BIN_DIR/enterprise-agent"

    # Test the wrapper
    if "$BIN_DIR/enterprise-agent" --version &>/dev/null 2>&1; then
        print_success "Wrapper script verified"
    else
        print_warning "Wrapper script created but could not be verified"
    fi
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

    echo "This will install Enterprise Agent v3.4"
    echo "Installation directory: $INSTALL_DIR"
    echo "Binary directory: $BIN_DIR"
    echo "Log file: $LOG_FILE"
    echo ""

    # Check if running as root
    if [ "$EUID" -eq 0 ]; then
        print_warning "Running as root is not recommended"
        read -p "Continue anyway? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 0
        fi
    fi

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

    # Clean up backup if successful
    if [ -d "$INSTALL_DIR.install_backup" ]; then
        rm -rf "$INSTALL_DIR.install_backup"
    fi

    # Success message
    echo ""
    print_success "Installation complete!"
    echo ""
    echo "${GREEN}═══════════════════════════════════════════${NC}"
    echo "${GREEN}   Enterprise Agent v3.4 Ready!${NC}"
    echo "${GREEN}═══════════════════════════════════════════${NC}"
    echo ""
    echo "Quick Start:"
    echo "  1. Reload your shell or run: ${YELLOW}source ~/.bashrc${NC}"
    echo "  2. Test installation: ${YELLOW}enterprise-agent --version${NC}"
    echo "  3. Initialize in a project: ${YELLOW}enterprise-agent init${NC}"
    echo "  4. Run agent: ${YELLOW}enterprise-agent run --input \"Your prompt\"${NC}"
    echo "  5. Interactive mode: ${YELLOW}enterprise-agent interactive${NC}"
    echo ""
    echo "For Claude Code (zero API costs with Max subscription):"
    echo "  1. Install: ${YELLOW}npm install -g @anthropic-ai/claude-code${NC}"
    echo "  2. Login: ${YELLOW}claude login${NC}"
    echo "  3. Update config: ${YELLOW}~/.enterprise-agent/config.yml${NC}"
    echo "     Set: ${YELLOW}use_claude_code: true${NC}"
    echo ""
    echo "Documentation: https://github.com/ernijsansons/Enterprise-Agent-"
    echo "Report issues: https://github.com/ernijsansons/Enterprise-Agent-/issues"
    echo ""
    print_success "Happy coding with Enterprise Agent! 🚀"
    echo "Installation log saved to: $LOG_FILE"
}

# Run main
main "$@"