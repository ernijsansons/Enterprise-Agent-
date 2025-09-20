#!/usr/bin/env node
/**
 * Post-install script for Enterprise Agent
 * Sets up Python dependencies and Claude Code
 */

const { exec, spawn } = require('child_process');
const path = require('path');
const fs = require('fs');

console.log('\n🚀 Setting up Enterprise Agent...\n');

function runCommand(command, args = []) {
    return new Promise((resolve, reject) => {
        const proc = spawn(command, args, {
            stdio: 'inherit',
            shell: true
        });

        proc.on('close', (code) => {
            if (code === 0) {
                resolve();
            } else {
                reject(new Error(`Command failed: ${command} ${args.join(' ')}`));
            }
        });

        proc.on('error', reject);
    });
}

async function checkPython() {
    const pythonCmd = process.platform === 'win32' ? 'python' : 'python3';

    try {
        await runCommand(pythonCmd, ['--version']);
        return pythonCmd;
    } catch {
        return null;
    }
}

async function installPythonDeps(pythonCmd) {
    console.log('📦 Installing Python dependencies...');

    const requirementsPath = path.join(__dirname, '..', 'requirements.txt');

    // Create requirements.txt if it doesn't exist
    if (!fs.existsSync(requirementsPath)) {
        const requirements = `# Enterprise Agent Requirements
pyyaml>=6.0.1
python-dotenv>=1.0.1
requests>=2.31.0
pytest>=7.4.4
anthropic>=0.20.0
openai>=1.2.0
google-generativeai>=0.7.0
langgraph>=0.0.28
networkx>=3.3
pinecone-client>=3.2.2
`;
        fs.writeFileSync(requirementsPath, requirements);
    }

    try {
        await runCommand(pythonCmd, ['-m', 'pip', 'install', '-r', requirementsPath]);
        console.log('✅ Python dependencies installed\n');
    } catch (err) {
        console.warn('⚠️  Could not install Python dependencies');
        console.warn('   Run manually: pip install -r requirements.txt\n');
    }
}

async function checkClaudeCode() {
    console.log('🔍 Checking Claude Code CLI...');

    try {
        await runCommand('claude', ['--version']);
        console.log('✅ Claude Code CLI is installed\n');
        return true;
    } catch {
        console.log('ℹ️  Claude Code CLI not found');
        console.log('   Install for zero-cost Claude usage:');
        console.log('   npm install -g @anthropic-ai/claude-code\n');
        return false;
    }
}

async function createGlobalConfig() {
    const homeDir = process.env.HOME || process.env.USERPROFILE;
    const configDir = path.join(homeDir, '.enterprise-agent');
    const configFile = path.join(configDir, 'config.yml');

    if (!fs.existsSync(configDir)) {
        fs.mkdirSync(configDir, { recursive: true });
    }

    if (!fs.existsSync(configFile)) {
        const defaultConfig = `# Enterprise Agent Global Configuration
# Created by postinstall script

# Use Claude Code CLI for zero API costs (requires Max subscription)
use_claude_code: false  # Set to true after running 'claude login'

# Default models
models:
  primary: claude-3-5-sonnet-20241022
  fallback: gpt-4o-mini

# Default domain
default_domain: coding

# Cache settings
cache:
  enabled: true
  ttl: 3600
`;

        fs.writeFileSync(configFile, defaultConfig);
        console.log(`✅ Created global config: ${configFile}\n`);
    }
}

async function main() {
    try {
        // Check Python
        const pythonCmd = await checkPython();
        if (!pythonCmd) {
            console.error('❌ Python 3.10+ is required');
            console.error('   Download from: https://python.org\n');
            process.exit(1);
        }

        // Install Python dependencies
        await installPythonDeps(pythonCmd);

        // Check Claude Code
        const hasClaudeCode = await checkClaudeCode();

        // Create global config
        await createGlobalConfig();

        // Success message
        console.log('✅ Enterprise Agent setup complete!\n');
        console.log('Quick Start:');
        console.log('  1. Initialize in your project: enterprise-agent init');
        console.log('  2. Run agent: enterprise-agent run --input "Your prompt"');
        console.log('  3. Interactive mode: enterprise-agent interactive\n');

        if (!hasClaudeCode) {
            console.log('💡 Tip: Install Claude Code for zero API costs:');
            console.log('   npm install -g @anthropic-ai/claude-code');
            console.log('   claude login\n');
        }

        console.log('📚 Full docs: https://github.com/yourorg/enterprise-agent\n');

    } catch (err) {
        console.error('❌ Setup failed:', err.message);
        console.error('\nPlease complete setup manually:');
        console.error('  1. Install Python 3.10+');
        console.error('  2. Run: pip install -r requirements.txt');
        console.error('  3. (Optional) Install Claude Code CLI');
        process.exit(1);
    }
}

main();