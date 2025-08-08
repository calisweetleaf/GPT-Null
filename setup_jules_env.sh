#!/bin/bash

# GPT-Ø Environment Setup Script
# Minimal environment validation for GPT-Ø system
# Author: GPT-Ø Development Team
# Version: 1.0.0

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}                               GPT-Ø Environment Setup                                        ${NC}"
echo -e "${BLUE}                          Self-Modifying Multimodal Transformer                              ${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check Python version
check_python() {
    echo -e "${YELLOW}[INFO]${NC} Checking Python installation..."
    
    if command_exists python3; then
        PYTHON_CMD="python3"
    elif command_exists python; then
        PYTHON_CMD="python"
    else
        echo -e "${RED}[ERROR]${NC} Python not found. Please install Python 3.9+ first."
        exit 1
    fi
    
    # Check Python version
    PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
    PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
    PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)
    
    if [ "$PYTHON_MAJOR" -ge 3 ] && [ "$PYTHON_MINOR" -ge 9 ]; then
        echo -e "${GREEN}[OK]${NC} Python $PYTHON_VERSION found"
    else
        echo -e "${RED}[ERROR]${NC} Python 3.9+ required. Found: $PYTHON_VERSION"
        exit 1
    fi
}

# Function to check system requirements
check_system_requirements() {
    echo -e "${YELLOW}[INFO]${NC} Checking system requirements..."
    
    # Check available RAM
    if command_exists free; then
        TOTAL_RAM=$(free -g | awk '/^Mem:/{print $2}')
        if [ "$TOTAL_RAM" -ge 8 ]; then
            echo -e "${GREEN}[OK]${NC} RAM: ${TOTAL_RAM}GB (meets 8GB minimum)"
        else
            echo -e "${YELLOW}[WARNING]${NC} RAM: ${TOTAL_RAM}GB (below 8GB recommendation)"
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        TOTAL_RAM=$(sysctl -n hw.memsize | awk '{print int($1/1024/1024/1024)}')
        if [ "$TOTAL_RAM" -ge 8 ]; then
            echo -e "${GREEN}[OK]${NC} RAM: ${TOTAL_RAM}GB (meets 8GB minimum)"
        else
            echo -e "${YELLOW}[WARNING]${NC} RAM: ${TOTAL_RAM}GB (below 8GB recommendation)"
        fi
    else
        echo -e "${YELLOW}[INFO]${NC} Cannot detect RAM. Ensure you have at least 8GB RAM."
    fi
    
    # Check disk space
    AVAILABLE_SPACE=$(df -h . | awk 'NR==2 {print $4}' | sed 's/G//')
    if [ "${AVAILABLE_SPACE%.*}" -ge 20 ]; then
        echo -e "${GREEN}[OK]${NC} Disk space: ${AVAILABLE_SPACE}G available"
    else
        echo -e "${YELLOW}[WARNING]${NC} Disk space: ${AVAILABLE_SPACE}G (recommend 20GB+)"
    fi
}

# Function to validate GPT-Ø core files
check_core_files() {
    echo -e "${YELLOW}[INFO]${NC} Validating GPT-Ø core files..."
    
    CORE_FILES=(
        "gpt_model.py"
        "recursive_weights_core.py"
        "bayesian_config_orchestrator.py"
        "tokenizer_adapter.py"
        "tokenizer_mux.py"
        "run.py"
    )
    
    for file in "${CORE_FILES[@]}"; do
        if [ -f "$file" ]; then
            echo -e "${GREEN}[OK]${NC} $file"
        else
            echo -e "${RED}[MISSING]${NC} $file"
        fi
    done
    
    # Check CAS subsystem
    echo -e "${YELLOW}[INFO]${NC} Checking CAS subsystem..."
    CAS_FILES=(
        "cas/neural_memory_runtime.py"
        "cas/neural_model_manager.py"
        "cas/cas_system.py"
        "cas/cas_integration_bridge.py"
        "cas/model_creation.py"
    )
    
    for file in "${CAS_FILES[@]}"; do
        if [ -f "$file" ]; then
            echo -e "${GREEN}[OK]${NC} $file"
        else
            echo -e "${RED}[MISSING]${NC} $file"
        fi
    done
    
    # Check output heads
    echo -e "${YELLOW}[INFO]${NC} Checking specialized output heads..."
    OUTPUT_HEAD_FILES=(
        "extra_output_heads/tool_output_head.py"
        "extra_output_heads/eyes_outputs.py"
        "extra_output_heads/ears_outputs.py"
    )
    
    for file in "${OUTPUT_HEAD_FILES[@]}"; do
        if [ -f "$file" ]; then
            echo -e "${GREEN}[OK]${NC} $file"
        else
            echo -e "${RED}[MISSING]${NC} $file"
        fi
    done
}

# Function to check Python dependencies
check_dependencies() {
    echo -e "${YELLOW}[INFO]${NC} Checking Python dependencies..."
    
    if [ -f "requirements.txt" ]; then
        echo -e "${GREEN}[OK]${NC} requirements.txt found"
        
        # Check if pip is available
        if command_exists pip3; then
            PIP_CMD="pip3"
        elif command_exists pip; then
            PIP_CMD="pip"
        else
            echo -e "${YELLOW}[WARNING]${NC} pip not found. Cannot verify dependencies."
            return
        fi
        
        # Check key dependencies
        KEY_DEPS=("torch" "numpy" "scipy" "cryptography" "pyyaml")
        
        for dep in "${KEY_DEPS[@]}"; do
            if $PIP_CMD show "$dep" >/dev/null 2>&1; then
                echo -e "${GREEN}[OK]${NC} $dep installed"
            else
                echo -e "${YELLOW}[MISSING]${NC} $dep (run: $PIP_CMD install $dep)"
            fi
        done
    else
        echo -e "${YELLOW}[WARNING]${NC} requirements.txt not found"
    fi
}

# Function to check GPU availability (optional)
check_gpu() {
    echo -e "${YELLOW}[INFO]${NC} Checking GPU availability..."
    
    if command_exists nvidia-smi; then
        GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits 2>/dev/null)
        if [ $? -eq 0 ] && [ -n "$GPU_INFO" ]; then
            echo -e "${GREEN}[OK]${NC} NVIDIA GPU detected:"
            echo "$GPU_INFO" | while IFS=, read -r name memory; do
                echo -e "      ${BLUE}→${NC} $name (${memory}MB VRAM)"
            done
        else
            echo -e "${YELLOW}[INFO]${NC} No NVIDIA GPU detected (CPU-only mode)"
        fi
    else
        echo -e "${YELLOW}[INFO]${NC} nvidia-smi not found (CPU-only mode)"
    fi
}

# Function to create necessary directories
create_directories() {
    echo -e "${YELLOW}[INFO]${NC} Creating necessary directories..."
    
    DIRS=(
        "logs"
        "model_state"
        "neural_memory"
        "cas_runtime"
        "tool_outputs"
    )
    
    for dir in "${DIRS[@]}"; do
        if [ ! -d "$dir" ]; then
            mkdir -p "$dir"
            echo -e "${GREEN}[CREATED]${NC} $dir/"
        else
            echo -e "${GREEN}[EXISTS]${NC} $dir/"
        fi
    done
}

# Function to validate configuration files
check_config() {
    echo -e "${YELLOW}[INFO]${NC} Checking configuration files..."
    
    if [ -f "config/cas_specification.yaml" ]; then
        echo -e "${GREEN}[OK]${NC} CAS specification found"
    else
        echo -e "${YELLOW}[INFO]${NC} CAS specification not found (will use defaults)"
    fi
    
    if [ -f "JULES.md" ]; then
        echo -e "${GREEN}[OK]${NC} JULES context file found"
    else
        echo -e "${YELLOW}[INFO]${NC} JULES context file not found"
    fi
}

# Main execution
main() {
    echo -e "${BLUE}Starting GPT-Ø environment validation...${NC}"
    echo ""
    
    check_python
    echo ""
    
    check_system_requirements
    echo ""
    
    check_core_files
    echo ""
    
    check_dependencies
    echo ""
    
    check_gpu
    echo ""
    
    create_directories
    echo ""
    
    check_config
    echo ""
    
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}                              Environment Validation Complete                               ${NC}"
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    echo -e "${BLUE}Next steps:${NC}"
    echo -e "  1. Install missing dependencies: ${YELLOW}pip install -r requirements.txt${NC}"
    echo -e "  2. Start GPT-Ø system: ${YELLOW}python run.py --mode chat${NC}"
    echo -e "  3. Validate system: ${YELLOW}python run.py --validate-system${NC}"
    echo ""
    echo -e "${BLUE}GPT-Ø is ready for interaction-based evolution!${NC}"
}

# Run main function
main "$@"
