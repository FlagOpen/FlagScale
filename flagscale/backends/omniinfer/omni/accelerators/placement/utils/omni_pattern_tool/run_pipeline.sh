#!/bin/bash
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2012-2025. All rights reserved.

# run_pipeline.sh
# Shell script to run pipeline.py with specified or default parameters

# Default parameters
INPUT_LOG_FILES=("./dump_to_log-1.log" "./dump_to_log-2.log")  # Default to multiple log files, can be overridden
INPUT_TXT_FOLDER="./decode"
INPUT_MODE="txt"  # Default to txt mode
TOPK_ID_COUNT_DIR="${TOPK_ID_COUNT_DIR:-./topk_id_count}"
PLACEMENT_PATTERN_DIR="${PLACEMENT_PATTERN_DIR:-./placement_pattern}"
PLACEMENT_PATTERN_VIEW_DIR="${PLACEMENT_PATTERN_VIEW_DIR:-./placement_pattern_view}"
PLACEMENT_PATTERN_ANALYSIS_DIR="${PLACEMENT_PATTERN_ANALYSIS_DIR:-./placement_pattern_analysis}"
OUTPUT_CSV=""
NUM_LAYERS=58
NUM_RANKS_OF_COLLECTING_DATA=32
NUM_POSITIONS_OF_ROUTED_EXPERTS=256
NUM_RANKS_TARGET_PATTERN=64
NUM_REDUNDANT_LAYERS="0 10 58"
EXPERT_REDUNDANT_LIMIT=11
NUM_LAYERS_TARGET_PATTERN=58
NUM_EPS_TARGET_PATTERN=256
DATASET_NAME="longbench_3.5k_decode_0524"
OUTPUT_FILE_PREFIX="DSV3_0524_longbench_3.5k_decode"
PATTERN_MODE="all"
COLLECTING_MODES="decode"

# Function to display usage
usage() {
    echo "Usage: $0 [options]"
    echo "Note: Parameters containing spaces (e.g., filenames) must be enclosed in quotes, e.g., --output_csv \"my output.csv\""
    echo "Options:"
    echo "  --input_log_files <log_file1> [<log_file2> ...]  Input log files (for log mode)"
    echo "  --input_txt_folder <folder>                     Input text folder (for txt mode)"
    echo "  --input_mode <log|txt>                          Input mode (default: txt)"
    echo "  --topk_id_count_dir <dir>                       Directory for topk ID count"
    echo "  --placement_pattern_dir <dir>                   Directory for placement patterns"
    echo "  --placement_pattern_view_dir <dir>              Directory for placement pattern views"
    echo "  --placement_pattern_analysis_dir <dir>          Directory for load analysis"
    echo "  --output_csv <output_csv>                       Output CSV file path"
    echo "  --num_layers <num_layers>                       Number of layers"
    echo "  --num_ranks_of_collecting_data <num_ranks>      Number of ranks for data collection"
    echo "  --num_positions_of_routed_experts <num_positions> Number of positions for routed experts"
    echo "  --num_ranks_target_pattern <num_ranks>          Number of ranks for target pattern"
    echo "  --num_redundant_layers <num_redundant_layers>   Number of redundant layers"
    echo "  --expert_redundant_limit <limit>                Expert redundant limit"
    echo "  --num_layers_target_pattern <num_layers>        Number of layers for target pattern"
    echo "  --num_eps_target_pattern <num_eps>              Number of experts for target pattern"
    echo "  --dataset_name <dataset_name>                   Dataset name"
    echo "  --output_file_prefix <prefix>                   Output file prefix"
    echo "  --pattern_mode <rearrange|redundant|all>        Pattern generation mode"
    echo "  --collecting_modes <prefill|decode|all>         Data collecting modes"
    echo "  -h, --help                                      Display this help message"
    echo "Example: $0 --input_log_files \"log-1.log\" \"log-2.log\" --input_mode log --num_ranks_of_collecting_data 64"
    echo "Example: $0 --input_txt_folder \"./activation_datas\" --input_mode txt --pattern_mode all"
    exit 1
}

# Function to parse command-line arguments using GNU getopt
parse_arguments() {
    TEMP=$(getopt -o h --long input_log_files:,input_txt_folder:,input_mode:,topk_id_count_dir:,placement_pattern_dir:,placement_pattern_view_dir:,placement_pattern_analysis_dir:,output_csv:,num_layers:,num_ranks_of_collecting_data:,num_positions_of_routed_experts:,num_ranks_target_pattern:,num_redundant_layers:,expert_redundant_limit:,num_layers_target_pattern:,num_eps_target_pattern:,dataset_name:,output_file_prefix:,pattern_mode:,collecting_modes:,help -n "$0" -- "$@")
    if [ $? != 0 ]; then echo "Error: Failed to parse arguments!" >&2; exit 1; fi
    eval set -- "$TEMP"

    # Only reset INPUT_LOG_FILES if --input_log_files is provided
    local input_log_files_provided=false
    while true; do
        case "$1" in
            --input_log_files)
                input_log_files_provided=true
                INPUT_LOG_FILES=()  # Reset only when --input_log_files is specified
                INPUT_LOG_FILES+=("$2")
                shift 2
                while [[ "$1" != "--"* && -n "$1" ]]; do
                    INPUT_LOG_FILES+=("$1")
                    shift
                done
                ;;
            --input_txt_folder) INPUT_TXT_FOLDER="$2"; shift 2 ;;
            --input_mode) INPUT_MODE="$2"; shift 2 ;;
            --topk_id_count_dir) TOPK_ID_COUNT_DIR="$2"; shift 2 ;;
            --placement_pattern_dir) PLACEMENT_PATTERN_DIR="$2"; shift 2 ;;
            --placement_pattern_view_dir) PLACEMENT_PATTERN_VIEW_DIR="$2"; shift 2 ;;
            --placement_pattern_analysis_dir) PLACEMENT_PATTERN_ANALYSIS_DIR="$2"; shift 2 ;;
            --output_csv) OUTPUT_CSV="$2"; shift 2 ;;
            --num_layers) NUM_LAYERS="$2"; shift 2 ;;
            --num_ranks_of_collecting_data) NUM_RANKS_OF_COLLECTING_DATA="$2"; shift 2 ;;
            --num_positions_of_routed_experts) NUM_POSITIONS_OF_ROUTED_EXPERTS="$2"; shift 2 ;;
            --num_ranks_target_pattern) NUM_RANKS_TARGET_PATTERN="$2"; shift 2 ;;
            --num_redundant_layers) NUM_REDUNDANT_LAYERS="$2"; shift 2 ;;
            --expert_redundant_limit) EXPERT_REDUNDANT_LIMIT="$2"; shift 2 ;;
            --num_layers_target_pattern) NUM_LAYERS_TARGET_PATTERN="$2"; shift 2 ;;
            --num_eps_target_pattern) NUM_EPS_TARGET_PATTERN="$2"; shift 2 ;;
            --dataset_name) DATASET_NAME="$2"; shift 2 ;;
            --output_file_prefix) OUTPUT_FILE_PREFIX="$2"; shift 2 ;;
            --pattern_mode) PATTERN_MODE="$2"; shift 2 ;;
            --collecting_modes) COLLECTING_MODES="$2"; shift 2 ;;
            -h|--help) usage ;;
            --) shift; break ;;
            *) echo "Error: Unknown parameter: $1" >&2; usage ;;
        esac
    done

    # Debug output (optional, uncomment to enable)
    # echo "Debug: Parsed INPUT_LOG_FILES: ${INPUT_LOG_FILES[*]}" >&2
}

# Function to validate input parameters
validate_inputs() {
    # Validate input_mode
    if [[ "$INPUT_MODE" != "log" && "$INPUT_MODE" != "txt" ]]; then
        echo "Error: input_mode must be 'log' or 'txt'." >&2
        exit 1
    fi

    # Validate inputs based on input_mode
    if [[ "$INPUT_MODE" == "log" ]]; then
        if [ ${#INPUT_LOG_FILES[@]} -eq 0 ]; then
            echo "Error: At least one log file must be provided when input_mode='log'." >&2
            exit 1
        fi
        for log_file in "${INPUT_LOG_FILES[@]}"; do
            if [[ ! -f "$log_file" ]]; then
                echo "Error: Log file '$log_file' does not exist." >&2
                exit 1
            fi
        done
    else
        if [[ -z "$INPUT_TXT_FOLDER" || ! -d "$INPUT_TXT_FOLDER" ]]; then
            echo "Error: input_txt_folder must be a valid directory when input_mode='txt'." >&2
            exit 1
        fi
    fi

    # Validate pattern_mode
    if [[ "$PATTERN_MODE" != "rearrange" && "$PATTERN_MODE" != "redundant" && "$PATTERN_MODE" != "all" ]]; then
        echo "Error: pattern_mode must be 'rearrange', 'redundant', or 'all'." >&2
        exit 1
    fi

    # Validate collecting_modes
    if [[ "$COLLECTING_MODES" != "prefill" && "$COLLECTING_MODES" != "decode" && "$COLLECTING_MODES" != "all" ]]; then
        echo "Error: collecting_modes must be 'prefill', 'decode', or 'all'." >&2
        exit 1
    fi

    # Validate num_ranks_of_collecting_data
    if ! [[ "$NUM_RANKS_OF_COLLECTING_DATA" =~ ^[0-9]+$ ]] || [ "$NUM_RANKS_OF_COLLECTING_DATA" -le 0 ]; then
        echo "Error: num_ranks_of_collecting_data must be a positive integer." >&2
        exit 1
    fi
}

# Function to check dependencies
check_dependencies() {
    # Check Python availability
    if ! command -v python &> /dev/null; then
        echo "Error: Python not found or not installed." >&2
        exit 1
    fi

    # Check required Python scripts
    local required_scripts=(
        "pipeline.py"
        "step_1_generate_csv_with_ceiling.py"
        "step_2_placement_pattern_generation.py"
        "step_3_placement_pattern_checking_and_plot.py"
        "step_4_load_analysis_and_plot.py"
    )

    for script in "${required_scripts[@]}"; do
        if [[ ! -f "$script" ]]; then
            echo "Error: Required script '$script' is missing." >&2
            exit 1
        fi
    done
}

# Function to print parameters
print_parameters() {
    echo "Running pipeline.py with the following parameters:"
    if [[ "$INPUT_MODE" == "log" ]]; then
        echo "Log files: ${INPUT_LOG_FILES[*]}"
    else
        echo "Text folder: $INPUT_TXT_FOLDER"
    fi
    echo "Input mode: $INPUT_MODE"
    echo "Topk ID count directory: $TOPK_ID_COUNT_DIR"
    echo "Placement pattern directory: $PLACEMENT_PATTERN_DIR"
    echo "Placement pattern view directory: $PLACEMENT_PATTERN_VIEW_DIR"
    echo "Load analysis directory: $PLACEMENT_PATTERN_ANALYSIS_DIR"
    echo "Output CSV: ${OUTPUT_CSV:-'default timestamp naming'}"
    echo "Number of layers: $NUM_LAYERS"
    echo "Number of ranks for data collection: $NUM_RANKS_OF_COLLECTING_DATA"
    echo "Number of positions for routed experts: $NUM_POSITIONS_OF_ROUTED_EXPERTS"
    echo "Number of ranks for target pattern: $NUM_RANKS_TARGET_PATTERN"
    echo "Number of redundant layers: $NUM_REDUNDANT_LAYERS"
    echo "Expert redundant limit: $EXPERT_REDUNDANT_LIMIT"
    echo "Number of layers for target pattern: $NUM_LAYERS_TARGET_PATTERN"
    echo "Number of experts for target pattern: $NUM_EPS_TARGET_PATTERN"
    echo "Dataset name: $DATASET_NAME"
    echo "Output file prefix: $OUTPUT_FILE_PREFIX"
    echo "Pattern mode: $PATTERN_MODE"
    echo "Data collecting modes: $COLLECTING_MODES"
}

# Function to run the pipeline
run_pipeline() {
    python pipeline.py \
        --input_log_files "${INPUT_LOG_FILES[@]}" \
        --input_txt_folder "$INPUT_TXT_FOLDER" \
        --input_mode "$INPUT_MODE" \
        --topk_id_count_dir "$TOPK_ID_COUNT_DIR" \
        --placement_pattern_dir "$PLACEMENT_PATTERN_DIR" \
        --placement_pattern_view_dir "$PLACEMENT_PATTERN_VIEW_DIR" \
        --placement_pattern_analysis_dir "$PLACEMENT_PATTERN_ANALYSIS_DIR" \
        --output_csv "$OUTPUT_CSV" \
        --num_layers "$NUM_LAYERS" \
        --num_ranks_of_collecting_data "$NUM_RANKS_OF_COLLECTING_DATA" \
        --num_positions_of_routed_experts "$NUM_POSITIONS_OF_ROUTED_EXPERTS" \
        --num_ranks_target_pattern "$NUM_RANKS_TARGET_PATTERN" \
        --num_redundant_layers $NUM_REDUNDANT_LAYERS \
        --expert_redundant_limit "$EXPERT_REDUNDANT_LIMIT" \
        --num_layers_target_pattern "$NUM_LAYERS_TARGET_PATTERN" \
        --num_eps_target_pattern "$NUM_EPS_TARGET_PATTERN" \
        --dataset_name "$DATASET_NAME" \
        --output_file_prefix "$OUTPUT_FILE_PREFIX" \
        --pattern_mode "$PATTERN_MODE" \
        --collecting_modes "$COLLECTING_MODES"

    if [[ $? -eq 0 ]]; then
        echo "Pipeline executed successfully."
    else
        echo "Error: Pipeline execution failed. Please check the output logs." >&2
        exit 1
    fi
}

# Main function
main() {
    parse_arguments "$@"
    validate_inputs
    check_dependencies
    print_parameters
    run_pipeline
}

# Execute main function
main "$@"