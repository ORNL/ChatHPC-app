#!/usr/bin/env bash
set -e

# Test help functions
echo '*** Test -h for scripts ***'

echo ChatHPC -h
ChatHPC -h
echo
echo chathpc -h
chathpc -h

# Test default arguments
echo
echo '*** Test basic script functionality ***'

echo CHATHPC_DATA_FILE="/home/7ry/Data/ellora/kokkos-data/kokkos_create_context.json"\
    CHATHPC_FINETUNED_MODEL_PATH="./app/peft_adapter"\
    CHATHPC_MERGED_MODEL_PATH="./app/merged_adapters"\
    CHATHPC_TRAINING_OUTPUT_DIR="./app/kokkos-code-llama"\
    CHATHPC_TRAINING_PROMPT="You are a powerful LLM model for Kokkos. Your job is to answer questions about Kokkos programming model. You are given a question and context regarding Kokkos programming model.\n\nYou must output the Kokkos question that answers the question.\n\n### Input:\n{question}\n\n### Context:\n{context}\n\n### Response:\n{answer}\n"\
    CHATHPC_INFERENCE_PROMPT="You are a powerful LLM model for Kokkos. Your job is to answer questions about Kokkos programming model. You are given a question and context regarding Kokkos programming model.\n\nYou must output the Kokkos question that answers the question.\n\n### Input:\n{question}\n\n### Context:\n{context}\n\n### Response:\n"\
    chathpc config
CHATHPC_DATA_FILE="/home/7ry/Data/ellora/kokkos-data/kokkos_create_context.json"\
    CHATHPC_FINETUNED_MODEL_PATH="./app/peft_adapter"\
    CHATHPC_MERGED_MODEL_PATH="./app/merged_adapters"\
    CHATHPC_TRAINING_OUTPUT_DIR="./app/kokkos-code-llama"\
    CHATHPC_TRAINING_PROMPT="You are a powerful LLM model for Kokkos. Your job is to answer questions about Kokkos programming model. You are given a question and context regarding Kokkos programming model.\n\nYou must output the Kokkos question that answers the question.\n\n### Input:\n{question}\n\n### Context:\n{context}\n\n### Response:\n{answer}\n"\
    CHATHPC_INFERENCE_PROMPT="You are a powerful LLM model for Kokkos. Your job is to answer questions about Kokkos programming model. You are given a question and context regarding Kokkos programming model.\n\nYou must output the Kokkos question that answers the question.\n\n### Input:\n{question}\n\n### Context:\n{context}\n\n### Response:\n"\
    chathpc config
