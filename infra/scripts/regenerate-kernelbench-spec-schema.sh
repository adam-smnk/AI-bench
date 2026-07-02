#!/usr/bin/env bash
#
# Regenerate KernelBench spec schema.

SCRIPTS_DIR=$(realpath $(dirname $0))
REPO_ROOT_DIR=$(git rev-parse --show-toplevel)

AI_BENCH_UV=${HOME}/.local/bin/uv
SCHEMA_FILE=${REPO_ROOT_DIR}/problems/specs/kernelbench-spec.schema.json

echo "Regenerating KernelBench spec schema..."
echo "  - Schema file: ${SCHEMA_FILE}"
${AI_BENCH_UV} run python -m ai_bench.harness.core.schema ${SCHEMA_FILE}

EXIT_CODE=$?
exit ${EXIT_CODE}
