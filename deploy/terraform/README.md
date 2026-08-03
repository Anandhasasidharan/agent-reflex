# Usage:
#
#   1. Put secrets in SSM Parameter Store (never in git):
#
#      aws ssm put-parameter --name /agent_reflex/llm_api_key \
#          --value "$DEEPSEEK_API_KEY" --type SecureString
#      aws ssm put-parameter --name /agent_reflex/db_password \
#          --value "$POSTGRES_PASSWORD" --type SecureString
#      aws ssm put-parameter --name /agent_reflex/neo4j_pass \
#          --value "$NEO4J_PASSWORD" --type SecureString
#      aws ssm put-parameter --name /agent_reflex/api_key \
#          --value "$AGENT_REFLEX_API_KEY" --type SecureString
#
#   2. Create the API key inside the deployed DB (run once, e.g. from the
#      task via ECS Exec, or against the RDS instance from a bastion):
#
#      python -m agent_reflex.api.auth create deploy --scope=write
#      python -m agent_reflex.api.auth create grafana --scope=read
#
#   3. terraform init && terraform plan -var image=ghcr.io/<org>/<repo>:<sha> \
#      && terraform apply -var image=...
#
# Notes:
#   - The image is built and pushed by .github/workflows/cd.yml (GHCR).
#   - AGENT_REFLEX_API_KEY (for the OTel collector/Grafana) must match a key
#     in the api_keys table; the collector reads it from the same SSM
#     parameter when deployed alongside the app.
#   - Production fail-fast (Settings.validate_production) requires
#     AGENT_REFLEX_ENV=production plus every secret above — the task
#     definition sets them, so a misconfigured deploy fails loudly at start.
#   - Terraform state contains the resolved RDS password; restrict access or
#     move state to a remote backend (s3 + dynamodb lock) for real use.
