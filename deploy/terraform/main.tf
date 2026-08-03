# AgentReflex on AWS: ECS Fargate + RDS Postgres.
#
# Secrets never appear in this file: LLM key, DB password, Neo4j password,
# and the API key for the collector/Grafana are stored in SSM Parameter
# Store and referenced by name (created with aws ssm put-parameter).
#
#   terraform init && terraform plan && terraform apply
#
# Image: ghcr.io/<org>/<repo>:<sha> built by .github/workflows/cd.yml.

terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

variable "region" {
  default = "us-east-1"
}

variable "image" {
  description = "Container image (e.g. ghcr.io/org/agent_reflex:sha-xxxx)"
}

variable "ssm_llm_api_key" {
  description = "SSM parameter name holding the LLM API key"
  default     = "/agent_reflex/llm_api_key"
}

variable "ssm_db_password" {
  description = "SSM parameter name holding the Postgres password"
  default     = "/agent_reflex/db_password"
}

variable "ssm_neo4j_pass" {
  description = "SSM parameter name holding the Neo4j password"
  default     = "/agent_reflex/neo4j_pass"
}

variable "ssm_api_key" {
  description = "SSM parameter name holding the API key used by the OTel collector/Grafana"
  default     = "/agent_reflex/api_key"
}

variable "db_name" {
  default = "agent_reflex"
}

variable "db_user" {
  default = "reflex"
}

provider "aws" {
  region = var.region
}

data "aws_vpc" "default" {}

data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
}

# ---------------------------------------------------------------------------
# Postgres (RDS)
# ---------------------------------------------------------------------------

resource "aws_db_instance" "agent_reflex" {
  identifier     = "agent-reflex"
  engine         = "postgres"
  engine_version = "16.3"
  instance_class = "db.t4g.micro"
  allocated_storage = 20
  db_name  = var.db_name
  username = var.db_user
  password = data.aws_ssm_parameter.db_password.value

  skip_final_snapshot = true
  vpc_security_group_ids = [aws_security_group.postgres.id]
  db_subnet_group_name   = aws_db_subnet_group.agent_reflex.name
}

resource "aws_db_subnet_group" "agent_reflex" {
  name       = "agent-reflex"
  subnet_ids = data.aws_subnets.default.ids
}

# ---------------------------------------------------------------------------
# Networking
# ---------------------------------------------------------------------------

resource "aws_security_group" "postgres" {
  name        = "agent-reflex-postgres"
  description = "Postgres reachable from the ECS tasks only"
  vpc_id      = data.aws_vpc.default.id
}

resource "aws_security_group" "alb" {
  name        = "agent-reflex-alb"
  description = "Public HTTPS/HTTP access to the API"
  vpc_id      = data.aws_vpc.default.id

  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_security_group" "app" {
  name        = "agent-reflex-app"
  description = "App receives traffic from the ALB"
  vpc_id      = data.aws_vpc.default.id

  ingress {
    from_port       = 8000
    to_port         = 8000
    protocol        = "tcp"
    security_groups = [aws_security_group.alb.id]
  }
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_security_group_rule" "postgres_from_app" {
  type                     = "ingress"
  from_port                = 5432
  to_port                  = 5432
  protocol                 = "tcp"
  security_group_id        = aws_security_group.postgres.id
  source_security_group_id = aws_security_group.app.id
}

# ---------------------------------------------------------------------------
# ECS
# ---------------------------------------------------------------------------

resource "aws_ecs_cluster" "agent_reflex" {
  name = "agent-reflex"
}

resource "aws_ecs_task_definition" "agent_reflex" {
  family                   = "agent-reflex"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = 512
  memory                   = 1024
  execution_role_arn       = aws_iam_role.ecs_execution.arn
  task_role_arn            = aws_iam_role.ecs_task.arn

  container_definitions = jsonencode([
    {
      name      = "app"
      image     = var.image
      essential = true
      portMappings = [{ containerPort = 8000, protocol = "tcp" }]
      environment = [
        { name = "AGENT_REFLEX_ENV", value = "production" },
        { name = "AGENT_REFLEX_DB_URL", value = "postgresql://${var.db_user}:${data.aws_ssm_parameter.db_password.value}@${aws_db_instance.agent_reflex.address}:5432/${var.db_name}" },
        { name = "AGENT_REFLEX_OTEL_ENDPOINT", value = "http://localhost:4318" },
        { name = "AGENT_REFLEX_NEO4J_URI", value = "" },
        { name = "AGENT_REFLEX_NEO4J_USER", value = "neo4j" },
        { name = "AGENT_REFLEX_NEO4J_PASS", value = data.aws_ssm_parameter.neo4j_pass.value },
      ]
      secrets = [
        { name = "AGENT_REFLEX_LLM_API_KEY", valueFrom = var.ssm_llm_api_key },
      ]
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = "/ecs/agent-reflex"
          "awslogs-region"        = var.region
          "awslogs-stream-prefix" = "app"
        }
      }
      healthCheck = {
        command     = ["CMD-SHELL", "python -c \"import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=5)\""]
        interval    = 30
        timeout     = 10
        retries     = 3
        startPeriod = 60
      }
    }
  ])
}

resource "aws_ecs_service" "agent_reflex" {
  name            = "agent-reflex"
  cluster         = aws_ecs_cluster.agent_reflex.id
  task_definition = aws_ecs_task_definition.agent_reflex.arn
  desired_count   = 1
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = data.aws_subnets.default.ids
    security_groups  = [aws_security_group.app.id]
    assign_public_ip = true
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.agent_reflex.arn
    container_name   = "app"
    container_port   = 8000
  }
}

# ---------------------------------------------------------------------------
# Load balancer
# ---------------------------------------------------------------------------

resource "aws_lb" "agent_reflex" {
  name               = "agent-reflex"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb.id]
  subnets            = data.aws_subnets.default.ids
}

resource "aws_lb_target_group" "agent_reflex" {
  name        = "agent-reflex"
  port        = 8000
  protocol    = "HTTP"
  vpc_id      = data.aws_vpc.default.id
  target_type = "ip"

  health_check {
    path                = "/ready"
    interval            = 30
    timeout             = 5
    healthy_threshold   = 2
    unhealthy_threshold = 3
    matcher             = "200"
  }
}

resource "aws_lb_listener" "http" {
  load_balancer_arn = aws_lb.agent_reflex.arn
  port              = 80
  protocol          = "HTTP"

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.agent_reflex.arn
  }
}

# ---------------------------------------------------------------------------
# IAM
# ---------------------------------------------------------------------------

resource "aws_iam_role" "ecs_execution" {
  name = "agent-reflex-ecs-execution"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "ecs-tasks.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy_attachment" "ecs_execution" {
  role       = aws_iam_role.ecs_execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

resource "aws_iam_role" "ecs_task" {
  name = "agent-reflex-ecs-task"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "ecs-tasks.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_policy" "ssm_read" {
  name = "agent-reflex-ssm-read"
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = ["ssm:GetParameters"]
      Resource = ["arn:aws:ssm:${var.region}:*:parameter/agent_reflex/*"]
    }]
  })
}

resource "aws_iam_role_policy_attachment" "ssm_read" {
  role       = aws_iam_role.ecs_task.name
  policy_arn = aws_iam_policy.ssm_read.arn
}
