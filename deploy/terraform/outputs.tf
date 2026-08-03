output "alb_dns" {
  value = aws_lb.agent_reflex.dns_name
}

output "db_address" {
  value = aws_db_instance.agent_reflex.address
}

output "ecs_cluster" {
  value = aws_ecs_cluster.agent_reflex.name
}
